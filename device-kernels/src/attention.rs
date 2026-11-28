//! Attention glue: query/gate split, RoPE in its NEOX and interleaved
//! multimodal forms, KV-cache appends and the decode combine.

use crate::common::{f2h, kvq8_quantize};
use cuda_device::{SharedArray, device, kernel, thread, warp};

/// Production attention RoPE, using two token groups on underfilled grids.
///
/// # Safety
/// input/output are disjoint and cover tokens * heads * head_dim floats.
/// position covers one i32; grid is (ceil(tokens/16), heads), blockDim.x
/// equals head_dim, and rotated is even and between zero and head_dim.
#[kernel]
pub unsafe fn attn_rope_partial_neox(
    input: *const f32,
    output: *mut f32,
    heads: i32,
    head_dim: i32,
    rotated: i32,
    position: *const i32,
    theta: f32,
    tokens: i32,
) {
    let t0 = thread::blockIdx_x() as i32 * 16;
    // One loop bound, also valid for a partial tile at i32::MAX.
    let end = t0 + (tokens - t0).min(16);
    let h = thread::blockIdx_y() as usize;
    let d = thread::threadIdx_x() as i32;
    unsafe {
        if d >= rotated {
            let mut t = t0;
            while t < end {
                let base = (t as usize * heads as usize + h) * head_dim as usize;
                *output.add(base + d as usize) = *input.add(base + d as usize);
                t += 1;
            }
            return;
        }
        let half = rotated / 2;
        // On the 72-SM target, a grid with at most one block per SM
        // needs more active rotary warps. Both halves then own alternating
        // tokens. Larger grids retain one frequency calculation per pair.
        let split_tokens = thread::gridDim_x() as u64 * thread::gridDim_y() as u64 <= 72;
        let stripe = if d < half { 0 } else { 1 };
        if stripe == 1 && !split_tokens {
            return;
        }
        let step = if split_tokens { 2 } else { 1 };
        let pair = if d < half { d } else { d - half };
        let mut u = stripe;
        let count = end - t0;
        if u >= count {
            return;
        }
        let freq = (theta as f64).powf(-2.0 * pair as f64 / rotated as f64);
        let pos = *position as f64;
        while u < count {
            let t = t0 + u;
            let base = (t as usize * heads as usize + h) * head_dim as usize;
            let angle = (pos + t as f64) * freq;
            let sin_a = angle.sin() as f32;
            let cos_a = angle.cos() as f32;
            let x0 = *input.add(base + pair as usize);
            let x1 = *input.add(base + (pair + half) as usize);
            *output.add(base + pair as usize) = x0.mul_add(cos_a, -(x1 * sin_a));
            *output.add(base + (pair + half) as usize) = x0.mul_add(sin_a, x1 * cos_a);
            u += step;
        }
    }
}

/// De-interleave the packed query/gate projection, one head per block.
///
/// `blk.N.attn_q.weight` emits, per token, `[q_h0, gate_h0, q_h1,
/// gate_h1, ...]`: a per-head stride of 2*head_dim with the query first,
/// as llama.cpp's `src/models/qwen35moe.cpp` views it. It is NOT
/// `[all queries | all gates]`: reading it that way feeds query heads
/// 8..15 the gates of heads 0..7, stays finite, and is a different model.
///
/// # Safety
/// packed covers gridDim.x * q_heads * 2 * head_dim floats; q and gate
/// each cover gridDim.x * q_heads * head_dim and are disjoint from packed
/// and each other. gridDim.y == q_heads and blockDim.x == head_dim.
#[kernel]
pub unsafe fn attn_split_query_gate(
    packed: *const f32,
    q: *mut f32,
    gate: *mut f32,
    q_heads: i32,
    head_dim: i32,
) {
    let t = thread::blockIdx_x() as i64;
    let h = thread::blockIdx_y() as i64;
    let d = thread::threadIdx_x() as i64;
    let row = t * q_heads as i64 * 2 * head_dim as i64;
    let dst = (t * q_heads as i64 + h) * head_dim as i64 + d;
    unsafe {
        *q.add(dst as usize) = *packed.add((row + 2 * h * head_dim as i64 + d) as usize);
        *gate.add(dst as usize) = *packed.add((row + (2 * h + 1) * head_dim as i64 + d) as usize);
    }
}

/// Row positions `position[0] + i`, on device: a host position cannot
/// be baked into a captured launch.
///
/// # Safety
/// position covers one int, out covers n, and the 1-D launch is nonempty.
#[kernel]
pub unsafe fn attn_row_positions(position: *const i32, out: *mut i32, n: i32) {
    let i = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as i32;
    if i < n {
        unsafe {
            *out.add(i as usize) = (*position).wrapping_add(i);
        }
    }
}

/// Merge the per-split partial softmaxes of one sequence's decode
/// attention into this block's head and this thread's dimension.
///
/// Each split carries its running maximum `m`, its normalizer `l` and its
/// weighted value sum, all relative to its own maximum; they are rescaled
/// to the global maximum and summed in ascending split order. Split 0
/// always holds at least key 0, so the global maximum is never -inf and
/// the subtraction never forms inf - inf; a split with no keys has
/// `m = -inf` and contributes nothing.
///
/// A split's rescale `exp(m - max)` depends on the head alone, so it is
/// taken once a block and read from shared memory by every dimension;
/// taking it in each thread was half of the kernel. Each thread still
/// fuses its own numerator and denominator in ascending split order.
///
/// # Safety
/// part_acc covers n_splits * q_heads * head_dim floats, part_m and
/// part_l n_splits * q_heads, out q_heads * head_dim; out is disjoint
/// from the inputs. n_splits >= 1, blockIdx.x < q_heads and threadIdx.x <
/// head_dim. The block is 1-D, a multiple of 32 and at most 1024 threads.
#[device]
unsafe fn decode_combine(
    part_acc: *const f32,
    part_m: *const f32,
    part_l: *const f32,
    out: *mut f32,
    q_heads: i32,
    head_dim: i32,
    n_splits: i32,
) {
    static mut FACTOR: SharedArray<f32, 1024> = SharedArray::UNINIT;
    static mut WARP_MAX: SharedArray<f32, 32> = SharedArray::UNINIT;
    let h = thread::blockIdx_x() as usize;
    let tid = thread::threadIdx_x() as i32;
    let block = thread::blockDim_x() as i32;
    let d = tid as usize;
    let row = q_heads as usize;
    unsafe {
        let factor = SharedArray::as_raw_mut_ptr(&raw mut FACTOR);
        let warp_max = SharedArray::as_raw_mut_ptr(&raw mut WARP_MAX);

        // The global maximum: each thread's share of the splits, then the
        // warp, then the block. A maximum does not depend on its order.
        let mut gm = f32::NEG_INFINITY;
        let mut s = tid;
        while s < n_splits {
            gm = gm.max(*part_m.add(h + s as usize * row));
            s += block;
        }
        let mut off = 16;
        while off > 0 {
            gm = gm.max(warp::shuffle_xor_f32_sync(u32::MAX, gm, off));
            off >>= 1;
        }
        if tid & 31 == 0 {
            *warp_max.add((tid >> 5) as usize) = gm;
        }
        thread::sync_threads();
        let mut w = 0;
        while w < block >> 5 {
            gm = gm.max(*warp_max.add(w as usize));
            w += 1;
        }

        // A split's rescale is the same number in every thread of the
        // block, so one thread takes its exponential and the rest read it.
        // Splits go a block-width at a time, in ascending order.
        let mut l = part_l.add(h);
        let mut acc = part_acc.add(h * head_dim as usize + d);
        let step = row * head_dim as usize;
        let (mut num, mut den) = (0.0f32, 0.0f32);
        let mut base = 0;
        while base < n_splits {
            let mine = base + tid;
            if mine < n_splits {
                let ms = *part_m.add(h + mine as usize * row);
                *factor.add(d) = if ms == f32::NEG_INFINITY {
                    0.0
                } else {
                    (ms - gm).exp()
                };
            }
            thread::sync_threads();
            let end = (base + block).min(n_splits);
            let mut f = factor as *const f32;
            let mut s = base;
            while s + 3 < end {
                let fv = (*f, *f.add(1), *f.add(2), *f.add(3));
                let av = (*acc, *acc.add(step), *acc.add(2 * step), *acc.add(3 * step));
                let lv = (*l, *l.add(row), *l.add(2 * row), *l.add(3 * row));
                num = fv.0.mul_add(av.0, num);
                den = fv.0.mul_add(lv.0, den);
                num = fv.1.mul_add(av.1, num);
                den = fv.1.mul_add(lv.1, den);
                num = fv.2.mul_add(av.2, num);
                den = fv.2.mul_add(lv.2, den);
                num = fv.3.mul_add(av.3, num);
                den = fv.3.mul_add(lv.3, den);
                f = f.add(4);
                acc = acc.wrapping_add(4 * step);
                l = l.wrapping_add(4 * row);
                s += 4;
            }
            while s < end {
                num = (*f).mul_add(*acc, num);
                den = (*f).mul_add(*l, den);
                f = f.add(1);
                acc = acc.wrapping_add(step);
                l = l.wrapping_add(row);
                s += 1;
            }
            // Before the next chunk overwrites the factors.
            thread::sync_threads();
            base += block;
        }
        *out.add(h * head_dim as usize + d) = num / den;
    }
}

/// Merge the per-split partial softmaxes of split decode attention.
///
/// grid: (q_heads,). block: head_dim threads, one per output dimension.
///
/// # Safety
/// As `decode_combine`.
#[kernel]
pub unsafe fn attn_flash_decode_combine(
    part_acc: *const f32,
    part_m: *const f32,
    part_l: *const f32,
    out: *mut f32,
    q_heads: i32,
    head_dim: i32,
    n_splits: i32,
) {
    unsafe { decode_combine(part_acc, part_m, part_l, out, q_heads, head_dim, n_splits) }
}

/// `attn_flash_decode_combine` for a batch: `gridDim.y` sequences, each
/// with its own partials and its own output row block.
///
/// grid: (q_heads, sequences). block: head_dim.
///
/// # Safety
/// As `decode_combine`, with every buffer `gridDim.y` times as long.
#[kernel]
pub unsafe fn attn_flash_decode_combine_k2_batch(
    part_acc: *const f32,
    part_m: *const f32,
    part_l: *const f32,
    out: *mut f32,
    q_heads: i32,
    head_dim: i32,
    n_splits: i32,
) {
    let seq = thread::blockIdx_y() as usize;
    let per_seq = n_splits as usize * q_heads as usize;
    unsafe {
        decode_combine(
            part_acc.add(seq * per_seq * head_dim as usize),
            part_m.add(seq * per_seq),
            part_l.add(seq * per_seq),
            out.add(seq * q_heads as usize * head_dim as usize),
            q_heads,
            head_dim,
            n_splits,
        )
    }
}

/// Append this batch's roped keys and raw values to the fp16 cache, at
/// the absolute position the sequence has reached.
///
/// This is a kernel rather than two device-to-device copies for one
/// reason: the destination offset was computed on the host, and a CUDA
/// graph records addresses, so a captured decode step would write the
/// same position forever. Reading the position from device memory is what
/// makes the step replayable (AGENTS.md rule 5). Both halves go in one
/// launch: same shape, same stride, independent copies.
///
/// `span` is `n_tokens * row`; the cache is indexed by position, so the
/// destination base is one row per position and not one span.
///
/// grid: (ceil(2 * span / block),).
///
/// # Safety
/// key and value cover span floats; each cache covers (position + tokens)
/// rows of `row` halves; position covers one int.
#[kernel]
pub unsafe fn attn_kv_append(
    key: *const f32,
    value: *const f32,
    k_cache: *mut u16,
    v_cache: *mut u16,
    position: *const i32,
    span: i32,
    row: i32,
) {
    let i = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as i32;
    if i >= 2 * span {
        return;
    }
    unsafe {
        let at = *position as i64 * row as i64;
        if i < span {
            *k_cache.add((at + i as i64) as usize) = f2h(*key.add(i as usize));
        } else {
            let j = i - span;
            *v_cache.add((at + j as i64) as usize) = f2h(*value.add(j as usize));
        }
    }
}

/// Interleaved multimodal rotary (IMROPE) on the leading `rope_dim`
/// dimensions of each head: pair `d` takes the temporal, height or width
/// position by `d % 3`, the spatial ones only below their bound. The
/// tail is copied.
///
/// grid: (tokens, heads). block: head_dim.
///
/// # Safety
/// input and out are disjoint and cover n_tokens * n_heads * head_dim
/// floats; mrope_pos covers three ints a token.
#[kernel]
pub unsafe fn attn_rope_partial_imrope(
    input: *const f32,
    out: *mut f32,
    n_heads: i32,
    head_dim: i32,
    rope_dim: i32,
    mrope_pos: *const i32,
    h_bound: i32,
    w_bound: i32,
    theta_base: f32,
    n_tokens: i32,
) {
    let t = thread::blockIdx_x() as i64;
    let h = thread::blockIdx_y() as i64;
    let d = thread::threadIdx_x() as i32;
    if t >= n_tokens as i64 {
        return;
    }
    let half = rope_dim >> 1;
    let base = ((t * n_heads as i64 + h) * head_dim as i64) as usize;
    unsafe {
        if d >= rope_dim {
            *out.add(base + d as usize) = *input.add(base + d as usize);
            return;
        }
        if d >= half {
            return;
        }
        let c = d % 3;
        let component = if c == 1 && d < h_bound {
            *mrope_pos.add(3 * t as usize + 1)
        } else if c == 2 && d < w_bound {
            *mrope_pos.add(3 * t as usize + 2)
        } else {
            *mrope_pos.add(3 * t as usize)
        };
        // Loads before the transcendental chain, not behind it.
        let x0 = *input.add(base + d as usize);
        let x1 = *input.add(base + (d + half) as usize);
        let freq = (theta_base as f64).powf(-2.0 * d as f64 / rope_dim as f64);
        let angle = component as f64 * freq;
        let (sin_a, cos_a) = (angle.sin() as f32, angle.cos() as f32);
        *out.add(base + d as usize) = x0.mul_add(cos_a, -(x1 * sin_a));
        *out.add(base + (d + half) as usize) = x0.mul_add(sin_a, x1 * cos_a);
    }
}

/// `attn_kv_append` with either half of the cache in q8_0. One thread per
/// element, so with `row` a multiple of 32 and 256-thread blocks each
/// warp holds exactly one 32-element block of one half of one position:
/// the early return, the K/V choice and the format choice are all
/// warp-uniform and `kvq8_quantize` sees all 32 lanes. A binary16 half is
/// written exactly as `attn_kv_append` writes it.
///
/// A q8_0 row of n elements is n int8 codes followed by n / 32 binary16
/// scales.
///
/// # Safety
/// key and value cover span floats; each cache covers every position's
/// row in its own format; position covers one int. row and span are
/// multiples of 32.
#[kernel]
pub unsafe fn attn_kv_append_kv8(
    key: *const f32,
    value: *const f32,
    k_cache: *mut u16,
    v_cache: *mut u16,
    position: *const i32,
    span: i32,
    row: i32,
    kq8: i32,
    vq8: i32,
) {
    let i = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as i32;
    if i >= 2 * span {
        return;
    }
    let is_k = i < span;
    let j = if is_k { i } else { i - span };
    unsafe {
        let x = if is_k {
            *key.add(j as usize)
        } else {
            *value.add(j as usize)
        };
        let cache = if is_k { k_cache } else { v_cache };
        let pos = *position as i64 + (j / row) as i64;
        let e = j % row;
        if (if is_k { kq8 } else { vq8 }) == 0 {
            *cache.add((pos * row as i64 + e as i64) as usize) = f2h(x);
            return;
        }
        let (code, d) = kvq8_quantize(x);
        let row_bytes = row as i64 + (row >> 4) as i64;
        let dst = (cache as *mut u8).add((pos * row_bytes) as usize);
        *dst.add(e as usize) = code as u8;
        if e & 31 == 0 {
            *(dst.add((row + 2 * (e >> 5)) as usize) as *mut u16) = d;
        }
    }
}

/// One decode token for up to eight sequences: rotate the query and key
/// rows and append the rotated key and the raw value to each sequence's
/// fp16 cache at its own position, in one launch.
///
/// The first `q_heads` blocks of a sequence rotate a query head; the rest
/// rotate a key head and write the cache. Positions and cache addresses
/// ride in as pointer slots, so a captured graph replays at any position.
///
/// grid: (q_heads + kv_heads, sequences <= 8). block: head_dim.
///
/// # Safety
/// q_in/q_out cover sequences * q_heads * head_dim floats, k_in/k_out and
/// v_in sequences * kv_heads * head_dim; each used slot holds a live
/// pointer (rope position, append position, K cache, V cache) for its
/// sequence, the caches covering position + 1 rows. Outputs are disjoint
/// from inputs.
#[kernel]
pub unsafe fn attn_decode_rope_append_batch(
    q_in: *const f32,
    q_out: *mut f32,
    k_in: *const f32,
    k_out: *mut f32,
    v_in: *const f32,
    rp0: u64,
    rp1: u64,
    rp2: u64,
    rp3: u64,
    rp4: u64,
    rp5: u64,
    rp6: u64,
    rp7: u64,
    ap0: u64,
    ap1: u64,
    ap2: u64,
    ap3: u64,
    ap4: u64,
    ap5: u64,
    ap6: u64,
    ap7: u64,
    kc0: u64,
    kc1: u64,
    kc2: u64,
    kc3: u64,
    kc4: u64,
    kc5: u64,
    kc6: u64,
    kc7: u64,
    vc0: u64,
    vc1: u64,
    vc2: u64,
    vc3: u64,
    vc4: u64,
    vc5: u64,
    vc6: u64,
    vc7: u64,
    q_heads: i32,
    kv_heads: i32,
    head_dim: i32,
    rope_dim: i32,
    theta_base: f32,
) {
    let seq = thread::blockIdx_y() as i32;
    let h = thread::blockIdx_x() as i32;
    let d = thread::threadIdx_x() as i32;
    let rope_pos = match seq {
        0 => rp0,
        1 => rp1,
        2 => rp2,
        3 => rp3,
        4 => rp4,
        5 => rp5,
        6 => rp6,
        _ => rp7,
    } as *const i32;
    let is_q = h < q_heads;
    let (input, out) = if is_q { (q_in, q_out) } else { (k_in, k_out) };
    let heads = if is_q { q_heads } else { kv_heads };
    let hh = if is_q { h } else { h - q_heads };
    let base = ((seq as i64 * heads as i64 + hh as i64) * head_dim as i64) as usize;
    unsafe {
        let (mut k_cache, mut v_cache) =
            (core::ptr::null_mut::<u16>(), core::ptr::null_mut::<u16>());
        let mut at = 0usize;
        if !is_q {
            let pos = match seq {
                0 => ap0,
                1 => ap1,
                2 => ap2,
                3 => ap3,
                4 => ap4,
                5 => ap5,
                6 => ap6,
                _ => ap7,
            } as *const i32;
            k_cache = match seq {
                0 => kc0,
                1 => kc1,
                2 => kc2,
                3 => kc3,
                4 => kc4,
                5 => kc5,
                6 => kc6,
                _ => kc7,
            } as *mut u16;
            v_cache = match seq {
                0 => vc0,
                1 => vc1,
                2 => vc2,
                3 => vc3,
                4 => vc4,
                5 => vc5,
                6 => vc6,
                _ => vc7,
            } as *mut u16;
            at = (*pos as i64 * kv_heads as i64 * head_dim as i64 + hh as i64 * head_dim as i64)
                as usize;
        }
        let half = rope_dim >> 1;
        let du = d as usize;
        if d >= rope_dim {
            let x = *input.add(base + du);
            *out.add(base + du) = x;
            if !is_q {
                *k_cache.add(at + du) = f2h(x);
                *v_cache.add(at + du) = f2h(*v_in.add(base + du));
            }
            return;
        }
        if d >= half {
            return;
        }
        // Every load before the transcendental chain, not behind it.
        let hu = half as usize;
        let position = *rope_pos;
        let x0 = *input.add(base + du);
        let x1 = *input.add(base + du + hu);
        let (v0, v1) = if is_q {
            (0.0, 0.0)
        } else {
            (*v_in.add(base + du), *v_in.add(base + du + hu))
        };
        let freq = (theta_base as f64).powf(-2.0 * d as f64 / rope_dim as f64);
        let angle = (position as f64 + 0.0) * freq;
        let (sin_a, cos_a) = (angle.sin() as f32, angle.cos() as f32);
        let y0 = x0.mul_add(cos_a, -(x1 * sin_a));
        let y1 = x0.mul_add(sin_a, x1 * cos_a);
        *out.add(base + du) = y0;
        *out.add(base + du + hu) = y1;
        if !is_q {
            *k_cache.add(at + du) = f2h(y0);
            *k_cache.add(at + du + hu) = f2h(y1);
            *v_cache.add(at + du) = f2h(v0);
            *v_cache.add(at + du + hu) = f2h(v1);
        }
    }
}
