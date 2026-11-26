//! `inkling2_loop`: the embedding gather, depth mixing, loop controllers,
//! short convolutions, head clipping, relative-position attention and the
//! expert glue around the grouped GEMMs.

use crate::common::{block_reduce_sum, f2h, h2f, load_half_le};
use cuda_device::{SharedArray, device, kernel, thread, warp};

/// The most states `ink_mix` blends and the most queries it answers.
const MIX_STATES: usize = 8;
const MIX_QUERIES: usize = 4;

/// Token rows of a Q8_0 (`bf16 == 0`) or bf16 embedding table as fp32.
///
/// grid: (ceil(hidden / block), tokens).
///
/// # Safety
/// table holds whole rows of `hidden` elements in its format, ids covers
/// gridDim.y entries naming rows it holds, out covers gridDim.y * hidden.
#[kernel]
pub unsafe fn ink_embed(table: *const u8, ids: *const i32, out: *mut f32, hidden: i32, bf16: i32) {
    let i = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as i64;
    let t = thread::blockIdx_y() as i64;
    if i >= hidden as i64 {
        return;
    }
    unsafe {
        let row = *ids.add(t as usize) as i64;
        let v = if bf16 != 0 {
            let p = table.add(((row * hidden as i64 + i) * 2) as usize);
            f32::from_bits(((*p as u32) | ((*p.add(1) as u32) << 8)) << 16)
        } else {
            let block = table.add(((row * (hidden as i64 / 32) + i / 32) * 34) as usize);
            (*block.add(2 + (i % 32) as usize) as i8) as f32 * load_half_le(block)
        };
        *out.add((t * hidden as i64 + i) as usize) = v;
    }
}

/// `out[t][j] = sum_i w[j][i] * x[t][i]`, fp32, ascending i, each product
/// fused into the running sum.
///
/// grid: (ceil(rows / block), tokens).
///
/// # Safety
/// w covers rows * k floats, x gridDim.y rows of stride `ldx >= k`, out
/// gridDim.y rows of `rows`.
#[kernel]
pub unsafe fn ink_linear(
    w: *const f32,
    x: *const f32,
    out: *mut f32,
    rows: i32,
    k: i32,
    ldx: i32,
) {
    let j = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as i64;
    let t = thread::blockIdx_y() as i64;
    if j >= rows as i64 {
        return;
    }
    unsafe {
        let wr = w.add((j * k as i64) as usize);
        let xr = x.add((t * ldx as i64) as usize);
        let mut acc = 0.0f32;
        let mut i = 0usize;
        while i < k as usize {
            acc = (*wr.add(i)).mul_add(*xr.add(i), acc);
            i += 1;
        }
        *out.add((t * rows as i64 + j) as usize) = acc;
    }
}

/// Attention over the depth: each state row is RMS-normalized (epsilon
/// 1e-6), scored against each query at `1 / sqrt(n)`, and the queries'
/// softmax-weighted sums of the normalized states are written out.
///
/// `stack` is `[states][tokens][n]`, `queries` `[queries][n]`, `out`
/// `[queries][tokens][n]`. grid: (tokens), 256 threads, dynamic shared
/// memory of one f32 per warp.
///
/// # Safety
/// The buffers cover those shapes; states <= 8 and queries <= 4; out is
/// disjoint from the inputs.
#[kernel]
pub unsafe fn ink_mix(
    stack: *const f32,
    queries: *const f32,
    out: *mut f32,
    n: i32,
    states: i32,
    n_queries: i32,
    tokens: i32,
) {
    static mut INV: SharedArray<f32, MIX_STATES> = SharedArray::UNINIT;
    static mut COEF: SharedArray<f32, { MIX_STATES * MIX_QUERIES }> = SharedArray::UNINIT;
    let tid = thread::threadIdx_x() as i32;
    let step = thread::blockDim_x() as i32;
    let t = thread::blockIdx_x() as i64;
    let plane = tokens as i64 * n as i64;
    unsafe {
        let mut s = 0;
        while s < states {
            let row = stack.add((s as i64 * plane + t * n as i64) as usize);
            let mut partial = 0.0f32;
            let mut i = tid;
            while i < n {
                let v = *row.add(i as usize);
                partial = v.mul_add(v, partial);
                i += step;
            }
            let total = block_reduce_sum(partial);
            if tid == 0 {
                INV[s as usize] = 1.0 / (total / n as f32 + 1e-6).sqrt();
            }
            s += 1;
        }
        let scale = 1.0 / (n as f32).sqrt();
        let mut q = 0;
        while q < n_queries {
            let query = queries.add((q as i64 * n as i64) as usize);
            let mut s = 0;
            while s < states {
                let row = stack.add((s as i64 * plane + t * n as i64) as usize);
                let mut partial = 0.0f32;
                let mut i = tid;
                while i < n {
                    partial = (*row.add(i as usize)).mul_add(*query.add(i as usize), partial);
                    i += step;
                }
                let total = block_reduce_sum(partial);
                if tid == 0 {
                    COEF[(q as usize) * MIX_STATES + s as usize] = total * INV[s as usize] * scale;
                }
                s += 1;
            }
            q += 1;
        }
        if tid == 0 {
            let mut q = 0usize;
            while q < n_queries as usize {
                let mut max = COEF[q * MIX_STATES];
                let mut s = 1usize;
                while s < states as usize {
                    max = max.max(COEF[q * MIX_STATES + s]);
                    s += 1;
                }
                let mut sum = 0.0f32;
                s = 0;
                while s < states as usize {
                    let e = (COEF[q * MIX_STATES + s] - max).exp();
                    COEF[q * MIX_STATES + s] = e;
                    sum += e;
                    s += 1;
                }
                s = 0;
                while s < states as usize {
                    COEF[q * MIX_STATES + s] = COEF[q * MIX_STATES + s] / sum * INV[s];
                    s += 1;
                }
                q += 1;
            }
        }
        thread::sync_threads();
        let mut q = 0;
        while q < n_queries {
            let dst = out.add((q as i64 * plane + t * n as i64) as usize);
            let mut i = tid;
            while i < n {
                let mut acc = 0.0f32;
                let mut s = 0;
                while s < states {
                    let v = *stack.add((s as i64 * plane + t * n as i64 + i as i64) as usize);
                    acc = COEF[(q as usize) * MIX_STATES + s as usize].mul_add(v, acc);
                    s += 1;
                }
                *dst.add(i as usize) = acc;
                i += step;
            }
            q += 1;
        }
    }
}

/// A loop controller's gate input: the RMS-normalized (epsilon 1e-6)
/// current row and message row, side by side, as binary16.
///
/// grid: (tokens), dynamic shared memory of one f32 per warp.
///
/// # Safety
/// cur and msg cover gridDim.x * n floats, out gridDim.x * 2 * n halves.
#[kernel]
pub unsafe fn ink_ctrl_pair(cur: *const f32, msg: *const f32, out: *mut u16, n: i32) {
    let tid = thread::threadIdx_x() as i32;
    let step = thread::blockDim_x() as i32;
    let t = thread::blockIdx_x() as usize;
    unsafe {
        let mut half = 0usize;
        while half < 2 {
            let src = if half == 0 { cur } else { msg }.add(t * n as usize);
            let mut partial = 0.0f32;
            let mut i = tid;
            while i < n {
                let v = *src.add(i as usize);
                partial = v.mul_add(v, partial);
                i += step;
            }
            let inv = 1.0 / (block_reduce_sum(partial) / n as f32 + 1e-6).sqrt();
            let dst = out.add((t * 2 + half) * n as usize);
            let mut i = tid;
            while i < n {
                *dst.add(i as usize) = f2h(*src.add(i as usize) * inv);
                i += step;
            }
            half += 1;
        }
    }
}

/// `out = cur - tanh(forget) * cur + tanh(gain) * msg`.
///
/// # Safety
/// Every buffer covers `count` floats; out may exactly alias cur.
#[kernel]
pub unsafe fn ink_ctrl_apply(
    cur: *const f32,
    msg: *const f32,
    gain: *const f32,
    forget: *const f32,
    out: *mut f32,
    count: i64,
) {
    let i =
        thread::blockIdx_x() as i64 * thread::blockDim_x() as i64 + thread::threadIdx_x() as i64;
    if i >= count {
        return;
    }
    unsafe {
        let i = i as usize;
        let c = *cur.add(i);
        *out.add(i) = c - (*forget.add(i)).tanh() * c + (*gain.add(i)).tanh() * *msg.add(i);
    }
}

/// One sequence's input `u` of a short convolution: the row itself, or
/// the carried state for the positions before this pass.
///
/// # Safety
/// `u >= -carry`; x holds the row and state the carry.
#[device]
unsafe fn conv_input(
    x: *const f32,
    state: *const f32,
    u: i32,
    c: usize,
    channels: usize,
    carry: i32,
) -> f32 {
    unsafe {
        if u >= 0 {
            *x.add(u as usize * channels + c)
        } else {
            *state.add(c * carry as usize + (carry + u) as usize)
        }
    }
}

/// Causal short convolution with a residual, over one sequence's `rows`
/// consecutive positions, then the carried state advanced past them.
///
/// `out[t][c] = (x[t][c] + sum_i in(t + i - carry)[c] * w[c][i]
///              + sum_lag in(t - lag)[c] * dynamic[t][head(c)][lag]) * norm`
/// with `carry = taps - 1` and `in` reading `state[c][carry + u]` for
/// `u < 0`. `dynamic` may be null. grid: (ceil(channels / block)).
///
/// # Safety
/// x and out cover rows * channels floats and are disjoint; weight covers
/// channels * taps; state covers channels * (taps - 1); dynamic, when not
/// null, rows * heads * taps with channels a multiple of heads.
#[kernel]
pub unsafe fn ink_conv(
    x: *const f32,
    weight: *const f32,
    dynamic: *const f32,
    state: *mut f32,
    out: *mut f32,
    channels: i32,
    taps: i32,
    heads: i32,
    rows: i32,
    norm: f32,
) {
    let c = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
    if c >= channels as usize {
        return;
    }
    let ch = channels as usize;
    let carry = taps - 1;
    unsafe {
        let head = if dynamic.is_null() {
            0
        } else {
            c / (ch / heads as usize)
        };
        let mut t = 0;
        while t < rows {
            let mut sum = 0.0f32;
            let mut i = 0;
            while i < taps {
                sum += conv_input(x, state, t + i - carry, c, ch, carry)
                    * *weight.add(c * taps as usize + i as usize);
                i += 1;
            }
            if !dynamic.is_null() {
                let tap = dynamic.add((t as usize * heads as usize + head) * taps as usize);
                let mut lag = 0;
                while lag < taps {
                    sum += conv_input(x, state, t - lag, c, ch, carry) * *tap.add(lag as usize);
                    lag += 1;
                }
            }
            *out.add(t as usize * ch + c) = (*x.add(t as usize * ch + c) + sum) * norm;
            t += 1;
        }
        // Slot i takes input rows - carry + i, which is a later slot of the
        // old state or a row of x: ascending i never reads a written slot.
        let mut i = 0;
        while i < carry {
            let v = conv_input(x, state, rows - carry + i, c, ch, carry);
            *state.add(c * carry as usize + i as usize) = v;
            i += 1;
        }
    }
}

/// Scale each `width`-long group of a row to an L2 norm of at most
/// `threshold`, after an optional per-channel scale (`scale[group][i]`),
/// then by the row's logarithmic position scale when `alpha >= 0`:
/// `1 + alpha * ln(max(1, (position + 1) / floor))`, the position being
/// `pos[0] + row`.
///
/// grid: (tokens), one thread a group.
///
/// # Safety
/// x covers gridDim.x * groups * width floats; scale, when not null,
/// groups * width; pos one i32; groups <= blockDim.x.
#[kernel]
pub unsafe fn ink_clip(
    x: *mut f32,
    scale: *const f32,
    pos: *const i32,
    width: i32,
    groups: i32,
    threshold: f32,
    alpha: f32,
    floor: f32,
) {
    let g = thread::threadIdx_x() as usize;
    let t = thread::blockIdx_x() as usize;
    if g >= groups as usize {
        return;
    }
    let w = width as usize;
    unsafe {
        let row = x.add((t * groups as usize + g) * w);
        let mut sq = 0.0f32;
        let mut i = 0usize;
        while i < w {
            let mut v = *row.add(i);
            if !scale.is_null() {
                v *= *scale.add(g * w + i);
                *row.add(i) = v;
            }
            sq = v.mul_add(v, sq);
            i += 1;
        }
        let mut factor = (threshold / (sq.sqrt() + 1e-6)).min(1.0);
        if alpha >= 0.0 {
            let p = (*pos + t as i32 + 1) as f32 / floor;
            factor *= 1.0 + alpha * p.max(1.0).ln();
        }
        i = 0;
        while i < w {
            *row.add(i) *= factor;
            i += 1;
        }
    }
}

/// RMS-normalize each `width`-long group in place, no weight.
///
/// # Safety
/// x covers groups * width floats.
#[kernel]
pub unsafe fn ink_rms_groups(x: *mut f32, width: i32, eps: f32, groups: i64) {
    let g = thread::blockIdx_x() as i64 * thread::blockDim_x() as i64 + thread::threadIdx_x() as i64;
    if g >= groups {
        return;
    }
    let w = width as usize;
    unsafe {
        let row = x.add(g as usize * w);
        let mut sq = 0.0f32;
        let mut i = 0usize;
        while i < w {
            let v = *row.add(i);
            sq = v.mul_add(v, sq);
            i += 1;
        }
        let inv = 1.0 / (sq / width as f32 + eps).sqrt();
        i = 0;
        while i < w {
            *row.add(i) *= inv;
            i += 1;
        }
    }
}

/// The relative-position bias of every distance, per row and head, with
/// the row's sink logit already subtracted:
/// `table[t][h][d] = bias[d] . r[t][h] - sink[t][h]` for `d < extent`,
/// and `-sink[t][h]` at `d == extent` (every distance past the table).
///
/// grid: (ceil((extent + 1) / block), heads, tokens).
///
/// # Safety
/// bias covers extent * n_rel floats, r gridDim.z * heads * n_rel, sink
/// gridDim.z * heads, table gridDim.z * heads * (extent + 1).
#[kernel]
pub unsafe fn ink_rel_table(
    bias: *const f32,
    r: *const f32,
    sink: *const f32,
    table: *mut f32,
    n_rel: i32,
    extent: i32,
) {
    let d = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as i32;
    let h = thread::blockIdx_y() as usize;
    let t = thread::blockIdx_z() as usize;
    let heads = thread::gridDim_y() as usize;
    if d > extent {
        return;
    }
    unsafe {
        let row = t * heads + h;
        let mut acc = 0.0f32;
        if d < extent {
            let b = bias.add(d as usize * n_rel as usize);
            let rv = r.add(row * n_rel as usize);
            let mut i = 0usize;
            while i < n_rel as usize {
                acc = (*b.add(i)).mul_add(*rv.add(i), acc);
                i += 1;
            }
        }
        *table.add(row * (extent as usize + 1) + d as usize) = acc - *sink.add(row);
    }
}

/// The largest value across a 256-thread block. Leaves the block
/// synchronized.
///
/// # Safety
/// Every thread of a 256-thread 1-D block participates.
#[device]
unsafe fn block_max_256(v: f32) -> f32 {
    static mut RED: SharedArray<f32, 8> = SharedArray::UNINIT;
    let tid = thread::threadIdx_x();
    let mut v = v;
    let mut off = 16;
    while off > 0 {
        v = v.max(warp::shuffle_xor_f32_sync(u32::MAX, v, off));
        off >>= 1;
    }
    unsafe {
        if tid & 31 == 0 {
            RED[(tid >> 5) as usize] = v;
        }
        thread::sync_threads();
        let mut m = RED[0];
        let mut w = 1usize;
        while w < 8 {
            m = m.max(RED[w]);
            w += 1;
        }
        thread::sync_threads();
        m
    }
}

/// The sum across a 256-thread block: an xor tree per warp, then the warp
/// totals in ascending order. Leaves the block synchronized.
///
/// # Safety
/// Every thread of a 256-thread 1-D block participates.
#[device]
unsafe fn block_sum_256(v: f32) -> f32 {
    static mut RED: SharedArray<f32, 8> = SharedArray::UNINIT;
    let tid = thread::threadIdx_x();
    let mut v = v;
    let mut off = 16;
    while off > 0 {
        v += warp::shuffle_xor_f32_sync(u32::MAX, v, off);
        off >>= 1;
    }
    unsafe {
        if tid & 31 == 0 {
            RED[(tid >> 5) as usize] = v;
        }
        thread::sync_threads();
        let mut s = RED[0];
        let mut w = 1usize;
        while w < 8 {
            s += RED[w];
            w += 1;
        }
        thread::sync_threads();
        s
    }
}

/// One query head's attention over its sequence's cache, with the
/// relative-position bias and a zero-logit sink:
/// `p_j = exp(s_j) / (1 + sum exp(s))`,
/// `s_j = q . k_j * scale + table[min(p - j, extent)]`, over cache rows
/// `j <= p` (and `j > p - window` when `window > 0`), `p = pos[0] + row`.
///
/// The cache is scanned 256 rows at a time with a running maximum, so the
/// result does not depend on the cache length through anything but the
/// rounding of the rescales. grid: (heads, tokens), 256 threads.
///
/// # Safety
/// q and out cover gridDim.y * heads * dim floats, table gridDim.y * heads
/// * (extent + 1); the caches cover (p + 1) * kv_heads * dim halves for
/// the last row's p; dim <= 128; heads a multiple of kv_heads.
#[kernel]
pub unsafe fn ink_attn(
    q: *const f32,
    k_cache: *const u16,
    v_cache: *const u16,
    table: *const f32,
    out: *mut f32,
    pos: *const i32,
    window: i32,
    extent: i32,
    kv_heads: i32,
    dim: i32,
    scale: f32,
) {
    static mut P: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut ACC: SharedArray<f32, 128> = SharedArray::UNINIT;
    static mut RUN: SharedArray<f32, 2> = SharedArray::UNINIT;
    const LOWEST: f32 = -3.0e38;
    let tid = thread::threadIdx_x() as i32;
    let h = thread::blockIdx_x() as usize;
    let t = thread::blockIdx_y() as usize;
    let heads = thread::gridDim_x() as usize;
    let d = dim as usize;
    let kv = h / (heads / kv_heads as usize);
    let stride = kv_heads as usize * d;
    unsafe {
        let p = *pos + t as i32;
        let lo = if window > 0 && p + 1 > window {
            p + 1 - window
        } else {
            0
        };
        let qh = q.add((t * heads + h) * d);
        let tb = table.add((t * heads + h) * (extent as usize + 1));
        if (tid as usize) < d {
            ACC[tid as usize] = 0.0;
        }
        if tid == 0 {
            // The sink: logit zero, attending to nothing.
            RUN[0] = 0.0;
            RUN[1] = 1.0;
        }
        thread::sync_threads();
        let mut j0 = lo;
        while j0 <= p {
            let j = j0 + tid;
            let mut s = LOWEST;
            if j <= p {
                let kr = k_cache.add(j as usize * stride + kv * d);
                let mut dot = 0.0f32;
                let mut i = 0usize;
                while i < d {
                    dot = (*qh.add(i)).mul_add(h2f(*kr.add(i)), dot);
                    i += 1;
                }
                let dist = (p - j).min(extent);
                s = dot * scale + *tb.add(dist as usize);
            }
            let chunk_max = block_max_256(s);
            let old = RUN[0];
            let top = old.max(chunk_max);
            let rescale = (old - top).exp();
            let pj = if j <= p { (s - top).exp() } else { 0.0 };
            P[tid as usize] = pj;
            let chunk_sum = block_sum_256(pj);
            if (tid as usize) < d {
                let live = (p - j0 + 1).min(256);
                let mut acc = ACC[tid as usize] * rescale;
                let mut jj = 0;
                while jj < live {
                    let v = *v_cache.add((j0 + jj) as usize * stride + kv * d + tid as usize);
                    acc = P[jj as usize].mul_add(h2f(v), acc);
                    jj += 1;
                }
                ACC[tid as usize] = acc;
            }
            if tid == 0 {
                RUN[0] = top;
                RUN[1] = RUN[1] * rescale + chunk_sum;
            }
            thread::sync_threads();
            j0 += 256;
        }
        if (tid as usize) < d {
            *out.add((t * heads + h) * d + tid as usize) = ACC[tid as usize] / RUN[1];
        }
    }
}

/// Expert-sorted activations for the grouped GEMM: slot `s` of a live
/// tile takes the row of the token its pair belongs to, as binary16, and
/// records itself in `slot_of`; a padding slot takes zeros.
///
/// grid: (ceil(n / block), slots).
///
/// # Safety
/// x covers the routed tokens' rows of n floats; sorted covers gridDim.y
/// entries, owners gridDim.y / tile; xs gridDim.y * n halves; slot_of
/// `pairs` entries. A live slot's entry is a pair id below `pairs`, or
/// `pairs` itself for padding.
#[kernel]
pub unsafe fn ink_gather_half(
    x: *const f32,
    sorted: *const i32,
    owners: *const i32,
    xs: *mut u16,
    slot_of: *mut i32,
    n: i32,
    topk: i32,
    pairs: i32,
    tile: i32,
) {
    let i = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
    let slot = thread::blockIdx_y() as usize;
    if i >= n as usize {
        return;
    }
    unsafe {
        if *owners.add(slot / tile as usize) < 0 {
            return;
        }
        let flat = *sorted.add(slot);
        let v = if flat < pairs {
            if i == 0 {
                *slot_of.add(flat as usize) = slot as i32;
            }
            f2h(*x.add((flat / topk) as usize * n as usize + i))
        } else {
            0
        };
        *xs.add(slot * n as usize + i) = v;
    }
}

/// The SiTU activation over the merged gate/up projection of every slot
/// of a live tile, as binary16 for the down projection:
/// `beta * tanh(gate / beta) * sigmoid(gate) * lb * tanh(up / lb)`.
///
/// grid: (ceil(width / block), slots).
///
/// # Safety
/// gate_up covers gridDim.y * 2 * width floats (gate then up per slot),
/// out gridDim.y * width halves, owners gridDim.y / tile entries.
#[kernel]
pub unsafe fn ink_situ_half(
    gate_up: *const f32,
    owners: *const i32,
    out: *mut u16,
    width: i32,
    tile: i32,
    beta: f32,
    linear_beta: f32,
) {
    let i = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
    let slot = thread::blockIdx_y() as usize;
    let w = width as usize;
    if i >= w {
        return;
    }
    unsafe {
        if *owners.add(slot / tile as usize) < 0 {
            return;
        }
        let gate = *gate_up.add(slot * 2 * w + i);
        let up = *gate_up.add(slot * 2 * w + w + i);
        let act = beta * (gate / beta).tanh() / (1.0 + (-gate).exp());
        let lin = if linear_beta > 0.0 {
            linear_beta * (up / linear_beta).tanh()
        } else {
            up
        };
        *out.add(slot * w + i) = f2h(act * lin);
    }
}

/// `out[t][i] = sum_j weights[t][j] * y[slot_of[t][j]][i]`, ascending j,
/// each product fused into the running sum.
///
/// grid: (ceil(n / block), tokens).
///
/// # Safety
/// weights and slot_of cover gridDim.y * topk entries, every slot naming
/// a row of y; out covers gridDim.y * n floats.
#[kernel]
pub unsafe fn ink_moe_combine(
    y: *const f32,
    weights: *const f32,
    slot_of: *const i32,
    out: *mut f32,
    n: i32,
    topk: i32,
) {
    let i = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
    let t = thread::blockIdx_y() as usize;
    if i >= n as usize {
        return;
    }
    unsafe {
        let mut acc = 0.0f32;
        let mut j = 0usize;
        while j < topk as usize {
            let pair = t * topk as usize + j;
            let slot = *slot_of.add(pair) as usize;
            acc = (*weights.add(pair)).mul_add(*y.add(slot * n as usize + i), acc);
            j += 1;
        }
        *out.add(t * n as usize + i) = acc;
    }
}

/// `x = cap * tanh(x / cap)`.
///
/// # Safety
/// x covers `count` floats.
#[kernel]
pub unsafe fn ink_softcap(x: *mut f32, count: i64, cap: f32) {
    let i =
        thread::blockIdx_x() as i64 * thread::blockDim_x() as i64 + thread::threadIdx_x() as i64;
    if i >= count {
        return;
    }
    unsafe {
        let p = x.add(i as usize);
        *p = cap * (*p / cap).tanh();
    }
}
