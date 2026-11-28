//! Mixture-of-experts routing, dispatch and alignment, and the block's
//! activation, reduce and gate-and-combine tails.

use crate::common::block_reduce_sum;
use cuda_device::{DynamicSharedArray, device, kernel, thread, warp};

/// SwiGLU for the shared expert's tensor-core path: one elementwise pass
/// between the two projections, which cannot fuse it because a projection
/// knows nothing about what its output feeds.
///
/// `g / (1 + exp(-g))`, as `llmcuda_kernels::norm::silu` and the fp32
/// shared-expert kernel write it, not the algebraically equal
/// `g * sigmoid(g)`.
///
/// # Safety
/// gate, up and out cover n floats; the 1-D launch is nonempty. Output
/// may exactly alias an input.
#[kernel]
pub unsafe fn moe_swiglu(gate: *const f32, up: *const f32, n: i64, out: *mut f32) {
    let i =
        thread::blockIdx_x() as i64 * thread::blockDim_x() as i64 + thread::threadIdx_x() as i64;
    if i >= n {
        return;
    }
    unsafe {
        let g = *gate.add(i as usize);
        *out.add(i as usize) = (g / (1.0 + (-g).exp())) * *up.add(i as usize);
    }
}

/// fp32 sum of each token's top-k routed contributions.
///
/// grid: (max_tokens, ceil(hidden / block)). The second dimension is the
/// point at one token: a block per token would be one block on a 72-SM
/// card summing eight 2,048-float vectors, which is all launch latency.
/// The k loop is ascending and inside one thread, so contributions are
/// added in selection order exactly as the reference adds them.
///
/// # Safety
/// partial covers max_tokens * top_k * hidden floats, out covers
/// max_tokens * hidden, valid_tokens one int; out is disjoint from
/// partial.
#[kernel]
pub unsafe fn moe_reduce(
    partial: *const f32,
    valid_tokens: *const i32,
    top_k: i32,
    hidden: i32,
    out: *mut f32,
) {
    let token = thread::blockIdx_x() as i64;
    let h = (thread::blockIdx_y() * thread::blockDim_x() + thread::threadIdx_x()) as i32;
    unsafe {
        if token >= *valid_tokens as i64 || h >= hidden {
            return;
        }
        let mut p = partial.add((token * top_k as i64 * hidden as i64 + h as i64) as usize);
        let mut acc = 0.0f32;
        let mut k = 0;
        while k < top_k {
            acc += *p;
            p = p.wrapping_add(hidden as usize);
            k += 1;
        }
        *out.add((token * hidden as i64 + h as i64) as usize) = acc;
    }
}

/// MoE alignment, pass one: fill the padded dispatch tables with their
/// sentinels and count the `(token, slot)` pairs routed to this block's
/// expert. Integer addition is associative, so the tree order of the
/// count cannot change it the way it would change a float sum.
///
/// grid: (num_experts,). block: a power of two; dynamic shared memory
/// holds one int per thread.
///
/// # Safety
/// topk_ids covers valid_tokens * top_k ints, sorted_token_ids
/// sorted_capacity, expert_ids expert_capacity, counts gridDim.x.
#[kernel]
pub unsafe fn moe_align_count(
    topk_ids: *const i32,
    valid_tokens: *const i32,
    top_k: i32,
    sorted_capacity: i32,
    expert_capacity: i32,
    sorted_token_ids: *mut i32,
    expert_ids: *mut i32,
    counts: *mut i32,
) {
    let scratch = DynamicSharedArray::<i32>::get();
    let e = thread::blockIdx_x() as i32;
    let tid = thread::threadIdx_x() as i32;
    let block = thread::blockDim_x() as i32;
    let gtid = e * block + tid;
    let gstride = thread::gridDim_x() as i32 * block;
    unsafe {
        let numel = *valid_tokens * top_k;
        let mut s = gtid;
        while s < sorted_capacity {
            *sorted_token_ids.add(s as usize) = numel;
            s += gstride;
        }
        let mut b = gtid;
        while b < expert_capacity {
            *expert_ids.add(b as usize) = -1;
            b += gstride;
        }
        let mut c = 0i32;
        let mut i = tid;
        while i < numel {
            if *topk_ids.add(i as usize) == e {
                c += 1;
            }
            i += block;
        }
        *scratch.add(tid as usize) = c;
        thread::sync_threads();
        let mut s = block >> 1;
        while s > 0 {
            if tid < s {
                *scratch.add(tid as usize) += *scratch.add((tid + s) as usize);
            }
            thread::sync_threads();
            s >>= 1;
        }
        if tid == 0 {
            *counts.add(e as usize) = *scratch;
        }
    }
}

/// The MoE block's tail in one launch: the shared expert's sigmoid gate,
/// the routed sum over the top-k contributions, and both adds.
///
/// ```text
/// g       = sigmoid(w . normed[t])
/// routed  = sum_k partial[t][k]          ascending k
/// ffn_out = routed + shexp[t] * g
/// l_out   = ffn_out + residual[t]
/// ```
///
/// The gate's dot product is taken by the first 256 threads, each over a
/// strided share, and reduced like RMSNorm's sum of squares; the routed
/// sum stays ascending inside one thread, so it matches the reference's
/// order. `routed` and `ffn_out` are still written out, because the
/// golden capture has them as waypoints.
///
/// grid: (max_tokens, ceil(hidden / block)). block: at least 256, with a
/// float of dynamic shared memory per warp.
///
/// # Safety
/// w covers hidden floats; normed, shexp, residual and the three row
/// outputs max_tokens * hidden; partial max_tokens * top_k * hidden;
/// gate max_tokens; valid_tokens one int. Outputs are disjoint from
/// inputs and each other.
#[kernel]
pub unsafe fn moe_block_gate_and_combine(
    w: *const f32,
    normed: *const f32,
    partial: *const f32,
    shexp: *const f32,
    residual: *const f32,
    valid_tokens: *const i32,
    top_k: i32,
    hidden: i32,
    gate: *mut f32,
    routed: *mut f32,
    ffn_out: *mut f32,
    l_out: *mut f32,
) {
    let t = thread::blockIdx_x() as i64;
    let tid = thread::threadIdx_x() as i32;
    unsafe {
        if t >= *valid_tokens as i64 {
            return;
        }
        let base = (t * hidden as i64) as usize;
        let xs = normed.add(base);
        let mut s = 0.0f32;
        if tid < 256 {
            // NVRTC contracts the original's s += w[j] * xs[j]. Eight
            // steps an iteration, loads first, so they issue together.
            let mut j = tid;
            let (mut wp, mut xp) = (w.add(tid as usize), xs.add(tid as usize));
            while j + 1792 < hidden {
                let a = (
                    *wp.add(0),
                    *wp.add(256),
                    *wp.add(512),
                    *wp.add(768),
                    *wp.add(1024),
                    *wp.add(1280),
                    *wp.add(1536),
                    *wp.add(1792),
                );
                let b = (
                    *xp.add(0),
                    *xp.add(256),
                    *xp.add(512),
                    *xp.add(768),
                    *xp.add(1024),
                    *xp.add(1280),
                    *xp.add(1536),
                    *xp.add(1792),
                );
                s = a.0.mul_add(b.0, s);
                s = a.1.mul_add(b.1, s);
                s = a.2.mul_add(b.2, s);
                s = a.3.mul_add(b.3, s);
                s = a.4.mul_add(b.4, s);
                s = a.5.mul_add(b.5, s);
                s = a.6.mul_add(b.6, s);
                s = a.7.mul_add(b.7, s);
                wp = wp.wrapping_add(2048);
                xp = xp.wrapping_add(2048);
                j += 2048;
            }
            while j < hidden {
                s = (*wp).mul_add(*xp, s);
                wp = wp.wrapping_add(256);
                xp = xp.wrapping_add(256);
                j += 256;
            }
        }
        s = block_reduce_sum(s);
        let g = 1.0 / (1.0 + (-s).exp());
        if tid == 0 && thread::blockIdx_y() == 0 {
            *gate.add(t as usize) = g;
        }

        let pbase = partial.add((t * top_k as i64 * hidden as i64) as usize);
        let stride = (thread::gridDim_y() * thread::blockDim_x()) as i32;
        let mut j = (thread::blockIdx_y() * thread::blockDim_x()) as i32 + tid;
        while j < hidden {
            let mut p = pbase.add(j as usize);
            let mut r = 0.0f32;
            let mut k = 0;
            let step = hidden as usize;
            while k + 3 < top_k {
                let v = (*p, *p.add(step), *p.add(2 * step), *p.add(3 * step));
                r += v.0;
                r += v.1;
                r += v.2;
                r += v.3;
                p = p.wrapping_add(4 * step);
                k += 4;
            }
            while k < top_k {
                r += *p;
                p = p.wrapping_add(step);
                k += 1;
            }
            let at = base + j as usize;
            let v = (*shexp.add(at)).mul_add(g, r);
            *routed.add(at) = r;
            *ffn_out.add(at) = v;
            *l_out.add(at) = v + *residual.add(at);
            j += stride;
        }
    }
}

/// How many of `n` ids equal `e`, reading eight ahead.
///
/// # Safety
/// ids covers n ints.
#[device]
unsafe fn count_equal(ids: *const i32, n: i32, e: i32) -> i32 {
    let mut c = 0i32;
    let mut i = 0;
    let mut p = ids;
    unsafe {
        while i + 7 < n {
            let v = (
                *p.add(0),
                *p.add(1),
                *p.add(2),
                *p.add(3),
                *p.add(4),
                *p.add(5),
                *p.add(6),
                *p.add(7),
            );
            c += (v.0 == e) as i32;
            c += (v.1 == e) as i32;
            c += (v.2 == e) as i32;
            c += (v.3 == e) as i32;
            c += (v.4 == e) as i32;
            c += (v.5 == e) as i32;
            c += (v.6 == e) as i32;
            c += (v.7 == e) as i32;
            p = p.wrapping_add(8);
            i += 8;
        }
        while i < n {
            c += (*p == e) as i32;
            p = p.wrapping_add(1);
            i += 1;
        }
    }
    c
}

/// Route one token: softmax over its expert logits, then the `top_k`
/// largest probabilities with their weights renormalized to sum to one.
///
/// This transcribes `llmcuda_kernels::moe::router::route_token`, tie
/// rules included: a tie in probability goes to the lower expert index
/// at every level, and if the selected mass is not positive the weights
/// fall back to uniform. The softmax subtracts the row maximum, so the
/// exponentials cannot overflow; its sum is reduced over shared memory
/// in a fixed tree. Selection runs in warp 0: each lane owns up to
/// sixteen experts (`lane + 32 i`), keeps their probabilities in
/// registers, and the warp picks a winner `top_k` times by xor shuffle.
///
/// # Safety
/// logits covers (token + 1) * num_experts floats; probs covers
/// num_experts and rval blockDim.x floats of shared memory; topk_ids and
/// topk_weights cover (token + 1) * top_k. num_experts <= 512, top_k <=
/// blockDim.x, and blockDim.x is a power of two of at least 32.
#[device]
unsafe fn route_token(
    logits: *const f32,
    token: i32,
    num_experts: i32,
    top_k: i32,
    probs: *mut f32,
    rval: *mut f32,
    topk_ids: *mut i32,
    topk_weights: *mut f32,
) {
    let tid = thread::threadIdx_x() as i32;
    let block = thread::blockDim_x() as i32;
    unsafe {
        let row = logits.add((token as i64 * num_experts as i64) as usize);
        let mut local = *row;
        let mut e = tid;
        while e < num_experts {
            local = local.max(*row.add(e as usize));
            e += block;
        }
        let mut off = 16;
        while off > 0 {
            local = local.max(warp::shuffle_down_f32_sync(u32::MAX, local, off));
            off >>= 1;
        }
        if tid & 31 == 0 {
            *rval.add((tid >> 5) as usize) = local;
        }
        thread::sync_threads();
        let mut max_logit = *rval;
        let mut w = 1;
        while w < block >> 5 {
            max_logit = max_logit.max(*rval.add(w as usize));
            w += 1;
        }
        thread::sync_threads();

        let mut esum = 0.0f32;
        let mut e = tid;
        while e < num_experts {
            let p = (*row.add(e as usize) - max_logit).exp();
            *probs.add(e as usize) = p;
            esum += p;
            e += block;
        }
        *rval.add(tid as usize) = esum;
        thread::sync_threads();
        let mut s = block >> 1;
        while s > 16 {
            if tid < s {
                *rval.add(tid as usize) += *rval.add((tid + s) as usize);
            }
            thread::sync_threads();
            s >>= 1;
        }
        if tid < 32 {
            let mut s = 16;
            while s > 0 {
                if tid < s {
                    *rval.add(tid as usize) += *rval.add((tid + s) as usize);
                }
                warp::sync_mask(u32::MAX);
                s >>= 1;
            }
        }
        thread::sync_threads();
        let sum_exp = *rval;

        let mut sel_sum = 0.0f32;
        if tid < 32 {
            let lane = tid;
            let mut p = [0.0f32; 16];
            p[0] = if lane + 0 < num_experts {
                *probs.add((lane + 0) as usize) / sum_exp
            } else {
                -1.0
            };
            p[1] = if lane + 32 < num_experts {
                *probs.add((lane + 32) as usize) / sum_exp
            } else {
                -1.0
            };
            p[2] = if lane + 64 < num_experts {
                *probs.add((lane + 64) as usize) / sum_exp
            } else {
                -1.0
            };
            p[3] = if lane + 96 < num_experts {
                *probs.add((lane + 96) as usize) / sum_exp
            } else {
                -1.0
            };
            p[4] = if lane + 128 < num_experts {
                *probs.add((lane + 128) as usize) / sum_exp
            } else {
                -1.0
            };
            p[5] = if lane + 160 < num_experts {
                *probs.add((lane + 160) as usize) / sum_exp
            } else {
                -1.0
            };
            p[6] = if lane + 192 < num_experts {
                *probs.add((lane + 192) as usize) / sum_exp
            } else {
                -1.0
            };
            p[7] = if lane + 224 < num_experts {
                *probs.add((lane + 224) as usize) / sum_exp
            } else {
                -1.0
            };
            p[8] = if lane + 256 < num_experts {
                *probs.add((lane + 256) as usize) / sum_exp
            } else {
                -1.0
            };
            p[9] = if lane + 288 < num_experts {
                *probs.add((lane + 288) as usize) / sum_exp
            } else {
                -1.0
            };
            p[10] = if lane + 320 < num_experts {
                *probs.add((lane + 320) as usize) / sum_exp
            } else {
                -1.0
            };
            p[11] = if lane + 352 < num_experts {
                *probs.add((lane + 352) as usize) / sum_exp
            } else {
                -1.0
            };
            p[12] = if lane + 384 < num_experts {
                *probs.add((lane + 384) as usize) / sum_exp
            } else {
                -1.0
            };
            p[13] = if lane + 416 < num_experts {
                *probs.add((lane + 416) as usize) / sum_exp
            } else {
                -1.0
            };
            p[14] = if lane + 448 < num_experts {
                *probs.add((lane + 448) as usize) / sum_exp
            } else {
                -1.0
            };
            p[15] = if lane + 480 < num_experts {
                *probs.add((lane + 480) as usize) / sum_exp
            } else {
                -1.0
            };
            let mut j = 0;
            while j < top_k {
                let (mut bv, mut bi) = (-1.0f32, -1i32);
                if p[0] > bv || (p[0] == bv && (bi < 0 || lane + 0 < bi)) {
                    bv = p[0];
                    bi = lane + 0;
                }
                if p[1] > bv || (p[1] == bv && (bi < 0 || lane + 32 < bi)) {
                    bv = p[1];
                    bi = lane + 32;
                }
                if p[2] > bv || (p[2] == bv && (bi < 0 || lane + 64 < bi)) {
                    bv = p[2];
                    bi = lane + 64;
                }
                if p[3] > bv || (p[3] == bv && (bi < 0 || lane + 96 < bi)) {
                    bv = p[3];
                    bi = lane + 96;
                }
                if p[4] > bv || (p[4] == bv && (bi < 0 || lane + 128 < bi)) {
                    bv = p[4];
                    bi = lane + 128;
                }
                if p[5] > bv || (p[5] == bv && (bi < 0 || lane + 160 < bi)) {
                    bv = p[5];
                    bi = lane + 160;
                }
                if p[6] > bv || (p[6] == bv && (bi < 0 || lane + 192 < bi)) {
                    bv = p[6];
                    bi = lane + 192;
                }
                if p[7] > bv || (p[7] == bv && (bi < 0 || lane + 224 < bi)) {
                    bv = p[7];
                    bi = lane + 224;
                }
                if p[8] > bv || (p[8] == bv && (bi < 0 || lane + 256 < bi)) {
                    bv = p[8];
                    bi = lane + 256;
                }
                if p[9] > bv || (p[9] == bv && (bi < 0 || lane + 288 < bi)) {
                    bv = p[9];
                    bi = lane + 288;
                }
                if p[10] > bv || (p[10] == bv && (bi < 0 || lane + 320 < bi)) {
                    bv = p[10];
                    bi = lane + 320;
                }
                if p[11] > bv || (p[11] == bv && (bi < 0 || lane + 352 < bi)) {
                    bv = p[11];
                    bi = lane + 352;
                }
                if p[12] > bv || (p[12] == bv && (bi < 0 || lane + 384 < bi)) {
                    bv = p[12];
                    bi = lane + 384;
                }
                if p[13] > bv || (p[13] == bv && (bi < 0 || lane + 416 < bi)) {
                    bv = p[13];
                    bi = lane + 416;
                }
                if p[14] > bv || (p[14] == bv && (bi < 0 || lane + 448 < bi)) {
                    bv = p[14];
                    bi = lane + 448;
                }
                if p[15] > bv || (p[15] == bv && (bi < 0 || lane + 480 < bi)) {
                    bv = p[15];
                    bi = lane + 480;
                }
                let mut off = 16;
                while off > 0 {
                    let cv = warp::shuffle_xor_f32_sync(u32::MAX, bv, off);
                    let ci = warp::shuffle_xor_sync(u32::MAX, bi as u32, off) as i32;
                    if cv > bv || (cv == bv && ci >= 0 && (bi < 0 || ci < bi)) {
                        bv = cv;
                        bi = ci;
                    }
                    off >>= 1;
                }
                if bi >= 0 && lane + 0 == bi {
                    p[0] = -1.0;
                }
                if bi >= 0 && lane + 32 == bi {
                    p[1] = -1.0;
                }
                if bi >= 0 && lane + 64 == bi {
                    p[2] = -1.0;
                }
                if bi >= 0 && lane + 96 == bi {
                    p[3] = -1.0;
                }
                if bi >= 0 && lane + 128 == bi {
                    p[4] = -1.0;
                }
                if bi >= 0 && lane + 160 == bi {
                    p[5] = -1.0;
                }
                if bi >= 0 && lane + 192 == bi {
                    p[6] = -1.0;
                }
                if bi >= 0 && lane + 224 == bi {
                    p[7] = -1.0;
                }
                if bi >= 0 && lane + 256 == bi {
                    p[8] = -1.0;
                }
                if bi >= 0 && lane + 288 == bi {
                    p[9] = -1.0;
                }
                if bi >= 0 && lane + 320 == bi {
                    p[10] = -1.0;
                }
                if bi >= 0 && lane + 352 == bi {
                    p[11] = -1.0;
                }
                if bi >= 0 && lane + 384 == bi {
                    p[12] = -1.0;
                }
                if bi >= 0 && lane + 416 == bi {
                    p[13] = -1.0;
                }
                if bi >= 0 && lane + 448 == bi {
                    p[14] = -1.0;
                }
                if bi >= 0 && lane + 480 == bi {
                    p[15] = -1.0;
                }
                if lane == 0 {
                    *topk_ids.add((token as i64 * top_k as i64 + j as i64) as usize) = bi;
                    *rval.add(j as usize) = bv;
                    sel_sum += bv;
                }
                j += 1;
            }
        }

        if tid == 0 {
            let out = topk_weights.add((token as i64 * top_k as i64) as usize);
            let mut j = 0;
            while j < top_k {
                *out.add(j as usize) = if sel_sum > 0.0 {
                    *rval.add(j as usize) / sel_sum
                } else {
                    1.0 / top_k as f32
                };
                j += 1;
            }
        }
    }
}

/// Build one block's dispatch tables from the routed ids: every
/// `(token, slot)` pair sorted by expert into runs padded to `block_size`
/// with the sentinel `numel`, the expert owning each bucket (`-1` for an
/// unused one), and the padded total. One thread per expert counts its
/// pairs in buckets, an in-warp prefix scan turns those into offsets,
/// and each thread then writes its own run, so the original order is
/// kept without an atomic.
///
/// # Safety
/// topk_ids covers valid_tokens * top_k ints; cumsum covers num_experts +
/// 1 ints and warp_sums one int per warp, both shared; sorted_token_ids
/// covers sorted_capacity, expert_ids expert_capacity, and
/// num_tokens_post_pad one int. block_size > 0.
#[device]
unsafe fn dispatch_token(
    topk_ids: *const i32,
    valid_tokens: *const i32,
    top_k: i32,
    num_experts: i32,
    block_size: i32,
    sorted_capacity: i32,
    expert_capacity: i32,
    cumsum: *mut i32,
    warp_sums: *mut i32,
    sorted_token_ids: *mut i32,
    expert_ids: *mut i32,
    num_tokens_post_pad: *mut i32,
) {
    let tid = thread::threadIdx_x() as i32;
    let block = thread::blockDim_x() as i32;
    unsafe {
        let numel = *valid_tokens * top_k;
        let mut s = tid;
        while s < sorted_capacity {
            *sorted_token_ids.add(s as usize) = numel;
            s += block;
        }
        let mut b = tid;
        while b < expert_capacity {
            *expert_ids.add(b as usize) = -1;
            b += block;
        }
        let mut e = tid;
        while e < num_experts {
            let c = count_equal(topk_ids, numel, e);
            *cumsum.add(e as usize + 1) = (c + block_size - 1) / block_size;
            e += block;
        }
        thread::sync_threads();

        if num_experts <= block {
            let (lane, warp_i) = (tid & 31, tid >> 5);
            let mut v = if tid < num_experts {
                *cumsum.add(tid as usize + 1)
            } else {
                0
            };
            let mut o = 1;
            while o < 32 {
                let n = warp::shuffle_up_sync(u32::MAX, v as u32, o as u32) as i32;
                if lane >= o {
                    v += n;
                }
                o <<= 1;
            }
            if lane == 31 {
                *warp_sums.add(warp_i as usize) = v;
            }
            thread::sync_threads();
            let mut w = 0;
            while w < warp_i {
                v += *warp_sums.add(w as usize);
                w += 1;
            }
            if tid < num_experts {
                *cumsum.add(tid as usize + 1) = v;
            }
            if tid == 0 {
                *cumsum = 0;
            }
            if tid == num_experts - 1 {
                *num_tokens_post_pad = v * block_size;
            }
        } else if tid == 0 {
            *cumsum = 0;
            let mut i = 0;
            while i < num_experts {
                *cumsum.add(i as usize + 1) += *cumsum.add(i as usize);
                i += 1;
            }
            *num_tokens_post_pad = *cumsum.add(num_experts as usize) * block_size;
        }
        thread::sync_threads();

        let mut e = tid;
        while e < num_experts {
            let first = *cumsum.add(e as usize);
            let base = first * block_size;
            let mut w = 0i32;
            let mut i = 0;
            while i < numel {
                if *topk_ids.add(i as usize) == e {
                    let slot = base + w;
                    if slot < sorted_capacity {
                        *sorted_token_ids.add(slot as usize) = i;
                    }
                    w += 1;
                }
                i += 1;
            }
            let last = *cumsum.add(e as usize + 1);
            let mut b = first;
            while b < last && b < expert_capacity {
                *expert_ids.add(b as usize) = e;
                b += 1;
            }
            e += block;
        }
    }
}

/// Route every live token: one block a token.
///
/// grid: (max_tokens,). block: 256, with num_experts + 2 * blockDim.x
/// floats of dynamic shared memory.
///
/// # Safety
/// As `route_token` for every token below `*valid_tokens`.
#[kernel]
pub unsafe fn moe_route(
    logits: *const f32,
    valid_tokens: *const i32,
    num_experts: i32,
    top_k: i32,
    topk_ids: *mut i32,
    topk_weights: *mut f32,
) {
    let token = thread::blockIdx_x() as i32;
    let shared = DynamicSharedArray::<f32>::get();
    unsafe {
        if token >= *valid_tokens {
            return;
        }
        route_token(
            logits,
            token,
            num_experts,
            top_k,
            shared,
            shared.add(num_experts as usize),
            topk_ids,
            topk_weights,
        );
    }
}

/// Dispatch a decode step's routes in one block.
///
/// grid: (1,). block: 256, with num_experts + 1 + blockDim.x / 32 ints of
/// dynamic shared memory.
///
/// # Safety
/// As `dispatch_token`.
#[kernel]
pub unsafe fn moe_dispatch_t1(
    topk_ids: *const i32,
    valid_tokens: *const i32,
    top_k: i32,
    num_experts: i32,
    block_size: i32,
    sorted_capacity: i32,
    expert_capacity: i32,
    sorted_token_ids: *mut i32,
    expert_ids: *mut i32,
    num_tokens_post_pad: *mut i32,
) {
    let cumsum = DynamicSharedArray::<i32>::get();
    unsafe {
        dispatch_token(
            topk_ids,
            valid_tokens,
            top_k,
            num_experts,
            block_size,
            sorted_capacity,
            expert_capacity,
            cumsum,
            cumsum.add(num_experts as usize + 1),
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
        );
    }
}

/// Route and dispatch one token in one launch: the decode step's router.
///
/// grid: (1,). block: 256. Shared memory holds the probabilities, the
/// reduction scratch, a spare block of ints and the dispatch prefix.
///
/// # Safety
/// As `route_token` at token 0 and `dispatch_token`.
#[kernel]
pub unsafe fn moe_route_dispatch_t1(
    logits: *const f32,
    valid_tokens: *const i32,
    num_experts: i32,
    top_k: i32,
    block_size: i32,
    sorted_capacity: i32,
    expert_capacity: i32,
    topk_ids: *mut i32,
    topk_weights: *mut f32,
    sorted_token_ids: *mut i32,
    expert_ids: *mut i32,
    num_tokens_post_pad: *mut i32,
) {
    let shared = DynamicSharedArray::<f32>::get();
    let block = thread::blockDim_x() as usize;
    unsafe {
        if *valid_tokens < 1 {
            return;
        }
        let rval = shared.add(num_experts as usize);
        let cumsum = rval.add(2 * block) as *mut i32;
        route_token(
            logits,
            0,
            num_experts,
            top_k,
            shared,
            rval,
            topk_ids,
            topk_weights,
        );
        thread::sync_threads();
        dispatch_token(
            topk_ids,
            valid_tokens,
            top_k,
            num_experts,
            block_size,
            sorted_capacity,
            expert_capacity,
            cumsum,
            cumsum.add(num_experts as usize + 1),
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
        );
    }
}

/// MoE alignment, pass two, for a batch: block `e` places expert `e`'s
/// `(token, slot)` pairs, in order, at its padded offset, and marks the
/// buckets it owns with how many live pairs each holds.
///
/// Every block recomputes the bucket prefix from `counts` (it is 257
/// ints). Pairs are scanned a warp-wide chunk at a time: a ballot of the
/// lanes that hit, and each hit's slot is the running total plus the
/// population count of the lower lanes, which keeps the original order.
///
/// grid: (num_experts,). block: 256, with num_experts + 1 + blockDim.x
/// ints of dynamic shared memory.
///
/// # Safety
/// topk_ids covers valid_tokens * top_k ints, counts num_experts;
/// sorted_token_ids covers sorted_capacity; expert_ids and bucket_live
/// expert_capacity; num_tokens_post_pad one int. block_size > 0.
#[kernel]
pub unsafe fn moe_align_block_size(
    topk_ids: *const i32,
    valid_tokens: *const i32,
    counts: *const i32,
    top_k: i32,
    num_experts: i32,
    block_size: i32,
    sorted_capacity: i32,
    expert_capacity: i32,
    sorted_token_ids: *mut i32,
    expert_ids: *mut i32,
    num_tokens_post_pad: *mut i32,
    bucket_live: *mut i32,
) {
    let cumsum = DynamicSharedArray::<i32>::get();
    let e = thread::blockIdx_x() as i32;
    let tid = thread::threadIdx_x() as i32;
    let block = thread::blockDim_x() as i32;
    unsafe {
        let scan = cumsum.add(num_experts as usize + 1);
        let numel = *valid_tokens * top_k;
        let mut i = tid;
        while i < num_experts {
            let c = *counts.add(i as usize);
            *cumsum.add(i as usize + 1) = (c + block_size - 1) / block_size;
            i += block;
        }
        thread::sync_threads();
        if tid == 0 {
            *cumsum = 0;
            let mut i = 0;
            while i < num_experts {
                *cumsum.add(i as usize + 1) += *cumsum.add(i as usize);
                i += 1;
            }
            if e == 0 {
                *num_tokens_post_pad = *cumsum.add(num_experts as usize) * block_size;
            }
        }
        thread::sync_threads();

        let first = *cumsum.add(e as usize);
        let base = first * block_size;
        let (lane, warp_i, n_warps) = (tid & 31, tid >> 5, block >> 5);
        let chunks = (numel + 31) >> 5;
        let per_warp = (chunks + n_warps - 1) / n_warps;
        let c0 = warp_i * per_warp;
        let c1 = if c0 + per_warp < chunks {
            c0 + per_warp
        } else {
            chunks
        };

        // Four chunks' ids are read before any ballot: a ballot is a
        // warp-wide barrier, and a load queued behind one waits for it.
        let mut found = 0i32;
        let mut c = c0;
        while c + 3 < c1 {
            let i = c * 32 + lane;
            let h0 = i < numel && *topk_ids.add(i as usize) == e;
            let h1 = i + 32 < numel && *topk_ids.add(i as usize + 32) == e;
            let h2 = i + 64 < numel && *topk_ids.add(i as usize + 64) == e;
            let h3 = i + 96 < numel && *topk_ids.add(i as usize + 96) == e;
            found += warp::ballot_sync(u32::MAX, h0).count_ones() as i32;
            found += warp::ballot_sync(u32::MAX, h1).count_ones() as i32;
            found += warp::ballot_sync(u32::MAX, h2).count_ones() as i32;
            found += warp::ballot_sync(u32::MAX, h3).count_ones() as i32;
            c += 4;
        }
        while c < c1 {
            let i = c * 32 + lane;
            let hit = i < numel && *topk_ids.add(i as usize) == e;
            found += warp::ballot_sync(u32::MAX, hit).count_ones() as i32;
            c += 1;
        }
        if lane == 0 {
            *scan.add(warp_i as usize) = found;
        }
        thread::sync_threads();

        let mut written = base;
        let mut w = 0;
        while w < warp_i {
            written += *scan.add(w as usize);
            w += 1;
        }
        let below = (1u32 << lane) - 1;
        macro_rules! place {
            ($hit:expr, $i:expr) => {
                let hits = warp::ballot_sync(u32::MAX, $hit);
                if $hit {
                    let slot = written + (hits & below).count_ones() as i32;
                    if slot < sorted_capacity {
                        *sorted_token_ids.add(slot as usize) = $i;
                    }
                }
                written += hits.count_ones() as i32;
            };
        }
        let mut c = c0;
        while c + 3 < c1 {
            let i = c * 32 + lane;
            let h0 = i < numel && *topk_ids.add(i as usize) == e;
            let h1 = i + 32 < numel && *topk_ids.add(i as usize + 32) == e;
            let h2 = i + 64 < numel && *topk_ids.add(i as usize + 64) == e;
            let h3 = i + 96 < numel && *topk_ids.add(i as usize + 96) == e;
            place!(h0, i);
            place!(h1, i + 32);
            place!(h2, i + 64);
            place!(h3, i + 96);
            c += 4;
        }
        while c < c1 {
            let i = c * 32 + lane;
            let hit = i < numel && *topk_ids.add(i as usize) == e;
            place!(hit, i);
            c += 1;
        }

        let last = *cumsum.add(e as usize + 1);
        let count = *counts.add(e as usize);
        let mut b = first + tid;
        while b < last && b < expert_capacity {
            *expert_ids.add(b as usize) = e;
            let live = count - (b - first) * block_size;
            let live = if live < 0 {
                0
            } else if live > block_size {
                block_size
            } else {
                live
            };
            *bucket_live.add(b as usize) = live;
            b += block;
        }
    }
}
