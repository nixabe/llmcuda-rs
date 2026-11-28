//! Layer ops: residual add, SwiGLU, sigmoid gating, RMSNorm and its fused
//! forms, softplus, standalone RoPE and the causal depthwise convolution.

use crate::common::{block_reduce_sum, conv_step_channel, inverse_rms};
use cuda_device::{kernel, thread, warp};

/// `out = a + b`, elementwise. The residual add.
///
/// One rounding, exactly as `llmcuda_kernels::norm::residual_add`, so it
/// is bit-identical to the reference and gated with `Tolerance::exact()`.
/// Grid-stride over a fixed grid, so the launch shape does not depend on
/// a host-side length and survives graph capture (AGENTS.md rule 5).
///
/// # Safety
/// All pointers cover `n` floats, n is nonnegative, and the launch is 1-D
/// with nonzero dimensions. Output may equal either input, but must not
/// partially overlap it. No other stream may access output concurrently.
#[kernel]
pub unsafe fn tensor_add(a: *const f32, b: *const f32, out: *mut f32, n: i64) {
    let stride = thread::blockDim_x() as i64 * thread::gridDim_x() as i64;
    let mut i =
        thread::blockIdx_x() as i64 * thread::blockDim_x() as i64 + thread::threadIdx_x() as i64;
    while i < n {
        // SAFETY: each thread owns its grid-stride indices; reads precede
        // the write so exact input/output aliasing is supported.
        unsafe {
            *out.add(i as usize) = *a.add(i as usize) + *b.add(i as usize);
        }
        i += stride;
    }
}

/// SwiGLU in the reference operand order, using libdevice exp rather than
/// an approximate exponential intrinsic.
///
/// # Safety
/// Pointers cover n floats, n >= 0, and the launch is nonempty and 1-D.
/// Output may exactly alias an input; partial overlap is forbidden.
#[kernel]
pub unsafe fn swiglu_mul(gate: *const f32, up: *const f32, out: *mut f32, n: i64) {
    let stride = thread::blockDim_x() as i64 * thread::gridDim_x() as i64;
    let mut i =
        thread::blockIdx_x() as i64 * thread::blockDim_x() as i64 + thread::threadIdx_x() as i64;
    // A capped grid makes each thread loop; four elements are read ahead
    // of the exponentials so the loads do not queue behind them. Each
    // element is still read before it is written, so exact aliasing holds.
    let st = stride as usize;
    unsafe {
        while i + 3 * stride < n {
            let (gp, up_p, op) = (
                gate.add(i as usize),
                up.add(i as usize),
                out.add(i as usize),
            );
            let g = (*gp, *gp.add(st), *gp.add(2 * st), *gp.add(3 * st));
            let u = (*up_p, *up_p.add(st), *up_p.add(2 * st), *up_p.add(3 * st));
            *op = (g.0 / (1.0 + (-g.0).exp())) * u.0;
            *op.add(st) = (g.1 / (1.0 + (-g.1).exp())) * u.1;
            *op.add(2 * st) = (g.2 / (1.0 + (-g.2).exp())) * u.2;
            *op.add(3 * st) = (g.3 / (1.0 + (-g.3).exp())) * u.3;
            i += 4 * stride;
        }
        while i < n {
            let g = *gate.add(i as usize);
            *out.add(i as usize) = (g / (1.0 + (-g).exp())) * *up.add(i as usize);
            i += stride;
        }
    }
}

/// Elementwise or per-row sigmoid gating, preserving the sigmoid waypoint.
///
/// With `broadcast`, row `i / width` gates every element of its row and
/// its sigmoid is stored once, by the thread holding the row's first
/// element. The row and the column are carried along the grid-stride
/// walk instead of being divided out per element: the 64-bit divide is
/// most of the loop otherwise.
///
/// # Safety
/// x/out cover n floats; gate/sig cover n floats or ceil(n/width) when
/// broadcast != 0. width > 0, n >= 0, and the launch is nonempty and 1-D.
/// Only out and x may exactly alias; all other overlaps are forbidden.
#[kernel]
pub unsafe fn sigmoid_gate_mul(
    x: *const f32,
    gate: *const f32,
    sig: *mut f32,
    out: *mut f32,
    n: i64,
    width: i32,
    broadcast: i32,
) {
    let stride = thread::blockDim_x() as i64 * thread::gridDim_x() as i64;
    let mut i =
        thread::blockIdx_x() as i64 * thread::blockDim_x() as i64 + thread::threadIdx_x() as i64;
    unsafe {
        if broadcast == 0 {
            while i < n {
                let s = 1.0 / (1.0 + (-*gate.add(i as usize)).exp());
                *sig.add(i as usize) = s;
                *out.add(i as usize) = s * *x.add(i as usize);
                i += stride;
            }
        } else {
            let width = width as i64;
            let row_step = stride / width;
            let col_step = stride - row_step * width;
            let mut row = i / width;
            let mut col = i - row * width;
            while i < n {
                let s = 1.0 / (1.0 + (-*gate.add(row as usize)).exp());
                if col == 0 {
                    *sig.add(row as usize) = s;
                }
                *out.add(i as usize) = s * *x.add(i as usize);
                i += stride;
                row += row_step;
                col += col_step;
                if col >= width {
                    col -= width;
                    row += 1;
                }
            }
        }
    }
}

/// `out = log(1 + exp(x))`, with a passthrough above 20. The GDN alpha
/// gate's nonlinearity: `a_softplus = softplus(alpha + ssm_dt.bias)`.
///
/// `if x > 20 { x } else { ln(1 + exp(x)) }` is ggml's formulation
/// verbatim (`op_softplus` in `ggml/src/ggml-cuda/unary.cu` and
/// `ggml/src/ggml-cpu/unary-ops.cpp`, `ggml_compute_softplus_f32` in
/// `ggml/src/ggml-impl.h`). **The threshold is a correctness guard, not
/// an optimization:** exp overflows fp32 above x ~ 88.7, so a branch-free
/// softplus returns inf where the answer is x, and that inf becomes a
/// non-finite log-decay that poisons the whole recurrence. Where the
/// branch is taken it is invisible: log(1 + e^20) - 20 is ~2.06e-9.
///
/// # Safety
/// x/out cover n floats, n >= 0, and the launch is nonempty and 1-D.
/// Output may exactly alias x; partial overlap is forbidden.
#[kernel]
pub unsafe fn softplus_elementwise(x: *const f32, out: *mut f32, n: i64) {
    let stride = thread::blockDim_x() as i64 * thread::gridDim_x() as i64;
    let mut i =
        thread::blockIdx_x() as i64 * thread::blockDim_x() as i64 + thread::threadIdx_x() as i64;
    while i < n {
        unsafe {
            let v = *x.add(i as usize);
            *out.add(i as usize) = if v > 20.0 { v } else { (1.0 + v.exp()).ln() };
        }
        i += stride;
    }
}

/// Partial rotary embedding, NEOX (split-half) pairing: dimension i
/// pairs with i + rope_dim/2, and dimensions [rope_dim, head_dim) pass
/// through untouched. Paired outputs share one angle.
///
/// **The tail is copied, not computed.** A load and a store with no
/// arithmetic between them preserve the bits, so the differential test
/// gates that span with `Tolerance::exact()`. Multiplying the tail by the
/// cosine of angle 0 would be right to an ulp and would fail that gate,
/// which is the point of having it.
///
/// Angles are double precision because the reference's are
/// (`f64::from(theta_base).powf(...)`, `pos * freq`, then `as f32`).
/// The first product of each output is fused and the second rounded on
/// its own; that is the rounding the fused attention kernels share.
///
/// # Safety
/// x/out are disjoint buffers covering gridDim.y * heads * head_dim
/// floats. positions covers gridDim.y. gridDim.x == heads, blockDim.x ==
/// head_dim, and 0 <= rope_dim <= head_dim is even. theta_base > 0.
#[kernel]
pub unsafe fn rope_partial(
    x: *const f32,
    positions: *const u32,
    out: *mut f32,
    heads: i32,
    head_dim: i32,
    rope_dim: i32,
    theta_base: f32,
) {
    let h = thread::blockIdx_x() as usize;
    let t = thread::blockIdx_y() as usize;
    let j = thread::threadIdx_x() as i32;
    let base = (t * heads as usize + h) * head_dim as usize;
    unsafe {
        if j >= rope_dim {
            *out.add(base + j as usize) = *x.add(base + j as usize);
        } else if j < rope_dim / 2 {
            let half = rope_dim / 2;
            let freq = (theta_base as f64).powf(-2.0 * j as f64 / rope_dim as f64);
            let angle = *positions.add(t) as f64 * freq;
            let sin_a = angle.sin() as f32;
            let cos_a = angle.cos() as f32;
            let x0 = *x.add(base + j as usize);
            let x1 = *x.add(base + (j + half) as usize);
            *out.add(base + j as usize) = x0.mul_add(cos_a, -(x1 * sin_a));
            *out.add(base + (j + half) as usize) = x0.mul_add(sin_a, x1 * cos_a);
        }
    }
}

/// `y = x / sqrt(mean(x^2) + eps) * weight`, one row a block.
///
/// Rows wider than the block are covered by a strided loop, which lets
/// one kernel serve hidden 2048 and head_dim 128. `weight` is one row of
/// `width` floats shared by every row of `x`.
///
/// The arithmetic transcribes the reference operand for operand: sum of
/// squares, divide by width, add eps, one `1 / sqrt`, then
/// `x * inv_rms * weight` in that order. A reciprocal square root would
/// be faster and is not IEEE-exact; matching the reference's division
/// keeps the disagreement attributable to the reduction order alone.
///
/// # Safety
/// x/out cover gridDim.x * width floats and weight covers width floats.
/// Only out may exactly alias x. The inverse_rms launch contract holds;
/// no other stream accesses output concurrently.
#[kernel]
pub unsafe fn rms_norm_rows(
    x: *const f32,
    weight: *const f32,
    out: *mut f32,
    width: i32,
    eps: f32,
) {
    unsafe {
        let inv = inverse_rms(x, width, eps);
        let base = thread::blockIdx_x() as usize * width as usize;
        let mut j = thread::threadIdx_x() as i32;
        while j < width {
            *out.add(base + j as usize) = *x.add(base + j as usize) * inv * *weight.add(j as usize);
            j += thread::blockDim_x() as i32;
        }
    }
}

/// RMSNorm and the SwiGLU multiply that consumes it, in one launch.
///
/// The Gated DeltaNet's output norm is immediately multiplied by
/// `silu(z)` over exactly the same elements, so the multiply rides along
/// in the norm's second pass. The reduction is `rms_norm_rows`', so
/// `normed` is bit-identical to what that kernel writes, and it is still
/// written: the block's differential test compares it.
///
/// # Safety
/// x/gate/normed/out cover gridDim.x * width floats, weight covers width,
/// outputs are disjoint from each other and inputs, and inverse_rms's
/// collective launch contract holds. No concurrent output access.
#[kernel]
pub unsafe fn rms_norm_swiglu_rows(
    x: *const f32,
    weight: *const f32,
    gate: *const f32,
    normed: *mut f32,
    out: *mut f32,
    width: i32,
    eps: f32,
) {
    unsafe {
        let inv = inverse_rms(x, width, eps);
        let base = thread::blockIdx_x() as usize * width as usize;
        let mut j = thread::threadIdx_x() as i32;
        while j < width {
            let nv = *x.add(base + j as usize) * inv * *weight.add(j as usize);
            *normed.add(base + j as usize) = nv;
            let g = *gate.add(base + j as usize);
            *out.add(base + j as usize) = (g / (1.0 + (-g).exp())) * nv;
            j += thread::blockDim_x() as i32;
        }
    }
}

/// `rms_norm_swiglu_rows` handing its output straight to the integer
/// tensor cores: int8 quants and one scale per 32 elements.
///
/// At prefill width the Gated DeltaNet's output has one reader, the
/// output projection, which reads it as int8. The row is a whole number
/// of 32-element blocks and the block is one thread an element, so each
/// warp holds one quantization block in its lanes and reduces the
/// absolute maximum and rounds in place; `normed` and `out` are never
/// written as floats. Both halves keep their own kernels' arithmetic
/// (absmax by xor butterfly, `a / 127` and `127 / a`, round to even,
/// clamp), so quants and scales are bit-identical to the two-launch form
/// (`gdn_tail_differential`).
///
/// # Safety
/// x/gate cover gridDim.x * width floats, weight covers width, q covers
/// gridDim.x * width bytes and scales gridDim.x * width / 32 floats.
/// blockDim.x == width, a multiple of 32 and at most 1024.
#[kernel]
pub unsafe fn rms_norm_swiglu_q8_rows(
    x: *const f32,
    weight: *const f32,
    gate: *const f32,
    q: *mut i8,
    scales: *mut f32,
    width: i32,
    eps: f32,
) {
    let j = thread::threadIdx_x() as usize;
    let base = thread::blockIdx_x() as usize * width as usize;
    unsafe {
        let v = *x.add(base + j);
        let sum_sq = block_reduce_sum(v * v);
        let inv_rms = 1.0 / (sum_sq / width as f32 + eps).sqrt();
        let nv = v * inv_rms * *weight.add(j);
        let g = *gate.add(base + j);
        let o = (g / (1.0 + (-g).exp())) * nv;

        let mut a = o.abs();
        let mut off = 16;
        while off > 0 {
            a = a.max(warp::shuffle_xor_f32_sync(u32::MAX, a, off));
            off >>= 1;
        }
        let d = if a > 0.0 { a / 127.0 } else { 1.0 };
        let inv = if a > 0.0 { 127.0 / a } else { 0.0 };
        let mut qi = (o * inv).round_ties_even() as i32;
        qi = if qi < -127 {
            -127
        } else if qi > 127 {
            127
        } else {
            qi
        };
        *q.add(base + j) = qi as i8;
        if j & 31 == 0 {
            *scales.add((base + j) >> 5) = d;
        }
    }
}

/// Causal depthwise convolution over the fused GDN q/k/v stream,
/// `[tokens][channels]`, reading the carried cache for positions before
/// this batch.
///
/// `out[t][ch] = sum_{i<K} w[ch][i] * xw(t - (K-1) + i, ch)`. Tap K-1
/// multiplies the *current* token and tap 0 the oldest, which is
/// `ggml_compute_forward_ssm_conv_f32`'s window order
/// (`ggml/src/ggml-cpu/ops.cpp`) and is what a reversed loop gets
/// silently backwards. Taps accumulate in
/// ascending order, each product rounded before it is added: the PTX is
/// built without implicit fusion, which is what keeps this bit-identical
/// to the scalar reference.
///
/// # Safety
/// x/out cover gridDim.y * channels floats, weight covers channels *
/// conv_kernel, state covers channels * (conv_kernel - 1); out is
/// disjoint from the inputs. 1 <= conv_kernel.
#[kernel]
pub unsafe fn conv1d_causal_depthwise(
    x: *const f32,
    weight: *const f32,
    state: *const f32,
    out: *mut f32,
    channels: i32,
    conv_kernel: i32,
) {
    let ch = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as i32;
    if ch >= channels {
        return;
    }
    let t = thread::blockIdx_y() as i32;
    let carry = conv_kernel - 1;
    // Tap i reads position `t - carry + i`: this batch when that is
    // non-negative, else cache slot `t + i`. One multiply each, then the
    // pointers walk.
    let first = t - carry;
    unsafe {
        let mut xp = x.offset((first as i64 * channels as i64 + ch as i64) as isize);
        let mut sp = state.offset((ch as i64 * carry as i64 + t as i64) as isize);
        let mut wp = weight.offset((ch as i64 * conv_kernel as i64) as isize);
        let mut acc = 0.0f32;
        let mut i = 0;
        while i < conv_kernel {
            let xv = if first + i >= 0 { *xp } else { *sp };
            acc += xv * *wp;
            xp = xp.wrapping_offset(channels as isize);
            sp = sp.wrapping_add(1);
            wp = wp.wrapping_add(1);
            i += 1;
        }
        *out.offset((t as i64 * channels as i64 + ch as i64) as isize) = acc;
    }
}

/// The whole convolution for a one-token step, cache advanced in place.
///
/// A batch needs a second kernel for the cache because its advance
/// depends on the last inputs of the whole batch. At one token the
/// window is the cache plus this token, and the new cache is the old one
/// shifted by one with this token appended.
///
/// # Safety
/// x/out cover channels floats, weight channels * conv_kernel, state
/// channels * (conv_kernel - 1); 1 <= conv_kernel.
#[kernel]
pub unsafe fn conv1d_step(
    x: *const f32,
    weight: *const f32,
    state: *mut f32,
    out: *mut f32,
    channels: i32,
    conv_kernel: i32,
) {
    let ch = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as i32;
    if ch >= channels {
        return;
    }
    let carry = conv_kernel - 1;
    unsafe {
        *out.add(ch as usize) = conv_step_channel(
            *x.add(ch as usize),
            weight.offset((ch as i64 * conv_kernel as i64) as isize),
            state.offset((ch as i64 * carry as i64) as isize),
            carry,
        );
    }
}

/// `conv1d_step` over up to eight sequences, one grid.y slice each; the
/// per-sequence caches arrive as eight address slots.
///
/// # Safety
/// As `conv1d_step` per slice; x/out cover gridDim.y * channels floats
/// and slots 0..gridDim.y hold distinct valid cache addresses.
#[kernel]
pub unsafe fn conv1d_step_batch(
    x: *const f32,
    weight: *const f32,
    s0: u64,
    s1: u64,
    s2: u64,
    s3: u64,
    s4: u64,
    s5: u64,
    s6: u64,
    s7: u64,
    out: *mut f32,
    channels: i32,
    conv_kernel: i32,
) {
    let ch = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as i32;
    if ch >= channels {
        return;
    }
    let z = thread::blockIdx_y() as i32;
    let state = match z {
        0 => s0,
        1 => s1,
        2 => s2,
        3 => s3,
        4 => s4,
        5 => s5,
        6 => s6,
        _ => s7,
    } as *mut f32;
    let carry = conv_kernel - 1;
    let at = (z as i64 * channels as i64 + ch as i64) as isize;
    unsafe {
        *out.offset(at) = conv_step_channel(
            *x.offset(at),
            weight.offset((ch as i64 * conv_kernel as i64) as isize),
            state.offset((ch as i64 * carry as i64) as isize),
            carry,
        );
    }
}

/// Advance the convolution cache to the last conv_kernel - 1 inputs.
///
/// A batch shorter than conv_kernel - 1 (every decode step) reads cache
/// entries it also overwrites. llama.cpp avoids that by taking the new
/// state as a view into a freshly concatenated buffer
/// (`llm_build_delta_net_base::build_conv_state`); there is no such
/// buffer here.
///
/// Safe in place without staging: slot j takes either a value of `x` or
/// cache slot `j + seq_len`, always a later slot than any already
/// written, and a thread owns its channel's whole cache row.
///
/// # Safety
/// x covers seq_len * channels floats with seq_len >= 1, state channels
/// * (conv_kernel - 1); 1 <= conv_kernel and the launch is nonempty, 1-D.
#[kernel]
pub unsafe fn conv1d_update_state(
    x: *const f32,
    state: *mut f32,
    seq_len: i32,
    channels: i32,
    conv_kernel: i32,
) {
    let carry = conv_kernel - 1;
    let stride = thread::blockDim_x() as i64 * thread::gridDim_x() as i64;
    let mut ch =
        thread::blockIdx_x() as i64 * thread::blockDim_x() as i64 + thread::threadIdx_x() as i64;
    unsafe {
        while ch < channels as i64 {
            let row = state.offset((ch * carry as i64) as isize);
            let mut j = 0;
            while j < carry {
                let u = seq_len - carry + j;
                *row.add(j as usize) = if u >= 0 {
                    *x.offset((u as i64 * channels as i64 + ch) as isize)
                } else {
                    *row.add((u + carry) as usize)
                };
                j += 1;
            }
            ch += stride;
        }
    }
}
