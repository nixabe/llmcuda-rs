//! The engine's Rust CUDA kernels: layer ops, greedy sampling, and the small
//! MoE, Gated DeltaNet, K2, attention and glue kernels.
//!
//! This is the only source for the kernels the engine launches from the
//! embedded PTX; there is no CUDA C++ twin. Raw pointers keep the cudarc
//! launch ABI, including in-place output. The CPU references are in
//! `llmcuda-kernels`, and the launchers and their differential tests in
//! `llmcuda-cuda` and `llmcuda-engine`.
//!
//! The PTX is built without implicit multiply-add fusion (`regenerate.py`),
//! so each `mul_add` here is one rounding and every other product is rounded
//! before it is added. Several kernels are compared bit for bit with fused
//! CUDA C++ kernels that share their arithmetic; do not reorder a sum or
//! change which products fuse without changing those too.

use cuda_device::{
    DynamicSharedArray, SharedArray, device, kernel, ptx_asm, thread, warp,
};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

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
        let mut i = thread::blockIdx_x() as i64 * thread::blockDim_x() as i64
            + thread::threadIdx_x() as i64;
        while i < n {
            // SAFETY: each thread owns its grid-stride indices; reads precede
            // the write so exact input/output aliasing is supported.
            unsafe {
                *out.add(i as usize) = *a.add(i as usize) + *b.add(i as usize);
            }
            i += stride;
        }
    }

    /// SwiGLU in the baseline operand order, using libdevice exp rather than
    /// an approximate exponential intrinsic.
    ///
    /// # Safety
    /// Pointers cover n floats, n >= 0, and the launch is nonempty and 1-D.
    /// Output may exactly alias an input; partial overlap is forbidden.
    #[kernel]
    pub unsafe fn swiglu_mul(gate: *const f32, up: *const f32, out: *mut f32, n: i64) {
        let stride = thread::blockDim_x() as i64 * thread::gridDim_x() as i64;
        let mut i = thread::blockIdx_x() as i64 * thread::blockDim_x() as i64
            + thread::threadIdx_x() as i64;
        // A capped grid makes each thread loop; four elements are read ahead
        // of the exponentials so the loads do not queue behind them. Each
        // element is still read before it is written, so exact aliasing holds.
        let st = stride as usize;
        unsafe {
            while i + 3 * stride < n {
                let (gp, up_p, op) = (gate.add(i as usize), up.add(i as usize), out.add(i as usize));
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
        let mut i = thread::blockIdx_x() as i64 * thread::blockDim_x() as i64
            + thread::threadIdx_x() as i64;
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
        let mut i = thread::blockIdx_x() as i64 * thread::blockDim_x() as i64
            + thread::threadIdx_x() as i64;
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
                // Match NVRTC's contraction while sharing the angle and loads.
                *out.add(base + j as usize) = x0.mul_add(cos_a, -(x1 * sin_a));
                *out.add(base + (j + half) as usize) = x0.mul_add(sin_a, x1 * cos_a);
            }
        }
    }

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

    /// `block_reduce_sum` from the CUDA C++ layer ops, operation for operation:
    /// a shuffle-down tree per warp, then thread 0 adds the warp totals in
    /// ascending order. The order is the contract -- the fused Gated DeltaNet
    /// kernels reduce the same way and are compared bit for bit.
    ///
    /// # Safety
    /// Every thread participates in a 1-D block of 32..=1024 threads, a
    /// multiple of 32. Dynamic shared memory holds one f32 per warp.
    #[device]
    unsafe fn block_reduce_sum(mut v: f32) -> f32 {
        let mut offset = 16;
        while offset > 0 {
            v += warp::shuffle_down_f32_sync(u32::MAX, v, offset);
            offset >>= 1;
        }
        let tid = thread::threadIdx_x();
        let scratch = DynamicSharedArray::<f32>::get();
        unsafe {
            if tid & 31 == 0 {
                *scratch.add((tid >> 5) as usize) = v;
            }
            thread::sync_threads();
            if tid == 0 {
                let warps = (thread::blockDim_x() + 31) >> 5;
                let mut total = 0.0f32;
                let mut w = 0;
                while w < warps {
                    total += *scratch.add(w as usize);
                    w += 1;
                }
                *scratch = total;
            }
            thread::sync_threads();
            *scratch
        }
    }

    /// Inverse RMS of this block's row, with the reference's exact division.
    ///
    /// # Safety
    /// x covers width floats for every grid row, width > 0, and
    /// block_reduce_sum's launch contract holds.
    #[device]
    unsafe fn inverse_rms(x: *const f32, width: i32, eps: f32) -> f32 {
        let base = thread::blockIdx_x() as usize * width as usize;
        let mut j = thread::threadIdx_x() as i32;
        let mut partial = 0.0f32;
        unsafe {
            while j < width {
                let v = *x.add(base + j as usize);
                // NVRTC contracts the baseline's partial += v * v.
                partial = v.mul_add(v, partial);
                j += thread::blockDim_x() as i32;
            }
            1.0 / (block_reduce_sum(partial) / width as f32 + eps).sqrt()
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
                *out.add(base + j as usize) =
                    *x.add(base + j as usize) * inv * *weight.add(j as usize);
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

    /// One channel's one-token convolution step: output, then the cache
    /// shifted by one with this token appended.
    ///
    /// The shift runs oldest slot first and each slot takes its successor,
    /// so no slot is read after it is written and nothing needs staging.
    ///
    /// # Safety
    /// state covers `carry` floats owned by this thread and weight covers
    /// `carry + 1`.
    #[device]
    unsafe fn conv_step_channel(xv0: f32, weight: *const f32, state: *mut f32, carry: i32) -> f32 {
        unsafe {
            let mut acc = 0.0f32;
            let mut i = 0;
            while i < carry {
                acc += *state.add(i as usize) * *weight.add(i as usize);
                i += 1;
            }
            acc += xv0 * *weight.add(carry as usize);
            let mut j = 0;
            while j + 1 < carry {
                *state.add(j as usize) = *state.add((j + 1) as usize);
                j += 1;
            }
            if carry > 0 {
                *state.add((carry - 1) as usize) = xv0;
            }
            acc
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
        let mut ch = thread::blockIdx_x() as i64 * thread::blockDim_x() as i64
            + thread::threadIdx_x() as i64;
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
            *gate.add(dst as usize) =
                *packed.add((row + (2 * h + 1) * head_dim as i64 + d) as usize);
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
        let i = thread::blockIdx_x() as i64 * thread::blockDim_x() as i64
            + thread::threadIdx_x() as i64;
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

    /// `out = silu(x)`, elementwise: `x / (1 + exp(-x))`, which is
    /// `ggml_cuda_op_silu_single` verbatim. libdevice exp, not an approximate
    /// one: the fast intrinsic's error is relative to the result rather than
    /// to the exponent, and this is not a bottleneck.
    ///
    /// # Safety
    /// x and out cover n floats; the 1-D launch is nonempty. Output may
    /// exactly alias the input.
    #[kernel]
    pub unsafe fn gdn_silu(x: *const f32, out: *mut f32, n: i64) {
        let stride = thread::blockDim_x() as i64 * thread::gridDim_x() as i64;
        let mut i = thread::blockIdx_x() as i64 * thread::blockDim_x() as i64
            + thread::threadIdx_x() as i64;
        while i < n {
            unsafe {
                let v = *x.add(i as usize);
                *out.add(i as usize) = v / (1.0 + (-v).exp());
            }
            i += stride;
        }
    }

    /// Split a convolved Gated DeltaNet row `[q | k | v]` into its three
    /// per-head tensors. Queries and keys have `qk_heads` heads, values
    /// `value_heads`; the grid covers the value heads and the first
    /// `qk_heads` blocks also write q and k.
    ///
    /// grid: (value_heads, tokens). block: head_dim.
    ///
    /// # Safety
    /// conv covers tokens * (2 * qk_heads + value_heads) * head_dim floats;
    /// q and k cover tokens * qk_heads * head_dim and v covers tokens *
    /// value_heads * head_dim; outputs are disjoint from conv and each other.
    #[kernel]
    pub unsafe fn gdn_split_qkv(
        conv: *const f32,
        q: *mut f32,
        k: *mut f32,
        v: *mut f32,
        head_dim: i32,
        qk_heads: i32,
        value_heads: i32,
    ) {
        let h = thread::blockIdx_x() as i64;
        let t = thread::blockIdx_y() as i64;
        let d = thread::threadIdx_x() as i64;
        let head_dim = head_dim as i64;
        let key_dim = qk_heads as i64 * head_dim;
        let conv_dim = 2 * key_dim + value_heads as i64 * head_dim;
        unsafe {
            let c = conv.add((t * conv_dim) as usize);
            *v.add(((t * value_heads as i64 + h) * head_dim + d) as usize) =
                *c.add((2 * key_dim + h * head_dim + d) as usize);
            if h < qk_heads as i64 {
                let o = ((t * qk_heads as i64 + h) * head_dim + d) as usize;
                *q.add(o) = *c.add((h * head_dim + d) as usize);
                *k.add(o) = *c.add((key_dim + h * head_dim + d) as usize);
            }
        }
    }

    /// The per-head decay and write gates.
    ///
    /// ```text
    /// a_softplus = softplus(alpha + ssm_dt.bias)
    /// log_decay  = a_softplus * ssm_a          <- already the log-decay
    /// beta       = sigmoid(beta_raw)
    /// ```
    ///
    /// `ssm_a` is stored already negated, so the product is the log of a
    /// decay in (0, 1] and must not be negated again. The softplus is
    /// `ggml_cuda_op_softplus` verbatim, including its x > 20 passthrough,
    /// which matters because a lost passthrough overflows exp rather than
    /// saturating.
    ///
    /// # Safety
    /// alpha, beta_raw and the three outputs cover n floats (n = tokens *
    /// heads); dt_bias and ssm_a cover heads. Outputs are disjoint from the
    /// inputs and each other. The 1-D launch is nonempty.
    #[kernel]
    pub unsafe fn gdn_gates(
        alpha: *const f32,
        beta_raw: *const f32,
        dt_bias: *const f32,
        ssm_a: *const f32,
        a_softplus: *mut f32,
        log_decay: *mut f32,
        beta: *mut f32,
        heads: i32,
        n: i64,
    ) {
        let stride = thread::blockDim_x() as i64 * thread::gridDim_x() as i64;
        let mut i = thread::blockIdx_x() as i64 * thread::blockDim_x() as i64
            + thread::threadIdx_x() as i64;
        while i < n {
            unsafe {
                let h = (i % heads as i64) as usize;
                let a = *alpha.add(i as usize) + *dt_bias.add(h);
                let sp = if a > 20.0 { a } else { (1.0 + a.exp()).ln() };
                *a_softplus.add(i as usize) = sp;
                *log_decay.add(i as usize) = sp * *ssm_a.add(h);
                *beta.add(i as usize) = 1.0 / (1.0 + (-*beta_raw.add(i as usize)).exp());
            }
            i += stride;
        }
    }

    /// `exp(log_decay)` for the whole pass, once.
    ///
    /// The unfolded scan takes this exponential in every lane of every warp
    /// of a head, on every token; the folded scan reads it from here instead.
    /// It is the same libdevice exp on the same operand, so the decay is the
    /// same bits.
    ///
    /// # Safety
    /// log_decay and decay cover n floats; the 1-D launch is nonempty.
    #[kernel]
    pub unsafe fn gdn_scan_decay(log_decay: *const f32, decay: *mut f32, n: i64) {
        let stride = thread::blockDim_x() as i64 * thread::gridDim_x() as i64;
        let mut i = thread::blockIdx_x() as i64 * thread::blockDim_x() as i64
            + thread::threadIdx_x() as i64;
        while i < n {
            unsafe {
                *decay.add(i as usize) = (*log_decay.add(i as usize)).exp();
            }
            i += stride;
        }
    }

    /// The weighted sum of one output column over a token's top-k
    /// projections, ascending k, each product fused into the running sum.
    ///
    /// # Safety
    /// p points at the token's first projection for this column and w at its
    /// first weight; p is read at topk strides of `rows`, w at topk floats.
    #[device]
    unsafe fn k2_weighted(p: *const f32, w: *const f32, rows: i32, topk: i32) -> f32 {
        let (mut p, mut w) = (p, w);
        let mut a = 0.0f32;
        let mut k = 0;
        while k < topk {
            unsafe {
                a = (*p).mul_add(*w, a);
            }
            p = p.wrapping_add(rows as usize);
            w = w.wrapping_add(1);
            k += 1;
        }
        a
    }

    /// `out[t][j] = sum_k p[t][k][j] * w[t][k]`.
    ///
    /// grid: (ceil(rows / block), tokens).
    ///
    /// # Safety
    /// p covers tokens * topk * rows floats, w tokens * topk, out tokens *
    /// rows; out is disjoint from the inputs.
    #[kernel]
    pub unsafe fn k2_combine(p: *const f32, w: *const f32, out: *mut f32, rows: i32, topk: i32) {
        let j = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as i32;
        let t = thread::blockIdx_y() as i64;
        if j >= rows {
            return;
        }
        unsafe {
            let a = k2_weighted(
                p.add((t * topk as i64 * rows as i64 + j as i64) as usize),
                w.add((t * topk as i64) as usize),
                rows,
                topk,
            );
            *out.add((t * rows as i64 + j as i64) as usize) = a;
        }
    }

    /// `k2_combine` followed by the two adds it used to feed: `ffn = routed +
    /// shared`, then `out = x + ffn`, each rounded as the separate
    /// `tensor_add` launches did.
    ///
    /// # Safety
    /// As `k2_combine`; shared, x, ffn and out cover tokens * rows floats.
    /// ffn and out are disjoint from each other and from p and w; out may
    /// exactly alias x.
    #[kernel]
    pub unsafe fn k2_combine_residual(
        p: *const f32,
        w: *const f32,
        shared: *const f32,
        x: *const f32,
        ffn: *mut f32,
        out: *mut f32,
        rows: i32,
        topk: i32,
    ) {
        let j = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as i32;
        let t = thread::blockIdx_y() as i64;
        if j >= rows {
            return;
        }
        unsafe {
            let a = k2_weighted(
                p.add((t * topk as i64 * rows as i64 + j as i64) as usize),
                w.add((t * topk as i64) as usize),
                rows,
                topk,
            );
            let i = (t * rows as i64 + j as i64) as usize;
            let f = a + *shared.add(i);
            *ffn.add(i) = f;
            *out.add(i) = *x.add(i) + f;
        }
    }

    /// `out[t][j] = sum_k silu(p[t][k][j]) * w[t][k]`, ascending k, each
    /// product fused into the running sum.
    ///
    /// # Safety
    /// As `k2_combine`.
    #[kernel]
    pub unsafe fn k2_values(
        projections: *const f32,
        weights: *const f32,
        out: *mut f32,
        rows: i32,
        topk: i32,
    ) {
        let j = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as i32;
        let token = thread::blockIdx_y() as i64;
        if j >= rows {
            return;
        }
        unsafe {
            let mut p = projections.add((token * topk as i64 * rows as i64 + j as i64) as usize);
            let mut w = weights.add((token * topk as i64) as usize);
            let mut acc = 0.0f32;
            let mut k = 0;
            while k < topk {
                let v = *p;
                acc = (v / (1.0 + (-v).exp())).mul_add(*w, acc);
                p = p.wrapping_add(rows as usize);
                w = w.wrapping_add(1);
                k += 1;
            }
            *out.add((token * rows as i64 + j as i64) as usize) = acc;
        }
    }

    /// `out = x * softplus2(gate)`, the base-2 softplus: the gate is scaled
    /// by ln 2, taken through the overflow-free `max(v, 0) + log1p(exp(-|v|))`
    /// and scaled back by 1 / ln 2.
    ///
    /// # Safety
    /// x, gate and out cover n floats; the 1-D launch is nonempty. Output may
    /// exactly alias an input.
    #[kernel]
    pub unsafe fn k2_gate(x: *const f32, gate: *const f32, out: *mut f32, n: i64) {
        let i = thread::blockIdx_x() as i64 * thread::blockDim_x() as i64
            + thread::threadIdx_x() as i64;
        if i >= n {
            return;
        }
        unsafe {
            let v = *gate.add(i as usize) * 0.693_147_2_f32;
            let sp = v.max(0.0) + (-v.abs()).exp().ln_1p();
            *out.add(i as usize) = *x.add(i as usize) * (sp * 1.442_695_f32);
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
    /// # Safety
    /// part_acc covers n_splits * q_heads * head_dim floats, part_m and
    /// part_l n_splits * q_heads, out q_heads * head_dim; out is disjoint
    /// from the inputs. n_splits >= 1, blockIdx.x < q_heads and threadIdx.x <
    /// head_dim.
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
        let h = thread::blockIdx_x() as usize;
        let d = thread::threadIdx_x() as usize;
        let row = q_heads as usize;
        unsafe {
            let mut gm = f32::NEG_INFINITY;
            let mut m = part_m.add(h);
            let mut s = 0;
            while s < n_splits {
                gm = gm.max(*m);
                m = m.wrapping_add(row);
                s += 1;
            }

            let mut m = part_m.add(h);
            let mut l = part_l.add(h);
            let mut acc = part_acc.add(h * head_dim as usize + d);
            let (mut num, mut den) = (0.0f32, 0.0f32);
            let mut s = 0;
            while s < n_splits {
                let ms = *m;
                let f = if ms == f32::NEG_INFINITY {
                    0.0
                } else {
                    (ms - gm).exp()
                };
                num = f.mul_add(*acc, num);
                den = f.mul_add(*l, den);
                m = m.wrapping_add(row);
                l = l.wrapping_add(row);
                acc = acc.wrapping_add(row * head_dim as usize);
                s += 1;
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

    /// Keep the larger of two argmax candidates, the lower index on a tie.
    ///
    /// A NaN never wins (`NaN > x` and `NaN == x` are both false), which is
    /// what the scalar reference does.
    #[device]
    fn argmax_merge(bv: f32, bi: i32, v: f32, i: i32) -> (f32, i32) {
        if v > bv || (v == bv && i < bi) {
            (v, i)
        } else {
            (bv, bi)
        }
    }

    /// Reduce one block's per-thread candidates to thread 0.
    ///
    /// Every thread seeds from element 0 rather than from a -inf sentinel, so
    /// a vector that is entirely NaN answers 0, as the scalar reference does:
    /// its running best never moves off index 0. A sentinel would answer "no
    /// candidate" instead, and the two would disagree on the one input where
    /// disagreement is hardest to spot.
    ///
    /// # Safety
    /// Every thread of a 256-thread 1-D block participates.
    #[device]
    unsafe fn argmax_reduce_block(bv: f32, bi: i32) -> (f32, i32) {
        static mut SV: SharedArray<f32, 8> = SharedArray::UNINIT;
        static mut SI: SharedArray<i32, 8> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x();
        let (mut bv, mut bi) = (bv, bi);
        let mut off = 16;
        while off > 0 {
            let ov = warp::shuffle_down_f32_sync(u32::MAX, bv, off);
            let oi = warp::shuffle_down_sync(u32::MAX, bi as u32, off) as i32;
            (bv, bi) = argmax_merge(bv, bi, ov, oi);
            off >>= 1;
        }
        unsafe {
            if tid & 31 == 0 {
                SV[(tid >> 5) as usize] = bv;
                SI[(tid >> 5) as usize] = bi;
            }
            thread::sync_threads();
            if tid == 0 {
                let mut w = 1;
                while w < 8 {
                    (bv, bi) = argmax_merge(bv, bi, SV[w], SI[w]);
                    w += 1;
                }
            }
        }
        (bv, bi)
    }

    /// Greedy sampling, first pass: each block's best `(value, index)` over a
    /// grid-strided share of `x`. Lowest index wins a tie at every level, so
    /// the answer does not depend on the grid.
    ///
    /// # Safety
    /// x covers n >= 1 floats; pv and pi cover gridDim.x entries. The block
    /// is 256 threads.
    #[kernel]
    pub unsafe fn argmax_partial(x: *const f32, n: i32, pv: *mut f32, pi: *mut i32) {
        let block = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        let stride = (thread::gridDim_x() * 256) as i32;
        unsafe {
            let (mut bv, mut bi) = (*x, 0i32);
            let mut i = (block * 256 + tid) as i32;
            while i < n {
                (bv, bi) = argmax_merge(bv, bi, *x.add(i as usize), i);
                i += stride;
            }
            (bv, bi) = argmax_reduce_block(bv, bi);
            if tid == 0 {
                *pv.add(block as usize) = bv;
                *pi.add(block as usize) = bi;
            }
        }
    }

    /// Greedy sampling, second pass: the best of the per-block candidates.
    ///
    /// # Safety
    /// pv and pi cover n >= 1 entries, out one int. One 256-thread block.
    #[kernel]
    pub unsafe fn argmax_final(pv: *const f32, pi: *const i32, n: i32, out: *mut i32) {
        let tid = thread::threadIdx_x();
        unsafe {
            let (mut bv, mut bi) = (*pv, *pi);
            let mut i = tid as i32;
            while i < n {
                (bv, bi) = argmax_merge(bv, bi, *pv.add(i as usize), *pi.add(i as usize));
                i += 256;
            }
            (bv, bi) = argmax_reduce_block(bv, bi);
            let _ = bv;
            if tid == 0 {
                *out = bi;
            }
        }
    }

    /// Sum the split-K partial products of a tensor-core GEMM, ascending
    /// split, two adjacent outputs a thread; optionally add into `out`.
    ///
    /// # Safety
    /// part covers splits * M * N floats; out covers M rows of stride ldo
    /// with N columns used. M * N is even. out is disjoint from part.
    #[kernel]
    pub unsafe fn hg_splitk_reduce(
        part: *const f32,
        out: *mut f32,
        m_rows: i32,
        n_cols: i32,
        ldo: i32,
        splits: i32,
        accumulate: i32,
    ) {
        let i = (thread::blockIdx_x() as i64 * thread::blockDim_x() as i64
            + thread::threadIdx_x() as i64)
            * 2;
        let mn = m_rows as i64 * n_cols as i64;
        if i >= mn {
            return;
        }
        let (m, n) = (i / n_cols as i64, i % n_cols as i64);
        unsafe {
            let mut p = part.add(i as usize);
            let (mut v0, mut v1) = (*p, *p.add(1));
            let mut z = 1;
            while z < splits {
                p = p.wrapping_add(mn as usize);
                v0 += *p;
                v1 += *p.add(1);
                z += 1;
            }
            let o = out.add((m * ldo as i64 + n) as usize);
            if accumulate != 0 {
                v0 = *o + v0;
                v1 = *o.add(1) + v1;
            }
            *o = v0;
            *o.add(1) = v1;
        }
    }

    /// K2 dispatch, pass one: how many of this lane's contiguous slice of
    /// flat `(token, slot)` ids route to this block's expert. The prefix of
    /// lane counts keeps the original route order without host counts or an
    /// atomic scatter.
    ///
    /// grid: (experts,). block: 128.
    ///
    /// # Safety
    /// ids covers pairs ints, counts gridDim.x * 128.
    #[kernel]
    pub unsafe fn k2_dispatch_count(ids: *const i32, counts: *mut i32, pairs: i32) {
        let e = thread::blockIdx_x() as i32;
        let t = thread::threadIdx_x() as i32;
        let chunk = (pairs + 127) / 128;
        let end = if (t + 1) * chunk < pairs { (t + 1) * chunk } else { pairs };
        let mut n = 0i32;
        let mut f = t * chunk;
        unsafe {
            while f < end {
                n += (*ids.add(f as usize) == e) as i32;
                f += 1;
            }
            *counts.add((e * 128 + t) as usize) = n;
        }
    }

    /// K2 dispatch, pass two: each expert's padded run start, and the owner
    /// of every tile (`-1` for an unused one).
    ///
    /// grid: (1,). block: 128, one thread per expert.
    ///
    /// # Safety
    /// counts covers experts * 128 ints, starts experts, owners capacity.
    /// experts <= 128 and tile > 0.
    #[kernel]
    pub unsafe fn k2_dispatch_prefix(
        counts: *const i32,
        starts: *mut i32,
        owners: *mut i32,
        experts: i32,
        capacity: i32,
        tile: i32,
    ) {
        static mut TOTALS: SharedArray<i32, 128> = SharedArray::UNINIT;
        let t = thread::threadIdx_x() as i32;
        unsafe {
            let mut n = 0i32;
            if t < experts {
                let mut j = 0;
                while j < 128 {
                    n += *counts.add((t * 128 + j) as usize);
                    j += 1;
                }
            }
            TOTALS[t as usize] = (n + tile - 1) / tile * tile;
            let mut b = t;
            while b < capacity {
                *owners.add(b as usize) = -1;
                b += 128;
            }
            thread::sync_threads();
            let mut start = 0i32;
            let mut e = 0;
            while e < t {
                start += TOTALS[e as usize];
                e += 1;
            }
            if t < experts {
                *starts.add(t as usize) = start;
                let end = start + TOTALS[t as usize];
                let mut f = start;
                while f < end {
                    *owners.add((f / tile) as usize) = t;
                    f += tile;
                }
            }
        }
    }

    /// K2 dispatch, pass three: write this expert's flat ids in their
    /// original order from its run start, then pad the run to a whole tile
    /// with `pairs`.
    ///
    /// grid: (experts,). block: 128.
    ///
    /// # Safety
    /// ids covers pairs ints, counts gridDim.x * 128, starts gridDim.x;
    /// sorted covers every padded run. tile > 0.
    #[kernel]
    pub unsafe fn k2_dispatch_gather(
        ids: *const i32,
        counts: *const i32,
        starts: *const i32,
        sorted: *mut i32,
        pairs: i32,
        tile: i32,
    ) {
        static mut PREFIX: SharedArray<i32, 128> = SharedArray::UNINIT;
        let e = thread::blockIdx_x() as i32;
        let t = thread::threadIdx_x() as i32;
        let chunk = (pairs + 127) / 128;
        unsafe {
            let own = *counts.add((e * 128 + t) as usize);
            PREFIX[t as usize] = own;
            thread::sync_threads();
            let mut d = 1;
            while d < 128 {
                let v = if t >= d { PREFIX[(t - d) as usize] } else { 0 };
                thread::sync_threads();
                PREFIX[t as usize] += v;
                thread::sync_threads();
                d *= 2;
            }
            let total = PREFIX[127];
            let start = *starts.add(e as usize);
            let mut cursor = start + PREFIX[t as usize] - own;
            let end = if (t + 1) * chunk < pairs { (t + 1) * chunk } else { pairs };
            let mut f = t * chunk;
            while f < end {
                if *ids.add(f as usize) == e {
                    *sorted.add(cursor as usize) = f;
                    cursor += 1;
                }
                f += 1;
            }
            let padded = (total + tile - 1) / tile * tile;
            let mut f = total + t;
            while f < padded {
                *sorted.add((start + f) as usize) = pairs;
                f += 128;
            }
        }
    }

    /// Grouped RMSNorm for K2: each block normalizes one group of one token,
    /// `y = (x / sqrt(mean(x^2) + eps)) * w`, with the group's own mean.
    ///
    /// The reduction is a shuffle-down tree per warp, then thread 0 adds the
    /// eight warp totals in ascending order.
    ///
    /// grid: (tokens * groups,). block: 256.
    ///
    /// # Safety
    /// x and y cover tokens * width floats and w covers width; groups divides
    /// width. Only y may exactly alias x.
    #[kernel]
    pub unsafe fn k2_norm(
        x: *const f32,
        w: *const f32,
        y: *mut f32,
        width: i32,
        groups: i32,
        eps: f32,
    ) {
        static mut SUMS: SharedArray<f32, 8> = SharedArray::UNINIT;
        let block = thread::blockIdx_x() as i32;
        let tid = thread::threadIdx_x();
        let step = thread::blockDim_x() as i32;
        let (g, token, part) = (block % groups, block / groups, width / groups);
        let base = token as i64 * width as i64 + (g * part) as i64;
        unsafe {
            let xb = x.add(base as usize);
            let mut acc = 0.0f32;
            let mut j = tid as i32;
            while j < part {
                let v = *xb.add(j as usize);
                // NVRTC contracts the original's acc += x * x.
                acc = v.mul_add(v, acc);
                j += step;
            }
            let mut d = 16;
            while d > 0 {
                acc += warp::shuffle_down_f32_sync(u32::MAX, acc, d);
                d >>= 1;
            }
            if tid & 31 == 0 {
                SUMS[(tid >> 5) as usize] = acc;
            }
            thread::sync_threads();
            if tid == 0 {
                let mut sum = 0.0f32;
                let mut i = 0;
                while i < 8 {
                    sum += SUMS[i];
                    i += 1;
                }
                SUMS[0] = 1.0 / (sum / part as f32 + eps).sqrt();
            }
            thread::sync_threads();
            let scale = SUMS[0];
            let wb = w.add((g * part) as usize);
            let yb = y.add(base as usize);
            let mut j = tid as i32;
            while j < part {
                *yb.add(j as usize) = (*xb.add(j as usize) * scale) * *wb.add(j as usize);
                j += step;
            }
        }
    }

    /// `l_out[t][j] = ffn_out[t][j] + residual[t][j]`, for live tokens only.
    ///
    /// The row is split over `gridDim.y` blocks for the same reason the MoE
    /// combine's is: one block per token would put a 20 KiB row on one SM.
    ///
    /// # Safety
    /// The three buffers cover gridDim.x * hidden floats, valid_tokens one
    /// int. l_out may exactly alias an input.
    #[kernel]
    pub unsafe fn dense_ffn_add_residual(
        ffn_out: *const f32,
        residual: *const f32,
        valid_tokens: *const i32,
        hidden: i32,
        l_out: *mut f32,
    ) {
        let t = thread::blockIdx_x() as i64;
        unsafe {
            if t >= *valid_tokens as i64 {
                return;
            }
            let base = (t * hidden as i64) as usize;
            let stride = (thread::gridDim_y() * thread::blockDim_x()) as i32;
            let mut j =
                (thread::blockIdx_y() * thread::blockDim_x() + thread::threadIdx_x()) as i32;
            while j < hidden {
                let at = base + j as usize;
                *l_out.add(at) = *ffn_out.add(at) + *residual.add(at);
                j += stride;
            }
        }
    }

    /// `out[t] = [e_norm[t] | h_norm[t]]`: the MTP head's concatenated input.
    ///
    /// grid: (n_tokens,). block: any, striding the row.
    ///
    /// # Safety
    /// e_norm and h_norm cover n_tokens * hidden floats, out twice that; out
    /// is disjoint from both.
    #[kernel]
    pub unsafe fn mtp_concat_eh(
        e_norm: *const f32,
        h_norm: *const f32,
        out: *mut f32,
        hidden: i32,
        n_tokens: i32,
    ) {
        let t = thread::blockIdx_x() as i64;
        if t >= n_tokens as i64 {
            return;
        }
        let step = thread::blockDim_x() as i32;
        unsafe {
            let e = e_norm.add((t * hidden as i64) as usize);
            let h = h_norm.add((t * hidden as i64) as usize);
            let dst = out.add((t * 2 * hidden as i64) as usize);
            let mut j = thread::threadIdx_x() as i32;
            while j < hidden {
                *dst.add(j as usize) = *e.add(j as usize);
                *dst.add((hidden + j) as usize) = *h.add(j as usize);
                j += step;
            }
        }
    }

    /// One target-layer tap into the concatenated feature rows:
    /// `aux[row][tap * hidden + j] = src[row][j]`.
    ///
    /// grid: (rows,). block: any, striding the row.
    ///
    /// # Safety
    /// src covers rows * hidden floats, aux rows * taps * hidden; disjoint.
    #[kernel]
    pub unsafe fn dflash_tap(
        src: *const f32,
        aux: *mut f32,
        hidden: i32,
        taps: i32,
        tap: i32,
        rows: i32,
    ) {
        let r = thread::blockIdx_x() as i64;
        if r >= rows as i64 {
            return;
        }
        let step = thread::blockDim_x() as i32;
        unsafe {
            let s = src.add((r * hidden as i64) as usize);
            let d = aux.add(((r * taps as i64 + tap as i64) * hidden as i64) as usize);
            let mut j = thread::threadIdx_x() as i32;
            while j < hidden {
                *d.add(j as usize) = *s.add(j as usize);
                j += step;
            }
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

    /// fp32 to binary16, round to nearest even.
    #[device]
    fn f2h(f: f32) -> u16 {
        let h: u16;
        // SAFETY: a register-only conversion.
        unsafe {
            ptx_asm!(
                "{ .reg .f16 a; cvt.rn.f16.f32 a, %1; mov.b16 %0, a; }",
                out("=h") h,
                in("f") f,
            );
        }
        h
    }

    /// binary16 to fp32, exact.
    #[device]
    fn h2f(h: u16) -> f32 {
        let f: f32;
        // SAFETY: a register-only conversion.
        unsafe {
            ptx_asm!(
                "{ .reg .f16 a; mov.b16 a, %1; cvt.f32.f16 %0, a; }",
                out("=f") f,
                in("h") h,
            );
        }
        f
    }

    /// fp32 to fp16, round to nearest even, `n` elements.
    ///
    /// # Safety
    /// x covers n floats and y n halves; the 1-D launch is nonempty.
    #[kernel]
    pub unsafe fn hg_to_half(x: *const f32, y: *mut u16, n: i64) {
        let i = thread::blockIdx_x() as i64 * thread::blockDim_x() as i64
            + thread::threadIdx_x() as i64;
        if i < n {
            unsafe {
                *y.add(i as usize) = f2h(*x.add(i as usize));
            }
        }
    }

    /// GELU in place on fp16 activations: ggml's tanh approximation,
    /// computed in fp32 and rounded back.
    ///
    /// # Safety
    /// x covers n halves; the 1-D launch is nonempty.
    #[kernel]
    pub unsafe fn vision_gelu(x: *mut u16, n: i64) {
        let i = thread::blockIdx_x() as i64 * thread::blockDim_x() as i64
            + thread::threadIdx_x() as i64;
        if i >= n {
            return;
        }
        unsafe {
            let v = h2f(*x.add(i as usize));
            // NVRTC contracts the original's 1 + 0.044715 * v * v.
            let inner = (0.797_884_56_f32 * v) * (0.044715_f32 * v).mul_add(v, 1.0);
            *x.add(i as usize) = f2h((0.5 * v) * (1.0 + inner.tanh()));
        }
    }

    /// `a += b` on fp16 activations, the sum taken in fp32 and rounded once.
    ///
    /// # Safety
    /// a and b cover n halves; b does not partially overlap a.
    #[kernel]
    pub unsafe fn vision_add_f16(a: *mut u16, b: *const u16, n: i64) {
        let i = thread::blockIdx_x() as i64 * thread::blockDim_x() as i64
            + thread::threadIdx_x() as i64;
        if i >= n {
            return;
        }
        unsafe {
            *a.add(i as usize) = f2h(h2f(*a.add(i as usize)) + h2f(*b.add(i as usize)));
        }
    }

    /// fp16 to fp32, `n` elements.
    ///
    /// # Safety
    /// src covers n halves and dst n floats.
    #[kernel]
    pub unsafe fn vision_half_to_float(src: *const u16, dst: *mut f32, n: i64) {
        let i = thread::blockIdx_x() as i64 * thread::blockDim_x() as i64
            + thread::threadIdx_x() as i64;
        if i >= n {
            return;
        }
        unsafe {
            *dst.add(i as usize) = h2f(*src.add(i as usize));
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

    /// `act_h[i] = half(silu(gate[i]) * up[i])`: SwiGLU narrowed to fp16 in
    /// the launch that computes it, because the down projection only ever
    /// reads the narrowed form.
    ///
    /// libdevice exp, as the fp32 SwiGLU uses, so the two residencies stay
    /// comparable byte for byte. It is not folded into the up projection's
    /// epilogue: that runs on lane 0 only, and the transcendentals of a whole
    /// row landing on one lane in thirty-two measured slower than this launch
    /// (docs/BENCHMARKS.md).
    ///
    /// # Safety
    /// gate and up cover n floats, act_h n halves.
    #[kernel]
    pub unsafe fn dense_swiglu_narrow(gate: *const f32, up: *const f32, act_h: *mut u16, n: i32) {
        let i = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as i32;
        if i >= n {
            return;
        }
        unsafe {
            let g = *gate.add(i as usize);
            *act_h.add(i as usize) = f2h((g / (1.0 + (-g).exp())) * *up.add(i as usize));
        }
    }

    /// `y[o][c] += src[owner(o)][c]`, the owner at `spans[o * stride + 2]`.
    ///
    /// grid: (ceil(cols / block), n).
    ///
    /// # Safety
    /// y covers n rows of stride ldy, src every owner row of stride lds,
    /// spans n * stride ints; y is disjoint from src.
    #[kernel]
    pub unsafe fn dec_add_owner_rows(
        y: *mut f32,
        ldy: i64,
        src: *const f32,
        lds: i64,
        spans: *const i32,
        stride: i32,
        n: i32,
        cols: i32,
    ) {
        let o = thread::blockIdx_y() as i32;
        let c = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as i32;
        if o >= n || c >= cols {
            return;
        }
        unsafe {
            let q = *spans.add((o * stride + 2) as usize);
            let at = y.add((o as i64 * ldy + c as i64) as usize);
            *at += *src.add((q as i64 * lds + c as i64) as usize);
        }
    }

    /// `out[s][c]` = the mean over rows t of span s of `LayerNorm(x[t])[c]`,
    /// from each row's precomputed `(mean, rstd)`. Spans are `(start, end)`
    /// pairs at `spans + s * stride`.
    ///
    /// grid: (ceil(cols / block), nspans).
    ///
    /// # Safety
    /// x covers every span row at stride ldx; stats holds a `(mean, rstd)`
    /// pair per row; g and b cover cols; out covers nspans rows of stride
    /// ldo.
    #[kernel]
    pub unsafe fn dec_span_mean(
        x: *const f32,
        ldx: i64,
        cols: i32,
        stats: *const f32,
        g: *const f32,
        b: *const f32,
        spans: *const i32,
        stride: i32,
        nspans: i32,
        out: *mut f32,
        ldo: i64,
    ) {
        let s = thread::blockIdx_y() as i32;
        let c = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as i32;
        if s >= nspans || c >= cols {
            return;
        }
        unsafe {
            let start = *spans.add((s * stride) as usize);
            let end = *spans.add((s * stride + 1) as usize);
            let mut acc = 0.0f32;
            let mut t = start;
            while t < end {
                let mean = *stats.add(2 * t as usize);
                let rstd = *stats.add(2 * t as usize + 1);
                acc = (*x.add((t as i64 * ldx + c as i64) as usize) - mean).mul_add(rstd, acc);
                t += 1;
            }
            *out.add((s as i64 * ldo + c as i64) as usize) =
                (acc / (end - start) as f32).mul_add(*g.add(c as usize), *b.add(c as usize));
        }
    }

    /// Merge the key-split partials of the decision head's attention.
    ///
    /// `part` holds `splits * nq * heads` weighted value sums of `dh`
    /// floats, then a `(max, normalizer)` pair per split, query and head.
    /// Splits are rescaled to the global maximum and summed in ascending
    /// order.
    ///
    /// grid: (nq, heads). block: one thread per head dimension.
    ///
    /// # Safety
    /// part covers splits * nq * heads * (dh + 2) floats; out covers nq rows
    /// of stride ldo with heads * dh columns used.
    #[kernel]
    pub unsafe fn dec_attention_combine(
        part: *const f32,
        splits: i32,
        nq: i32,
        heads: i32,
        dh: i32,
        out: *mut f32,
        ldo: i64,
    ) {
        let q = thread::blockIdx_x() as i64;
        let h = thread::blockIdx_y() as i64;
        let d = thread::threadIdx_x() as i64;
        if d >= dh as i64 {
            return;
        }
        let per_split = nq as i64 * heads as i64;
        let slot = q * heads as i64 + h;
        unsafe {
            let ml = part.add((splits as i64 * per_split * dh as i64) as usize);
            let mut mx = f32::NEG_INFINITY;
            let mut m = ml.add((slot * 2) as usize);
            let mut z = 0;
            while z < splits {
                mx = mx.max(*m);
                m = m.wrapping_add((per_split * 2) as usize);
                z += 1;
            }
            let (mut num, mut den) = (0.0f32, 0.0f32);
            let mut m = ml.add((slot * 2) as usize);
            let mut v = part.add((slot * dh as i64 + d) as usize);
            let mut z = 0;
            while z < splits {
                let w = (*m - mx).exp();
                den = (*m.add(1)).mul_add(w, den);
                num = (*v).mul_add(w, num);
                m = m.wrapping_add((per_split * 2) as usize);
                v = v.wrapping_add((per_split * dh as i64) as usize);
                z += 1;
            }
            *out.add((q * ldo + h * dh as i64 + d) as usize) = num / den;
        }
    }

    /// Rotate K2's attention rows by precomputed `(sin, cos)` coefficients,
    /// eight tokens a block; dimensions past `rope_dim` are copied.
    ///
    /// grid: (ceil(tokens / 8), heads). block: `dim` threads.
    ///
    /// # Safety
    /// input and out cover tokens * heads * dim floats and are disjoint;
    /// coefficients covers (offset + tokens) * rope_dim.
    #[kernel]
    pub unsafe fn k2_rotary_apply(
        input: *const f32,
        out: *mut f32,
        coefficients: *const f32,
        heads: i32,
        dim: i32,
        rope_dim: i32,
        offset: i32,
        tokens: i32,
    ) {
        let t0 = (thread::blockIdx_x() * 8) as i32;
        let h = thread::blockIdx_y() as i64;
        let d = thread::threadIdx_x() as i32;
        let half = rope_dim / 2;
        if d >= half && d < rope_dim {
            return;
        }
        let mut u = 0;
        while u < 8 {
            let t = t0 + u;
            if t >= tokens {
                break;
            }
            let base = ((t as i64 * heads as i64 + h) * dim as i64) as usize;
            unsafe {
                if d >= rope_dim {
                    *out.add(base + d as usize) = *input.add(base + d as usize);
                } else {
                    let at = ((offset as i64 + t as i64) * rope_dim as i64 + 2 * d as i64) as usize;
                    let (sin_a, cos_a) = (*coefficients.add(at), *coefficients.add(at + 1));
                    let x0 = *input.add(base + d as usize);
                    let x1 = *input.add(base + (d + half) as usize);
                    *out.add(base + d as usize) = x0.mul_add(cos_a, -(x1 * sin_a));
                    *out.add(base + (d + half) as usize) = x0.mul_add(sin_a, x1 * cos_a);
                }
            }
            u += 1;
        }
    }

    /// A binary16 block scale stored as two little-endian bytes, widened by
    /// the hardware conversion, so it is bit-identical to the `half` crate's
    /// `to_f32` on the host. The quantizer emits subnormal scales for
    /// near-zero blocks, so that path is exercised.
    ///
    /// # Safety
    /// p covers two bytes.
    #[device]
    unsafe fn load_half_le(p: *const u8) -> f32 {
        unsafe { h2f(*p as u16 | ((*p.add(1) as u16) << 8)) }
    }

    /// Look up token embeddings in a Q8_0 table stored as raw GGUF rows:
    /// 34-byte blocks of a binary16 scale and 32 int8 codes.
    /// `out[t][j] = scale * code`.
    ///
    /// grid: (n_tokens,). block: any, striding the row.
    ///
    /// # Safety
    /// table covers every row `ids` names, of hidden / 32 blocks; ids covers
    /// n_tokens ints and out n_tokens * hidden floats. hidden is a multiple
    /// of 32.
    #[kernel]
    pub unsafe fn embed_q8_0(
        table: *const u8,
        ids: *const i32,
        out: *mut f32,
        hidden: i32,
        n_tokens: i32,
    ) {
        let t = thread::blockIdx_x() as i64;
        if t >= n_tokens as i64 {
            return;
        }
        let step = thread::blockDim_x() as i32;
        unsafe {
            let blocks = (hidden / 32) as i64;
            let row = table.add((*ids.add(t as usize) as i64 * blocks * 34) as usize);
            let dst = out.add((t * hidden as i64) as usize);
            let mut j = thread::threadIdx_x() as i32;
            while j < hidden {
                let blk = row.add((j / 32) as usize * 34);
                let d = load_half_le(blk);
                let q = *blk.add(2 + (j % 32) as usize) as i8;
                *dst.add(j as usize) = d * q as f32;
                j += step;
            }
        }
    }

    /// Dequantize a flat Q8_0 tensor: `dst[i] = code * scale`.
    ///
    /// # Safety
    /// src covers ceil(n_elements / 32) blocks of 34 bytes, dst n_elements
    /// floats.
    #[kernel]
    pub unsafe fn dequantize_q8_0(src: *const u8, dst: *mut f32, n_elements: i64) {
        let i = thread::blockIdx_x() as i64 * thread::blockDim_x() as i64
            + thread::threadIdx_x() as i64;
        if i >= n_elements {
            return;
        }
        unsafe {
            let base = src.add((i / 32 * 34) as usize);
            let d = load_half_le(base);
            let q = *base.add(2 + (i % 32) as usize) as i8;
            *dst.add(i as usize) = q as f32 * d;
        }
    }

    /// `gdn_silu` and `gdn_split_qkv` in one launch: the activated row is
    /// written whole to `silu` and its three slices to q, k and v.
    ///
    /// The standalone pair wrote one buffer that the split immediately read.
    /// `v / (1 + exp(-v))` is the same expression in the same order, so the
    /// outputs are the same bits.
    ///
    /// grid: (value_heads, tokens). block: head_dim.
    ///
    /// # Safety
    /// As `gdn_split_qkv`; silu covers the same extent as conv and is
    /// disjoint from every other buffer.
    #[kernel]
    pub unsafe fn gdn_silu_split_qkv(
        conv: *const f32,
        silu: *mut f32,
        q: *mut f32,
        k: *mut f32,
        v: *mut f32,
        head_dim: i32,
        qk_heads: i32,
        value_heads: i32,
    ) {
        let h = thread::blockIdx_x() as i64;
        let t = thread::blockIdx_y() as i64;
        let d = thread::threadIdx_x() as i64;
        let head_dim = head_dim as i64;
        let key_dim = qk_heads as i64 * head_dim;
        let conv_dim = 2 * key_dim + value_heads as i64 * head_dim;
        unsafe {
            let c = conv.add((t * conv_dim) as usize);
            let sr = silu.add((t * conv_dim) as usize);
            let iv = (2 * key_dim + h * head_dim + d) as usize;
            let mut vv = *c.add(iv);
            vv = vv / (1.0 + (-vv).exp());
            *sr.add(iv) = vv;
            *v.add(((t * value_heads as i64 + h) * head_dim + d) as usize) = vv;
            if h < qk_heads as i64 {
                let iq = (h * head_dim + d) as usize;
                let ik = (key_dim + h * head_dim + d) as usize;
                let (mut qv, mut kv) = (*c.add(iq), *c.add(ik));
                qv = qv / (1.0 + (-qv).exp());
                kv = kv / (1.0 + (-kv).exp());
                *sr.add(iq) = qv;
                *sr.add(ik) = kv;
                let o = ((t * qk_heads as i64 + h) * head_dim + d) as usize;
                *q.add(o) = qv;
                *k.add(o) = kv;
            }
        }
    }

    /// The decode step's convolution, SiLU and q/k/v split in one launch for
    /// up to eight sequences, each with its own convolution cache.
    ///
    /// Per channel this is `conv1d_step_batch`'s arithmetic exactly (the
    /// same helper), then `gdn_silu_split_qkv`'s; the raw convolution is
    /// kept in `conv_raw` because verification replays from it.
    ///
    /// grid: (value_heads, sequences <= 8). block: head_dim.
    ///
    /// # Safety
    /// x, conv_raw and silu cover sequences * conv_dim floats; weight covers
    /// conv_dim * conv_kernel; each used state pointer covers conv_dim *
    /// (conv_kernel - 1) floats and belongs to one sequence; q, k and v as
    /// `gdn_split_qkv`. All outputs are mutually disjoint.
    #[kernel]
    pub unsafe fn gdn_conv_silu_split_step_batch(
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
        conv_raw: *mut f32,
        silu: *mut f32,
        q: *mut f32,
        k: *mut f32,
        v: *mut f32,
        head_dim: i32,
        qk_heads: i32,
        value_heads: i32,
        conv_kernel: i32,
    ) {
        let h = thread::blockIdx_x() as i64;
        let t = thread::blockIdx_y() as i64;
        let d = thread::threadIdx_x() as i64;
        let state = match t {
            0 => s0,
            1 => s1,
            2 => s2,
            3 => s3,
            4 => s4,
            5 => s5,
            6 => s6,
            _ => s7,
        } as *mut f32;
        let head_dim = head_dim as i64;
        let key_dim = qk_heads as i64 * head_dim;
        let conv_dim = 2 * key_dim + value_heads as i64 * head_dim;
        let carry = conv_kernel - 1;
        unsafe {
            let xt = x.add((t * conv_dim) as usize);
            let cr = conv_raw.add((t * conv_dim) as usize);
            let sr = silu.add((t * conv_dim) as usize);
            let iv = 2 * key_dim + h * head_dim + d;
            let mut vv = conv_step_channel(
                *xt.add(iv as usize),
                weight.add((iv * conv_kernel as i64) as usize),
                state.add((iv * carry as i64) as usize),
                carry,
            );
            *cr.add(iv as usize) = vv;
            vv = vv / (1.0 + (-vv).exp());
            *sr.add(iv as usize) = vv;
            *v.add(((t * value_heads as i64 + h) * head_dim + d) as usize) = vv;
            if h < qk_heads as i64 {
                let iq = h * head_dim + d;
                let ik = key_dim + h * head_dim + d;
                let mut qv = conv_step_channel(
                    *xt.add(iq as usize),
                    weight.add((iq * conv_kernel as i64) as usize),
                    state.add((iq * carry as i64) as usize),
                    carry,
                );
                let mut kv = conv_step_channel(
                    *xt.add(ik as usize),
                    weight.add((ik * conv_kernel as i64) as usize),
                    state.add((ik * carry as i64) as usize),
                    carry,
                );
                *cr.add(iq as usize) = qv;
                *cr.add(ik as usize) = kv;
                qv = qv / (1.0 + (-qv).exp());
                kv = kv / (1.0 + (-kv).exp());
                *sr.add(iq as usize) = qv;
                *sr.add(ik as usize) = kv;
                let o = ((t * qk_heads as i64 + h) * head_dim + d) as usize;
                *q.add(o) = qv;
                *k.add(o) = kv;
            }
        }
    }

    /// L2-normalize one head's query and key rows; the query is also scaled.
    ///
    /// `1 / sqrt(sum + eps)`: the epsilon is added to the sum of squares,
    /// which is flash-linear-attention's l2norm, transformers' and
    /// llama.cpp's since 5fdfa6282
    /// (`build_gdn_l2_norm`). The form that floors the norm instead,
    /// `1 / max(sqrt(sum), eps)`, is ggml_l2_norm's; on the real q_conv and
    /// k_conv of blocks 0/4/20 the two differ by up to 2.148e-4 relative.
    ///
    /// A reciprocal square root would be faster and is not IEEE-exact;
    /// dividing by sqrt keeps the kernel bit-comparable with its CPU
    /// reference up to the reduction order.
    ///
    /// # Safety
    /// base + threadIdx.x indexes all four buffers; block_reduce_sum's
    /// launch contract holds with blockDim.x == head_dim.
    #[device]
    unsafe fn normalize_qk_row(
        q_in: *const f32,
        k_in: *const f32,
        q_out: *mut f32,
        k_out: *mut f32,
        base: i64,
        eps: f32,
        scale: f32,
    ) {
        let at = (base + thread::threadIdx_x() as i64) as usize;
        unsafe {
            let (qv, kv) = (*q_in.add(at), *k_in.add(at));
            let q_sq = block_reduce_sum(qv * qv);
            // The next reduction overwrites the slot this one returned from.
            thread::sync_threads();
            let k_sq = block_reduce_sum(kv * kv);
            thread::sync_threads();
            *q_out.add(at) = qv * (1.0 / (q_sq + eps).sqrt()) * scale;
            *k_out.add(at) = kv * (1.0 / (k_sq + eps).sqrt());
        }
    }

    /// `normalize_qk_row` over one token's heads, for the recurrent step.
    ///
    /// grid: (heads,). block: head_dim, with a float of dynamic shared
    /// memory per warp.
    ///
    /// # Safety
    /// The four buffers cover gridDim.x * head_dim floats; outputs are
    /// disjoint from inputs and each other.
    #[kernel]
    pub unsafe fn gdn_normalize_qk(
        q_in: *const f32,
        k_in: *const f32,
        q_out: *mut f32,
        k_out: *mut f32,
        head_dim: i32,
        eps: f32,
        scale: f32,
    ) {
        let base = (thread::blockIdx_x() as i32 * head_dim) as i64;
        unsafe { normalize_qk_row(q_in, k_in, q_out, k_out, base, eps, scale) }
    }

    /// `normalize_qk_row` over every head of every token of a chunk.
    ///
    /// grid: (tokens * heads,). block: head_dim, with a float of dynamic
    /// shared memory per warp.
    ///
    /// # Safety
    /// As `gdn_normalize_qk`.
    #[kernel]
    pub unsafe fn gdn_chunk_normalize_qk(
        q_in: *const f32,
        k_in: *const f32,
        q_out: *mut f32,
        k_out: *mut f32,
        head_dim: i32,
        eps: f32,
        scale: f32,
    ) {
        let base = thread::blockIdx_x() as i64 * head_dim as i64;
        unsafe { normalize_qk_row(q_in, k_in, q_out, k_out, base, eps, scale) }
    }

    /// The dense model's post-mixer RMSNorm, emitting both widths in one
    /// pass: fp32 for the residual and binary16 for the tensor-core
    /// projections. Same reduction and same division as `rms_norm_rows`.
    ///
    /// # Safety
    /// As `rms_norm_rows`; out_h covers the same element count in halves.
    #[kernel]
    pub unsafe fn dense_rms_norm_narrow(
        x: *const f32,
        weight: *const f32,
        out: *mut f32,
        out_h: *mut u16,
        width: i32,
        eps: f32,
    ) {
        let base = thread::blockIdx_x() as usize * width as usize;
        let step = thread::blockDim_x() as i32;
        unsafe {
            let inv_rms = inverse_rms(x, width, eps);
            let mut j = thread::threadIdx_x() as i32;
            while j < width {
                let at = base + j as usize;
                let v = *x.add(at) * inv_rms * *weight.add(j as usize);
                *out.add(at) = v;
                *out_h.add(at) = f2h(v);
                j += step;
            }
        }
    }

    /// The decision head's block sum: an xor-shuffle tree per warp, the warp
    /// totals parked in shared memory, then a second tree over them.
    ///
    /// # Safety
    /// Every thread of a 1-D block of at most 1024 threads, a multiple of
    /// 32, participates.
    #[device]
    unsafe fn dec_block_sum(v: f32) -> f32 {
        static mut RED: SharedArray<f32, 32> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x();
        let (lane, w, nw) = (tid & 31, tid >> 5, thread::blockDim_x() >> 5);
        let mut v = v;
        let mut o = 16;
        while o > 0 {
            v += warp::shuffle_xor_f32_sync(u32::MAX, v, o);
            o >>= 1;
        }
        unsafe {
            thread::sync_threads();
            if lane == 0 {
                RED[w as usize] = v;
            }
            thread::sync_threads();
            v = if lane < nw { RED[lane as usize] } else { 0.0 };
        }
        let mut o = 16;
        while o > 0 {
            v += warp::shuffle_xor_f32_sync(u32::MAX, v, o);
            o >>= 1;
        }
        v
    }

    /// LayerNorm over each row, optionally recording the row's `(mean,
    /// rstd)` and optionally writing the normalized row.
    ///
    /// grid: (rows,). block: a multiple of 32.
    ///
    /// # Safety
    /// x covers rows of stride ldx with `cols` used; g and b cover cols.
    /// stats is null or covers two floats a row; y is null or covers rows of
    /// stride ldy and is disjoint from x.
    #[kernel]
    pub unsafe fn dec_layer_norm(
        x: *const f32,
        ldx: i64,
        rows: i32,
        cols: i32,
        eps: f32,
        g: *const f32,
        b: *const f32,
        stats: *mut f32,
        y: *mut f32,
        ldy: i64,
    ) {
        let r = thread::blockIdx_x() as i64;
        if r >= rows as i64 {
            return;
        }
        let tid = thread::threadIdx_x() as i32;
        let step = thread::blockDim_x() as i32;
        unsafe {
            let xr = x.add((r * ldx) as usize);
            let mut s = 0.0f32;
            let mut c = tid;
            while c < cols {
                s += *xr.add(c as usize);
                c += step;
            }
            let mean = dec_block_sum(s) / cols as f32;
            let mut v = 0.0f32;
            let mut c = tid;
            while c < cols {
                let d = *xr.add(c as usize) - mean;
                // NVRTC contracts the original's v += d * d.
                v = d.mul_add(d, v);
                c += step;
            }
            let var = dec_block_sum(v) / cols as f32;
            let rstd = 1.0 / (var + eps).sqrt();
            if !stats.is_null() && tid == 0 {
                *stats.add(2 * r as usize) = mean;
                *stats.add(2 * r as usize + 1) = rstd;
            }
            if !y.is_null() {
                let yr = y.add((r * ldy) as usize);
                let mut c = tid;
                while c < cols {
                    let at = c as usize;
                    *yr.add(at) = ((*xr.add(at) - mean) * rstd).mul_add(*g.add(at), *b.add(at));
                    c += step;
                }
            }
        }
    }

    /// Dequantize a flat Q6_K tensor, one thread per `(superblock, half,
    /// l)`: four elements, 32 apart, from the low nibbles and high bit pairs
    /// that share bytes. `y = (d * scale) * (code - 32)`, in that order.
    ///
    /// A superblock is 210 bytes: 128 of low nibbles, 64 of high pairs, 16
    /// signed scales and a binary16 super-scale.
    ///
    /// # Safety
    /// src covers n_superblocks * 210 bytes and dst n_superblocks * 256
    /// floats.
    #[kernel]
    pub unsafe fn dequantize_q6_k(src: *const u8, dst: *mut f32, n_superblocks: i64) {
        let t = thread::blockIdx_x() as i64 * thread::blockDim_x() as i64
            + thread::threadIdx_x() as i64;
        if t >= n_superblocks * 64 {
            return;
        }
        let sb = t / 64;
        let rem = (t % 64) as usize;
        let (half, l) = (rem / 32, rem % 32);
        unsafe {
            let base = src.add((sb * 210) as usize);
            let ql = base.add(half * 64);
            let qh = base.add(128 + half * 32);
            let sc = base.add(192 + half * 8) as *const i8;
            let d = load_half_le(base.add(208));
            let is = l / 16;
            let (lo0, lo1, hi) = (*ql.add(l) as i32, *ql.add(l + 32) as i32, *qh.add(l) as i32);
            let raw1 = (lo0 & 0xF) | ((hi & 3) << 4);
            let raw2 = (lo1 & 0xF) | (((hi >> 2) & 3) << 4);
            let raw3 = (lo0 >> 4) | (((hi >> 4) & 3) << 4);
            let raw4 = (lo1 >> 4) | (((hi >> 6) & 3) << 4);
            let y = dst.add((sb * 256) as usize + half * 128);
            *y.add(l) = d * *sc.add(is) as f32 * (raw1 - 32) as f32;
            *y.add(l + 32) = d * *sc.add(is + 2) as f32 * (raw2 - 32) as f32;
            *y.add(l + 64) = d * *sc.add(is + 4) as f32 * (raw3 - 32) as f32;
            *y.add(l + 96) = d * *sc.add(is + 6) as f32 * (raw4 - 32) as f32;
        }
    }

    /// Softmax in place over each row of fp16 attention scores: subtract the
    /// row maximum, exponentiate, and scale by the reciprocal of the sum,
    /// every step taken in fp32 and rounded back to fp16.
    ///
    /// Both reductions are a shuffle-down tree per warp and a second tree
    /// over the warp results.
    ///
    /// grid: (rows,). block: a multiple of 32, at most 1024.
    ///
    /// # Safety
    /// scores covers gridDim.x * row_len halves.
    #[kernel]
    pub unsafe fn vision_softmax_rows(scores: *mut u16, row_len: i32) {
        static mut RED: SharedArray<f32, 32> = SharedArray::UNINIT;
        const LOWEST: f32 = -3.4e38;
        let tid = thread::threadIdx_x();
        let step = thread::blockDim_x() as i32;
        let warps = (thread::blockDim_x() + 31) >> 5;
        unsafe {
            let row = scores.add(thread::blockIdx_x() as usize * row_len as usize);
            let mut local_max = LOWEST;
            let mut j = tid as i32;
            while j < row_len {
                local_max = local_max.max(h2f(*row.add(j as usize)));
                j += step;
            }
            let mut off = 16;
            while off > 0 {
                local_max = local_max.max(warp::shuffle_down_f32_sync(u32::MAX, local_max, off));
                off >>= 1;
            }
            if tid & 31 == 0 {
                RED[(tid >> 5) as usize] = local_max;
            }
            thread::sync_threads();
            if tid < 32 {
                let mut v = if tid < warps { RED[tid as usize] } else { LOWEST };
                let mut off = 16;
                while off > 0 {
                    v = v.max(warp::shuffle_down_f32_sync(u32::MAX, v, off));
                    off >>= 1;
                }
                if tid == 0 {
                    RED[0] = v;
                }
            }
            thread::sync_threads();
            let row_max = RED[0];
            thread::sync_threads();

            let mut total = 0.0f32;
            let mut j = tid as i32;
            while j < row_len {
                let e = (h2f(*row.add(j as usize)) - row_max).exp();
                *row.add(j as usize) = f2h(e);
                total += e;
                j += step;
            }
            let mut off = 16;
            while off > 0 {
                total += warp::shuffle_down_f32_sync(u32::MAX, total, off);
                off >>= 1;
            }
            if tid & 31 == 0 {
                RED[(tid >> 5) as usize] = total;
            }
            thread::sync_threads();
            if tid < 32 {
                let mut v = if tid < warps { RED[tid as usize] } else { 0.0 };
                let mut off = 16;
                while off > 0 {
                    v += warp::shuffle_down_f32_sync(u32::MAX, v, off);
                    off >>= 1;
                }
                if tid == 0 {
                    RED[0] = v;
                }
            }
            thread::sync_threads();
            let inv = 1.0 / RED[0];
            let mut j = tid as i32;
            while j < row_len {
                *row.add(j as usize) = f2h(h2f(*row.add(j as usize)) * inv);
                j += step;
            }
        }
    }

    /// Rotate the query and key slices of the vision tower's packed fp16
    /// `[q | k | v]` rows in place by 2-D rotary positions: the first quarter
    /// of each head's pairs takes the patch's first coordinate, the second
    /// quarter its second.
    ///
    /// grid: (tokens, 2, heads). block: at least head_dim / 2 threads.
    ///
    /// # Safety
    /// qkv covers tokens * 3 * hidden halves and positions two ints a token.
    #[kernel]
    pub unsafe fn vision_rope_qk(
        qkv: *mut u16,
        positions: *const i32,
        hidden: i32,
        head_dim: i32,
        theta_base: f32,
    ) {
        let token = thread::blockIdx_x() as i64;
        let slice = thread::blockIdx_y() as i32;
        let head = thread::blockIdx_z() as i32;
        let j = thread::threadIdx_x() as i32;
        let half_dim = head_dim >> 1;
        let quarter = head_dim >> 2;
        if j >= half_dim {
            return;
        }
        unsafe {
            let base = qkv.add(
                (token * 3 * hidden as i64) as usize + (slice * hidden + head * head_dim) as usize,
            );
            let pos = if j < quarter {
                *positions.add(2 * token as usize)
            } else {
                *positions.add(2 * token as usize + 1)
            };
            let k = if j < quarter { j } else { j - quarter };
            let freq = theta_base.powf(-(k as f32) / quarter as f32);
            let angle = pos as f32 * freq;
            let (sin_a, cos_a) = (angle.sin(), angle.cos());
            let x0 = h2f(*base.add(j as usize));
            let x1 = h2f(*base.add((j + half_dim) as usize));
            *base.add(j as usize) = f2h(x0.mul_add(cos_a, -(x1 * sin_a)));
            *base.add((j + half_dim) as usize) = f2h(x0.mul_add(sin_a, x1 * cos_a));
        }
    }

    /// Precompute K2's rotary `(sin, cos)` pairs for a chunk of tokens of up
    /// to eight sequences, each starting at its own device-side position.
    /// The angle is formed in double precision and narrowed once, as the
    /// reference does.
    ///
    /// grid: (ceil(chunk / 8), sequences). block: (rope_dim / 2 or more, 8).
    ///
    /// # Safety
    /// Each used position pointer covers one int; coefficients covers
    /// (offset + sequences * chunk) * rope_dim floats.
    #[kernel]
    pub unsafe fn k2_rotary_prepare(
        coefficients: *mut f32,
        chunk: i32,
        rope_dim: i32,
        theta: f32,
        offset: i32,
        p0: u64,
        p1: u64,
        p2: u64,
        p3: u64,
        p4: u64,
        p5: u64,
        p6: u64,
        p7: u64,
    ) {
        let t = (thread::blockIdx_x() * 8 + thread::threadIdx_y()) as i32;
        let d = thread::threadIdx_x() as i32;
        let seq = thread::blockIdx_y() as i32;
        if t >= chunk || d >= rope_dim / 2 {
            return;
        }
        let position = match seq {
            0 => p0,
            1 => p1,
            2 => p2,
            3 => p3,
            4 => p4,
            5 => p5,
            6 => p6,
            _ => p7,
        } as *const i32;
        unsafe {
            let freq = (theta as f64).powf(-2.0 * d as f64 / rope_dim as f64);
            let angle = (*position as f64 + t as f64) * freq;
            let at = ((offset as i64 + (seq * chunk) as i64 + t as i64) * rope_dim as i64
                + 2 * d as i64) as usize;
            *coefficients.add(at) = angle.sin() as f32;
            *coefficients.add(at + 1) = angle.cos() as f32;
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
                    let a = (*wp.add(0), *wp.add(256), *wp.add(512), *wp.add(768), *wp.add(1024), *wp.add(1280), *wp.add(1536), *wp.add(1792));
                    let b = (*xp.add(0), *xp.add(256), *xp.add(512), *xp.add(768), *xp.add(1024), *xp.add(1280), *xp.add(1536), *xp.add(1792));
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

    /// A dense fp32 projection for the Gated DeltaNet gates: one warp per
    /// output row, each lane contracting every 32nd element in ascending
    /// order, then an xor shuffle tree.
    ///
    /// The contraction is unrolled by hand, four fused steps an iteration:
    /// the compiler does not unroll it, and the single-step loop is a third
    /// slower at decode sizes. The order of the fused additions is unchanged.
    ///
    /// grid: (ceil(n_rows / blockDim.y), tokens). block: (32, rows per block).
    ///
    /// # Safety
    /// weight covers n_rows * k_dim floats, x gridDim.y * k_dim, out
    /// gridDim.y * n_rows; out is disjoint from the inputs.
    #[kernel]
    pub unsafe fn gdn_proj_f32(
        weight: *const f32,
        x: *const f32,
        out: *mut f32,
        k_dim: i32,
        n_rows: i32,
    ) {
        let lane = thread::threadIdx_x() as i32;
        let n = (thread::blockIdx_x() * thread::blockDim_y() + thread::threadIdx_y()) as i32;
        if n >= n_rows {
            return;
        }
        let t = thread::blockIdx_y() as i64;
        unsafe {
            let row = weight.add((n as i64 * k_dim as i64) as usize);
            let xr = x.add((t * k_dim as i64) as usize);
            let acc = strided_dot(row, xr, lane, k_dim);
            let acc = warp_xor_sum(acc);
            if lane == 0 {
                *out.add((t * n_rows as i64 + n as i64) as usize) = acc;
            }
        }
    }

    /// `sum a[i] * b[i]` over `i = first, first + 32, ...` below `len`, each
    /// product fused into the running sum in ascending order.
    ///
    /// Sixteen steps an iteration with every load ahead of the fused chain:
    /// a GPU thread issues in order, so loads interleaved with a dependent
    /// chain wait on each other and loads issued together do not. The
    /// plain loop is a third slower on a one-token launch.
    ///
    /// # Safety
    /// a and b cover len floats.
    #[device]
    unsafe fn strided_dot(a: *const f32, b: *const f32, first: i32, len: i32) -> f32 {
        let mut acc = 0.0f32;
        let mut i = first;
        unsafe {
            let (mut pa, mut pb) = (a.add(first as usize), b.add(first as usize));
            while i + 480 < len {
                let a = (*pa.add(0), *pa.add(32), *pa.add(64), *pa.add(96), *pa.add(128), *pa.add(160), *pa.add(192), *pa.add(224), *pa.add(256), *pa.add(288), *pa.add(320), *pa.add(352), *pa.add(384), *pa.add(416), *pa.add(448), *pa.add(480));
                let b = (*pb.add(0), *pb.add(32), *pb.add(64), *pb.add(96), *pb.add(128), *pb.add(160), *pb.add(192), *pb.add(224), *pb.add(256), *pb.add(288), *pb.add(320), *pb.add(352), *pb.add(384), *pb.add(416), *pb.add(448), *pb.add(480));
                acc = a.0.mul_add(b.0, acc);
                acc = a.1.mul_add(b.1, acc);
                acc = a.2.mul_add(b.2, acc);
                acc = a.3.mul_add(b.3, acc);
                acc = a.4.mul_add(b.4, acc);
                acc = a.5.mul_add(b.5, acc);
                acc = a.6.mul_add(b.6, acc);
                acc = a.7.mul_add(b.7, acc);
                acc = a.8.mul_add(b.8, acc);
                acc = a.9.mul_add(b.9, acc);
                acc = a.10.mul_add(b.10, acc);
                acc = a.11.mul_add(b.11, acc);
                acc = a.12.mul_add(b.12, acc);
                acc = a.13.mul_add(b.13, acc);
                acc = a.14.mul_add(b.14, acc);
                acc = a.15.mul_add(b.15, acc);
                pa = pa.wrapping_add(512);
                pb = pb.wrapping_add(512);
                i += 512;
            }
            while i < len {
                acc = (*pa).mul_add(*pb, acc);
                pa = pa.wrapping_add(32);
                pb = pb.wrapping_add(32);
                i += 32;
            }
        }
        acc
    }

    /// `strided_dot(a, x)` and `strided_dot(b, x)` in one pass over `x`.
    /// Each chain is fused in ascending order exactly as the single form.
    ///
    /// # Safety
    /// a, b and x cover len floats.
    #[device]
    unsafe fn strided_dot2(
        a: *const f32,
        b: *const f32,
        x: *const f32,
        first: i32,
        len: i32,
    ) -> (f32, f32) {
        let (mut aa, mut bb) = (0.0f32, 0.0f32);
        let mut i = first;
        unsafe {
            let mut pa = a.add(first as usize);
            let mut pb = b.add(first as usize);
            let mut px = x.add(first as usize);
            while i + 480 < len {
                let a = (*pa.add(0), *pa.add(32), *pa.add(64), *pa.add(96), *pa.add(128), *pa.add(160), *pa.add(192), *pa.add(224), *pa.add(256), *pa.add(288), *pa.add(320), *pa.add(352), *pa.add(384), *pa.add(416), *pa.add(448), *pa.add(480));
                let b = (*pb.add(0), *pb.add(32), *pb.add(64), *pb.add(96), *pb.add(128), *pb.add(160), *pb.add(192), *pb.add(224), *pb.add(256), *pb.add(288), *pb.add(320), *pb.add(352), *pb.add(384), *pb.add(416), *pb.add(448), *pb.add(480));
                let x = (*px.add(0), *px.add(32), *px.add(64), *px.add(96), *px.add(128), *px.add(160), *px.add(192), *px.add(224), *px.add(256), *px.add(288), *px.add(320), *px.add(352), *px.add(384), *px.add(416), *px.add(448), *px.add(480));
                aa = a.0.mul_add(x.0, aa);
                aa = a.1.mul_add(x.1, aa);
                aa = a.2.mul_add(x.2, aa);
                aa = a.3.mul_add(x.3, aa);
                aa = a.4.mul_add(x.4, aa);
                aa = a.5.mul_add(x.5, aa);
                aa = a.6.mul_add(x.6, aa);
                aa = a.7.mul_add(x.7, aa);
                aa = a.8.mul_add(x.8, aa);
                aa = a.9.mul_add(x.9, aa);
                aa = a.10.mul_add(x.10, aa);
                aa = a.11.mul_add(x.11, aa);
                aa = a.12.mul_add(x.12, aa);
                aa = a.13.mul_add(x.13, aa);
                aa = a.14.mul_add(x.14, aa);
                aa = a.15.mul_add(x.15, aa);
                bb = b.0.mul_add(x.0, bb);
                bb = b.1.mul_add(x.1, bb);
                bb = b.2.mul_add(x.2, bb);
                bb = b.3.mul_add(x.3, bb);
                bb = b.4.mul_add(x.4, bb);
                bb = b.5.mul_add(x.5, bb);
                bb = b.6.mul_add(x.6, bb);
                bb = b.7.mul_add(x.7, bb);
                bb = b.8.mul_add(x.8, bb);
                bb = b.9.mul_add(x.9, bb);
                bb = b.10.mul_add(x.10, bb);
                bb = b.11.mul_add(x.11, bb);
                bb = b.12.mul_add(x.12, bb);
                bb = b.13.mul_add(x.13, bb);
                bb = b.14.mul_add(x.14, bb);
                bb = b.15.mul_add(x.15, bb);
                pa = pa.wrapping_add(512);
                pb = pb.wrapping_add(512);
                px = px.wrapping_add(512);
                i += 512;
            }
            while i < len {
                aa = (*pa).mul_add(*px, aa);
                bb = (*pb).mul_add(*px, bb);
                pa = pa.wrapping_add(32);
                pb = pb.wrapping_add(32);
                px = px.wrapping_add(32);
                i += 32;
            }
        }
        (aa, bb)
    }

    /// The sum of a value over the warp, by an xor shuffle tree.
    #[device]
    fn warp_xor_sum(v: f32) -> f32 {
        let mut v = v;
        let mut offset = 16;
        while offset > 0 {
            v += warp::shuffle_xor_f32_sync(u32::MAX, v, offset);
            offset >>= 1;
        }
        v
    }

    /// One warp per option: the scorer's output projection, then the final
    /// combination `prior_scale * prior + gate * (joint_scale * cosine +
    /// residual)`.
    ///
    /// # Safety
    /// h covers n_opts rows of stride ldh with `width` columns used, w_out
    /// covers width, aux two floats per option, scores n_opts.
    #[kernel]
    pub unsafe fn dec_final_scores(
        h: *const f32,
        ldh: i64,
        n_opts: i32,
        width: i32,
        w_out: *const f32,
        b_out: f32,
        aux: *const f32,
        prior_scale: f32,
        joint_scale: f32,
        gate: f32,
        scores: *mut f32,
    ) {
        let flat = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let o = (flat >> 5) as i32;
        let lane = (thread::threadIdx_x() & 31) as i32;
        if o >= n_opts {
            return;
        }
        unsafe {
            let d = warp_xor_sum(strided_dot(h.add((o as i64 * ldh) as usize), w_out, lane, width));
            if lane == 0 {
                let joint = joint_scale.mul_add(*aux.add((o * 2 + 1) as usize), d + b_out);
                *scores.add(o as usize) =
                    prior_scale.mul_add(*aux.add((o * 2) as usize), gate * joint);
            }
        }
    }

    /// The Gated DeltaNet gate prelude in one launch: RMSNorm of the token's
    /// hidden row, the alpha and beta projections of the normalized row for
    /// one head, and that head's gates.
    ///
    /// Replaces two `gdn_proj_f32` launches and a `gdn_gates` launch and
    /// keeps their arithmetic: the norm reduces in four bands of eight warp
    /// totals that thread 0 adds in ascending order, each projection is the
    /// warp contraction of `gdn_proj_f32`, and the gates are `gdn_gates`'
    /// expressions. Block 0 of each token also writes the normalized row.
    ///
    /// grid: (heads, tokens). block: 256, with k_dim + 32 floats of dynamic
    /// shared memory.
    ///
    /// # Safety
    /// w_alpha and w_beta cover heads * k_dim floats; x and normed tokens *
    /// k_dim; norm_weight k_dim; dt_bias and ssm_a heads; the five per-head
    /// outputs tokens * heads. Outputs are disjoint from inputs and each
    /// other. k_dim is at most 4096.
    #[kernel]
    pub unsafe fn gdn_norm_alpha_beta_gates(
        w_alpha: *const f32,
        w_beta: *const f32,
        x: *const f32,
        dt_bias: *const f32,
        ssm_a: *const f32,
        alpha: *mut f32,
        beta_raw: *mut f32,
        a_softplus: *mut f32,
        log_decay: *mut f32,
        beta: *mut f32,
        k_dim: i32,
        heads: i32,
        _tokens: i32,
        norm_weight: *const f32,
        normed: *mut f32,
        eps: f32,
    ) {
        let tid = thread::threadIdx_x() as i32;
        let lane = tid & 31;
        let n = thread::blockIdx_x() as i64;
        let t = thread::blockIdx_y() as i64;
        let data = DynamicSharedArray::<f32>::get();
        unsafe {
            let scratch = data.add(k_dim as usize);
            let xr = x.add((t * k_dim as i64) as usize);
            // Eight loads issued together, then the stores: a thread issues
            // in order, and a load waiting on the previous store is the slow
            // way through a row.
            let mut i = tid;
            let (mut xp, mut dp) = (xr.add(tid as usize), data.add(tid as usize));
            while i + 1792 < k_dim {
                let v = (*xp.add(0), *xp.add(256), *xp.add(512), *xp.add(768), *xp.add(1024), *xp.add(1280), *xp.add(1536), *xp.add(1792));
                *dp.add(0) = v.0;
                *dp.add(256) = v.1;
                *dp.add(512) = v.2;
                *dp.add(768) = v.3;
                *dp.add(1024) = v.4;
                *dp.add(1280) = v.5;
                *dp.add(1536) = v.6;
                *dp.add(1792) = v.7;
                xp = xp.wrapping_add(2048);
                dp = dp.wrapping_add(2048);
                i += 2048;
            }
            while i < k_dim {
                *dp = *xp;
                xp = xp.wrapping_add(256);
                dp = dp.wrapping_add(256);
                i += 256;
            }
            thread::sync_threads();
            let mut band = 0;
            while band < 4 {
                let mut partial = 0.0f32;
                let mut j = tid + band * 256;
                while j < k_dim {
                    let v = *data.add(j as usize);
                    partial = v.mul_add(v, partial);
                    j += 1024;
                }
                let mut off = 16;
                while off > 0 {
                    partial += warp::shuffle_down_f32_sync(u32::MAX, partial, off);
                    off >>= 1;
                }
                if lane == 0 {
                    *scratch.add((band * 8 + (tid >> 5)) as usize) = partial;
                }
                band += 1;
            }
            thread::sync_threads();
            if tid == 0 {
                let mut sum_sq = 0.0f32;
                let mut w = 0;
                while w < 32 {
                    sum_sq += *scratch.add(w);
                    w += 1;
                }
                *scratch = 1.0 / (sum_sq / k_dim as f32 + eps).sqrt();
            }
            thread::sync_threads();
            let inv_rms = *scratch;
            let mut j = tid;
            let mut dp = data.add(tid as usize);
            let mut wp = norm_weight.add(tid as usize);
            let mut np = normed.add((t * k_dim as i64) as usize + tid as usize);
            while j + 1792 < k_dim {
                let w = (*wp.add(0), *wp.add(256), *wp.add(512), *wp.add(768), *wp.add(1024), *wp.add(1280), *wp.add(1536), *wp.add(1792));
                let d = (*dp.add(0), *dp.add(256), *dp.add(512), *dp.add(768), *dp.add(1024), *dp.add(1280), *dp.add(1536), *dp.add(1792));
                let y = (d.0 * inv_rms * w.0, d.1 * inv_rms * w.1, d.2 * inv_rms * w.2, d.3 * inv_rms * w.3, d.4 * inv_rms * w.4, d.5 * inv_rms * w.5, d.6 * inv_rms * w.6, d.7 * inv_rms * w.7);
                *dp.add(0) = y.0;
                *dp.add(256) = y.1;
                *dp.add(512) = y.2;
                *dp.add(768) = y.3;
                *dp.add(1024) = y.4;
                *dp.add(1280) = y.5;
                *dp.add(1536) = y.6;
                *dp.add(1792) = y.7;
                if n == 0 {
                    *np.add(0) = y.0;
                    *np.add(256) = y.1;
                    *np.add(512) = y.2;
                    *np.add(768) = y.3;
                    *np.add(1024) = y.4;
                    *np.add(1280) = y.5;
                    *np.add(1536) = y.6;
                    *np.add(1792) = y.7;
                }
                dp = dp.wrapping_add(2048);
                wp = wp.wrapping_add(2048);
                np = np.wrapping_add(2048);
                j += 2048;
            }
            while j < k_dim {
                let normalized = *dp * inv_rms * *wp;
                *dp = normalized;
                if n == 0 {
                    *np = normalized;
                }
                dp = dp.wrapping_add(256);
                wp = wp.wrapping_add(256);
                np = np.wrapping_add(256);
                j += 256;
            }
            thread::sync_threads();
            if tid >= 32 {
                return;
            }
            let row = (n * k_dim as i64) as usize;
            let (aa, bb) = strided_dot2(w_alpha.add(row), w_beta.add(row), data, lane, k_dim);
            let a_sum = warp_xor_sum(aa);
            let b_sum = warp_xor_sum(bb);
            if lane == 0 {
                let i = (t * heads as i64 + n) as usize;
                *alpha.add(i) = a_sum;
                *beta_raw.add(i) = b_sum;
                let a = a_sum + *dt_bias.add(n as usize);
                let sp = if a > 20.0 { a } else { (1.0 + a.exp()).ln() };
                *a_softplus.add(i) = sp;
                *log_decay.add(i) = sp * *ssm_a.add(n as usize);
                *beta.add(i) = 1.0 / (1.0 + (-b_sum).exp());
            }
        }
    }

    /// Four adjacent floats in one 128-bit load.
    ///
    /// # Safety
    /// p is 16-byte aligned and covers four floats.
    #[device]
    unsafe fn load4(p: *const f32) -> [f32; 4] {
        let (a, b, c, d): (f32, f32, f32, f32);
        unsafe {
            ptx_asm!(
                "ld.global.v4.f32 {%0, %1, %2, %3}, [%4];",
                out("=f") a,
                out("=f") b,
                out("=f") c,
                out("=f") d,
                in("l") p as u64,
            );
        }
        [a, b, c, d]
    }

    /// Four adjacent floats in one 128-bit store.
    ///
    /// # Safety
    /// p is 16-byte aligned and covers four floats this thread owns.
    #[device]
    unsafe fn store4(p: *mut f32, v: [f32; 4]) {
        unsafe {
            ptx_asm!(
                "st.global.v4.f32 [%0], {%1, %2, %3, %4};",
                in("l") p as u64,
                in("f") v[0],
                in("f") v[1],
                in("f") v[2],
                in("f") v[3],
            );
        }
    }

    /// One Gated DeltaNet decode step for up to eight sequences: decay the
    /// state, read the prediction, write the delta-rule correction and
    /// produce the output, one warp per state row.
    ///
    /// ```text
    /// S    = S * exp(log_decay)
    /// pred = S k
    /// S    = S + beta * (v - pred) k^T
    /// out  = S q
    /// ```
    ///
    /// The order follows the reference exactly: decay the state, form the
    /// correction against the *decayed* state, apply the outer-product
    /// update, then read the output from the *updated* state. Reordering any
    /// of these is a plausible-looking change that silently produces a
    /// different model.
    ///
    /// The kernel is bandwidth-bound: 2 MiB of state read and written per
    /// layer per sequence. Its first form gave each state row a block, one
    /// thread per key index, with two block-wide reductions on the hot path,
    /// and moved state at 44% of the card's streaming roofline. One warp per
    /// row is the structure that fits: a lane owns four adjacent columns of
    /// each 128-column chunk, read and written as one 16-byte vector, so a
    /// warp keeps 512 contiguous bytes in flight; both reductions are
    /// five-shuffle butterflies with no barrier and no shared memory; and
    /// the decayed row is carried between the two passes in registers, which
    /// is why the chunks are spelled out rather than looped.
    ///
    /// libdevice exp for the decay, not an approximate one: it multiplies
    /// the whole state every token, so a systematic bias compounds over the
    /// context rather than averaging out.
    ///
    /// grid: (ceil(head_dim / rows per block), value heads, sequences <= 8).
    /// block: 32 threads per row.
    ///
    /// # Safety
    /// Each used state pointer covers value_heads * head_dim * head_dim
    /// floats and belongs to one sequence; q and k cover sequences * qk_heads
    /// * head_dim; v and out sequences * value_heads * head_dim; log_decay
    /// and beta sequences * value_heads. head_dim is a multiple of 128, at
    /// most 512. out is disjoint from the inputs.
    #[kernel]
    pub unsafe fn gdn_recurrent_step(
        s0: u64,
        s1: u64,
        s2: u64,
        s3: u64,
        s4: u64,
        s5: u64,
        s6: u64,
        s7: u64,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        log_decay: *const f32,
        beta: *const f32,
        out: *mut f32,
        head_dim: i32,
        qk_heads: i32,
    ) {
        let tid = thread::threadIdx_x();
        let lane = (tid & 31) as i32;
        let vi = (thread::blockIdx_x() * (thread::blockDim_x() >> 5) + (tid >> 5)) as i32;
        let h = thread::blockIdx_y() as i32;
        let z = thread::blockIdx_z() as i32;
        if vi >= head_dim {
            return;
        }
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
        let n_v_heads = thread::gridDim_y() as i32;
        let qk_base = ((z * qk_heads + h % qk_heads) * head_dim) as usize;
        let v_base = ((z * n_v_heads + h) * head_dim) as usize;
        let g_at = (z * n_v_heads + h) as usize;
        let chunks = head_dim >> 7;
        let j0 = (4 * lane) as usize;
        unsafe {
            let decay = (*log_decay.add(g_at)).exp();
            let row = state.add(((h as i64 * head_dim as i64 + vi as i64) * head_dim as i64) as usize);
            let kq = k.add(qk_base);
            let qq = q.add(qk_base);
            let zero = [0.0f32; 4];
            let (mut s, mut kc) = ([zero; 4], [zero; 4]);
            let mut partial = 0.0f32;
            macro_rules! decay_and_predict {
                ($c:literal) => {
                    if $c < chunks {
                        let j = ($c << 7) + j0;
                        let kv = load4(kq.add(j));
                        let sv = load4(row.add(j));
                        let sv = [
                            sv[0] * decay,
                            sv[1] * decay,
                            sv[2] * decay,
                            sv[3] * decay,
                        ];
                        partial = sv[0].mul_add(kv[0], partial);
                        partial = sv[1].mul_add(kv[1], partial);
                        partial = sv[2].mul_add(kv[2], partial);
                        partial = sv[3].mul_add(kv[3], partial);
                        kc[$c] = kv;
                        s[$c] = sv;
                    }
                };
            }
            decay_and_predict!(0);
            decay_and_predict!(1);
            decay_and_predict!(2);
            decay_and_predict!(3);
            let predicted = warp_xor_sum(partial);
            let vcorr = *beta.add(g_at) * (*v.add(v_base + vi as usize) - predicted);

            partial = 0.0;
            macro_rules! correct_and_read {
                ($c:literal) => {
                    if $c < chunks {
                        let j = ($c << 7) + j0;
                        let (sv, kv) = (s[$c], kc[$c]);
                        let sv = [
                            vcorr.mul_add(kv[0], sv[0]),
                            vcorr.mul_add(kv[1], sv[1]),
                            vcorr.mul_add(kv[2], sv[2]),
                            vcorr.mul_add(kv[3], sv[3]),
                        ];
                        store4(row.add(j), sv);
                        let qv = load4(qq.add(j));
                        partial = sv[0].mul_add(qv[0], partial);
                        partial = sv[1].mul_add(qv[1], partial);
                        partial = sv[2].mul_add(qv[2], partial);
                        partial = sv[3].mul_add(qv[3], partial);
                    }
                };
            }
            correct_and_read!(0);
            correct_and_read!(1);
            correct_and_read!(2);
            correct_and_read!(3);
            let o = warp_xor_sum(partial);
            if lane == 0 {
                *out.add(v_base + vi as usize) = o;
            }
        }
    }

    /// The softmax probability of a row's maximum, `1 / sum(exp(x - max))`:
    /// the confidence of the greedy token.
    ///
    /// One block walks the whole row, so each thread's strided loop is long;
    /// it reads eight elements ahead of using them, which keeps the loads
    /// from queueing behind the exponentials.
    ///
    /// # Safety
    /// x covers n >= 1 floats, out one. One 256-thread block.
    #[kernel]
    pub unsafe fn argmax_prob(x: *const f32, n: i32, out: *mut f32) {
        static mut SUMS: SharedArray<f32, 8> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x();
        unsafe {
            let (mut bv, mut bi) = (*x, 0i32);
            let mut i = tid as i32;
            let mut p = x.add(tid as usize);
            while i + 1792 < n {
                let v = (*p.add(0), *p.add(256), *p.add(512), *p.add(768), *p.add(1024), *p.add(1280), *p.add(1536), *p.add(1792));
                (bv, bi) = argmax_merge(bv, bi, v.0, i + 0);
                (bv, bi) = argmax_merge(bv, bi, v.1, i + 256);
                (bv, bi) = argmax_merge(bv, bi, v.2, i + 512);
                (bv, bi) = argmax_merge(bv, bi, v.3, i + 768);
                (bv, bi) = argmax_merge(bv, bi, v.4, i + 1024);
                (bv, bi) = argmax_merge(bv, bi, v.5, i + 1280);
                (bv, bi) = argmax_merge(bv, bi, v.6, i + 1536);
                (bv, bi) = argmax_merge(bv, bi, v.7, i + 1792);
                p = p.wrapping_add(2048);
                i += 2048;
            }
            while i < n {
                (bv, bi) = argmax_merge(bv, bi, *p, i);
                p = p.wrapping_add(256);
                i += 256;
            }
            (bv, bi) = argmax_reduce_block(bv, bi);
            let _ = bi;
            if tid == 0 {
                SUMS[0] = bv;
            }
            thread::sync_threads();
            let row_max = SUMS[0];
            thread::sync_threads();

            let mut acc = 0.0f32;
            let mut i = tid as i32;
            let mut p = x.add(tid as usize);
            while i + 1792 < n {
                let v = (*p.add(0), *p.add(256), *p.add(512), *p.add(768), *p.add(1024), *p.add(1280), *p.add(1536), *p.add(1792));
                acc += (v.0 - row_max).exp();
                acc += (v.1 - row_max).exp();
                acc += (v.2 - row_max).exp();
                acc += (v.3 - row_max).exp();
                acc += (v.4 - row_max).exp();
                acc += (v.5 - row_max).exp();
                acc += (v.6 - row_max).exp();
                acc += (v.7 - row_max).exp();
                p = p.wrapping_add(2048);
                i += 2048;
            }
            while i < n {
                acc += (*p - row_max).exp();
                p = p.wrapping_add(256);
                i += 256;
            }
            let mut off = 16;
            while off > 0 {
                acc += warp::shuffle_down_f32_sync(u32::MAX, acc, off);
                off >>= 1;
            }
            if tid & 31 == 0 {
                SUMS[(tid >> 5) as usize] = acc;
            }
            thread::sync_threads();
            if tid == 0 {
                let mut total = 0.0f32;
                let mut w = 0;
                while w < 8 {
                    total += SUMS[w];
                    w += 1;
                }
                *out = 1.0 / total;
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
                let v = (*p.add(0), *p.add(1), *p.add(2), *p.add(3), *p.add(4), *p.add(5), *p.add(6), *p.add(7));
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
    /// pairs, an in-warp prefix scan turns the padded counts into offsets,
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
                *cumsum.add(e as usize + 1) = ((c + block_size - 1) / block_size) * block_size;
                e += block;
            }
            thread::sync_threads();

            if num_experts <= block {
                let (lane, warp_i) = (tid & 31, tid >> 5);
                let mut v = if tid < num_experts { *cumsum.add(tid as usize + 1) } else { 0 };
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
                    *num_tokens_post_pad = v;
                }
            } else if tid == 0 {
                *cumsum = 0;
                let mut i = 0;
                while i < num_experts {
                    *cumsum.add(i as usize + 1) += *cumsum.add(i as usize);
                    i += 1;
                }
                *num_tokens_post_pad = *cumsum.add(num_experts as usize);
            }
            thread::sync_threads();

            let mut e = tid;
            while e < num_experts {
                let base = *cumsum.add(e as usize);
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
                let last = *cumsum.add(e as usize + 1) / block_size;
                let mut b = base / block_size;
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
            route_token(logits, 0, num_experts, top_k, shared, rval, topk_ids, topk_weights);
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
    /// Every block recomputes the padded prefix from `counts` (it is 257
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
                *cumsum.add(i as usize + 1) = ((c + block_size - 1) / block_size) * block_size;
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
                    *num_tokens_post_pad = *cumsum.add(num_experts as usize);
                }
            }
            thread::sync_threads();

            let base = *cumsum.add(e as usize);
            let (lane, warp_i, n_warps) = (tid & 31, tid >> 5, block >> 5);
            let chunks = (numel + 31) >> 5;
            let per_warp = (chunks + n_warps - 1) / n_warps;
            let c0 = warp_i * per_warp;
            let c1 = if c0 + per_warp < chunks { c0 + per_warp } else { chunks };

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

            let first = base / block_size;
            let last = *cumsum.add(e as usize + 1) / block_size;
            let count = *counts.add(e as usize);
            let mut b = first + tid;
            while b < last && b < expert_capacity {
                *expert_ids.add(b as usize) = e;
                let live = count - (b - first) * block_size;
                let live = if live < 0 { 0 } else if live > block_size { block_size } else { live };
                *bucket_live.add(b as usize) = live;
                b += block;
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

    /// Quantize one element of a 32-element q8_0 block: the block's scale is
    /// its largest magnitude over 127, found across the warp, and the code is
    /// the element over the scale, rounded half away from zero. Returns the
    /// code's low byte and the scale as binary16 bits.
    ///
    /// This is llama.cpp's `quantize_row_q8_0_ref` (ggml-quants.c):
    /// `d = amax / 127`, `q = round(x * (1 / d))`. The maximum is exact and
    /// commutative, so the butterfly order cannot change it.
    ///
    /// # Safety
    /// All 32 lanes of the warp call this together, each with one element of
    /// the same block.
    #[device]
    unsafe fn kvq8_quantize(x: f32) -> (u32, u16) {
        let mut amax = x.abs();
        let mut off = 16;
        while off > 0 {
            amax = amax.max(warp::shuffle_xor_f32_sync(u32::MAX, amax, off));
            off >>= 1;
        }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        (((x * id).round() as i32 as u32) & 0xff, f2h(d))
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
            let x = if is_k { *key.add(j as usize) } else { *value.add(j as usize) };
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
        rp0: u64, rp1: u64, rp2: u64, rp3: u64, rp4: u64, rp5: u64, rp6: u64, rp7: u64,
        ap0: u64, ap1: u64, ap2: u64, ap3: u64, ap4: u64, ap5: u64, ap6: u64, ap7: u64,
        kc0: u64, kc1: u64, kc2: u64, kc3: u64, kc4: u64, kc5: u64, kc6: u64, kc7: u64,
        vc0: u64, vc1: u64, vc2: u64, vc3: u64, vc4: u64, vc5: u64, vc6: u64, vc7: u64,
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
            let (mut k_cache, mut v_cache) = (core::ptr::null_mut::<u16>(), core::ptr::null_mut::<u16>());
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
                at = (*pos as i64 * kv_heads as i64 * head_dim as i64
                    + hh as i64 * head_dim as i64) as usize;
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

    /// `1 / sqrt(x)` as the hardware approximates it: what CUDA's `rsqrtf`
    /// compiles to.
    #[device]
    fn rsqrt_approx(x: f32) -> f32 {
        let y: f32;
        // SAFETY: a register-only instruction.
        unsafe {
            ptx_asm!("rsqrt.approx.f32 %0, %1;", out("=f") y, in("f") x);
        }
        y
    }

    /// Block total for the vision LayerNorm: a shuffle-down tree in each
    /// warp, lane 0 parks its warp's total, and warp 0 runs the same tree
    /// over the parked totals. The result is left in `sums[0]`.
    ///
    /// # Safety
    /// sums covers 32 floats of shared memory; every thread of a 1-D block
    /// participates.
    #[device]
    unsafe fn vision_block_total(v: f32, sums: *mut f32) {
        let tid = thread::threadIdx_x() as i32;
        let mut v = v;
        let mut off = 16;
        while off > 0 {
            v += warp::shuffle_down_f32_sync(u32::MAX, v, off);
            off >>= 1;
        }
        unsafe {
            if tid & 31 == 0 {
                *sums.add((tid >> 5) as usize) = v;
            }
            thread::sync_threads();
            if tid < 32 {
                let warps = (thread::blockDim_x() as i32 + 31) >> 5;
                let mut t = if tid < warps { *sums.add(tid as usize) } else { 0.0 };
                let mut off = 16;
                while off > 0 {
                    t += warp::shuffle_down_f32_sync(u32::MAX, t, off);
                    off >>= 1;
                }
                if tid == 0 {
                    *sums = t;
                }
            }
            thread::sync_threads();
        }
    }

    /// LayerNorm over one fp16 row a block: population mean and variance in
    /// fp32, `(x - mean) * rsqrt(var + eps) * gamma + beta`, stored as fp16.
    ///
    /// grid: (rows,). block: 256.
    ///
    /// # Safety
    /// x and out cover rows * width halves; gamma and beta cover width.
    #[kernel]
    pub unsafe fn vision_layer_norm(
        x: *const u16,
        out: *mut u16,
        gamma: *const u16,
        beta: *const u16,
        width: i32,
        eps: f32,
    ) {
        static mut WARP_SUMS: SharedArray<f32, 32> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as i32;
        let step = thread::blockDim_x() as i32;
        unsafe {
            let sums = SharedArray::as_raw_mut_ptr(&raw mut WARP_SUMS);
            let at = (thread::blockIdx_x() as i64 * width as i64) as usize;
            let (row, dst) = (x.add(at), out.add(at));
            let mut sum = 0.0f32;
            let mut j = tid;
            while j < width {
                sum += h2f(*row.add(j as usize));
                j += step;
            }
            vision_block_total(sum, sums);
            let mean = *sums / width as f32;
            thread::sync_threads();

            let mut var = 0.0f32;
            let mut j = tid;
            while j < width {
                let d = h2f(*row.add(j as usize)) - mean;
                var = d.mul_add(d, var);
                j += step;
            }
            vision_block_total(var, sums);
            let inv = rsqrt_approx(*sums / width as f32 + eps);

            let mut j = tid;
            while j < width {
                let g = h2f(*gamma.add(j as usize));
                let b = h2f(*beta.add(j as usize));
                let centered = (h2f(*row.add(j as usize)) - mean) * inv;
                *dst.add(j as usize) = f2h(centered.mul_add(g, b));
                j += step;
            }
        }
    }

    /// Four Q8_0 weights to four fp16 values, each `code * scale` rounded
    /// once. A code byte beside 0x64 is the fp16 `1024 + (code + 128)`
    /// exactly; subtracting 1152 leaves the code exactly, and the multiply by
    /// the block's scale is the one rounding.
    ///
    /// grid: ceil(n / 4 / 256). block: 256.
    ///
    /// # Safety
    /// wq covers n int8 codes, ws n / 32 fp16 scales, w16 n halves; n is a
    /// multiple of 4 and all three are 4-byte aligned.
    #[kernel]
    pub unsafe fn hg_q8_to_half_w(wq: *const i8, ws: *const u16, w16: *mut u16, n: i64) {
        let i = (thread::blockIdx_x() as i64 * thread::blockDim_x() as i64
            + thread::threadIdx_x() as i64)
            * 4;
        if i >= n {
            return;
        }
        unsafe {
            let q = *(wq.add(i as usize) as *const u32) ^ 0x8080_8080;
            let sc = *ws.add((i >> 5) as usize) as u32;
            let (lo, hi): (u32, u32);
            ptx_asm!(
                "{ prmt.b32 %0, %2, %3, 0x5150; prmt.b32 %1, %2, %3, 0x5352; sub.rn.f16x2 %0, %0, %4; sub.rn.f16x2 %1, %1, %4; mul.rn.f16x2 %0, %0, %5; mul.rn.f16x2 %1, %1, %5; }",
                out("=r") lo,
                out("=r") hi,
                in("r") q,
                in("r") 0x6464_6464u32,
                in("r") 0x6480_6480u32,
                in("r") sc | (sc << 16),
            );
            *(w16.add(i as usize) as *mut u64) = lo as u64 | ((hi as u64) << 32);
        }
    }

    /// Block-wide maximum; as `dec_block_sum`.
    ///
    /// # Safety
    /// As `dec_block_sum`.
    #[device]
    unsafe fn dec_block_max(v: f32) -> f32 {
        static mut RED_MAX: SharedArray<f32, 32> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x();
        let (lane, w, nw) = (tid & 31, tid >> 5, thread::blockDim_x() >> 5);
        let mut v = v;
        let mut o = 16;
        while o > 0 {
            v = v.max(warp::shuffle_xor_f32_sync(u32::MAX, v, o));
            o >>= 1;
        }
        unsafe {
            let red = SharedArray::as_raw_mut_ptr(&raw mut RED_MAX);
            thread::sync_threads();
            if lane == 0 {
                *red.add(w as usize) = v;
            }
            thread::sync_threads();
            v = if lane < nw { *red.add(lane as usize) } else { f32::NEG_INFINITY };
        }
        let mut o = 16;
        while o > 0 {
            v = v.max(warp::shuffle_xor_f32_sync(u32::MAX, v, o));
            o >>= 1;
        }
        v
    }

    /// Element `i` of one row in GGUF storage format `fmt` (0 Q8_0, 1 BF16,
    /// 2 Q6_K, 3 F16, 4 Q4_0, 5 Q4_K, 6 Q5_K). Operand order follows
    /// `llmcuda_kernels::quant`; nothing here is fused.
    ///
    /// # Safety
    /// row covers the whole blocks of its format up to element i.
    #[device]
    unsafe fn dequant_element(row: *const u8, fmt: i32, i: i32) -> f32 {
        unsafe {
            match fmt {
                0 => {
                    let b = row.add(((i >> 5) * 34) as usize);
                    (*b.add((2 + (i & 31)) as usize) as i8 as f32) * load_half_le(b)
                }
                1 => {
                    let bits = *row.add((2 * i) as usize) as u32
                        | ((*row.add((2 * i + 1) as usize) as u32) << 8);
                    f32::from_bits(bits << 16)
                }
                3 => load_half_le(row.add((2 * i) as usize)),
                4 => {
                    let b = row.add(((i >> 5) * 18) as usize);
                    let e = i & 31;
                    let byte = *b.add((2 + (e & 15)) as usize) as i32;
                    let q = if e < 16 { byte & 0xF } else { byte >> 4 };
                    (q - 8) as f32 * load_half_le(b)
                }
                2 => {
                    let b = row.add(((i >> 8) * 210) as usize);
                    let e = i & 255;
                    let (half, r) = (e >> 7, e & 127);
                    let (j, l) = (r >> 5, r & 31);
                    let ql = b.add((half * 64) as usize);
                    let qh = b.add((128 + half * 32) as usize);
                    let sc = b.add((192 + half * 8) as usize) as *const i8;
                    let lowbyte = *ql.add((l + (j & 1) * 32) as usize) as i32;
                    let low = if j < 2 { lowbyte & 0xF } else { lowbyte >> 4 };
                    let high = ((*qh.add(l as usize) as i32) >> (2 * j)) & 3;
                    let d = load_half_le(b.add(208));
                    d * (*sc.add(((l >> 4) + 2 * j) as usize) as f32)
                        * ((low | (high << 4)) - 32) as f32
                }
                5 | 6 => {
                    let five = fmt == 6;
                    let b = row.add(((i >> 8) * if five { 176 } else { 144 }) as usize);
                    let e = i & 255;
                    let (g, sub, l) = (e >> 6, (e >> 5) & 1, e & 31);
                    let d = load_half_le(b);
                    let dmin = load_half_le(b.add(2));
                    // `get_scale_min_k4` from ggml-quants.c.
                    let (js, q) = ((2 * g + sub) as usize, b.add(4));
                    let (sc, m) = if js < 4 {
                        ((*q.add(js) & 63) as i32, (*q.add(js + 4) & 63) as i32)
                    } else {
                        (
                            ((*q.add(js + 4) & 0xF) | ((*q.add(js - 4) >> 6) << 4)) as i32,
                            ((*q.add(js + 4) >> 4) | ((*q.add(js) >> 6) << 4)) as i32,
                        )
                    };
                    let byte =
                        *b.add((if five { 48 } else { 16 } + g * 32 + l) as usize) as i32;
                    let mut code = if sub == 0 { byte & 0xF } else { byte >> 4 };
                    if five {
                        code += (((*b.add((16 + l) as usize) as i32) >> (2 * g + sub)) & 1) << 4;
                    }
                    let d1 = d * sc as f32;
                    let m1 = dmin * m as f32;
                    d1 * code as f32 - m1
                }
                _ => 0.0,
            }
        }
    }

    /// Mean of the dequantized embedding rows of each span's tokens.
    ///
    /// grid: (ceil(cols / 256), spans). block: 256.
    ///
    /// # Safety
    /// table covers every referenced row; tokens covers each span's
    /// [start, end); spans covers nspans * stride ints; out covers nspans
    /// rows of ldo floats.
    #[kernel]
    pub unsafe fn dec_lexical(
        table: *const u8,
        fmt: i32,
        row_bytes: i64,
        cols: i32,
        tokens: *const i32,
        spans: *const i32,
        stride: i32,
        nspans: i32,
        out: *mut f32,
        ldo: i64,
    ) {
        let sp = thread::blockIdx_y() as i32;
        let c = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as i32;
        if sp >= nspans || c >= cols {
            return;
        }
        unsafe {
            let start = *spans.add((sp * stride) as usize);
            let end = *spans.add((sp * stride + 1) as usize);
            let mut acc = 0.0f32;
            let mut t = start;
            while t < end {
                let row = table.add((*tokens.add(t as usize) as i64 * row_bytes) as usize);
                acc += dequant_element(row, fmt, c);
                t += 1;
            }
            *out.add((sp as i64 * ldo + c as i64) as usize) = acc / (end - start) as f32;
        }
    }

    /// Per question: attention of its base row over its own options (dot
    /// products scaled, softmax), the weighted option summary, a LayerNorm
    /// of that summary, and the field row
    /// `base + ln(summary) + gproj + type_embd[type]`.
    ///
    /// grid: (questions,). block: 256, with width floats of dynamic shared
    /// memory.
    ///
    /// # Safety
    /// opts covers n_opts rows of ldo; ospans n_opts * ostride ints with the
    /// owner at +2; base and fields one row per question; gproj, sg and sb
    /// width floats; type_embd one row per type; qspans the type at +2;
    /// weights n_opts floats.
    #[kernel]
    pub unsafe fn dec_question_fields(
        opts: *const f32,
        ldo: i64,
        n_opts: i32,
        ospans: *const i32,
        ostride: i32,
        base: *const f32,
        ldb: i64,
        gproj: *const f32,
        type_embd: *const f32,
        qspans: *const i32,
        qstride: i32,
        sg: *const f32,
        sb: *const f32,
        eps: f32,
        weights: *mut f32,
        fields: *mut f32,
        ldf: i64,
        width: i32,
        scale: f32,
    ) {
        let summary = DynamicSharedArray::<f32>::get();
        let q = thread::blockIdx_x() as i32;
        let tid = thread::threadIdx_x() as i32;
        let step = thread::blockDim_x() as i32;
        let (lane, warp_i, nw) = (tid & 31, tid >> 5, step >> 5);
        unsafe {
            let bq = base.add((q as i64 * ldb) as usize);
            let mut o = warp_i;
            while o < n_opts {
                if *ospans.add((o * ostride + 2) as usize) == q {
                    let oo = opts.add((o as i64 * ldo) as usize);
                    let mut d = 0.0f32;
                    let mut c = lane;
                    while c < width {
                        d = (*oo.add(c as usize)).mul_add(*bq.add(c as usize), d);
                        c += 32;
                    }
                    d = warp_xor_sum(d);
                    if lane == 0 {
                        *weights.add(o as usize) = d * scale;
                    }
                }
                o += nw;
            }
            thread::sync_threads();
            let mut mx = f32::NEG_INFINITY;
            let mut o = tid;
            while o < n_opts {
                if *ospans.add((o * ostride + 2) as usize) == q {
                    mx = mx.max(*weights.add(o as usize));
                }
                o += step;
            }
            mx = dec_block_max(mx);
            let mut se = 0.0f32;
            let mut o = tid;
            while o < n_opts {
                if *ospans.add((o * ostride + 2) as usize) == q {
                    se += (*weights.add(o as usize) - mx).exp();
                }
                o += step;
            }
            se = dec_block_sum(se);
            let mut o = tid;
            while o < n_opts {
                if *ospans.add((o * ostride + 2) as usize) == q {
                    *weights.add(o as usize) = (*weights.add(o as usize) - mx).exp() / se;
                }
                o += step;
            }
            thread::sync_threads();

            let mut s = 0.0f32;
            let mut c = tid;
            while c < width {
                let mut acc = 0.0f32;
                let mut o = 0;
                while o < n_opts {
                    if *ospans.add((o * ostride + 2) as usize) == q {
                        acc = (*weights.add(o as usize))
                            .mul_add(*opts.add((o as i64 * ldo + c as i64) as usize), acc);
                    }
                    o += 1;
                }
                *summary.add(c as usize) = acc;
                s += acc;
                c += step;
            }
            let mean = dec_block_sum(s) / width as f32;
            let mut v = 0.0f32;
            let mut c = tid;
            while c < width {
                let d = *summary.add(c as usize) - mean;
                v = d.mul_add(d, v);
                c += step;
            }
            let rstd = 1.0 / (dec_block_sum(v) / width as f32 + eps).sqrt();
            let ty = *qspans.add((q * qstride + 2) as usize);
            let fq = fields.add((q as i64 * ldf) as usize);
            let te = type_embd.add((ty as i64 * width as i64) as usize);
            let mut c = tid;
            while c < width {
                let cu = c as usize;
                let ln = ((*summary.add(cu) - mean) * rstd).mul_add(*sg.add(cu), *sb.add(cu));
                *fq.add(cu) = *bq.add(cu) + ln + *gproj.add(cu) + *te.add(cu);
                c += step;
            }
        }
    }

    /// The scorer's feature row for each option, `[field, option, field *
    /// option, |field - option|]`, and its two scalars: the cosine of the
    /// lexical row with `qmean + global` (the prior) and of field with
    /// option.
    ///
    /// grid: (options,). block: 256.
    ///
    /// # Safety
    /// fieldn and qmean cover one row per question, optn and lexical one per
    /// option, global hidden floats; feat covers n_opts rows of at least
    /// 4 * width; aux two floats an option; ospans the owner at +2.
    #[kernel]
    pub unsafe fn dec_features(
        fieldn: *const f32,
        ldf: i64,
        optn: *const f32,
        ldo: i64,
        ospans: *const i32,
        ostride: i32,
        n_opts: i32,
        qmean: *const f32,
        ldq: i64,
        global: *const f32,
        lexical: *const f32,
        ldl: i64,
        hidden: i32,
        width: i32,
        feat: *mut f32,
        ldfeat: i64,
        aux: *mut f32,
    ) {
        let o = thread::blockIdx_x() as i32;
        let tid = thread::threadIdx_x() as i32;
        let step = thread::blockDim_x() as i32;
        if o >= n_opts {
            return;
        }
        unsafe {
            let q = *ospans.add((o * ostride + 2) as usize);
            let f = fieldn.add((q as i64 * ldf) as usize);
            let x = optn.add((o as i64 * ldo) as usize);
            let out = feat.add((o as i64 * ldfeat) as usize);
            let w = width as usize;
            let (mut ff, mut xx, mut fx) = (0.0f32, 0.0f32, 0.0f32);
            let mut c = tid;
            while c < width {
                let cu = c as usize;
                let (a, b) = (*f.add(cu), *x.add(cu));
                // The product is also stored, so NVRTC does not contract it
                // into the running sum.
                let ab = a * b;
                ff = a.mul_add(a, ff);
                xx = b.mul_add(b, xx);
                fx += ab;
                *out.add(cu) = a;
                *out.add(w + cu) = b;
                *out.add(2 * w + cu) = ab;
                *out.add(3 * w + cu) = (a - b).abs();
                c += step;
            }
            let qm = qmean.add((q as i64 * ldq) as usize);
            let lx = lexical.add((o as i64 * ldl) as usize);
            let (mut aa, mut ll, mut al) = (0.0f32, 0.0f32, 0.0f32);
            let mut c = tid;
            while c < hidden {
                let cu = c as usize;
                let (a, b) = (*qm.add(cu) + *global.add(cu), *lx.add(cu));
                aa = a.mul_add(a, aa);
                ll = b.mul_add(b, ll);
                al = a.mul_add(b, al);
                c += step;
            }
            let ff = dec_block_sum(ff);
            let xx = dec_block_sum(xx);
            let fx = dec_block_sum(fx);
            let aa = dec_block_sum(aa);
            let ll = dec_block_sum(ll);
            let al = dec_block_sum(al);
            if tid == 0 {
                let cosine = fx / (ff.sqrt().max(1e-12) * xx.sqrt().max(1e-12));
                let prior = al / (ll.sqrt().max(1e-12) * aa.sqrt().max(1e-12));
                *aux.add((o * 2) as usize) = prior;
                *aux.add((o * 2 + 1) as usize) = cosine;
            }
        }
    }

    /// Token embedding gather from a table in Q6_K (format 0), Q4_0 (1),
    /// Q4_K (2), Q5_K (3) or F16 (4), one element a thread.
    ///
    /// The row is first copied into shared memory when it is whole aligned
    /// words and fits: a table mapped from host memory costs a round trip a
    /// read, and the gather reads a row byte by byte. Every thread issues all
    /// four of its word loads before storing any.
    ///
    /// Operand order follows `llmcuda_kernels::quant`. Nothing is fused; the
    /// affine k-quants are one multiply and one subtract.
    ///
    /// grid: (token slots,). block: any.
    ///
    /// # Safety
    /// table covers every row an id names in its format; ids covers n_tokens;
    /// out covers n_tokens * hidden floats. hidden is a whole number of the
    /// format's blocks.
    #[kernel]
    pub unsafe fn fwd_embed_quant(
        table: *const u8,
        ids: *const i32,
        out: *mut f32,
        hidden: i32,
        n_tokens: i32,
        format: i32,
    ) {
        static mut STAGE: SharedArray<u32, 4096> = SharedArray::UNINIT;
        let t = thread::blockIdx_x() as i32;
        if t >= n_tokens {
            return;
        }
        let tid = thread::threadIdx_x() as i32;
        let step = thread::blockDim_x() as i32;
        let row_bytes = match format {
            0 => (hidden / 256) as i64 * 210,
            1 => (hidden / 32) as i64 * 18,
            2 => (hidden / 256) as i64 * 144,
            3 => (hidden / 256) as i64 * 176,
            _ => hidden as i64 * 2,
        };
        unsafe {
            let mut row = table.add((*ids.add(t as usize) as i64 * row_bytes) as usize);
            if row_bytes & 3 == 0 && row_bytes <= 4096 * 4 && (row as usize) & 3 == 0 {
                let stage = SharedArray::as_raw_mut_ptr(&raw mut STAGE);
                let words = (row_bytes >> 2) as i32;
                let src = row as *const u32;
                let mut base = 0;
                while base < words {
                    let w0 = base + tid;
                    let (w1, w2, w3) = (w0 + step, w0 + 2 * step, w0 + 3 * step);
                    let v0 = if w0 < words { *src.add(w0 as usize) } else { 0 };
                    let v1 = if w1 < words { *src.add(w1 as usize) } else { 0 };
                    let v2 = if w2 < words { *src.add(w2 as usize) } else { 0 };
                    let v3 = if w3 < words { *src.add(w3 as usize) } else { 0 };
                    if w0 < words {
                        *stage.add(w0 as usize) = v0;
                    }
                    if w1 < words {
                        *stage.add(w1 as usize) = v1;
                    }
                    if w2 < words {
                        *stage.add(w2 as usize) = v2;
                    }
                    if w3 < words {
                        *stage.add(w3 as usize) = v3;
                    }
                    base += 4 * step;
                }
                thread::sync_threads();
                row = stage as *const u8;
            }
            let dst = out.add((t as i64 * hidden as i64) as usize);
            let mut j = tid;
            while j < hidden {
                let v = if format == 0 {
                    let sb = row.add(((j >> 8) as i64 * 210) as usize);
                    let r = j & 255;
                    let (half, grp, l) = (r >> 7, (r >> 5) & 3, r & 31);
                    let ql = *sb.add((half * 64 + if grp & 1 != 0 { l + 32 } else { l }) as usize)
                        as i32;
                    let qh = *sb.add((128 + half * 32 + l) as usize) as i32;
                    let sc = sb.add((192 + half * 8) as usize) as *const i8;
                    let d = load_half_le(sb.add(208));
                    let lo4 = if grp < 2 { ql & 0xF } else { ql >> 4 };
                    let raw = lo4 | (((qh >> (2 * grp)) & 3) << 4);
                    d * (*sc.add(((l >> 4) + 2 * grp) as usize) as f32) * (raw - 32) as f32
                } else if format == 1 {
                    let blk = row.add(((j >> 5) as i64 * 18) as usize);
                    let r = j & 31;
                    let d = load_half_le(blk);
                    let q = *blk.add((2 + (r & 15)) as usize) as u32;
                    let code = if r < 16 { (q & 0xF) as i32 } else { (q >> 4) as i32 };
                    (code - 8) as f32 * d
                } else if format == 4 {
                    load_half_le(row.add((j as i64 * 2) as usize))
                } else {
                    let q5 = format == 3;
                    let (sb_bytes, qs_off) = if q5 { (176, 48) } else { (144, 16) };
                    let sb = row.add(((j >> 8) as i64 * sb_bytes) as usize);
                    let r = j & 255;
                    let (g, sub, l) = (r >> 6, (r >> 5) & 1, r & 31);
                    let js = (2 * g + sub) as usize;
                    let d = load_half_le(sb);
                    let dmin = load_half_le(sb.add(2));
                    // `get_scale_min_k4` from ggml-quants.c. The `js >= 4`
                    // branch is the one a wrong shift breaks silently.
                    let q = sb.add(4);
                    let (sc, m) = if js < 4 {
                        ((*q.add(js) & 63) as i32, (*q.add(js + 4) & 63) as i32)
                    } else {
                        (
                            ((*q.add(js + 4) & 0xF) | ((*q.add(js - 4) >> 6) << 4)) as i32,
                            ((*q.add(js + 4) >> 4) | ((*q.add(js) >> 6) << 4)) as i32,
                        )
                    };
                    let byte = *sb.add((qs_off + g * 32 + l) as usize) as u32;
                    let mut code = if sub != 0 { (byte >> 4) as i32 } else { (byte & 0xF) as i32 };
                    if q5 {
                        code |= (((*sb.add((16 + l) as usize) as i32) >> js) & 1) << 4;
                    }
                    (d * sc as f32) * code as f32 - dmin * m as f32
                };
                *dst.add(j as usize) = v;
                j += step;
            }
        }
    }

    /// K2's decode-step rotary and cache append for up to eight sequences,
    /// from the prepared sine/cosine table: rotate the query and key rows,
    /// and write the rotated key and the raw value to each sequence's fp16
    /// cache at its own position.
    ///
    /// grid: (qh + kh, sequences <= 8). block: dim.
    ///
    /// # Safety
    /// q/qr cover sequences * qh * dim floats, k/kr/v sequences * kh * dim;
    /// coefficients covers sequences * rope_dim; each used slot holds a live
    /// pointer (append position, K cache, V cache), the caches covering
    /// position + 1 rows of kh * dim halves.
    #[kernel]
    pub unsafe fn k2_rotary_append(
        q: *const f32,
        qr: *mut f32,
        k: *const f32,
        kr: *mut f32,
        v: *const f32,
        coefficients: *const f32,
        ap0: u64, ap1: u64, ap2: u64, ap3: u64, ap4: u64, ap5: u64, ap6: u64, ap7: u64,
        kc0: u64, kc1: u64, kc2: u64, kc3: u64, kc4: u64, kc5: u64, kc6: u64, kc7: u64,
        vc0: u64, vc1: u64, vc2: u64, vc3: u64, vc4: u64, vc5: u64, vc6: u64, vc7: u64,
        qh: i32,
        kh: i32,
        dim: i32,
        rope_dim: i32,
    ) {
        let seq = thread::blockIdx_y() as i32;
        let h = thread::blockIdx_x() as i32;
        let d = thread::threadIdx_x() as i32;
        let is_q = h < qh;
        let hh = if is_q { h } else { h - qh };
        let heads = if is_q { qh } else { kh };
        let (input, out) = if is_q { (q, qr) } else { (k, kr) };
        let base = ((seq as i64 * heads as i64 + hh as i64) * dim as i64) as usize;
        let (du, hu) = (d as usize, (rope_dim / 2) as usize);
        let half = rope_dim / 2;
        unsafe {
            let (mut k_cache, mut v_cache) = (core::ptr::null_mut::<u16>(), core::ptr::null_mut::<u16>());
            let mut at = 0i64;
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
                at = *pos as i64 * kh as i64 * dim as i64 + hh as i64 * dim as i64;
            }
            let co = coefficients.add((seq as i64 * rope_dim as i64) as usize + 2 * du);
            if d >= rope_dim {
                let x = *input.add(base + du);
                *out.add(base + du) = x;
                if !is_q {
                    *k_cache.add(at as usize + du) = f2h(x);
                    *v_cache.add(at as usize + du) = f2h(*v.add(base + du));
                }
                return;
            }
            if d >= half {
                return;
            }
            let (sin_a, cos_a) = (*co, *co.add(1));
            let (x0, x1) = (*input.add(base + du), *input.add(base + du + hu));
            let y0 = x0.mul_add(cos_a, -(x1 * sin_a));
            let y1 = x0.mul_add(sin_a, x1 * cos_a);
            *out.add(base + du) = y0;
            *out.add(base + du + hu) = y1;
            if !is_q {
                *k_cache.add(at as usize + du) = f2h(y0);
                *k_cache.add(at as usize + du + hu) = f2h(y1);
                *v_cache.add(at as usize + du) = f2h(*v.add(base + du));
                *v_cache.add(at as usize + du + hu) = f2h(*v.add(base + du + hu));
            }
        }
    }

    /// Store element `e` of position `pos`'s row: binary16, or a q8_0 code
    /// plus, from each block's first lane, its scale.
    ///
    /// # Safety
    /// All 32 lanes of the warp call this together with one q8_0 block's
    /// elements; cache covers the row in its format.
    #[device]
    unsafe fn kv8_store(cache: *mut u16, q8: i32, pos: i64, row: i32, e: i32, x: f32) {
        unsafe {
            if q8 == 0 {
                *cache.add((pos * row as i64 + e as i64) as usize) = f2h(x);
                return;
            }
            let (code, d) = kvq8_quantize(x);
            let dst = (cache as *mut u8).add((pos * (row as i64 + (row >> 4) as i64)) as usize);
            *dst.add(e as usize) = code as u8;
            if e & 31 == 0 {
                *(dst.add((row + 2 * (e >> 5)) as usize) as *mut u16) = d;
            }
        }
    }

    /// `k2_rotary_append` with either half of the cache in q8_0. A rotary
    /// pair is computed by one thread, so the key and value rows are parked
    /// in shared memory and every thread then stores its own element, which
    /// puts one 32-element block in each warp for the q8_0 scale.
    ///
    /// grid: (qh + kh, sequences <= 8). block: dim, a multiple of 32, at
    /// most 1024.
    ///
    /// # Safety
    /// As `k2_rotary_append`, each cache covering its rows in its own format.
    #[kernel]
    pub unsafe fn k2_rotary_append_kv8(
        q: *const f32,
        qr: *mut f32,
        k: *const f32,
        kr: *mut f32,
        v: *const f32,
        coefficients: *const f32,
        ap0: u64, ap1: u64, ap2: u64, ap3: u64, ap4: u64, ap5: u64, ap6: u64, ap7: u64,
        kc0: u64, kc1: u64, kc2: u64, kc3: u64, kc4: u64, kc5: u64, kc6: u64, kc7: u64,
        vc0: u64, vc1: u64, vc2: u64, vc3: u64, vc4: u64, vc5: u64, vc6: u64, vc7: u64,
        qh: i32,
        kh: i32,
        dim: i32,
        rope_dim: i32,
        kq8: i32,
        vq8: i32,
    ) {
        static mut KROW: SharedArray<f32, 1024> = SharedArray::UNINIT;
        static mut VROW: SharedArray<f32, 1024> = SharedArray::UNINIT;
        let seq = thread::blockIdx_y() as i32;
        let h = thread::blockIdx_x() as i32;
        let d = thread::threadIdx_x() as i32;
        let is_q = h < qh;
        let hh = if is_q { h } else { h - qh };
        let heads = if is_q { qh } else { kh };
        let (input, out) = if is_q { (q, qr) } else { (k, kr) };
        let base = ((seq as i64 * heads as i64 + hh as i64) * dim as i64) as usize;
        let (du, hu) = (d as usize, (rope_dim / 2) as usize);
        let half = rope_dim / 2;
        unsafe {
            let (mut k_cache, mut v_cache) = (core::ptr::null_mut::<u16>(), core::ptr::null_mut::<u16>());
            let mut at = 0i64;
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
                at = *pos as i64 * kh as i64 * dim as i64 + hh as i64 * dim as i64;
            }
            let co = coefficients.add((seq as i64 * rope_dim as i64) as usize + 2 * du);
            let krow = SharedArray::as_raw_mut_ptr(&raw mut KROW);
            let vrow = SharedArray::as_raw_mut_ptr(&raw mut VROW);
            if d >= rope_dim {
                let x = *input.add(base + du);
                *out.add(base + du) = x;
                if !is_q {
                    *krow.add(du) = x;
                    *vrow.add(du) = *v.add(base + du);
                }
            } else if d < half {
                let (sin_a, cos_a) = (*co, *co.add(1));
                let (x0, x1) = (*input.add(base + du), *input.add(base + du + hu));
                let y0 = x0.mul_add(cos_a, -(x1 * sin_a));
                let y1 = x0.mul_add(sin_a, x1 * cos_a);
                *out.add(base + du) = y0;
                *out.add(base + du + hu) = y1;
                if !is_q {
                    *krow.add(du) = y0;
                    *krow.add(du + hu) = y1;
                    *vrow.add(du) = *v.add(base + du);
                    *vrow.add(du + hu) = *v.add(base + du + hu);
                }
            }
            if is_q {
                return;
            }
            thread::sync_threads();
            let row = kh * dim;
            let e = hh * dim + d;
            let pos = at / row as i64;
            kv8_store(k_cache, kq8, pos, row, e, *krow.add(du));
            kv8_store(v_cache, vq8, pos, row, e, *vrow.add(du));
        }
    }

    /// Quantize fp32 activations to int8 a 32-element block at a time: each
    /// block's scale is its largest magnitude over 127, and a code is the
    /// element times `127 / amax`, rounded to nearest even and clamped to
    /// +-127. An all-zero block gets scale 1 and zero codes.
    ///
    /// grid: (rows, ceil(k_dim / 32 / blockDim.y)). block: (32, warps): a
    /// warp is one block.
    ///
    /// # Safety
    /// x and q cover rows * k_dim elements; scales covers rows * k_dim / 32.
    /// k_dim is a multiple of 32.
    #[kernel]
    pub unsafe fn mma_quantize_rows_q8(x: *const f32, q: *mut i8, scales: *mut f32, k_dim: i32) {
        let row = thread::blockIdx_x() as i64;
        let block = (thread::blockIdx_y() * thread::blockDim_y() + thread::threadIdx_y()) as i32;
        let lane = thread::threadIdx_x() as usize;
        let blocks = k_dim / 32;
        if block >= blocks {
            return;
        }
        unsafe {
            let base = (row * k_dim as i64 + (block * 32) as i64) as usize;
            let v = *x.add(base + lane);
            let mut a = v.abs();
            let mut off = 16;
            while off > 0 {
                a = a.max(warp::shuffle_xor_f32_sync(u32::MAX, a, off));
                off >>= 1;
            }
            let d = if a > 0.0 { a / 127.0 } else { 1.0 };
            let inv = if a > 0.0 { 127.0 / a } else { 0.0 };
            let qi = (v * inv).round_ties_even() as i32;
            let qi = if qi < -127 { -127 } else if qi > 127 { 127 } else { qi };
            *q.add(base + lane) = qi as i8;
            if lane == 0 {
                *scales.add((row * blocks as i64 + block as i64) as usize) = d;
            }
        }
    }

    /// Split GGUF Q8_0 blocks (a binary16 scale and 32 int8 codes) into a
    /// contiguous code array and fp32 scales.
    ///
    /// grid: ceil(n_blocks / blockDim.y). block: (32, warps).
    ///
    /// # Safety
    /// src covers n_blocks * 34 bytes, q n_blocks * 32, scales n_blocks.
    #[kernel]
    pub unsafe fn mma_repack_q8_0(src: *const u8, q: *mut i8, scales: *mut f32, n_blocks: i64) {
        let b = thread::blockIdx_x() as i64 * thread::blockDim_y() as i64
            + thread::threadIdx_y() as i64;
        if b >= n_blocks {
            return;
        }
        let lane = thread::threadIdx_x() as usize;
        unsafe {
            let blk = src.add((b * 34) as usize);
            *q.add((b * 32) as usize + lane) = *blk.add(2 + lane) as i8;
            if lane == 0 {
                *scales.add(b as usize) = load_half_le(blk);
            }
        }
    }

    /// `mma_repack_q8_0` keeping the scales as binary16 bits.
    ///
    /// # Safety
    /// As `mma_repack_q8_0`, scales covering n_blocks halves.
    #[kernel]
    pub unsafe fn mma_repack_q8_0_h(src: *const u8, q: *mut i8, scales: *mut u16, n_blocks: i64) {
        let b = thread::blockIdx_x() as i64 * thread::blockDim_y() as i64
            + thread::threadIdx_y() as i64;
        if b >= n_blocks {
            return;
        }
        let lane = thread::threadIdx_x() as usize;
        unsafe {
            let blk = src.add((b * 34) as usize);
            *q.add((b * 32) as usize + lane) = *blk.add(2 + lane) as i8;
            if lane == 0 {
                *scales.add(b as usize) = *blk as u16 | ((*blk.add(1) as u16) << 8);
            }
        }
    }
}

fn main() {}
