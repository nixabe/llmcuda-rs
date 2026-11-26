//! The engine's Rust CUDA kernels: layer ops and the small MoE, Gated
//! DeltaNet, K2 and attention kernels.
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

use cuda_device::{DynamicSharedArray, device, kernel, thread, warp};
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
        while i < n {
            unsafe {
                let g = *gate.add(i as usize);
                *out.add(i as usize) = (g / (1.0 + (-g).exp())) * *up.add(i as usize);
            }
            i += stride;
        }
    }

    /// Elementwise or per-row sigmoid gating, preserving the sigmoid waypoint.
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
        while i < n {
            let g = if broadcast != 0 { i / width as i64 } else { i };
            unsafe {
                let s = 1.0 / (1.0 + (-*gate.add(g as usize)).exp());
                if broadcast == 0 || i % width as i64 == 0 {
                    *sig.add(g as usize) = s;
                }
                *out.add(i as usize) = s * *x.add(i as usize);
            }
            i += stride;
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

    /// Merge the per-split partial softmaxes of split decode attention.
    ///
    /// Each split carries its running maximum `m`, its normalizer `l` and its
    /// weighted value sum, all relative to its own maximum; they are rescaled
    /// to the global maximum and summed in ascending split order. Split 0
    /// always holds at least key 0, so the global maximum is never -inf and
    /// the subtraction never forms inf - inf; a split with no keys has
    /// `m = -inf` and contributes nothing.
    ///
    /// grid: (q_heads,). block: head_dim threads, one per output dimension.
    ///
    /// # Safety
    /// part_acc covers n_splits * q_heads * head_dim floats, part_m and
    /// part_l n_splits * q_heads, out q_heads * head_dim; out is disjoint
    /// from the inputs. n_splits >= 1.
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
                let f = if ms == f32::NEG_INFINITY { 0.0 } else { (ms - gm).exp() };
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
}

fn main() {}
