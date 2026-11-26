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

    /// The binary16 scale at the head of a Q8_0 block, little-endian.
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
    /// which is flash-linear-attention's l2norm, transformers'
    /// (`x * rsqrt(sum + eps)`) and llama.cpp's since 5fdfa6282
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
}

fn main() {}
