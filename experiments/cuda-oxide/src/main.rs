//! Rust implementations of the layer-op and attention helper kernels.
//! Raw pointers preserve the existing cudarc ABI, including in-place output.

use cuda_device::{DynamicSharedArray, device, kernel, thread, warp};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// Add a residual with the same grid-stride loop as the NVRTC baseline.
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

    /// Softplus with the baseline overflow guard and libdevice exp/log.
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

    /// Split-half partial RoPE; paired outputs share double-precision angles.
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

    /// Weighted RMSNorm with the existing per-row launch ABI.
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

    /// Fused RMSNorm/SwiGLU, retaining the normalized intermediate waypoint.
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

    /// Fused RMSNorm/SwiGLU that hands int8 quants and one scale per 32
    /// elements to the integer tensor cores, one element a thread.
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

    /// Causal depthwise convolution over `[tokens][channels]`, reading the
    /// carried cache for positions before this batch. Taps accumulate in
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

    /// Row positions `position[0] + i` for a prefill chunk.
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
}

fn main() {}
