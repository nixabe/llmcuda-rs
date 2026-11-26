//! The dense (`qwen35`) FFN's narrow-layout kernels.

use crate::common::{f2h, inverse_rms};
use cuda_device::{kernel, thread};

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
        let mut j = (thread::blockIdx_y() * thread::blockDim_x() + thread::threadIdx_x()) as i32;
        while j < hidden {
            let at = base + j as usize;
            *l_out.add(at) = *ffn_out.add(at) + *residual.add(at);
            j += stride;
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
