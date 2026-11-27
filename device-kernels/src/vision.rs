//! Vision tower elementwise and row kernels over binary16 activations.

use crate::common::{f2h, h2f};
use cuda_device::{SharedArray, device, kernel, ptx_asm, thread, warp};

/// In-place exact GELU: `x / 2 * (1 + erf(x / sqrt 2))`, the activation of
/// the `inkling2v` tower (ggml's `gelu_erf`).
///
/// erf is Abramowitz and Stegun 7.1.26, within 1.5e-7 everywhere: the
/// result is rounded to binary16, whose spacing at these magnitudes is a
/// thousand times that.
///
/// grid: (ceil(n / block)), any block.
///
/// # Safety
/// x covers `n` halves.
#[kernel]
pub unsafe fn vision_gelu_erf(x: *mut u16, n: i64) {
    let i =
        thread::blockIdx_x() as i64 * thread::blockDim_x() as i64 + thread::threadIdx_x() as i64;
    if i >= n {
        return;
    }
    unsafe {
        let v = h2f(*x.add(i as usize));
        let z = v * 0.707_106_77_f32;
        let a = z.abs();
        let t = 1.0 / (1.0 + 0.327_591_1_f32 * a);
        let poly = ((((1.061_405_4_f32 * t - 1.453_152_f32) * t + 1.421_413_7_f32) * t
            - 0.284_496_74_f32)
            * t
            + 0.254_829_6_f32)
            * t;
        let magnitude = 1.0 - poly * (-a * a).exp();
        let erf = if z < 0.0 { -magnitude } else { magnitude };
        *x.add(i as usize) = f2h((0.5 * v) * (1.0 + erf));
    }
}

/// GELU in place on fp16 activations: ggml's tanh approximation,
/// computed in fp32 and rounded back.
///
/// # Safety
/// x covers n halves; the 1-D launch is nonempty.
#[kernel]
pub unsafe fn vision_gelu(x: *mut u16, n: i64) {
    let i =
        thread::blockIdx_x() as i64 * thread::blockDim_x() as i64 + thread::threadIdx_x() as i64;
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
    let i =
        thread::blockIdx_x() as i64 * thread::blockDim_x() as i64 + thread::threadIdx_x() as i64;
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
    let i =
        thread::blockIdx_x() as i64 * thread::blockDim_x() as i64 + thread::threadIdx_x() as i64;
    if i >= n {
        return;
    }
    unsafe {
        *dst.add(i as usize) = h2f(*src.add(i as usize));
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
            let mut v = if tid < warps {
                RED[tid as usize]
            } else {
                LOWEST
            };
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
            let mut t = if tid < warps {
                *sums.add(tid as usize)
            } else {
                0.0
            };
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
