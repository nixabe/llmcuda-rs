//! Helpers more than one kernel family uses: block and warp reductions,
//! binary16 conversion, strided contractions and the convolution step.

use cuda_device::{DynamicSharedArray, device, ptx_asm, thread, warp};

/// `block_reduce_sum` from the CUDA C++ layer ops, operation for operation:
/// a shuffle-down tree per warp, then thread 0 adds the warp totals in
/// ascending order. The order is the contract -- the fused Gated DeltaNet
/// kernels reduce the same way and are compared bit for bit.
///
/// # Safety
/// Every thread participates in a 1-D block of 32..=1024 threads, a
/// multiple of 32. Dynamic shared memory holds one f32 per warp.
#[device]
pub(crate) unsafe fn block_reduce_sum(mut v: f32) -> f32 {
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
pub(crate) unsafe fn inverse_rms(x: *const f32, width: i32, eps: f32) -> f32 {
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
pub(crate) unsafe fn conv_step_channel(
    xv0: f32,
    weight: *const f32,
    state: *mut f32,
    carry: i32,
) -> f32 {
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

/// fp32 to binary16, round to nearest even.
#[device]
pub(crate) fn f2h(f: f32) -> u16 {
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
pub(crate) fn h2f(h: u16) -> f32 {
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

/// A binary16 block scale stored as two little-endian bytes, widened by
/// the hardware conversion, so it is bit-identical to the `half` crate's
/// `to_f32` on the host. The quantizer emits subnormal scales for
/// near-zero blocks, so that path is exercised.
///
/// # Safety
/// p covers two bytes.
#[device]
pub(crate) unsafe fn load_half_le(p: *const u8) -> f32 {
    unsafe { h2f(*p as u16 | ((*p.add(1) as u16) << 8)) }
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
pub(crate) unsafe fn strided_dot(a: *const f32, b: *const f32, first: i32, len: i32) -> f32 {
    let mut acc = 0.0f32;
    let mut i = first;
    unsafe {
        let (mut pa, mut pb) = (a.add(first as usize), b.add(first as usize));
        while i + 480 < len {
            let a = (
                *pa.add(0),
                *pa.add(32),
                *pa.add(64),
                *pa.add(96),
                *pa.add(128),
                *pa.add(160),
                *pa.add(192),
                *pa.add(224),
                *pa.add(256),
                *pa.add(288),
                *pa.add(320),
                *pa.add(352),
                *pa.add(384),
                *pa.add(416),
                *pa.add(448),
                *pa.add(480),
            );
            let b = (
                *pb.add(0),
                *pb.add(32),
                *pb.add(64),
                *pb.add(96),
                *pb.add(128),
                *pb.add(160),
                *pb.add(192),
                *pb.add(224),
                *pb.add(256),
                *pb.add(288),
                *pb.add(320),
                *pb.add(352),
                *pb.add(384),
                *pb.add(416),
                *pb.add(448),
                *pb.add(480),
            );
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

/// The sum of a value over the warp, by an xor shuffle tree.
#[device]
pub(crate) fn warp_xor_sum(v: f32) -> f32 {
    let mut v = v;
    let mut offset = 16;
    while offset > 0 {
        v += warp::shuffle_xor_f32_sync(u32::MAX, v, offset);
        offset >>= 1;
    }
    v
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
pub(crate) unsafe fn kvq8_quantize(x: f32) -> (u32, u16) {
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
