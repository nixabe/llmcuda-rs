//! Support kernels for the half and int8 GEMMs: conversions to binary16,
//! the split-K reduce, activation quantization and Q8_0 repacks.

use crate::common::{f2h, load_half_le};
use cuda_device::{kernel, ptx_asm, thread, warp};

/// Sum the split-K partial products of a tensor-core GEMM, ascending
/// split, two adjacent outputs a thread; optionally add into `out`.
///
/// # Safety
/// part covers splits * M * N floats; out covers M rows of stride ldo
/// with N columns used. M * N is even and below 2^32, so the row and
/// column are a 32-bit divide. out is disjoint from part.
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
    let m = i as u32 / n_cols as u32;
    let n = i as u32 - m * n_cols as u32;
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
        let o = out.add((m as i64 * ldo as i64 + n as i64) as usize);
        if accumulate != 0 {
            v0 = *o + v0;
            v1 = *o.add(1) + v1;
        }
        *o = v0;
        *o.add(1) = v1;
    }
}

/// fp32 to fp16, round to nearest even, `n` elements.
///
/// # Safety
/// x covers n floats and y n halves; the 1-D launch is nonempty.
#[kernel]
pub unsafe fn hg_to_half(x: *const f32, y: *mut u16, n: i64) {
    let i =
        thread::blockIdx_x() as i64 * thread::blockDim_x() as i64 + thread::threadIdx_x() as i64;
    if i < n {
        unsafe {
            *y.add(i as usize) = f2h(*x.add(i as usize));
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

/// Quantize fp32 activations to int8 a 32-element block at a time: each
/// block's scale is its largest magnitude over 127, and a code is the
/// element times `127 / amax`, rounded to nearest even and clamped to
/// +-127. An all-zero block gets scale 1 and zero codes.
///
/// Symmetric, absmax, round-to-nearest: llama.cpp's `quantize_q8_1`
/// shape minus the sum term this formulation does not need (no zero
/// point, so no correction term).
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
        let qi = if qi < -127 {
            -127
        } else if qi > 127 {
            127
        } else {
            qi
        };
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
    let b =
        thread::blockIdx_x() as i64 * thread::blockDim_y() as i64 + thread::threadIdx_y() as i64;
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
/// Which width the scales get is a bandwidth decision, not a numeric
/// one. Q8_0 stores an fp16 scale per 32 quants, so the fp32 form widens
/// a value with no more precision to give: 1.125 bytes an element
/// against Q8_0's 1.0625, 5.9% more traffic. That is noise for a shared
/// expert and 1.01 GiB of every decoded token for a 17,408-wide dense
/// FFN. This form keeps the file's own bits, so it is neither a
/// requantization nor a rounding.
///
/// # Safety
/// As `mma_repack_q8_0`, scales covering n_blocks halves.
#[kernel]
pub unsafe fn mma_repack_q8_0_h(src: *const u8, q: *mut i8, scales: *mut u16, n_blocks: i64) {
    let b =
        thread::blockIdx_x() as i64 * thread::blockDim_y() as i64 + thread::threadIdx_y() as i64;
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
