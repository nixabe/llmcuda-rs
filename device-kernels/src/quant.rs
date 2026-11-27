//! Embedding gathers and whole-tensor dequantization from GGUF storage
//! formats.

use crate::common::load_half_le;
use cuda_device::{SharedArray, kernel, thread};

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
    let i =
        thread::blockIdx_x() as i64 * thread::blockDim_x() as i64 + thread::threadIdx_x() as i64;
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
    let t =
        thread::blockIdx_x() as i64 * thread::blockDim_x() as i64 + thread::threadIdx_x() as i64;
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
                let ql =
                    *sb.add((half * 64 + if grp & 1 != 0 { l + 32 } else { l }) as usize) as i32;
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
                let code = if r < 16 {
                    (q & 0xF) as i32
                } else {
                    (q >> 4) as i32
                };
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
                let mut code = if sub != 0 {
                    (byte >> 4) as i32
                } else {
                    (byte & 0xF) as i32
                };
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
