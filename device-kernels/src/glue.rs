//! Small glue kernels for the MTP head and the DFlash tap.

use cuda_device::{kernel, thread};

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
