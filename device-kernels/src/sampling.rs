//! Greedy sampling: block and final argmax, and the argmax with its
//! probability.

use cuda_device::{SharedArray, device, kernel, thread, warp};

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
            let v = (
                *p.add(0),
                *p.add(256),
                *p.add(512),
                *p.add(768),
                *p.add(1024),
                *p.add(1280),
                *p.add(1536),
                *p.add(1792),
            );
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
            let v = (
                *p.add(0),
                *p.add(256),
                *p.add(512),
                *p.add(768),
                *p.add(1024),
                *p.add(1280),
                *p.add(1536),
                *p.add(1792),
            );
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
