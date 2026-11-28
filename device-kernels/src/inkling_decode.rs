//! `inkling2_loop` decode glue: what a layer does between its GEMMs when a
//! pass is a handful of rows, each the next position of its own sequence.
//!
//! A block per row does every small step that row needs in turn, so that a
//! layer is a dozen launches rather than sixty. What streams megabytes of
//! weights (the loop controllers, the router) is its own kernel on a grid
//! wide enough to use the whole card: one SM reads at about a seventieth of
//! the bus, and a block a row doing those measured 1.2-1.8 ms a launch.

//! Per-row state (cache and convolution carries) is reached through a
//! pointer table, `[rows][6]`: key cache, value cache, then the key, value,
//! attention-output and feed-forward carries. A cache holds position `j` at
//! row `j & mask`: a ring of `mask + 1` rows on a sliding layer, every
//! position (`mask = i32::MAX`) on a full one.
//!
//! **Windows.** A step may hold several consecutive positions of one
//! sequence, to verify drafted tokens: `woff[t]` is how many rows before
//! row `t` belong to the same sequence. The only state rows share is a
//! convolution's carry, so the three kernels that read one take a tap from
//! an earlier row of the step where that is what the carry would have held
//! by then - the same values in the same order as a step a row - and, since
//! how many rows are kept is decided after the step, leave the carry alone
//! and save each row's raw input for `ink_dec_commit`. `woff` and `save`
//! are both null on an ordinary step.

use crate::common::{block_reduce_sum, f2h, h2f};
use crate::inkling::{block_max_256, block_sum_256};
use cuda_device::{SharedArray, device, kernel, thread, warp};

const MIX_STATES: usize = 8;
const LOWEST: f32 = -3.0e38;

/// Slot of a row's pointer table.
const P_K_CACHE: usize = 0;
const P_V_CACHE: usize = 1;
const P_K_CARRY: usize = 2;
const P_ATTN_CARRY: usize = 4;
const P_FFN_CARRY: usize = 5;

/// Where element `i` goes in byte order: pair `i & 7` of its 16-element
/// group, half `(i & 15) >> 3`.
#[device]
fn byte_order(i: usize) -> usize {
    (i & !15) + ((i & 7) << 1) + ((i & 15) >> 3)
}

/// Two adjacent floats in one load. These kernels wait on their load
/// pipeline, not on arithmetic, and an element a load is twice the loads.
///
/// # Safety
/// p is 8-byte aligned and covers two floats.
#[device]
unsafe fn pair(p: *const f32) -> (f32, f32) {
    let w = unsafe { *(p as *const u64) };
    (f32::from_bits(w as u32), f32::from_bits((w >> 32) as u32))
}

/// One channel's one-row step of a short convolution with a residual, and
/// its carry advanced: `(x + sum_i carry[i] * w[i] + x * w[carry]) * norm`
/// before any input-dependent taps.
///
/// # Safety
/// carry covers `taps - 1` floats owned by this thread and w `taps`.
#[device]
unsafe fn conv_row(x: f32, w: *const f32, carry: *const f32, taps: i32) -> f32 {
    let n = (taps - 1) as usize;
    unsafe {
        let mut sum = 0.0f32;
        let mut i = 0usize;
        while i < n {
            sum += *carry.add(i) * *w.add(i);
            i += 1;
        }
        sum + x * *w.add(n)
    }
}

/// Shift a carry by one and append `x`.
///
/// # Safety
/// carry covers `taps - 1` floats owned by this thread.
#[device]
unsafe fn carry_push(x: f32, carry: *mut f32, taps: i32) {
    let n = (taps - 1) as usize;
    unsafe {
        let mut i = 0usize;
        while i + 1 < n {
            *carry.add(i) = *carry.add(i + 1);
            i += 1;
        }
        *carry.add(n - 1) = x;
    }
}

/// Depth mixing of one row over `states` stacked planes plus `cur`, for
/// `n_queries` queries, into `out[q]` planes. `ink_mix`'s values, its sums
/// taken a warp a state: the block is at least `32 * (states + 1)` threads.
///
/// # Safety
/// Every thread of the block calls this together; stack holds `states`
/// planes `plane` floats apart, each with this row at `row * n`; queries
/// n_queries * n; out n_queries planes `out_plane` apart; states + 1 <= 8
/// and n_queries <= 4; block_reduce_sum's contract holds.
#[device]
unsafe fn mix_row(
    stack: *const f32,
    cur: *const f32,
    queries: *const f32,
    out: *mut f32,
    row: usize,
    n: i32,
    states: i32,
    n_queries: i32,
    plane: i64,
    out_plane: i64,
) {
    static mut INV: SharedArray<f32, MIX_STATES> = SharedArray::UNINIT;
    static mut COEF: SharedArray<f32, { MIX_STATES * 4 }> = SharedArray::UNINIT;
    let tid = thread::threadIdx_x() as i32;
    let step = thread::blockDim_x() as i32;
    let total = states + 1;
    let (warp_id, lane) = (tid >> 5, tid & 31);
    unsafe {
        let base = row * n as usize;
        // A warp a state: its sum of squares and its product with every
        // query in one walk of the row, reduced by shuffle. Taking each of
        // the up to forty sums as a pass of the block with a barrier of its
        // own would be most of this kernel's time.
        if warp_id < total {
            let src = if warp_id < states {
                stack.add((warp_id as i64 * plane) as usize + base)
            } else {
                cur.add(base)
            };
            let nn = n as usize;
            let (mut sq, mut d0, mut d1, mut d2, mut d3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
            let mut i = lane as usize;
            while i < nn {
                let v = *src.add(i);
                sq = v.mul_add(v, sq);
                d0 = v.mul_add(*queries.add(i), d0);
                if n_queries > 1 {
                    d1 = v.mul_add(*queries.add(nn + i), d1);
                }
                if n_queries > 2 {
                    d2 = v.mul_add(*queries.add(2 * nn + i), d2);
                }
                if n_queries > 3 {
                    d3 = v.mul_add(*queries.add(3 * nn + i), d3);
                }
                i += 32;
            }
            let mut offset = 16u32;
            while offset > 0 {
                sq += warp::shuffle_xor_f32_sync(u32::MAX, sq, offset);
                d0 += warp::shuffle_xor_f32_sync(u32::MAX, d0, offset);
                d1 += warp::shuffle_xor_f32_sync(u32::MAX, d1, offset);
                d2 += warp::shuffle_xor_f32_sync(u32::MAX, d2, offset);
                d3 += warp::shuffle_xor_f32_sync(u32::MAX, d3, offset);
                offset >>= 1;
            }
            if lane == 0 {
                let s = warp_id as usize;
                let inv = 1.0 / (sq / n as f32 + 1e-6).sqrt();
                let scale = inv / (n as f32).sqrt();
                INV[s] = inv;
                COEF[s] = d0 * scale;
                COEF[MIX_STATES + s] = d1 * scale;
                COEF[2 * MIX_STATES + s] = d2 * scale;
                COEF[3 * MIX_STATES + s] = d3 * scale;
            }
        }
        thread::sync_threads();
        if tid == 0 {
            let mut q = 0usize;
            while q < n_queries as usize {
                let mut max = COEF[q * MIX_STATES];
                let mut s = 1usize;
                while s < total as usize {
                    max = max.max(COEF[q * MIX_STATES + s]);
                    s += 1;
                }
                let mut sum = 0.0f32;
                s = 0;
                while s < total as usize {
                    let e = (COEF[q * MIX_STATES + s] - max).exp();
                    COEF[q * MIX_STATES + s] = e;
                    sum += e;
                    s += 1;
                }
                s = 0;
                while s < total as usize {
                    COEF[q * MIX_STATES + s] = COEF[q * MIX_STATES + s] / sum * INV[s];
                    s += 1;
                }
                q += 1;
            }
        }
        thread::sync_threads();
        let mut q = 0;
        while q < n_queries {
            let dst = out.add((q as i64 * out_plane) as usize + base);
            let mut i = tid;
            while i < n {
                let mut acc = 0.0f32;
                let mut s = 0;
                while s < total {
                    let v = if s < states {
                        *stack.add((s as i64 * plane) as usize + base + i as usize)
                    } else {
                        *cur.add(base + i as usize)
                    };
                    acc = COEF[q as usize * MIX_STATES + s as usize].mul_add(v, acc);
                    s += 1;
                }
                *dst.add(i as usize) = acc;
                i += step;
            }
            q += 1;
        }
        thread::sync_threads();
    }
}

/// RMS-normalize a row in place with a weight, and write it as binary16,
/// in byte order when `nv != 0`.
///
/// # Safety
/// Every thread of the block calls this together; x, weight and out cover
/// n elements; block_reduce_sum's contract holds.
#[device]
unsafe fn norm_row(x: *mut f32, weight: *const f32, out: *mut u16, n: i32, eps: f32, nv: i32) {
    let tid = thread::threadIdx_x() as i32;
    let step = thread::blockDim_x() as i32;
    unsafe {
        let mut partial = 0.0f32;
        let mut i = tid;
        while i < n {
            let v = *x.add(i as usize);
            partial = v.mul_add(v, partial);
            i += step;
        }
        let inv = 1.0 / (block_reduce_sum(partial) / n as f32 + eps).sqrt();
        let mut i = tid;
        while i < n {
            let v = *x.add(i as usize) * inv * *weight.add(i as usize);
            *x.add(i as usize) = v;
            let at = if nv != 0 {
                byte_order(i as usize)
            } else {
                i as usize
            };
            *out.add(at) = f2h(v);
            i += step;
        }
        thread::sync_threads();
    }
}

/// The inverse RMS of a row of `x` and of `msg`, written to `inv[0..2]`.
///
/// # Safety
/// Every thread of the block calls this together; x and msg cover n floats;
/// block_reduce_sum's contract holds.
#[device]
unsafe fn pair_inv(x: *const f32, msg: *const f32, inv: *mut f32, n: i32) {
    let tid = thread::threadIdx_x() as i32;
    let step = thread::blockDim_x() as i32;
    unsafe {
        let mut sq = 0.0f32;
        let mut mq = 0.0f32;
        let mut i = tid;
        while i < n {
            let a = *x.add(i as usize);
            let b = *msg.add(i as usize);
            sq = a.mul_add(a, sq);
            mq = b.mul_add(b, mq);
            i += step;
        }
        let inv_x = 1.0 / (block_reduce_sum(sq) / n as f32 + 1e-6).sqrt();
        let inv_m = 1.0 / (block_reduce_sum(mq) / n as f32 + 1e-6).sqrt();
        if tid == 0 {
            *inv = inv_x;
            *inv.add(1) = inv_m;
        }
    }
}

/// A layer's attention inputs for every row: depth mixing over the
/// residual blocks, then either the input norms (binary16 planes
/// `[3][rows][n]`) or, when loop messages are given, the three inputs laid
/// out for the controllers.
///
/// Without `queries` the row itself is the one input; with them a sliding
/// layer mixes one and a full layer three. Without `msgs` each mixed plane
/// j is normalized with weight `min(j, n_norms - 1)` into `out`. With
/// `msgs` (`[3][rows][n]`, planes Q, K, V) all three planes of `work` are
/// filled and `inv[row][3][2]` gets the inverse RMS of each plane and its
/// message; `ink_dec_gate`, `ink_dec_ctrl` and `ink_dec_norm` finish.
///
/// grid: (rows), 256 threads, one f32 of dynamic shared memory per warp.
///
/// # Safety
/// The buffers cover the shapes above; work is `[3][rows][n]`; states + 1
/// <= 8.
#[kernel]
pub unsafe fn ink_dec_inputs(
    cur: *const f32,
    stack: *const f32,
    queries: *const f32,
    msgs: *const f32,
    norms: *const f32,
    work: *mut f32,
    out: *mut u16,
    inv: *mut f32,
    n: i32,
    states: i32,
    n_queries: i32,
    n_norms: i32,
    eps: f32,
) {
    let tid = thread::threadIdx_x() as i32;
    let step = thread::blockDim_x() as i32;
    let t = thread::blockIdx_x() as usize;
    let rows = thread::gridDim_x() as i64;
    let plane = rows * n as i64;
    let nu = n as usize;
    unsafe {
        let mixed = if queries.is_null() {
            let mut i = tid;
            while i < n {
                *work.add(t * nu + i as usize) = *cur.add(t * nu + i as usize);
                i += step;
            }
            thread::sync_threads();
            1
        } else {
            mix_row(stack, cur, queries, work, t, n, states, n_queries, plane, plane);
            n_queries
        };
        if msgs.is_null() {
            let mut j = 0;
            while j < mixed {
                let w = if j < n_norms { j } else { n_norms - 1 };
                norm_row(
                    work.add((j as i64 * plane) as usize + t * nu),
                    norms.add(w as usize * nu),
                    out.add((j as i64 * plane) as usize + t * nu),
                    n,
                    eps,
                    0,
                );
                j += 1;
            }
        } else {
            // Descending, so the plane the others copy is read before it
            // could be anything else.
            let mut j = 2;
            while j >= 0 {
                let dst = work.add((j as i64 * plane) as usize + t * nu);
                let src_plane = if j < mixed { j } else { mixed - 1 };
                if src_plane != j {
                    let src = work.add((src_plane as i64 * plane) as usize + t * nu);
                    let mut i = tid;
                    while i < n {
                        *dst.add(i as usize) = *src.add(i as usize);
                        i += step;
                    }
                    thread::sync_threads();
                }
                pair_inv(
                    dst,
                    msgs.add((j as i64 * plane) as usize + t * nu),
                    inv.add((t * 3 + j as usize) * 2),
                    n,
                );
                j -= 1;
            }
        }
    }
}

/// One gate value of a loop controller: row `r` of `down` against the
/// RMS-normalized pair `[x, msg]`.
///
/// One block a gate row, so the megabytes of `down` are streamed by as many
/// SMs as there are rows; a block a *sequence* row reading every gate row
/// is held to one SM's share of the bus.
///
/// grid: (rank, rows * ctrls), 256 threads, dynamic shared memory for the
/// reduction. Block y is `row * ctrls + controller`.
///
/// # Safety
/// x and msg are `[ctrls][rows][n]`; inv `[rows][ctrls][2]`; down
/// `[ctrls][rank][2n]`; gate `[rows][ctrls][rank]`; n is even.
#[kernel]
pub unsafe fn ink_dec_gate(
    x: *const f32,
    msg: *const f32,
    inv: *const f32,
    down: *const f32,
    gate: *mut f32,
    n: i32,
    rank: i32,
    ctrls: i32,
) {
    let tid = thread::threadIdx_x() as i32;
    let step = thread::blockDim_x() as i32;
    let r = thread::blockIdx_x() as usize;
    let y = thread::blockIdx_y() as usize;
    let j = y % ctrls as usize;
    let t = y / ctrls as usize;
    let rows = thread::gridDim_y() as usize / ctrls as usize;
    let nu = n as usize;
    unsafe {
        let at = (j * rows + t) * nu;
        let inv_x = *inv.add(y * 2);
        let inv_m = *inv.add(y * 2 + 1);
        let row = down.add((j * rank as usize + r) * 2 * nu);
        // A thread takes pairs 2 * tid, 2 * tid + 2 * step, ..: n is even,
        // so a pair is on one side of the [x, msg] seam.
        let mut acc = 0.0f32;
        let mut i = 2 * tid;
        while i < 2 * n {
            let (v0, v1, s) = if i < n {
                let (a, b) = pair(x.add(at + i as usize));
                (a, b, inv_x)
            } else {
                let (a, b) = pair(msg.add(at + (i - n) as usize));
                (a, b, inv_m)
            };
            let (r0, r1) = pair(row.add(i as usize));
            acc = r0.mul_add(v0 * s, acc);
            acc = r1.mul_add(v1 * s, acc);
            i += 2 * step;
        }
        let sum = block_reduce_sum(acc);
        if tid == 0 {
            *gate.add(y * rank as usize + r) = sum;
        }
    }
}

/// The loop controller applied in place:
/// `x = x - tanh(forget . gate) * x + tanh(gain . gate) * msg`.
///
/// grid: (ceil(n / 256), rows * ctrls), 256 threads; a thread a channel.
///
/// # Safety
/// As `ink_dec_gate`; gain and forget are `[ctrls][n][rank]`; rank <= 64.
#[kernel]
pub unsafe fn ink_dec_ctrl(
    x: *mut f32,
    msg: *const f32,
    gate: *const f32,
    gain: *const f32,
    forget: *const f32,
    n: i32,
    rank: i32,
    ctrls: i32,
) {
    static mut GATE: SharedArray<f32, 64> = SharedArray::UNINIT;
    let tid = thread::threadIdx_x() as usize;
    let i = thread::blockIdx_x() as usize * thread::blockDim_x() as usize + tid;
    let y = thread::blockIdx_y() as usize;
    let j = y % ctrls as usize;
    let t = y / ctrls as usize;
    let rows = thread::gridDim_y() as usize / ctrls as usize;
    let nu = n as usize;
    let ru = rank as usize;
    unsafe {
        if tid < ru {
            GATE[tid] = *gate.add(y * ru + tid);
        }
        thread::sync_threads();
        if i < nu {
            let gr = gain.add((j * nu + i) * ru);
            let fr = forget.add((j * nu + i) * ru);
            let mut g = 0.0f32;
            let mut f = 0.0f32;
            let mut k = 0usize;
            while ru & 1 == 0 && k + 1 < ru {
                let (g0, g1) = pair(gr.add(k));
                let (f0, f1) = pair(fr.add(k));
                let (a0, a1) = (GATE[k], GATE[k + 1]);
                g = g0.mul_add(a0, g);
                f = f0.mul_add(a0, f);
                g = g1.mul_add(a1, g);
                f = f1.mul_add(a1, f);
                k += 2;
            }
            if k < ru {
                g = (*gr.add(k)).mul_add(GATE[k], g);
                f = (*fr.add(k)).mul_add(GATE[k], f);
            }
            let at = (j * rows + t) * nu + i;
            let a = *x.add(at);
            *x.add(at) = a - f.tanh() * a + g.tanh() * *msg.add(at);
        }
    }
}

/// RMS-normalize planes of rows in place and write them as binary16 (in
/// byte order when `nv != 0`); plane j takes weight `min(j, n_norms - 1)`.
///
/// grid: (planes, rows), 256 threads, dynamic shared memory for the
/// reduction.
///
/// # Safety
/// x and out are `[planes][rows][n]`; weights `[n_norms][n]`.
#[kernel]
pub unsafe fn ink_dec_norm(
    x: *mut f32,
    weights: *const f32,
    out: *mut u16,
    n: i32,
    n_norms: i32,
    eps: f32,
    nv: i32,
) {
    let j = thread::blockIdx_x() as usize;
    let t = thread::blockIdx_y() as usize;
    let rows = thread::gridDim_y() as usize;
    let nu = n as usize;
    let w = if (j as i32) < n_norms { j } else { n_norms as usize - 1 };
    unsafe {
        let at = (j * rows + t) * nu;
        norm_row(x.add(at), weights.add(w * nu), out.add(at), n, eps, nv);
    }
}

/// Keys and values of every row: the short convolution over the row's
/// carry (with input-dependent taps when `taps_off[which] >= 0`), the key
/// clip or the value norm, and the append to the row's cache at its
/// position.
///
/// `proj` is the projection pack's output, `ldo` floats a row; the key
/// projection starts at `k_off`, the value's at `v_off`, their taps at
/// `k_taps` / `v_taps` (negative: none). grid: (2 * kv_heads, rows), `dim`
/// threads: block x is the head, keys first.
///
/// # Safety
/// The pointer table and positions cover gridDim.y rows and name caches
/// with room at each position; the convolution weights cover kv_heads *
/// dim * taps; dim is a multiple of 32 and at most 128.
#[kernel]
pub unsafe fn ink_dec_kv(
    proj: *const f32,
    ptrs: *const u64,
    pos: *const i32,
    k_conv: *const f32,
    v_conv: *const f32,
    ldo: i32,
    k_off: i32,
    v_off: i32,
    k_taps: i32,
    v_taps: i32,
    kv_heads: i32,
    taps: i32,
    clip: f32,
    eps: f32,
    mask: i32,
    woff: *const i32,
    save: *mut f32,
    save_width: i32,
) {
    static mut RED: SharedArray<f32, 4> = SharedArray::UNINIT;
    let tid = thread::threadIdx_x() as usize;
    let dim = thread::blockDim_x() as usize;
    let which = thread::blockIdx_x() as usize / kv_heads as usize;
    let head = thread::blockIdx_x() as usize % kv_heads as usize;
    let t = thread::blockIdx_y() as usize;
    let c = head * dim + tid;
    unsafe {
        let row = proj.add(t * ldo as usize);
        let (off, tap_off, weight) = if which == 0 {
            (k_off, k_taps, k_conv)
        } else {
            (v_off, v_taps, v_conv)
        };
        let x = *row.add(off as usize + c);
        let carry = (*ptrs.add(t * 6 + P_K_CARRY + which) as *mut f32).add(c * (taps as usize - 1));
        let r = if woff.is_null() { 0 } else { *woff.add(t) as usize };
        let n = taps as usize - 1;
        let w = weight.add(c * taps as usize);
        let mut span = taps;
        let mut sum;
        if r == 0 {
            sum = conv_row(x, w, carry, taps);
            if tap_off >= 0 {
                let tap = row.add(tap_off as usize + head * taps as usize);
                sum += x * *tap;
                let mut lag = 1;
                while lag < taps {
                    sum += *carry.add((taps - 1 - lag) as usize) * *tap.add(lag as usize);
                    lag += 1;
                }
                span = 2 * taps;
            }
        } else {
            // Carry element i as a step a row would find it: the carry
            // shifted by the r rows before this one, their inputs behind it.
            sum = 0.0;
            let mut i = 0usize;
            while i < n {
                let held = if i + r < n {
                    *carry.add(i + r)
                } else {
                    *proj.add((t + i - n) * ldo as usize + off as usize + c)
                };
                sum += held * *w.add(i);
                i += 1;
            }
            sum = sum + x * *w.add(n);
            if tap_off >= 0 {
                let tap = row.add(tap_off as usize + head * taps as usize);
                sum += x * *tap;
                let mut lag = 1usize;
                while lag <= n {
                    let i = n - lag;
                    let held = if i + r < n {
                        *carry.add(i + r)
                    } else {
                        *proj.add((t + i - n) * ldo as usize + off as usize + c)
                    };
                    sum += held * *tap.add(lag);
                    lag += 1;
                }
                span = 2 * taps;
            }
        }
        let v = (x + sum) / ((span + 1) as f32).sqrt();
        if save.is_null() {
            carry_push(x, carry, taps);
        } else {
            *save.add(t * save_width as usize + which * kv_heads as usize * dim + c) = x;
        }

        let mut sq = v * v;
        let mut off = 16;
        while off > 0 {
            sq += warp::shuffle_xor_f32_sync(u32::MAX, sq, off);
            off >>= 1;
        }
        if tid & 31 == 0 {
            RED[tid >> 5] = sq;
        }
        thread::sync_threads();
        let mut total = 0.0f32;
        let mut w = 0usize;
        while w < dim / 32 {
            total += RED[w];
            w += 1;
        }
        let factor = if which == 0 {
            (clip / (total.sqrt() + 1e-6)).min(1.0)
        } else {
            1.0 / (total / dim as f32 + eps).sqrt()
        };
        let cache = *ptrs.add(t * 6 + P_K_CACHE + which) as *mut u16;
        let at = (*pos.add(t) & mask) as usize * kv_heads as usize * dim + c;
        *cache.add(at) = f2h(v * factor);
    }
}

/// One query head of one row over a share of the row's cache: the query
/// and relative-position clips (with the logarithmic position scale when
/// `alpha >= 0`) and attention with the bias computed per distance, as an
/// unnormalized partial for `ink_dec_attn_merge`:
/// `part[row][head][z] = [max logit, sum exp(logit - max), dim sums of
/// exp(logit - max) * value]`.
///
/// Block z of a (head, row) takes the 256-position chunks z, z + Z, z + 2Z
/// .. of the visible span, Z = gridDim.z, and one with no chunk writes the
/// empty partial (lowest max, zero sums). A chunk is a serial chain of
/// loads and barriers, so one block walking a 1,024-row window is four of
/// them in a row on one SM while the rest of the card idles: 90 us a
/// launch at three rows, and barely more than at one.
///
/// grid: (heads, rows, Z), 256 threads.
///
/// # Safety
/// As `ink_attn` and `ink_clip`; `proj` rows hold the query and relative
/// projections at their offsets; dim is a multiple of 4 and at most 128,
/// n_rel <= 32; part is `[rows][heads][Z][dim + 2]`.
#[kernel]
pub unsafe fn ink_dec_attn(
    proj: *const f32,
    ptrs: *const u64,
    pos: *const i32,
    q_clip: *const f32,
    rel_clip: *const f32,
    rel_bias: *const f32,
    part: *mut f32,
    ldo: i32,
    q_off: i32,
    rel_off: i32,
    sink_off: i32,
    kv_heads: i32,
    dim: i32,
    n_rel: i32,
    window: i32,
    extent: i32,
    qk_clip: f32,
    rel_bound: f32,
    alpha: f32,
    floor: f32,
    mask: i32,
) {
    static mut Q: SharedArray<f32, 128> = SharedArray::UNINIT;
    static mut R: SharedArray<f32, 32> = SharedArray::UNINIT;
    static mut P: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut ACC: SharedArray<f32, 128> = SharedArray::UNINIT;
    static mut PV: SharedArray<f32, 1024> = SharedArray::UNINIT;
    static mut RUN: SharedArray<f32, 2> = SharedArray::UNINIT;
    let tid = thread::threadIdx_x() as i32;
    let h = thread::blockIdx_x() as usize;
    let t = thread::blockIdx_y() as usize;
    let heads = thread::gridDim_x() as usize;
    let z = thread::blockIdx_z() as usize;
    let nz = thread::gridDim_z() as usize;
    let d = dim as usize;
    let kv = h / (heads / kv_heads as usize);
    let stride = kv_heads as usize * d;
    unsafe {
        let row = proj.add(t * ldo as usize);
        let p = *pos.add(t);
        let lo = if window > 0 && p + 1 > window {
            p + 1 - window
        } else {
            0
        };
        let result = part.add(((t * heads + h) * nz + z) * (d + 2));
        let first = lo + 256 * z as i32;
        if first > p {
            if (tid as usize) < d + 2 {
                *result.add(tid as usize) = if tid == 0 { LOWEST } else { 0.0 };
            }
            return;
        }
        let tau = if alpha >= 0.0 {
            1.0 + alpha * (((p + 1) as f32) / floor).max(1.0).ln()
        } else {
            1.0
        };
        let mut qv = 0.0f32;
        if (tid as usize) < d {
            qv = *row.add(q_off as usize + h * d + tid as usize) * *q_clip.add(h * d + tid as usize);
        }
        let q_norm = block_sum_256(qv * qv).sqrt();
        let mut rv = 0.0f32;
        if tid < n_rel {
            rv = *row.add(rel_off as usize + h * n_rel as usize + tid as usize)
                * *rel_clip.add(h * n_rel as usize + tid as usize);
        }
        let r_norm = block_sum_256(rv * rv).sqrt();
        if (tid as usize) < d {
            Q[tid as usize] = qv * ((qk_clip / (q_norm + 1e-6)).min(1.0) * tau);
            ACC[tid as usize] = 0.0;
        }
        if tid < n_rel {
            R[tid as usize] = rv * ((rel_bound / (r_norm + 1e-6)).min(1.0) * tau);
        }
        if tid == 0 {
            RUN[0] = LOWEST;
            RUN[1] = 0.0;
        }
        thread::sync_threads();

        let sink = *row.add(sink_off as usize + h);
        let scale = 1.0 / (dim as f32).sqrt();
        let k_cache = *ptrs.add(t * 6 + P_K_CACHE) as *const u16;
        let v_cache = *ptrs.add(t * 6 + P_V_CACHE) as *const u16;
        // A lane keeps four channels of the query and one feature of the
        // relative vector in registers.
        let lane = (tid & 31) as usize;
        let wp = tid >> 5;
        let q4i = lane * 4;
        let mut q0 = 0.0f32;
        let mut q1 = 0.0f32;
        let mut q2 = 0.0f32;
        let mut q3 = 0.0f32;
        if q4i < d {
            q0 = Q[q4i];
            q1 = Q[q4i + 1];
            q2 = Q[q4i + 2];
            q3 = Q[q4i + 3];
        }
        let rl = if (lane as i32) < n_rel { R[lane] } else { 0.0 };
        let mut j0 = first;
        while j0 <= p {
            // Logits. Warp w takes the keys j = w mod 8 of the chunk, a
            // lane four channels of each: one 8-byte load a lane a key, the
            // lane's share of the bias beside it, and the warp's sum by
            // shuffle. A thread a key walking all 128 channels reads the
            // query back from shared memory once a channel, and it is the
            // load pipeline this kernel waits on.
            let mut warp_max = LOWEST;
            let mut i = 0i32;
            while i < 32 {
                let jj = wp + 8 * i;
                let j = j0 + jj;
                let mut s = LOWEST;
                if j <= p {
                    let mut v = 0.0f32;
                    if q4i < d {
                        let w = *(k_cache.add((j & mask) as usize * stride + kv * d + q4i) as *const u64);
                        let mut a0 = q0 * h2f(w as u16);
                        let mut a1 = q1 * h2f((w >> 16) as u16);
                        a0 = q2.mul_add(h2f((w >> 32) as u16), a0);
                        a1 = q3.mul_add(h2f((w >> 48) as u16), a1);
                        v = (a0 + a1) * scale;
                    }
                    let dist = p - j;
                    if dist < extent && (lane as i32) < n_rel {
                        v = (*rel_bias.add(dist as usize * n_rel as usize + lane)).mul_add(rl, v);
                    }
                    let mut off = 16;
                    while off > 0 {
                        v += warp::shuffle_xor_f32_sync(u32::MAX, v, off);
                        off >>= 1;
                    }
                    s = v - sink;
                }
                if lane == 0 {
                    P[jj as usize] = s;
                }
                warp_max = warp_max.max(s);
                i += 1;
            }
            let chunk_max = block_max_256(warp_max);
            let j = j0 + tid;
            let s = P[tid as usize];

            let old = RUN[0];
            let top = old.max(chunk_max);
            let rescale = (old - top).exp();
            let pj = if j <= p { (s - top).exp() } else { 0.0 };
            P[tid as usize] = pj;
            let chunk_sum = block_sum_256(pj);
            // Values: thread (class c, quad q) sums four channels over the
            // keys j = c mod 8, so every thread reads and a read is four
            // channels wide; the eight classes are added in order after.
            let class = tid >> 5;
            let q4 = ((tid & 31) * 4) as usize;
            if q4 < d {
                let live = (p - j0 + 1).min(256);
                let mut v0 = 0.0f32;
                let mut v1 = 0.0f32;
                let mut v2 = 0.0f32;
                let mut v3 = 0.0f32;
                let mut jj = class;
                while jj < live {
                    let w = *(v_cache.add(((j0 + jj) & mask) as usize * stride + kv * d + q4)
                        as *const u64);
                    let pj = P[jj as usize];
                    v0 = pj.mul_add(h2f(w as u16), v0);
                    v1 = pj.mul_add(h2f((w >> 16) as u16), v1);
                    v2 = pj.mul_add(h2f((w >> 32) as u16), v2);
                    v3 = pj.mul_add(h2f((w >> 48) as u16), v3);
                    jj += 8;
                }
                let at = class as usize * 128 + q4;
                PV[at] = v0;
                PV[at + 1] = v1;
                PV[at + 2] = v2;
                PV[at + 3] = v3;
            }
            thread::sync_threads();
            if (tid as usize) < d {
                let mut acc = ACC[tid as usize] * rescale;
                let mut c = 0usize;
                while c < 8 {
                    acc += PV[c * 128 + tid as usize];
                    c += 1;
                }
                ACC[tid as usize] = acc;
            }
            if tid == 0 {
                RUN[0] = top;
                RUN[1] = RUN[1] * rescale + chunk_sum;
            }
            thread::sync_threads();
            j0 += 256 * nz as i32;
        }
        if (tid as usize) < d {
            *result.add(2 + tid as usize) = ACC[tid as usize];
        }
        if tid < 2 {
            *result.add(tid as usize) = RUN[tid as usize];
        }
    }
}

/// The partials of `ink_dec_attn` joined: the zero-logit sink (weight one
/// at logit zero, already folded into the partials' logits), the softmax
/// normalization and the output gate. Partials are added in z order. The
/// result is binary16, `[rows][heads][dim]`.
///
/// grid: (heads, rows), `dim` threads.
///
/// # Safety
/// part is `[rows][heads][nz][dim + 2]`; `proj` rows hold the gate
/// projection at its offset; out is `[rows][heads][dim]`.
#[kernel]
pub unsafe fn ink_dec_attn_merge(
    part: *const f32,
    proj: *const f32,
    out: *mut u16,
    ldo: i32,
    gate_off: i32,
    dim: i32,
    nz: i32,
) {
    let tid = thread::threadIdx_x() as usize;
    let h = thread::blockIdx_x() as usize;
    let t = thread::blockIdx_y() as usize;
    let heads = thread::gridDim_x() as usize;
    let d = dim as usize;
    if tid >= d {
        return;
    }
    unsafe {
        let base = part.add((t * heads + h) * nz as usize * (d + 2));
        let mut top = 0.0f32;
        let mut z = 0usize;
        while z < nz as usize {
            top = top.max(*base.add(z * (d + 2)));
            z += 1;
        }
        let mut total = (-top).exp();
        let mut acc = 0.0f32;
        let mut z = 0usize;
        while z < nz as usize {
            let b = base.add(z * (d + 2));
            let w = (*b - top).exp();
            total = (*b.add(1)).mul_add(w, total);
            acc = (*b.add(2 + tid)).mul_add(w, acc);
            z += 1;
        }
        let g = *proj.add(t * ldo as usize + gate_off as usize + h * d + tid);
        *out.add((t * heads + h) * d + tid) = f2h(acc / total / (1.0 + (-g).exp()));
    }
}

/// From the attention output projection to the feed-forward input, for
/// every row: the output convolution and residual (or the start of a new
/// residual block, whose input joins the stack), depth mixing into `work`,
/// then either the norm (in place, and binary16 into `out`, in byte order
/// when `nv != 0`) or, when a loop message is given, `inv[row][2]` for the
/// controller kernels to finish.
///
/// grid: (rows), 256 threads, one f32 of dynamic shared memory per warp.
///
/// # Safety
/// Row buffers cover gridDim.x rows of n; stack has room for states + 1
/// planes.
#[kernel]
pub unsafe fn ink_dec_ffn_in(
    o_raw: *const f32,
    input: *const f32,
    conv_w: *const f32,
    ptrs: *const u64,
    stack: *mut f32,
    query: *const f32,
    msg: *const f32,
    norm_w: *const f32,
    ffn_inp: *mut f32,
    work: *mut f32,
    out: *mut u16,
    inv: *mut f32,
    n: i32,
    states: i32,
    opens: i32,
    taps: i32,
    nv: i32,
    eps: f32,
    woff: *const i32,
    save: *mut f32,
    save_width: i32,
    save_off: i32,
) {
    let tid = thread::threadIdx_x() as i32;
    let step = thread::blockDim_x() as i32;
    let t = thread::blockIdx_x() as usize;
    let rows = thread::gridDim_x() as i64;
    let plane = rows * n as i64;
    let nu = n as usize;
    unsafe {
        let carry = *ptrs.add(t * 6 + P_ATTN_CARRY) as *mut f32;
        let norm = 1.0 / ((taps + 1) as f32).sqrt();
        let r = if woff.is_null() { 0 } else { *woff.add(t) as usize };
        let held = (taps - 1) as usize;
        let mut i = tid;
        while i < n {
            let iu = i as usize;
            let x = *o_raw.add(t * nu + iu);
            let c = carry.add(iu * (taps as usize - 1));
            let w = conv_w.add(iu * taps as usize);
            let conv = if r == 0 {
                conv_row(x, w, c, taps)
            } else {
                let mut sum = 0.0f32;
                let mut k = 0usize;
                while k < held {
                    let v = if k + r < held {
                        *c.add(k + r)
                    } else {
                        *o_raw.add((t + k - held) * nu + iu)
                    };
                    sum += v * *w.add(k);
                    k += 1;
                }
                sum + x * *w.add(held)
            };
            let a = (x + conv) * norm;
            if save.is_null() {
                carry_push(x, c, taps);
            } else {
                *save.add(t * save_width as usize + save_off as usize + iu) = x;
            }
            let inp = *input.add(t * nu + iu);
            if opens != 0 {
                *stack.add((states as i64 * plane) as usize + t * nu + iu) = inp;
                *ffn_inp.add(t * nu + iu) = a;
            } else {
                *ffn_inp.add(t * nu + iu) = inp + a;
            }
            i += step;
        }
        thread::sync_threads();
        let live = states + opens;
        mix_row(stack, ffn_inp, query, work, t, n, live, 1, plane, plane);
        let x = work.add(t * nu);
        if msg.is_null() {
            norm_row(x, norm_w, out.add(t * nu), n, eps, nv);
        } else {
            pair_inv(x, msg.add(t * nu), inv.add(t * 2), n);
        }
    }
}

/// Router logits, fp32: `logits[row][e] = router[e] . x[row]`.
///
/// grid: (experts / 8, rows), 256 threads: a block owns eight experts, a
/// thread one expert over one of 32 spans of the row.
///
/// # Safety
/// x is `[rows][n]`, router `[experts][n]`, logits `[rows][experts]`;
/// experts is a multiple of 8.
#[kernel]
pub unsafe fn ink_dec_router(x: *const f32, router: *const f32, logits: *mut f32, n: i32, experts: i32) {
    static mut PART: SharedArray<f32, 256> = SharedArray::UNINIT;
    let tid = thread::threadIdx_x() as usize;
    let t = thread::blockIdx_y() as usize;
    let e = thread::blockIdx_x() as usize * 8 + (tid & 7);
    let part = tid >> 3;
    let nu = n as usize;
    let span = nu / 32;
    let first = part * span;
    let last = if part == 31 { nu } else { first + span };
    unsafe {
        let wr = router.add(e * nu);
        let xr = x.add(t * nu);
        let mut acc = 0.0f32;
        let mut i = first;
        if (first | nu) & 1 == 0 {
            while i + 1 < last {
                let (w0, w1) = pair(wr.add(i));
                let (x0, x1) = pair(xr.add(i));
                acc = w0.mul_add(x0, acc);
                acc = w1.mul_add(x1, acc);
                i += 2;
            }
        }
        while i < last {
            acc = (*wr.add(i)).mul_add(*xr.add(i), acc);
            i += 1;
        }
        PART[tid] = acc;
        thread::sync_threads();
        if tid < 8 {
            let mut logit = 0.0f32;
            let mut p = 0usize;
            while p < 32 {
                logit += PART[tid + p * 8];
                p += 1;
            }
            *logits.add(t * experts as usize + e) = logit;
        }
    }
}

/// Sigmoid top-k routing of every row from its logits, with normalized,
/// scaled weights. Selection adds `bias`; the lower expert wins a tie.
///
/// grid: (rows), 256 threads.
///
/// # Safety
/// logits `[rows][experts]`, bias `[experts]`, ids and weights
/// `[rows][topk]`; experts <= 256, topk <= 32.
#[kernel]
pub unsafe fn ink_dec_route(
    logits: *const f32,
    bias: *const f32,
    ids: *mut i32,
    weights: *mut f32,
    experts: i32,
    topk: i32,
    weight_scale: f32,
) {
    static mut PROB: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut BV: SharedArray<f32, 8> = SharedArray::UNINIT;
    static mut BI: SharedArray<i32, 8> = SharedArray::UNINIT;
    static mut PICK: SharedArray<i32, 1> = SharedArray::UNINIT;
    static mut CHOSEN: SharedArray<f32, 32> = SharedArray::UNINIT;
    let tid = thread::threadIdx_x() as i32;
    let t = thread::blockIdx_x() as usize;
    unsafe {
        let mut sel = LOWEST;
        if tid < experts {
            let prob = 1.0 / (1.0 + (-*logits.add(t * experts as usize + tid as usize)).exp());
            PROB[tid as usize] = prob;
            sel = prob + *bias.add(tid as usize);
        }
        thread::sync_threads();
        let mut round = 0;
        while round < topk {
            let mut bv = sel;
            let mut bi = tid;
            let mut off = 16;
            while off > 0 {
                let ov = warp::shuffle_xor_f32_sync(u32::MAX, bv, off);
                let oi = warp::shuffle_xor_sync(u32::MAX, bi as u32, off) as i32;
                if ov > bv || (ov == bv && oi < bi) {
                    bv = ov;
                    bi = oi;
                }
                off >>= 1;
            }
            if tid & 31 == 0 {
                BV[(tid >> 5) as usize] = bv;
                BI[(tid >> 5) as usize] = bi;
            }
            thread::sync_threads();
            if tid == 0 {
                let mut w = 1usize;
                while w < 8 {
                    if BV[w] > bv || (BV[w] == bv && BI[w] < bi) {
                        bv = BV[w];
                        bi = BI[w];
                    }
                    w += 1;
                }
                PICK[0] = bi;
                *ids.add(t * topk as usize + round as usize) = bi;
                CHOSEN[round as usize] = PROB[bi as usize];
            }
            thread::sync_threads();
            if tid == PICK[0] {
                sel = LOWEST;
            }
            thread::sync_threads();
            round += 1;
        }
        if tid == 0 {
            let mut sum = 0.0f32;
            let mut j = 0usize;
            while j < topk as usize {
                sum += CHOSEN[j];
                j += 1;
            }
            j = 0;
            while j < topk as usize {
                *weights.add(t * topk as usize + j) = CHOSEN[j] / (sum + 1e-20) * weight_scale;
                j += 1;
            }
        }
    }
}

/// The end of a layer for every row: the weighted expert outputs summed
/// in pair order, the feed-forward convolution over the row's carry, and
/// the residual.
///
/// grid: (ceil(n / block), rows).
///
/// # Safety
/// partial covers gridDim.y * topk rows of n floats, already scaled by
/// their router weights; ffn_inp and h gridDim.y rows of n; h may alias
/// neither.
#[kernel]
pub unsafe fn ink_dec_ffn_out(
    partial: *const f32,
    ffn_inp: *const f32,
    conv_w: *const f32,
    ptrs: *const u64,
    h: *mut f32,
    n: i32,
    topk: i32,
    taps: i32,
    woff: *const i32,
    save: *mut f32,
    save_width: i32,
    save_off: i32,
) {
    let i = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
    let t = thread::blockIdx_y() as usize;
    let nu = n as usize;
    if i >= nu {
        return;
    }
    unsafe {
        let mut x = 0.0f32;
        let mut j = 0usize;
        while j < topk as usize {
            x += *partial.add((t * topk as usize + j) * nu + i);
            j += 1;
        }
        let carry = (*ptrs.add(t * 6 + P_FFN_CARRY) as *mut f32).add(i * (taps as usize - 1));
        let r = if woff.is_null() { 0 } else { *woff.add(t) as usize };
        let w = conv_w.add(i * taps as usize);
        let conv = if r == 0 {
            conv_row(x, w, carry, taps)
        } else {
            // An earlier row's input is its own sum of the same pairs.
            let held = (taps - 1) as usize;
            let mut sum = 0.0f32;
            let mut k = 0usize;
            while k < held {
                let value = if k + r < held {
                    *carry.add(k + r)
                } else {
                    let row = t + k - held;
                    let mut earlier = 0.0f32;
                    let mut j = 0usize;
                    while j < topk as usize {
                        earlier += *partial.add((row * topk as usize + j) * nu + i);
                        j += 1;
                    }
                    earlier
                };
                sum += value * *w.add(k);
                k += 1;
            }
            sum + x * *w.add(held)
        };
        let v = (x + conv) / ((taps + 1) as f32).sqrt();
        if save.is_null() {
            carry_push(x, carry, taps);
        } else {
            *save.add(t * save_width as usize + save_off as usize + i) = x;
        }
        *h.add(t * nu + i) = *ffn_inp.add(t * nu + i) + v;
    }
}

/// The carries of a window's sequence advanced by the rows that were kept:
/// each of a slot's four carries takes the saved inputs of rows
/// `t .. t + keep[t]` in order, as that many steps a row would have pushed
/// them. Rows with `keep[t] == 0` are not the first of a window, or are of
/// a sequence that keeps nothing.
///
/// grid: (slots, rows), 256 threads. `ptrs` is `[slots][rows][6]` and
/// `save` `[slots][rows][width]`, a row's fields being its key, value,
/// attention-output and feed-forward inputs (`kv`, `kv`, `n`, `n` wide).
///
/// # Safety
/// The tables cover the grid; `keep[t]` rows from `t` are one sequence's.
#[kernel]
pub unsafe fn ink_dec_commit(
    ptrs: *const u64,
    save: *const f32,
    keep: *const i32,
    kv: i32,
    n: i32,
    taps: i32,
) {
    let tid = thread::threadIdx_x() as usize;
    let step = thread::blockDim_x() as usize;
    let slot = thread::blockIdx_x() as usize;
    let t = thread::blockIdx_y() as usize;
    let rows = thread::gridDim_y() as usize;
    let width = 2 * (kv + n) as usize;
    unsafe {
        let kept = *keep.add(t) as usize;
        if kept == 0 {
            return;
        }
        let table = ptrs.add((slot * rows + t) * 6 + P_K_CARRY);
        let mut c = tid;
        while c < width {
            // Which carry this channel is in, and where in it.
            let (field, at) = if c < kv as usize {
                (0usize, c)
            } else if c < 2 * kv as usize {
                (1usize, c - kv as usize)
            } else if c < (2 * kv + n) as usize {
                (2usize, c - 2 * kv as usize)
            } else {
                (3usize, c - (2 * kv + n) as usize)
            };
            let carry = (*table.add(field) as *mut f32).add(at * (taps as usize - 1));
            let mut j = 0usize;
            while j < kept {
                carry_push(*save.add((slot * rows + t + j) * width + c), carry, taps);
                j += 1;
            }
            c += step;
        }
    }
}
