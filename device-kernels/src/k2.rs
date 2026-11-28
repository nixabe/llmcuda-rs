//! K2: combine, values, gate and norm, the dispatch passes and the rotary
//! table, apply and cache append.

use crate::common::{f2h, kvq8_quantize};
use cuda_device::{SharedArray, device, kernel, thread, warp};

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

/// `k2_combine` followed by the two adds it feeds: `ffn = routed +
/// shared`, then `out = x + ffn`, each rounded as a separate
/// `tensor_add` launch rounds it.
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
    let i =
        thread::blockIdx_x() as i64 * thread::blockDim_x() as i64 + thread::threadIdx_x() as i64;
    if i >= n {
        return;
    }
    unsafe {
        let v = *gate.add(i as usize) * 0.693_147_2_f32;
        let sp = v.max(0.0) + (-v.abs()).exp().ln_1p();
        *out.add(i as usize) = *x.add(i as usize) * (sp * 1.442_695_f32);
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
    let end = if (t + 1) * chunk < pairs {
        (t + 1) * chunk
    } else {
        pairs
    };
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
        let end = if (t + 1) * chunk < pairs {
            (t + 1) * chunk
        } else {
            pairs
        };
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

/// Precompute K2's rotary `(sin, cos)` pairs for a chunk of tokens of up
/// to eight sequences, each starting at its own device-side position.
/// The angle is formed in double precision and narrowed once, as the
/// reference does.
///
/// grid: (ceil(chunk / 8), sequences). block: (rope_dim / 2 or more, 8).
///
/// # Safety
/// Each used position pointer covers one int; coefficients covers
/// (offset + sequences * chunk) * rope_dim floats.
#[kernel]
pub unsafe fn k2_rotary_prepare(
    coefficients: *mut f32,
    chunk: i32,
    rope_dim: i32,
    theta: f32,
    offset: i32,
    p0: u64,
    p1: u64,
    p2: u64,
    p3: u64,
    p4: u64,
    p5: u64,
    p6: u64,
    p7: u64,
) {
    let t = (thread::blockIdx_x() * 8 + thread::threadIdx_y()) as i32;
    let d = thread::threadIdx_x() as i32;
    let seq = thread::blockIdx_y() as i32;
    if t >= chunk || d >= rope_dim / 2 {
        return;
    }
    let position = match seq {
        0 => p0,
        1 => p1,
        2 => p2,
        3 => p3,
        4 => p4,
        5 => p5,
        6 => p6,
        _ => p7,
    } as *const i32;
    unsafe {
        let freq = (theta as f64).powf(-2.0 * d as f64 / rope_dim as f64);
        let angle = (*position as f64 + t as f64) * freq;
        let at = ((offset as i64 + (seq * chunk) as i64 + t as i64) * rope_dim as i64
            + 2 * d as i64) as usize;
        *coefficients.add(at) = angle.sin() as f32;
        *coefficients.add(at + 1) = angle.cos() as f32;
    }
}

/// K2's decode-step rotary and cache append for up to eight sequences,
/// from the prepared sine/cosine table: rotate the query and key rows,
/// and write the rotated key and the raw value to each sequence's fp16
/// cache at its own position.
///
/// grid: (qh + kh, sequences <= 8). block: dim.
///
/// # Safety
/// q/qr cover sequences * qh * dim floats, k/kr/v sequences * kh * dim;
/// coefficients covers sequences * rope_dim; each used slot holds a live
/// pointer (append position, K cache, V cache), the caches covering
/// position + 1 rows of kh * dim halves.
#[kernel]
pub unsafe fn k2_rotary_append(
    q: *const f32,
    qr: *mut f32,
    k: *const f32,
    kr: *mut f32,
    v: *const f32,
    coefficients: *const f32,
    ap0: u64,
    ap1: u64,
    ap2: u64,
    ap3: u64,
    ap4: u64,
    ap5: u64,
    ap6: u64,
    ap7: u64,
    kc0: u64,
    kc1: u64,
    kc2: u64,
    kc3: u64,
    kc4: u64,
    kc5: u64,
    kc6: u64,
    kc7: u64,
    vc0: u64,
    vc1: u64,
    vc2: u64,
    vc3: u64,
    vc4: u64,
    vc5: u64,
    vc6: u64,
    vc7: u64,
    qh: i32,
    kh: i32,
    dim: i32,
    rope_dim: i32,
) {
    let seq = thread::blockIdx_y() as i32;
    let h = thread::blockIdx_x() as i32;
    let d = thread::threadIdx_x() as i32;
    let is_q = h < qh;
    let hh = if is_q { h } else { h - qh };
    let heads = if is_q { qh } else { kh };
    let (input, out) = if is_q { (q, qr) } else { (k, kr) };
    let base = ((seq as i64 * heads as i64 + hh as i64) * dim as i64) as usize;
    let (du, hu) = (d as usize, (rope_dim / 2) as usize);
    let half = rope_dim / 2;
    unsafe {
        let (mut k_cache, mut v_cache) =
            (core::ptr::null_mut::<u16>(), core::ptr::null_mut::<u16>());
        let mut at = 0i64;
        if !is_q {
            let pos = match seq {
                0 => ap0,
                1 => ap1,
                2 => ap2,
                3 => ap3,
                4 => ap4,
                5 => ap5,
                6 => ap6,
                _ => ap7,
            } as *const i32;
            k_cache = match seq {
                0 => kc0,
                1 => kc1,
                2 => kc2,
                3 => kc3,
                4 => kc4,
                5 => kc5,
                6 => kc6,
                _ => kc7,
            } as *mut u16;
            v_cache = match seq {
                0 => vc0,
                1 => vc1,
                2 => vc2,
                3 => vc3,
                4 => vc4,
                5 => vc5,
                6 => vc6,
                _ => vc7,
            } as *mut u16;
            at = *pos as i64 * kh as i64 * dim as i64 + hh as i64 * dim as i64;
        }
        let co = coefficients.add((seq as i64 * rope_dim as i64) as usize + 2 * du);
        if d >= rope_dim {
            let x = *input.add(base + du);
            *out.add(base + du) = x;
            if !is_q {
                *k_cache.add(at as usize + du) = f2h(x);
                *v_cache.add(at as usize + du) = f2h(*v.add(base + du));
            }
            return;
        }
        if d >= half {
            return;
        }
        let (sin_a, cos_a) = (*co, *co.add(1));
        let (x0, x1) = (*input.add(base + du), *input.add(base + du + hu));
        let y0 = x0.mul_add(cos_a, -(x1 * sin_a));
        let y1 = x0.mul_add(sin_a, x1 * cos_a);
        *out.add(base + du) = y0;
        *out.add(base + du + hu) = y1;
        if !is_q {
            *k_cache.add(at as usize + du) = f2h(y0);
            *k_cache.add(at as usize + du + hu) = f2h(y1);
            *v_cache.add(at as usize + du) = f2h(*v.add(base + du));
            *v_cache.add(at as usize + du + hu) = f2h(*v.add(base + du + hu));
        }
    }
}

/// Store element `e` of position `pos`'s row: binary16, or a q8_0 code
/// plus, from each block's first lane, its scale.
///
/// # Safety
/// All 32 lanes of the warp call this together with one q8_0 block's
/// elements; cache covers the row in its format.
#[device]
unsafe fn kv8_store(cache: *mut u16, q8: i32, pos: i64, row: i32, e: i32, x: f32) {
    unsafe {
        if q8 == 0 {
            *cache.add((pos * row as i64 + e as i64) as usize) = f2h(x);
            return;
        }
        let (code, d) = kvq8_quantize(x);
        let dst = (cache as *mut u8).add((pos * (row as i64 + (row >> 4) as i64)) as usize);
        *dst.add(e as usize) = code as u8;
        if e & 31 == 0 {
            *(dst.add((row + 2 * (e >> 5)) as usize) as *mut u16) = d;
        }
    }
}

/// `k2_rotary_append` with either half of the cache in q8_0. A rotary
/// pair is computed by one thread, so the key and value rows are parked
/// in shared memory and every thread then stores its own element, which
/// puts one 32-element block in each warp for the q8_0 scale.
///
/// grid: (qh + kh, sequences <= 8). block: dim, a multiple of 32, at
/// most 1024.
///
/// # Safety
/// As `k2_rotary_append`, each cache covering its rows in its own format.
#[kernel]
pub unsafe fn k2_rotary_append_kv8(
    q: *const f32,
    qr: *mut f32,
    k: *const f32,
    kr: *mut f32,
    v: *const f32,
    coefficients: *const f32,
    ap0: u64,
    ap1: u64,
    ap2: u64,
    ap3: u64,
    ap4: u64,
    ap5: u64,
    ap6: u64,
    ap7: u64,
    kc0: u64,
    kc1: u64,
    kc2: u64,
    kc3: u64,
    kc4: u64,
    kc5: u64,
    kc6: u64,
    kc7: u64,
    vc0: u64,
    vc1: u64,
    vc2: u64,
    vc3: u64,
    vc4: u64,
    vc5: u64,
    vc6: u64,
    vc7: u64,
    qh: i32,
    kh: i32,
    dim: i32,
    rope_dim: i32,
    kq8: i32,
    vq8: i32,
) {
    static mut KROW: SharedArray<f32, 1024> = SharedArray::UNINIT;
    static mut VROW: SharedArray<f32, 1024> = SharedArray::UNINIT;
    let seq = thread::blockIdx_y() as i32;
    let h = thread::blockIdx_x() as i32;
    let d = thread::threadIdx_x() as i32;
    let is_q = h < qh;
    let hh = if is_q { h } else { h - qh };
    let heads = if is_q { qh } else { kh };
    let (input, out) = if is_q { (q, qr) } else { (k, kr) };
    let base = ((seq as i64 * heads as i64 + hh as i64) * dim as i64) as usize;
    let (du, hu) = (d as usize, (rope_dim / 2) as usize);
    let half = rope_dim / 2;
    unsafe {
        let (mut k_cache, mut v_cache) =
            (core::ptr::null_mut::<u16>(), core::ptr::null_mut::<u16>());
        let mut at = 0i64;
        if !is_q {
            let pos = match seq {
                0 => ap0,
                1 => ap1,
                2 => ap2,
                3 => ap3,
                4 => ap4,
                5 => ap5,
                6 => ap6,
                _ => ap7,
            } as *const i32;
            k_cache = match seq {
                0 => kc0,
                1 => kc1,
                2 => kc2,
                3 => kc3,
                4 => kc4,
                5 => kc5,
                6 => kc6,
                _ => kc7,
            } as *mut u16;
            v_cache = match seq {
                0 => vc0,
                1 => vc1,
                2 => vc2,
                3 => vc3,
                4 => vc4,
                5 => vc5,
                6 => vc6,
                _ => vc7,
            } as *mut u16;
            at = *pos as i64 * kh as i64 * dim as i64 + hh as i64 * dim as i64;
        }
        let co = coefficients.add((seq as i64 * rope_dim as i64) as usize + 2 * du);
        let krow = SharedArray::as_raw_mut_ptr(&raw mut KROW);
        let vrow = SharedArray::as_raw_mut_ptr(&raw mut VROW);
        if d >= rope_dim {
            let x = *input.add(base + du);
            *out.add(base + du) = x;
            if !is_q {
                *krow.add(du) = x;
                *vrow.add(du) = *v.add(base + du);
            }
        } else if d < half {
            let (sin_a, cos_a) = (*co, *co.add(1));
            let (x0, x1) = (*input.add(base + du), *input.add(base + du + hu));
            let y0 = x0.mul_add(cos_a, -(x1 * sin_a));
            let y1 = x0.mul_add(sin_a, x1 * cos_a);
            *out.add(base + du) = y0;
            *out.add(base + du + hu) = y1;
            if !is_q {
                *krow.add(du) = y0;
                *krow.add(du + hu) = y1;
                *vrow.add(du) = *v.add(base + du);
                *vrow.add(du + hu) = *v.add(base + du + hu);
            }
        }
        if is_q {
            return;
        }
        thread::sync_threads();
        let row = kh * dim;
        let e = hh * dim + d;
        let pos = at / row as i64;
        kv8_store(k_cache, kq8, pos, row, e, *krow.add(du));
        kv8_store(v_cache, vq8, pos, row, e, *vrow.add(du));
    }
}
