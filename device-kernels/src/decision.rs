//! Decision head: LayerNorm, span and owner reductions, the lexical
//! gather, question fields, features and final scores.

use crate::common::{load_half_le, strided_dot, warp_xor_sum};
use cuda_device::{DynamicSharedArray, SharedArray, device, kernel, thread, warp};

/// `y[o][c] += src[owner(o)][c]`, the owner at `spans[o * stride + 2]`.
///
/// grid: (ceil(cols / block), n).
///
/// # Safety
/// y covers n rows of stride ldy, src every owner row of stride lds,
/// spans n * stride ints; y is disjoint from src.
#[kernel]
pub unsafe fn dec_add_owner_rows(
    y: *mut f32,
    ldy: i64,
    src: *const f32,
    lds: i64,
    spans: *const i32,
    stride: i32,
    n: i32,
    cols: i32,
) {
    let o = thread::blockIdx_y() as i32;
    let c = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as i32;
    if o >= n || c >= cols {
        return;
    }
    unsafe {
        let q = *spans.add((o * stride + 2) as usize);
        let at = y.add((o as i64 * ldy + c as i64) as usize);
        *at += *src.add((q as i64 * lds + c as i64) as usize);
    }
}

/// `out[s][c]` = the mean over rows t of span s of `LayerNorm(x[t])[c]`,
/// from each row's precomputed `(mean, rstd)`. Spans are `(start, end)`
/// pairs at `spans + s * stride`.
///
/// grid: (ceil(cols / block), nspans).
///
/// # Safety
/// x covers every span row at stride ldx; stats holds a `(mean, rstd)`
/// pair per row; g and b cover cols; out covers nspans rows of stride
/// ldo.
#[kernel]
pub unsafe fn dec_span_mean(
    x: *const f32,
    ldx: i64,
    cols: i32,
    stats: *const f32,
    g: *const f32,
    b: *const f32,
    spans: *const i32,
    stride: i32,
    nspans: i32,
    out: *mut f32,
    ldo: i64,
) {
    let s = thread::blockIdx_y() as i32;
    let c = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as i32;
    if s >= nspans || c >= cols {
        return;
    }
    unsafe {
        let start = *spans.add((s * stride) as usize);
        let end = *spans.add((s * stride + 1) as usize);
        let mut acc = 0.0f32;
        let mut t = start;
        while t < end {
            let mean = *stats.add(2 * t as usize);
            let rstd = *stats.add(2 * t as usize + 1);
            acc = (*x.add((t as i64 * ldx + c as i64) as usize) - mean).mul_add(rstd, acc);
            t += 1;
        }
        *out.add((s as i64 * ldo + c as i64) as usize) =
            (acc / (end - start) as f32).mul_add(*g.add(c as usize), *b.add(c as usize));
    }
}

/// Merge the key-split partials of the decision head's attention.
///
/// `part` holds `splits * nq * heads` weighted value sums of `dh`
/// floats, then a `(max, normalizer)` pair per split, query and head.
/// Splits are rescaled to the global maximum and summed in ascending
/// order.
///
/// grid: (nq, heads). block: one thread per head dimension.
///
/// # Safety
/// part covers splits * nq * heads * (dh + 2) floats; out covers nq rows
/// of stride ldo with heads * dh columns used.
#[kernel]
pub unsafe fn dec_attention_combine(
    part: *const f32,
    splits: i32,
    nq: i32,
    heads: i32,
    dh: i32,
    out: *mut f32,
    ldo: i64,
) {
    let q = thread::blockIdx_x() as i64;
    let h = thread::blockIdx_y() as i64;
    let d = thread::threadIdx_x() as i64;
    if d >= dh as i64 {
        return;
    }
    let per_split = nq as i64 * heads as i64;
    let slot = q * heads as i64 + h;
    unsafe {
        let ml = part.add((splits as i64 * per_split * dh as i64) as usize);
        let mut mx = f32::NEG_INFINITY;
        let mut m = ml.add((slot * 2) as usize);
        let mut z = 0;
        while z < splits {
            mx = mx.max(*m);
            m = m.wrapping_add((per_split * 2) as usize);
            z += 1;
        }
        let (mut num, mut den) = (0.0f32, 0.0f32);
        let mut m = ml.add((slot * 2) as usize);
        let mut v = part.add((slot * dh as i64 + d) as usize);
        let mut z = 0;
        while z < splits {
            let w = (*m - mx).exp();
            den = (*m.add(1)).mul_add(w, den);
            num = (*v).mul_add(w, num);
            m = m.wrapping_add((per_split * 2) as usize);
            v = v.wrapping_add((per_split * dh as i64) as usize);
            z += 1;
        }
        *out.add((q * ldo + h * dh as i64 + d) as usize) = num / den;
    }
}

/// The decision head's block sum: an xor-shuffle tree per warp, the warp
/// totals parked in shared memory, then a second tree over them.
///
/// # Safety
/// Every thread of a 1-D block of at most 1024 threads, a multiple of
/// 32, participates.
#[device]
unsafe fn dec_block_sum(v: f32) -> f32 {
    static mut RED: SharedArray<f32, 32> = SharedArray::UNINIT;
    let tid = thread::threadIdx_x();
    let (lane, w, nw) = (tid & 31, tid >> 5, thread::blockDim_x() >> 5);
    let mut v = v;
    let mut o = 16;
    while o > 0 {
        v += warp::shuffle_xor_f32_sync(u32::MAX, v, o);
        o >>= 1;
    }
    unsafe {
        thread::sync_threads();
        if lane == 0 {
            RED[w as usize] = v;
        }
        thread::sync_threads();
        v = if lane < nw { RED[lane as usize] } else { 0.0 };
    }
    let mut o = 16;
    while o > 0 {
        v += warp::shuffle_xor_f32_sync(u32::MAX, v, o);
        o >>= 1;
    }
    v
}

/// LayerNorm over each row, optionally recording the row's `(mean,
/// rstd)` and optionally writing the normalized row.
///
/// grid: (rows,). block: a multiple of 32.
///
/// # Safety
/// x covers rows of stride ldx with `cols` used; g and b cover cols.
/// stats is null or covers two floats a row; y is null or covers rows of
/// stride ldy and is disjoint from x.
#[kernel]
pub unsafe fn dec_layer_norm(
    x: *const f32,
    ldx: i64,
    rows: i32,
    cols: i32,
    eps: f32,
    g: *const f32,
    b: *const f32,
    stats: *mut f32,
    y: *mut f32,
    ldy: i64,
) {
    let r = thread::blockIdx_x() as i64;
    if r >= rows as i64 {
        return;
    }
    let tid = thread::threadIdx_x() as i32;
    let step = thread::blockDim_x() as i32;
    unsafe {
        let xr = x.add((r * ldx) as usize);
        let mut s = 0.0f32;
        let mut c = tid;
        while c < cols {
            s += *xr.add(c as usize);
            c += step;
        }
        let mean = dec_block_sum(s) / cols as f32;
        let mut v = 0.0f32;
        let mut c = tid;
        while c < cols {
            let d = *xr.add(c as usize) - mean;
            v = d.mul_add(d, v);
            c += step;
        }
        let var = dec_block_sum(v) / cols as f32;
        let rstd = 1.0 / (var + eps).sqrt();
        if !stats.is_null() && tid == 0 {
            *stats.add(2 * r as usize) = mean;
            *stats.add(2 * r as usize + 1) = rstd;
        }
        if !y.is_null() {
            let yr = y.add((r * ldy) as usize);
            let mut c = tid;
            while c < cols {
                let at = c as usize;
                *yr.add(at) = ((*xr.add(at) - mean) * rstd).mul_add(*g.add(at), *b.add(at));
                c += step;
            }
        }
    }
}

/// One warp per option: the scorer's output projection, then the final
/// combination `prior_scale * prior + gate * (joint_scale * cosine +
/// residual)`.
///
/// # Safety
/// h covers n_opts rows of stride ldh with `width` columns used, w_out
/// covers width, aux two floats per option, scores n_opts.
#[kernel]
pub unsafe fn dec_final_scores(
    h: *const f32,
    ldh: i64,
    n_opts: i32,
    width: i32,
    w_out: *const f32,
    b_out: f32,
    aux: *const f32,
    prior_scale: f32,
    joint_scale: f32,
    gate: f32,
    scores: *mut f32,
) {
    let flat = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
    let o = (flat >> 5) as i32;
    let lane = (thread::threadIdx_x() & 31) as i32;
    if o >= n_opts {
        return;
    }
    unsafe {
        let d = warp_xor_sum(strided_dot(
            h.add((o as i64 * ldh) as usize),
            w_out,
            lane,
            width,
        ));
        if lane == 0 {
            let joint = joint_scale.mul_add(*aux.add((o * 2 + 1) as usize), d + b_out);
            *scores.add(o as usize) = prior_scale.mul_add(*aux.add((o * 2) as usize), gate * joint);
        }
    }
}

/// Block-wide maximum; as `dec_block_sum`.
///
/// # Safety
/// As `dec_block_sum`.
#[device]
unsafe fn dec_block_max(v: f32) -> f32 {
    static mut RED_MAX: SharedArray<f32, 32> = SharedArray::UNINIT;
    let tid = thread::threadIdx_x();
    let (lane, w, nw) = (tid & 31, tid >> 5, thread::blockDim_x() >> 5);
    let mut v = v;
    let mut o = 16;
    while o > 0 {
        v = v.max(warp::shuffle_xor_f32_sync(u32::MAX, v, o));
        o >>= 1;
    }
    unsafe {
        let red = SharedArray::as_raw_mut_ptr(&raw mut RED_MAX);
        thread::sync_threads();
        if lane == 0 {
            *red.add(w as usize) = v;
        }
        thread::sync_threads();
        v = if lane < nw {
            *red.add(lane as usize)
        } else {
            f32::NEG_INFINITY
        };
    }
    let mut o = 16;
    while o > 0 {
        v = v.max(warp::shuffle_xor_f32_sync(u32::MAX, v, o));
        o >>= 1;
    }
    v
}

/// Element `i` of one row in GGUF storage format `fmt` (0 Q8_0, 1 BF16,
/// 2 Q6_K, 3 F16, 4 Q4_0, 5 Q4_K, 6 Q5_K). Operand order follows
/// `llmcuda_kernels::quant`; nothing here is fused.
///
/// # Safety
/// row covers the whole blocks of its format up to element i.
#[device]
unsafe fn dequant_element(row: *const u8, fmt: i32, i: i32) -> f32 {
    unsafe {
        match fmt {
            0 => {
                let b = row.add(((i >> 5) * 34) as usize);
                (*b.add((2 + (i & 31)) as usize) as i8 as f32) * load_half_le(b)
            }
            1 => {
                let bits = *row.add((2 * i) as usize) as u32
                    | ((*row.add((2 * i + 1) as usize) as u32) << 8);
                f32::from_bits(bits << 16)
            }
            3 => load_half_le(row.add((2 * i) as usize)),
            4 => {
                let b = row.add(((i >> 5) * 18) as usize);
                let e = i & 31;
                let byte = *b.add((2 + (e & 15)) as usize) as i32;
                let q = if e < 16 { byte & 0xF } else { byte >> 4 };
                (q - 8) as f32 * load_half_le(b)
            }
            2 => {
                let b = row.add(((i >> 8) * 210) as usize);
                let e = i & 255;
                let (half, r) = (e >> 7, e & 127);
                let (j, l) = (r >> 5, r & 31);
                let ql = b.add((half * 64) as usize);
                let qh = b.add((128 + half * 32) as usize);
                let sc = b.add((192 + half * 8) as usize) as *const i8;
                let lowbyte = *ql.add((l + (j & 1) * 32) as usize) as i32;
                let low = if j < 2 { lowbyte & 0xF } else { lowbyte >> 4 };
                let high = ((*qh.add(l as usize) as i32) >> (2 * j)) & 3;
                let d = load_half_le(b.add(208));
                d * (*sc.add(((l >> 4) + 2 * j) as usize) as f32)
                    * ((low | (high << 4)) - 32) as f32
            }
            5 | 6 => {
                let five = fmt == 6;
                let b = row.add(((i >> 8) * if five { 176 } else { 144 }) as usize);
                let e = i & 255;
                let (g, sub, l) = (e >> 6, (e >> 5) & 1, e & 31);
                let d = load_half_le(b);
                let dmin = load_half_le(b.add(2));
                // `get_scale_min_k4` from ggml-quants.c.
                let (js, q) = ((2 * g + sub) as usize, b.add(4));
                let (sc, m) = if js < 4 {
                    ((*q.add(js) & 63) as i32, (*q.add(js + 4) & 63) as i32)
                } else {
                    (
                        ((*q.add(js + 4) & 0xF) | ((*q.add(js - 4) >> 6) << 4)) as i32,
                        ((*q.add(js + 4) >> 4) | ((*q.add(js) >> 6) << 4)) as i32,
                    )
                };
                let byte = *b.add((if five { 48 } else { 16 } + g * 32 + l) as usize) as i32;
                let mut code = if sub == 0 { byte & 0xF } else { byte >> 4 };
                if five {
                    code += (((*b.add((16 + l) as usize) as i32) >> (2 * g + sub)) & 1) << 4;
                }
                let d1 = d * sc as f32;
                let m1 = dmin * m as f32;
                d1 * code as f32 - m1
            }
            _ => 0.0,
        }
    }
}

/// Mean of the dequantized embedding rows of each span's tokens.
///
/// grid: (ceil(cols / 256), spans). block: 256.
///
/// # Safety
/// table covers every referenced row; tokens covers each span's
/// [start, end); spans covers nspans * stride ints; out covers nspans
/// rows of ldo floats.
#[kernel]
pub unsafe fn dec_lexical(
    table: *const u8,
    fmt: i32,
    row_bytes: i64,
    cols: i32,
    tokens: *const i32,
    spans: *const i32,
    stride: i32,
    nspans: i32,
    out: *mut f32,
    ldo: i64,
) {
    let sp = thread::blockIdx_y() as i32;
    let c = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as i32;
    if sp >= nspans || c >= cols {
        return;
    }
    unsafe {
        let start = *spans.add((sp * stride) as usize);
        let end = *spans.add((sp * stride + 1) as usize);
        let mut acc = 0.0f32;
        let mut t = start;
        while t < end {
            let row = table.add((*tokens.add(t as usize) as i64 * row_bytes) as usize);
            acc += dequant_element(row, fmt, c);
            t += 1;
        }
        *out.add((sp as i64 * ldo + c as i64) as usize) = acc / (end - start) as f32;
    }
}

/// Per question: attention of its base row over its own options (dot
/// products scaled, softmax), the weighted option summary, a LayerNorm
/// of that summary, and the field row
/// `base + ln(summary) + gproj + type_embd[type]`.
///
/// grid: (questions,). block: 256, with width floats of dynamic shared
/// memory.
///
/// # Safety
/// opts covers n_opts rows of ldo; ospans n_opts * ostride ints with the
/// owner at +2; base and fields one row per question; gproj, sg and sb
/// width floats; type_embd one row per type; qspans the type at +2;
/// weights n_opts floats.
#[kernel]
pub unsafe fn dec_question_fields(
    opts: *const f32,
    ldo: i64,
    n_opts: i32,
    ospans: *const i32,
    ostride: i32,
    base: *const f32,
    ldb: i64,
    gproj: *const f32,
    type_embd: *const f32,
    qspans: *const i32,
    qstride: i32,
    sg: *const f32,
    sb: *const f32,
    eps: f32,
    weights: *mut f32,
    fields: *mut f32,
    ldf: i64,
    width: i32,
    scale: f32,
) {
    let summary = DynamicSharedArray::<f32>::get();
    let q = thread::blockIdx_x() as i32;
    let tid = thread::threadIdx_x() as i32;
    let step = thread::blockDim_x() as i32;
    let (lane, warp_i, nw) = (tid & 31, tid >> 5, step >> 5);
    unsafe {
        let bq = base.add((q as i64 * ldb) as usize);
        let mut o = warp_i;
        while o < n_opts {
            if *ospans.add((o * ostride + 2) as usize) == q {
                let oo = opts.add((o as i64 * ldo) as usize);
                let mut d = 0.0f32;
                let mut c = lane;
                while c < width {
                    d = (*oo.add(c as usize)).mul_add(*bq.add(c as usize), d);
                    c += 32;
                }
                d = warp_xor_sum(d);
                if lane == 0 {
                    *weights.add(o as usize) = d * scale;
                }
            }
            o += nw;
        }
        thread::sync_threads();
        let mut mx = f32::NEG_INFINITY;
        let mut o = tid;
        while o < n_opts {
            if *ospans.add((o * ostride + 2) as usize) == q {
                mx = mx.max(*weights.add(o as usize));
            }
            o += step;
        }
        mx = dec_block_max(mx);
        let mut se = 0.0f32;
        let mut o = tid;
        while o < n_opts {
            if *ospans.add((o * ostride + 2) as usize) == q {
                se += (*weights.add(o as usize) - mx).exp();
            }
            o += step;
        }
        se = dec_block_sum(se);
        let mut o = tid;
        while o < n_opts {
            if *ospans.add((o * ostride + 2) as usize) == q {
                *weights.add(o as usize) = (*weights.add(o as usize) - mx).exp() / se;
            }
            o += step;
        }
        thread::sync_threads();

        let mut s = 0.0f32;
        let mut c = tid;
        while c < width {
            let mut acc = 0.0f32;
            let mut o = 0;
            while o < n_opts {
                if *ospans.add((o * ostride + 2) as usize) == q {
                    acc = (*weights.add(o as usize))
                        .mul_add(*opts.add((o as i64 * ldo + c as i64) as usize), acc);
                }
                o += 1;
            }
            *summary.add(c as usize) = acc;
            s += acc;
            c += step;
        }
        let mean = dec_block_sum(s) / width as f32;
        let mut v = 0.0f32;
        let mut c = tid;
        while c < width {
            let d = *summary.add(c as usize) - mean;
            v = d.mul_add(d, v);
            c += step;
        }
        let rstd = 1.0 / (dec_block_sum(v) / width as f32 + eps).sqrt();
        let ty = *qspans.add((q * qstride + 2) as usize);
        let fq = fields.add((q as i64 * ldf) as usize);
        let te = type_embd.add((ty as i64 * width as i64) as usize);
        let mut c = tid;
        while c < width {
            let cu = c as usize;
            let ln = ((*summary.add(cu) - mean) * rstd).mul_add(*sg.add(cu), *sb.add(cu));
            *fq.add(cu) = *bq.add(cu) + ln + *gproj.add(cu) + *te.add(cu);
            c += step;
        }
    }
}

/// The scorer's feature row for each option, `[field, option, field *
/// option, |field - option|]`, and its two scalars: the cosine of the
/// lexical row with `qmean + global` (the prior) and of field with
/// option.
///
/// grid: (options,). block: 256.
///
/// # Safety
/// fieldn and qmean cover one row per question, optn and lexical one per
/// option, global hidden floats; feat covers n_opts rows of at least
/// 4 * width; aux two floats an option; ospans the owner at +2.
#[kernel]
pub unsafe fn dec_features(
    fieldn: *const f32,
    ldf: i64,
    optn: *const f32,
    ldo: i64,
    ospans: *const i32,
    ostride: i32,
    n_opts: i32,
    qmean: *const f32,
    ldq: i64,
    global: *const f32,
    lexical: *const f32,
    ldl: i64,
    hidden: i32,
    width: i32,
    feat: *mut f32,
    ldfeat: i64,
    aux: *mut f32,
) {
    let o = thread::blockIdx_x() as i32;
    let tid = thread::threadIdx_x() as i32;
    let step = thread::blockDim_x() as i32;
    if o >= n_opts {
        return;
    }
    unsafe {
        let q = *ospans.add((o * ostride + 2) as usize);
        let f = fieldn.add((q as i64 * ldf) as usize);
        let x = optn.add((o as i64 * ldo) as usize);
        let out = feat.add((o as i64 * ldfeat) as usize);
        let w = width as usize;
        let (mut ff, mut xx, mut fx) = (0.0f32, 0.0f32, 0.0f32);
        let mut c = tid;
        while c < width {
            let cu = c as usize;
            let (a, b) = (*f.add(cu), *x.add(cu));
            // The product is also stored, so it joins the running sum
            // already rounded, not fused.
            let ab = a * b;
            ff = a.mul_add(a, ff);
            xx = b.mul_add(b, xx);
            fx += ab;
            *out.add(cu) = a;
            *out.add(w + cu) = b;
            *out.add(2 * w + cu) = ab;
            *out.add(3 * w + cu) = (a - b).abs();
            c += step;
        }
        let qm = qmean.add((q as i64 * ldq) as usize);
        let lx = lexical.add((o as i64 * ldl) as usize);
        let (mut aa, mut ll, mut al) = (0.0f32, 0.0f32, 0.0f32);
        let mut c = tid;
        while c < hidden {
            let cu = c as usize;
            let (a, b) = (*qm.add(cu) + *global.add(cu), *lx.add(cu));
            aa = a.mul_add(a, aa);
            ll = b.mul_add(b, ll);
            al = a.mul_add(b, al);
            c += step;
        }
        let ff = dec_block_sum(ff);
        let xx = dec_block_sum(xx);
        let fx = dec_block_sum(fx);
        let aa = dec_block_sum(aa);
        let ll = dec_block_sum(ll);
        let al = dec_block_sum(al);
        if tid == 0 {
            let cosine = fx / (ff.sqrt().max(1e-12) * xx.sqrt().max(1e-12));
            let prior = al / (ll.sqrt().max(1e-12) * aa.sqrt().max(1e-12));
            *aux.add((o * 2) as usize) = prior;
            *aux.add((o * 2 + 1) as usize) = cosine;
        }
    }
}
