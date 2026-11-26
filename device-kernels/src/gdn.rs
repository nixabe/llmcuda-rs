//! Gated DeltaNet: the prelude's splits, gates and normalizers, the fp32
//! projection and the recurrent decode step.

use crate::common::{block_reduce_sum, conv_step_channel, strided_dot, warp_xor_sum};
use cuda_device::{DynamicSharedArray, device, kernel, ptx_asm, thread, warp};

/// `out = silu(x)`, elementwise: `x / (1 + exp(-x))`, which is
/// `ggml_cuda_op_silu_single` verbatim. libdevice exp, not an approximate
/// one: the fast intrinsic's error is relative to the result rather than
/// to the exponent, and this is not a bottleneck.
///
/// # Safety
/// x and out cover n floats; the 1-D launch is nonempty. Output may
/// exactly alias the input.
#[kernel]
pub unsafe fn gdn_silu(x: *const f32, out: *mut f32, n: i64) {
    let stride = thread::blockDim_x() as i64 * thread::gridDim_x() as i64;
    let mut i =
        thread::blockIdx_x() as i64 * thread::blockDim_x() as i64 + thread::threadIdx_x() as i64;
    while i < n {
        unsafe {
            let v = *x.add(i as usize);
            *out.add(i as usize) = v / (1.0 + (-v).exp());
        }
        i += stride;
    }
}

/// Split a convolved Gated DeltaNet row `[q | k | v]` into its three
/// per-head tensors. Queries and keys have `qk_heads` heads, values
/// `value_heads`; the grid covers the value heads and the first
/// `qk_heads` blocks also write q and k.
///
/// grid: (value_heads, tokens). block: head_dim.
///
/// # Safety
/// conv covers tokens * (2 * qk_heads + value_heads) * head_dim floats;
/// q and k cover tokens * qk_heads * head_dim and v covers tokens *
/// value_heads * head_dim; outputs are disjoint from conv and each other.
#[kernel]
pub unsafe fn gdn_split_qkv(
    conv: *const f32,
    q: *mut f32,
    k: *mut f32,
    v: *mut f32,
    head_dim: i32,
    qk_heads: i32,
    value_heads: i32,
) {
    let h = thread::blockIdx_x() as i64;
    let t = thread::blockIdx_y() as i64;
    let d = thread::threadIdx_x() as i64;
    let head_dim = head_dim as i64;
    let key_dim = qk_heads as i64 * head_dim;
    let conv_dim = 2 * key_dim + value_heads as i64 * head_dim;
    unsafe {
        let c = conv.add((t * conv_dim) as usize);
        *v.add(((t * value_heads as i64 + h) * head_dim + d) as usize) =
            *c.add((2 * key_dim + h * head_dim + d) as usize);
        if h < qk_heads as i64 {
            let o = ((t * qk_heads as i64 + h) * head_dim + d) as usize;
            *q.add(o) = *c.add((h * head_dim + d) as usize);
            *k.add(o) = *c.add((key_dim + h * head_dim + d) as usize);
        }
    }
}

/// The per-head decay and write gates.
///
/// ```text
/// a_softplus = softplus(alpha + ssm_dt.bias)
/// log_decay  = a_softplus * ssm_a          <- already the log-decay
/// beta       = sigmoid(beta_raw)
/// ```
///
/// `ssm_a` is stored already negated, so the product is the log of a
/// decay in (0, 1] and must not be negated again. The softplus is
/// `ggml_cuda_op_softplus` verbatim, including its x > 20 passthrough,
/// which matters because a lost passthrough overflows exp rather than
/// saturating.
///
/// # Safety
/// alpha, beta_raw and the three outputs cover n floats (n = tokens *
/// heads); dt_bias and ssm_a cover heads. Outputs are disjoint from the
/// inputs and each other. The 1-D launch is nonempty.
#[kernel]
pub unsafe fn gdn_gates(
    alpha: *const f32,
    beta_raw: *const f32,
    dt_bias: *const f32,
    ssm_a: *const f32,
    a_softplus: *mut f32,
    log_decay: *mut f32,
    beta: *mut f32,
    heads: i32,
    n: i64,
) {
    let stride = thread::blockDim_x() as i64 * thread::gridDim_x() as i64;
    let mut i =
        thread::blockIdx_x() as i64 * thread::blockDim_x() as i64 + thread::threadIdx_x() as i64;
    while i < n {
        unsafe {
            let h = (i % heads as i64) as usize;
            let a = *alpha.add(i as usize) + *dt_bias.add(h);
            let sp = if a > 20.0 { a } else { (1.0 + a.exp()).ln() };
            *a_softplus.add(i as usize) = sp;
            *log_decay.add(i as usize) = sp * *ssm_a.add(h);
            *beta.add(i as usize) = 1.0 / (1.0 + (-*beta_raw.add(i as usize)).exp());
        }
        i += stride;
    }
}

/// `exp(log_decay)` for the whole pass, once.
///
/// The unfolded scan takes this exponential in every lane of every warp
/// of a head, on every token; the folded scan reads it from here instead.
/// It is the same libdevice exp on the same operand, so the decay is the
/// same bits.
///
/// # Safety
/// log_decay and decay cover n floats; the 1-D launch is nonempty.
#[kernel]
pub unsafe fn gdn_scan_decay(log_decay: *const f32, decay: *mut f32, n: i64) {
    let stride = thread::blockDim_x() as i64 * thread::gridDim_x() as i64;
    let mut i =
        thread::blockIdx_x() as i64 * thread::blockDim_x() as i64 + thread::threadIdx_x() as i64;
    while i < n {
        unsafe {
            *decay.add(i as usize) = (*log_decay.add(i as usize)).exp();
        }
        i += stride;
    }
}

/// `gdn_silu` and `gdn_split_qkv` in one launch: the activated row is
/// written whole to `silu` and its three slices to q, k and v.
///
/// The standalone pair wrote one buffer that the split immediately read.
/// `v / (1 + exp(-v))` is the same expression in the same order, so the
/// outputs are the same bits.
///
/// grid: (value_heads, tokens). block: head_dim.
///
/// # Safety
/// As `gdn_split_qkv`; silu covers the same extent as conv and is
/// disjoint from every other buffer.
#[kernel]
pub unsafe fn gdn_silu_split_qkv(
    conv: *const f32,
    silu: *mut f32,
    q: *mut f32,
    k: *mut f32,
    v: *mut f32,
    head_dim: i32,
    qk_heads: i32,
    value_heads: i32,
) {
    let h = thread::blockIdx_x() as i64;
    let t = thread::blockIdx_y() as i64;
    let d = thread::threadIdx_x() as i64;
    let head_dim = head_dim as i64;
    let key_dim = qk_heads as i64 * head_dim;
    let conv_dim = 2 * key_dim + value_heads as i64 * head_dim;
    unsafe {
        let c = conv.add((t * conv_dim) as usize);
        let sr = silu.add((t * conv_dim) as usize);
        let iv = (2 * key_dim + h * head_dim + d) as usize;
        let mut vv = *c.add(iv);
        vv = vv / (1.0 + (-vv).exp());
        *sr.add(iv) = vv;
        *v.add(((t * value_heads as i64 + h) * head_dim + d) as usize) = vv;
        if h < qk_heads as i64 {
            let iq = (h * head_dim + d) as usize;
            let ik = (key_dim + h * head_dim + d) as usize;
            let (mut qv, mut kv) = (*c.add(iq), *c.add(ik));
            qv = qv / (1.0 + (-qv).exp());
            kv = kv / (1.0 + (-kv).exp());
            *sr.add(iq) = qv;
            *sr.add(ik) = kv;
            let o = ((t * qk_heads as i64 + h) * head_dim + d) as usize;
            *q.add(o) = qv;
            *k.add(o) = kv;
        }
    }
}

/// The decode step's convolution, SiLU and q/k/v split in one launch for
/// up to eight sequences, each with its own convolution cache.
///
/// Per channel this is `conv1d_step_batch`'s arithmetic exactly (the
/// same helper), then `gdn_silu_split_qkv`'s; the raw convolution is
/// kept in `conv_raw` because verification replays from it.
///
/// grid: (value_heads, sequences <= 8). block: head_dim.
///
/// # Safety
/// x, conv_raw and silu cover sequences * conv_dim floats; weight covers
/// conv_dim * conv_kernel; each used state pointer covers conv_dim *
/// (conv_kernel - 1) floats and belongs to one sequence; q, k and v as
/// `gdn_split_qkv`. All outputs are mutually disjoint.
#[kernel]
pub unsafe fn gdn_conv_silu_split_step_batch(
    x: *const f32,
    weight: *const f32,
    s0: u64,
    s1: u64,
    s2: u64,
    s3: u64,
    s4: u64,
    s5: u64,
    s6: u64,
    s7: u64,
    conv_raw: *mut f32,
    silu: *mut f32,
    q: *mut f32,
    k: *mut f32,
    v: *mut f32,
    head_dim: i32,
    qk_heads: i32,
    value_heads: i32,
    conv_kernel: i32,
) {
    let h = thread::blockIdx_x() as i64;
    let t = thread::blockIdx_y() as i64;
    let d = thread::threadIdx_x() as i64;
    let state = match t {
        0 => s0,
        1 => s1,
        2 => s2,
        3 => s3,
        4 => s4,
        5 => s5,
        6 => s6,
        _ => s7,
    } as *mut f32;
    let head_dim = head_dim as i64;
    let key_dim = qk_heads as i64 * head_dim;
    let conv_dim = 2 * key_dim + value_heads as i64 * head_dim;
    let carry = conv_kernel - 1;
    unsafe {
        let xt = x.add((t * conv_dim) as usize);
        let cr = conv_raw.add((t * conv_dim) as usize);
        let sr = silu.add((t * conv_dim) as usize);
        let iv = 2 * key_dim + h * head_dim + d;
        let mut vv = conv_step_channel(
            *xt.add(iv as usize),
            weight.add((iv * conv_kernel as i64) as usize),
            state.add((iv * carry as i64) as usize),
            carry,
        );
        *cr.add(iv as usize) = vv;
        vv = vv / (1.0 + (-vv).exp());
        *sr.add(iv as usize) = vv;
        *v.add(((t * value_heads as i64 + h) * head_dim + d) as usize) = vv;
        if h < qk_heads as i64 {
            let iq = h * head_dim + d;
            let ik = key_dim + h * head_dim + d;
            let mut qv = conv_step_channel(
                *xt.add(iq as usize),
                weight.add((iq * conv_kernel as i64) as usize),
                state.add((iq * carry as i64) as usize),
                carry,
            );
            let mut kv = conv_step_channel(
                *xt.add(ik as usize),
                weight.add((ik * conv_kernel as i64) as usize),
                state.add((ik * carry as i64) as usize),
                carry,
            );
            *cr.add(iq as usize) = qv;
            *cr.add(ik as usize) = kv;
            qv = qv / (1.0 + (-qv).exp());
            kv = kv / (1.0 + (-kv).exp());
            *sr.add(iq as usize) = qv;
            *sr.add(ik as usize) = kv;
            let o = ((t * qk_heads as i64 + h) * head_dim + d) as usize;
            *q.add(o) = qv;
            *k.add(o) = kv;
        }
    }
}

/// L2-normalize one head's query and key rows; the query is also scaled.
///
/// `1 / sqrt(sum + eps)`: the epsilon is added to the sum of squares,
/// which is flash-linear-attention's l2norm, transformers' and
/// llama.cpp's since 5fdfa6282
/// (`build_gdn_l2_norm`). The form that floors the norm instead,
/// `1 / max(sqrt(sum), eps)`, is ggml_l2_norm's; on the real q_conv and
/// k_conv of blocks 0/4/20 the two differ by up to 2.148e-4 relative.
///
/// A reciprocal square root would be faster and is not IEEE-exact;
/// dividing by sqrt keeps the kernel bit-comparable with its CPU
/// reference up to the reduction order.
///
/// # Safety
/// base + threadIdx.x indexes all four buffers; block_reduce_sum's
/// launch contract holds with blockDim.x == head_dim.
#[device]
unsafe fn normalize_qk_row(
    q_in: *const f32,
    k_in: *const f32,
    q_out: *mut f32,
    k_out: *mut f32,
    base: i64,
    eps: f32,
    scale: f32,
) {
    let at = (base + thread::threadIdx_x() as i64) as usize;
    unsafe {
        let (qv, kv) = (*q_in.add(at), *k_in.add(at));
        let q_sq = block_reduce_sum(qv * qv);
        // The next reduction overwrites the slot this one returned from.
        thread::sync_threads();
        let k_sq = block_reduce_sum(kv * kv);
        thread::sync_threads();
        *q_out.add(at) = qv * (1.0 / (q_sq + eps).sqrt()) * scale;
        *k_out.add(at) = kv * (1.0 / (k_sq + eps).sqrt());
    }
}

/// `normalize_qk_row` over one token's heads, for the recurrent step.
///
/// grid: (heads,). block: head_dim, with a float of dynamic shared
/// memory per warp.
///
/// # Safety
/// The four buffers cover gridDim.x * head_dim floats; outputs are
/// disjoint from inputs and each other.
#[kernel]
pub unsafe fn gdn_normalize_qk(
    q_in: *const f32,
    k_in: *const f32,
    q_out: *mut f32,
    k_out: *mut f32,
    head_dim: i32,
    eps: f32,
    scale: f32,
) {
    let base = (thread::blockIdx_x() as i32 * head_dim) as i64;
    unsafe { normalize_qk_row(q_in, k_in, q_out, k_out, base, eps, scale) }
}

/// `normalize_qk_row` over every head of every token of a chunk.
///
/// grid: (tokens * heads,). block: head_dim, with a float of dynamic
/// shared memory per warp.
///
/// # Safety
/// As `gdn_normalize_qk`.
#[kernel]
pub unsafe fn gdn_chunk_normalize_qk(
    q_in: *const f32,
    k_in: *const f32,
    q_out: *mut f32,
    k_out: *mut f32,
    head_dim: i32,
    eps: f32,
    scale: f32,
) {
    let base = thread::blockIdx_x() as i64 * head_dim as i64;
    unsafe { normalize_qk_row(q_in, k_in, q_out, k_out, base, eps, scale) }
}

/// A dense fp32 projection for the Gated DeltaNet gates: one warp per
/// output row, each lane contracting every 32nd element in ascending
/// order, then an xor shuffle tree.
///
/// The contraction is unrolled by hand, four fused steps an iteration:
/// the compiler does not unroll it, and the single-step loop is a third
/// slower at decode sizes. The order of the fused additions is unchanged.
///
/// grid: (ceil(n_rows / blockDim.y), tokens). block: (32, rows per block).
///
/// # Safety
/// weight covers n_rows * k_dim floats, x gridDim.y * k_dim, out
/// gridDim.y * n_rows; out is disjoint from the inputs.
#[kernel]
pub unsafe fn gdn_proj_f32(
    weight: *const f32,
    x: *const f32,
    out: *mut f32,
    k_dim: i32,
    n_rows: i32,
) {
    let lane = thread::threadIdx_x() as i32;
    let n = (thread::blockIdx_x() * thread::blockDim_y() + thread::threadIdx_y()) as i32;
    if n >= n_rows {
        return;
    }
    let t = thread::blockIdx_y() as i64;
    unsafe {
        let row = weight.add((n as i64 * k_dim as i64) as usize);
        let xr = x.add((t * k_dim as i64) as usize);
        let acc = strided_dot(row, xr, lane, k_dim);
        let acc = warp_xor_sum(acc);
        if lane == 0 {
            *out.add((t * n_rows as i64 + n as i64) as usize) = acc;
        }
    }
}

/// `strided_dot(a, x)` and `strided_dot(b, x)` in one pass over `x`.
/// Each chain is fused in ascending order exactly as the single form.
///
/// # Safety
/// a, b and x cover len floats.
#[device]
unsafe fn strided_dot2(
    a: *const f32,
    b: *const f32,
    x: *const f32,
    first: i32,
    len: i32,
) -> (f32, f32) {
    let (mut aa, mut bb) = (0.0f32, 0.0f32);
    let mut i = first;
    unsafe {
        let mut pa = a.add(first as usize);
        let mut pb = b.add(first as usize);
        let mut px = x.add(first as usize);
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
            let x = (
                *px.add(0),
                *px.add(32),
                *px.add(64),
                *px.add(96),
                *px.add(128),
                *px.add(160),
                *px.add(192),
                *px.add(224),
                *px.add(256),
                *px.add(288),
                *px.add(320),
                *px.add(352),
                *px.add(384),
                *px.add(416),
                *px.add(448),
                *px.add(480),
            );
            aa = a.0.mul_add(x.0, aa);
            aa = a.1.mul_add(x.1, aa);
            aa = a.2.mul_add(x.2, aa);
            aa = a.3.mul_add(x.3, aa);
            aa = a.4.mul_add(x.4, aa);
            aa = a.5.mul_add(x.5, aa);
            aa = a.6.mul_add(x.6, aa);
            aa = a.7.mul_add(x.7, aa);
            aa = a.8.mul_add(x.8, aa);
            aa = a.9.mul_add(x.9, aa);
            aa = a.10.mul_add(x.10, aa);
            aa = a.11.mul_add(x.11, aa);
            aa = a.12.mul_add(x.12, aa);
            aa = a.13.mul_add(x.13, aa);
            aa = a.14.mul_add(x.14, aa);
            aa = a.15.mul_add(x.15, aa);
            bb = b.0.mul_add(x.0, bb);
            bb = b.1.mul_add(x.1, bb);
            bb = b.2.mul_add(x.2, bb);
            bb = b.3.mul_add(x.3, bb);
            bb = b.4.mul_add(x.4, bb);
            bb = b.5.mul_add(x.5, bb);
            bb = b.6.mul_add(x.6, bb);
            bb = b.7.mul_add(x.7, bb);
            bb = b.8.mul_add(x.8, bb);
            bb = b.9.mul_add(x.9, bb);
            bb = b.10.mul_add(x.10, bb);
            bb = b.11.mul_add(x.11, bb);
            bb = b.12.mul_add(x.12, bb);
            bb = b.13.mul_add(x.13, bb);
            bb = b.14.mul_add(x.14, bb);
            bb = b.15.mul_add(x.15, bb);
            pa = pa.wrapping_add(512);
            pb = pb.wrapping_add(512);
            px = px.wrapping_add(512);
            i += 512;
        }
        while i < len {
            aa = (*pa).mul_add(*px, aa);
            bb = (*pb).mul_add(*px, bb);
            pa = pa.wrapping_add(32);
            pb = pb.wrapping_add(32);
            px = px.wrapping_add(32);
            i += 32;
        }
    }
    (aa, bb)
}

/// The Gated DeltaNet gate prelude in one launch: RMSNorm of the token's
/// hidden row, the alpha and beta projections of the normalized row for
/// one head, and that head's gates.
///
/// Replaces two `gdn_proj_f32` launches and a `gdn_gates` launch and
/// keeps their arithmetic: the norm reduces in four bands of eight warp
/// totals that thread 0 adds in ascending order, each projection is the
/// warp contraction of `gdn_proj_f32`, and the gates are `gdn_gates`'
/// expressions. Block 0 of each token also writes the normalized row.
///
/// grid: (heads, tokens). block: 256, with k_dim + 32 floats of dynamic
/// shared memory.
///
/// # Safety
/// w_alpha and w_beta cover heads * k_dim floats; x and normed tokens *
/// k_dim; norm_weight k_dim; dt_bias and ssm_a heads; the five per-head
/// outputs tokens * heads. Outputs are disjoint from inputs and each
/// other. k_dim is at most 4096.
#[kernel]
pub unsafe fn gdn_norm_alpha_beta_gates(
    w_alpha: *const f32,
    w_beta: *const f32,
    x: *const f32,
    dt_bias: *const f32,
    ssm_a: *const f32,
    alpha: *mut f32,
    beta_raw: *mut f32,
    a_softplus: *mut f32,
    log_decay: *mut f32,
    beta: *mut f32,
    k_dim: i32,
    heads: i32,
    _tokens: i32,
    norm_weight: *const f32,
    normed: *mut f32,
    eps: f32,
) {
    let tid = thread::threadIdx_x() as i32;
    let lane = tid & 31;
    let n = thread::blockIdx_x() as i64;
    let t = thread::blockIdx_y() as i64;
    let data = DynamicSharedArray::<f32>::get();
    unsafe {
        let scratch = data.add(k_dim as usize);
        let xr = x.add((t * k_dim as i64) as usize);
        // Eight loads issued together, then the stores: a thread issues
        // in order, and a load waiting on the previous store is the slow
        // way through a row.
        let mut i = tid;
        let (mut xp, mut dp) = (xr.add(tid as usize), data.add(tid as usize));
        while i + 1792 < k_dim {
            let v = (
                *xp.add(0),
                *xp.add(256),
                *xp.add(512),
                *xp.add(768),
                *xp.add(1024),
                *xp.add(1280),
                *xp.add(1536),
                *xp.add(1792),
            );
            *dp.add(0) = v.0;
            *dp.add(256) = v.1;
            *dp.add(512) = v.2;
            *dp.add(768) = v.3;
            *dp.add(1024) = v.4;
            *dp.add(1280) = v.5;
            *dp.add(1536) = v.6;
            *dp.add(1792) = v.7;
            xp = xp.wrapping_add(2048);
            dp = dp.wrapping_add(2048);
            i += 2048;
        }
        while i < k_dim {
            *dp = *xp;
            xp = xp.wrapping_add(256);
            dp = dp.wrapping_add(256);
            i += 256;
        }
        thread::sync_threads();
        let mut band = 0;
        while band < 4 {
            let mut partial = 0.0f32;
            let mut j = tid + band * 256;
            while j < k_dim {
                let v = *data.add(j as usize);
                partial = v.mul_add(v, partial);
                j += 1024;
            }
            let mut off = 16;
            while off > 0 {
                partial += warp::shuffle_down_f32_sync(u32::MAX, partial, off);
                off >>= 1;
            }
            if lane == 0 {
                *scratch.add((band * 8 + (tid >> 5)) as usize) = partial;
            }
            band += 1;
        }
        thread::sync_threads();
        if tid == 0 {
            let mut sum_sq = 0.0f32;
            let mut w = 0;
            while w < 32 {
                sum_sq += *scratch.add(w);
                w += 1;
            }
            *scratch = 1.0 / (sum_sq / k_dim as f32 + eps).sqrt();
        }
        thread::sync_threads();
        let inv_rms = *scratch;
        let mut j = tid;
        let mut dp = data.add(tid as usize);
        let mut wp = norm_weight.add(tid as usize);
        let mut np = normed.add((t * k_dim as i64) as usize + tid as usize);
        while j + 1792 < k_dim {
            let w = (
                *wp.add(0),
                *wp.add(256),
                *wp.add(512),
                *wp.add(768),
                *wp.add(1024),
                *wp.add(1280),
                *wp.add(1536),
                *wp.add(1792),
            );
            let d = (
                *dp.add(0),
                *dp.add(256),
                *dp.add(512),
                *dp.add(768),
                *dp.add(1024),
                *dp.add(1280),
                *dp.add(1536),
                *dp.add(1792),
            );
            let y = (
                d.0 * inv_rms * w.0,
                d.1 * inv_rms * w.1,
                d.2 * inv_rms * w.2,
                d.3 * inv_rms * w.3,
                d.4 * inv_rms * w.4,
                d.5 * inv_rms * w.5,
                d.6 * inv_rms * w.6,
                d.7 * inv_rms * w.7,
            );
            *dp.add(0) = y.0;
            *dp.add(256) = y.1;
            *dp.add(512) = y.2;
            *dp.add(768) = y.3;
            *dp.add(1024) = y.4;
            *dp.add(1280) = y.5;
            *dp.add(1536) = y.6;
            *dp.add(1792) = y.7;
            if n == 0 {
                *np.add(0) = y.0;
                *np.add(256) = y.1;
                *np.add(512) = y.2;
                *np.add(768) = y.3;
                *np.add(1024) = y.4;
                *np.add(1280) = y.5;
                *np.add(1536) = y.6;
                *np.add(1792) = y.7;
            }
            dp = dp.wrapping_add(2048);
            wp = wp.wrapping_add(2048);
            np = np.wrapping_add(2048);
            j += 2048;
        }
        while j < k_dim {
            let normalized = *dp * inv_rms * *wp;
            *dp = normalized;
            if n == 0 {
                *np = normalized;
            }
            dp = dp.wrapping_add(256);
            wp = wp.wrapping_add(256);
            np = np.wrapping_add(256);
            j += 256;
        }
        thread::sync_threads();
        if tid >= 32 {
            return;
        }
        let row = (n * k_dim as i64) as usize;
        let (aa, bb) = strided_dot2(w_alpha.add(row), w_beta.add(row), data, lane, k_dim);
        let a_sum = warp_xor_sum(aa);
        let b_sum = warp_xor_sum(bb);
        if lane == 0 {
            let i = (t * heads as i64 + n) as usize;
            *alpha.add(i) = a_sum;
            *beta_raw.add(i) = b_sum;
            let a = a_sum + *dt_bias.add(n as usize);
            let sp = if a > 20.0 { a } else { (1.0 + a.exp()).ln() };
            *a_softplus.add(i) = sp;
            *log_decay.add(i) = sp * *ssm_a.add(n as usize);
            *beta.add(i) = 1.0 / (1.0 + (-b_sum).exp());
        }
    }
}

/// Four adjacent floats in one 128-bit load.
///
/// # Safety
/// p is 16-byte aligned and covers four floats.
#[device]
unsafe fn load4(p: *const f32) -> [f32; 4] {
    let (a, b, c, d): (f32, f32, f32, f32);
    unsafe {
        ptx_asm!(
            "ld.global.v4.f32 {%0, %1, %2, %3}, [%4];",
            out("=f") a,
            out("=f") b,
            out("=f") c,
            out("=f") d,
            in("l") p as u64,
        );
    }
    [a, b, c, d]
}

/// Four adjacent floats in one 128-bit store.
///
/// # Safety
/// p is 16-byte aligned and covers four floats this thread owns.
#[device]
unsafe fn store4(p: *mut f32, v: [f32; 4]) {
    unsafe {
        ptx_asm!(
            "st.global.v4.f32 [%0], {%1, %2, %3, %4};",
            in("l") p as u64,
            in("f") v[0],
            in("f") v[1],
            in("f") v[2],
            in("f") v[3],
        );
    }
}

/// One Gated DeltaNet decode step for up to eight sequences: decay the
/// state, read the prediction, write the delta-rule correction and
/// produce the output, one warp per state row.
///
/// ```text
/// S    = S * exp(log_decay)
/// pred = S k
/// S    = S + beta * (v - pred) k^T
/// out  = S q
/// ```
///
/// The order follows the reference exactly: decay the state, form the
/// correction against the *decayed* state, apply the outer-product
/// update, then read the output from the *updated* state. Reordering any
/// of these is a plausible-looking change that silently produces a
/// different model.
///
/// The kernel is bandwidth-bound: 2 MiB of state read and written per
/// layer per sequence. Its first form gave each state row a block, one
/// thread per key index, with two block-wide reductions on the hot path,
/// and moved state at 44% of the card's streaming roofline. One warp per
/// row is the structure that fits: a lane owns four adjacent columns of
/// each 128-column chunk, read and written as one 16-byte vector, so a
/// warp keeps 512 contiguous bytes in flight; both reductions are
/// five-shuffle butterflies with no barrier and no shared memory; and
/// the decayed row is carried between the two passes in registers, which
/// is why the chunks are spelled out rather than looped.
///
/// libdevice exp for the decay, not an approximate one: it multiplies
/// the whole state every token, so a systematic bias compounds over the
/// context rather than averaging out.
///
/// grid: (ceil(head_dim / rows per block), value heads, sequences <= 8).
/// block: 32 threads per row.
///
/// # Safety
/// Each used state pointer covers value_heads * head_dim * head_dim
/// floats and belongs to one sequence; q and k cover sequences * qk_heads
/// * head_dim; v and out sequences * value_heads * head_dim; log_decay
/// and beta sequences * value_heads. head_dim is a multiple of 128, at
/// most 512. out is disjoint from the inputs.
#[kernel]
pub unsafe fn gdn_recurrent_step(
    s0: u64,
    s1: u64,
    s2: u64,
    s3: u64,
    s4: u64,
    s5: u64,
    s6: u64,
    s7: u64,
    q: *const f32,
    k: *const f32,
    v: *const f32,
    log_decay: *const f32,
    beta: *const f32,
    out: *mut f32,
    head_dim: i32,
    qk_heads: i32,
) {
    let tid = thread::threadIdx_x();
    let lane = (tid & 31) as i32;
    let vi = (thread::blockIdx_x() * (thread::blockDim_x() >> 5) + (tid >> 5)) as i32;
    let h = thread::blockIdx_y() as i32;
    let z = thread::blockIdx_z() as i32;
    if vi >= head_dim {
        return;
    }
    let state = match z {
        0 => s0,
        1 => s1,
        2 => s2,
        3 => s3,
        4 => s4,
        5 => s5,
        6 => s6,
        _ => s7,
    } as *mut f32;
    let n_v_heads = thread::gridDim_y() as i32;
    let qk_base = ((z * qk_heads + h % qk_heads) * head_dim) as usize;
    let v_base = ((z * n_v_heads + h) * head_dim) as usize;
    let g_at = (z * n_v_heads + h) as usize;
    let chunks = head_dim >> 7;
    let j0 = (4 * lane) as usize;
    unsafe {
        let decay = (*log_decay.add(g_at)).exp();
        let row = state.add(((h as i64 * head_dim as i64 + vi as i64) * head_dim as i64) as usize);
        let kq = k.add(qk_base);
        let qq = q.add(qk_base);
        let zero = [0.0f32; 4];
        let (mut s, mut kc) = ([zero; 4], [zero; 4]);
        let mut partial = 0.0f32;
        macro_rules! decay_and_predict {
            ($c:literal) => {
                if $c < chunks {
                    let j = ($c << 7) + j0;
                    let kv = load4(kq.add(j));
                    let sv = load4(row.add(j));
                    let sv = [sv[0] * decay, sv[1] * decay, sv[2] * decay, sv[3] * decay];
                    partial = sv[0].mul_add(kv[0], partial);
                    partial = sv[1].mul_add(kv[1], partial);
                    partial = sv[2].mul_add(kv[2], partial);
                    partial = sv[3].mul_add(kv[3], partial);
                    kc[$c] = kv;
                    s[$c] = sv;
                }
            };
        }
        decay_and_predict!(0);
        decay_and_predict!(1);
        decay_and_predict!(2);
        decay_and_predict!(3);
        let predicted = warp_xor_sum(partial);
        let vcorr = *beta.add(g_at) * (*v.add(v_base + vi as usize) - predicted);

        partial = 0.0;
        macro_rules! correct_and_read {
            ($c:literal) => {
                if $c < chunks {
                    let j = ($c << 7) + j0;
                    let (sv, kv) = (s[$c], kc[$c]);
                    let sv = [
                        vcorr.mul_add(kv[0], sv[0]),
                        vcorr.mul_add(kv[1], sv[1]),
                        vcorr.mul_add(kv[2], sv[2]),
                        vcorr.mul_add(kv[3], sv[3]),
                    ];
                    store4(row.add(j), sv);
                    let qv = load4(qq.add(j));
                    partial = sv[0].mul_add(qv[0], partial);
                    partial = sv[1].mul_add(qv[1], partial);
                    partial = sv[2].mul_add(qv[2], partial);
                    partial = sv[3].mul_add(qv[3], partial);
                }
            };
        }
        correct_and_read!(0);
        correct_and_read!(1);
        correct_and_read!(2);
        correct_and_read!(3);
        let o = warp_xor_sum(partial);
        if lane == 0 {
            *out.add(v_base + vi as usize) = o;
        }
    }
}
