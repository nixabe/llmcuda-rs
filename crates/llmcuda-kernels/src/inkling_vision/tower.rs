//! Reference forward pass through the `inkling2v` tower and projector.
//!
//! Ported from llama.cpp at ae6a7ad2c: `mtmd_graph_inkling2v::build`
//! (`tools/mtmd/models/inkling2v.cpp`); `mtmd_graph::build_inp`,
//! `resize_position_embeddings`, `build_vit`, `build_norm`, `build_ffn`,
//! `build_attn`, `build_rope_2d` and `build_patch_merge_permute`
//! (`tools/mtmd/mtmd-graph.cpp`); the position fill for
//! `PROJECTOR_TYPE_INKLING2V` in `tools/mtmd/mtmd-subcontext.cpp`; and the
//! ggml CPU operators those nodes run (`ggml/src/ggml-cpu/ops.cpp`).
//!
//! # Layout
//!
//! Activations are row-major `[rows][channels]` in one flat `Vec<f32>`.
//! Every matrix is row-major `[out][in]`: `w[o * n_in + i]` multiplies
//! input `i` into output `o`, the layout `y = W x` wants. That is the byte
//! order of the GGUF tensors as stored (ggml dims `[in, out]`, `in`
//! fastest), so loading is a widening copy and nothing is transposed.
//!
//! # Precision
//!
//! Sums of products are fp32 with a fixed order (see [`dot`]). The pieces
//! where fp32 intermediates would be the dominant error are computed in
//! f64 and rounded once: LayerNorm statistics, rotary angles, the softmax
//! normalizer and the error function. Half precision enters only where
//! [`InklingVisionNumerics`] says llama.cpp's graph puts it.

use llmcuda_gguf::{GgmlType, GgufFile};
use llmcuda_model::InklingVisionConfig;
use llmcuda_model::inkling_vision::{InklingVisionRole, InklingVisionWeightSchema};

use super::preprocess::PreprocessedInklingImage;
use crate::f16::round_through_f16;

/// One transformer block's weights, fp32.
#[derive(Debug, Clone)]
pub struct InklingVisionBlockWeights {
    /// `v.blk.N.ln1.weight` — LayerNorm before attention, `[hidden]`.
    pub ln1_w: Vec<f32>,
    /// `v.blk.N.ln1.bias`.
    pub ln1_b: Vec<f32>,
    /// `v.blk.N.attn_q.weight`, `[hidden][hidden]`.
    pub q_w: Vec<f32>,
    /// `v.blk.N.attn_q.bias`.
    pub q_b: Vec<f32>,
    /// `v.blk.N.attn_k.weight`, `[hidden][hidden]`.
    pub k_w: Vec<f32>,
    /// `v.blk.N.attn_k.bias`.
    pub k_b: Vec<f32>,
    /// `v.blk.N.attn_v.weight`, `[hidden][hidden]`.
    pub v_w: Vec<f32>,
    /// `v.blk.N.attn_v.bias`.
    pub v_b: Vec<f32>,
    /// `v.blk.N.attn_out.weight`, `[hidden][hidden]`.
    pub out_w: Vec<f32>,
    /// `v.blk.N.attn_out.bias`.
    pub out_b: Vec<f32>,
    /// `v.blk.N.ls1.weight` — LayerScale on the attention branch.
    pub ls1: Vec<f32>,
    /// `v.blk.N.ln2.weight` — LayerNorm before the MLP.
    pub ln2_w: Vec<f32>,
    /// `v.blk.N.ln2.bias`.
    pub ln2_b: Vec<f32>,
    /// `v.blk.N.ffn_up.weight`, `[ffn][hidden]`.
    pub up_w: Vec<f32>,
    /// `v.blk.N.ffn_up.bias`, `[ffn]`.
    pub up_b: Vec<f32>,
    /// `v.blk.N.ffn_down.weight`, `[hidden][ffn]`.
    pub down_w: Vec<f32>,
    /// `v.blk.N.ffn_down.bias`.
    pub down_b: Vec<f32>,
    /// `v.blk.N.ls2.weight` — LayerScale on the MLP branch.
    pub ls2: Vec<f32>,
}

/// The whole tower's weights, fp32. See the module docs for the layout.
#[derive(Debug, Clone)]
pub struct InklingVisionWeights {
    /// `v.class_embd` — the class token, `[hidden]`.
    pub class_embed: Vec<f32>,
    /// `v.class_position_embd.weight` — added to the class token, `[hidden]`.
    pub class_pos_embed: Vec<f32>,
    /// `v.patch_embd.weight`, `[hidden][3 * patch * patch]`; one row's
    /// layout is `c * patch^2 + ky * patch + kx`. There is no bias.
    pub patch_embed: Vec<f32>,
    /// `v.position_embd.weight`, `[edge * edge][hidden]` — the learned
    /// grid, row-major: position `row * edge + col`.
    pub pos_embed: Vec<f32>,
    /// `v.pre_ln.weight` — LayerNorm ahead of the first block.
    pub pre_ln_w: Vec<f32>,
    /// `v.pre_ln.bias`.
    pub pre_ln_b: Vec<f32>,
    /// The blocks, in order.
    pub blocks: Vec<InklingVisionBlockWeights>,
    /// `mm.1.weight`, `[4 * hidden][4 * hidden]`. No bias.
    pub mm1_w: Vec<f32>,
    /// `mm.2.weight`, `[projection_dim][4 * hidden]`. No bias.
    pub mm2_w: Vec<f32>,
}

/// Why an mmproj file could not be loaded as an `inkling2v` tower.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum InklingVisionLoadError {
    /// Tensors are missing or misshapen, or the file is another projector.
    #[error("mmproj does not match the inkling2v schema: {}", .0.join("; "))]
    Schema(Vec<String>),
    /// The file has tensors the reference would ignore but llama.cpp's
    /// graph would use (a patch bias, a post-LayerNorm, projector biases).
    #[error("mmproj holds tensors the inkling2v reference does not implement: {}", .0.join(", "))]
    Unclaimed(Vec<String>),
    /// A tensor is neither F32 nor F16.
    #[error("tensor `{name}` has type {found:?}; only F32 and F16 are read")]
    TensorType {
        /// The tensor.
        name: String,
        /// Its stored type.
        found: GgmlType,
    },
    /// The file ends before a tensor's data does.
    #[error("tensor `{0}` has no readable data")]
    Truncated(String),
}

/// One tensor, widened to fp32.
fn tensor_f32(file: &GgufFile, name: &str) -> Result<Vec<f32>, InklingVisionLoadError> {
    let truncated = || InklingVisionLoadError::Truncated(name.to_string());
    let info = file.tensor(name).ok_or_else(truncated)?;
    let bytes = file.tensor_bytes(name).ok_or_else(truncated)?;
    match info.ggml_type {
        GgmlType::F32 => Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()),
        GgmlType::F16 => Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half::f16::from_le_bytes(*c).to_f32())
            .collect()),
        found => Err(InklingVisionLoadError::TensorType {
            name: name.to_string(),
            found,
        }),
    }
}

impl InklingVisionWeights {
    /// Load every tensor of an `inkling2v` mmproj, widened to fp32.
    ///
    /// `file` is checked against [`InklingVisionWeightSchema`] in both
    /// directions first: every tensor the tower needs, at the shape `cfg`
    /// implies, and none the reference would silently ignore.
    pub fn from_gguf(
        file: &GgufFile,
        cfg: &InklingVisionConfig,
    ) -> Result<Self, InklingVisionLoadError> {
        use InklingVisionRole::*;
        let schema = InklingVisionWeightSchema::new(cfg);
        schema.resolve(file).map_err(|errors| {
            InklingVisionLoadError::Schema(errors.iter().map(ToString::to_string).collect())
        })?;
        let unclaimed = schema.unclaimed(file);
        if !unclaimed.is_empty() {
            return Err(InklingVisionLoadError::Unclaimed(
                unclaimed.into_iter().map(String::from).collect(),
            ));
        }

        let load = |role: InklingVisionRole, layer: Option<u32>| {
            let spec = schema.find(role, layer).expect("role is in the schema");
            tensor_f32(file, &spec.name)
        };
        let blocks = (0..cfg.num_layers)
            .map(|l| {
                let t = |role| load(role, Some(l));
                Ok(InklingVisionBlockWeights {
                    ln1_w: t(Ln1Weight)?,
                    ln1_b: t(Ln1Bias)?,
                    q_w: t(AttnQWeight)?,
                    q_b: t(AttnQBias)?,
                    k_w: t(AttnKWeight)?,
                    k_b: t(AttnKBias)?,
                    v_w: t(AttnVWeight)?,
                    v_b: t(AttnVBias)?,
                    out_w: t(AttnOutWeight)?,
                    out_b: t(AttnOutBias)?,
                    ls1: t(Ls1)?,
                    ln2_w: t(Ln2Weight)?,
                    ln2_b: t(Ln2Bias)?,
                    up_w: t(FfnUpWeight)?,
                    up_b: t(FfnUpBias)?,
                    down_w: t(FfnDownWeight)?,
                    down_b: t(FfnDownBias)?,
                    ls2: t(Ls2)?,
                })
            })
            .collect::<Result<Vec<_>, InklingVisionLoadError>>()?;
        Ok(Self {
            class_embed: load(ClassEmbed, None)?,
            class_pos_embed: load(ClassPositionEmbed, None)?,
            patch_embed: load(PatchEmbed, None)?,
            pos_embed: load(PositionEmbed, None)?,
            pre_ln_w: load(PreLnWeight, None)?,
            pre_ln_b: load(PreLnBias, None)?,
            blocks,
            mm1_w: load(Projector1, None)?,
            mm2_w: load(Projector2, None)?,
        })
    }
}

/// Which arithmetic [`encode_tapped`] reproduces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InklingVisionNumerics {
    /// fp32 from the normalized pixels on: the graph with no half-precision
    /// step anywhere.
    Fp32,
    /// The graph as llama.cpp's CPU backend runs it with flash attention
    /// off: fp32, except that the normalized pixels are rounded to F16
    /// before the patch convolution.
    ///
    /// That rounding is in the graph, not the backend: `ggml_conv_2d`
    /// builds its `im2col` in F16 whenever the kernel is not BF16. It moves
    /// the patch embedding by up to 3e-3 and is the whole of the
    /// difference from [`Self::Fp32`]: the CPU backend multiplies an F16
    /// matrix against fp32 activations as they are (`ggml_vec_dot_f16_f32`).
    LlamaCpp,
    /// [`Self::LlamaCpp`] with flash attention on, which is what llama.cpp
    /// picks when the backend supports it: the graph additionally casts the
    /// rotated keys and the values to F16.
    LlamaCppFlashAttn,
}

/// Names a tap: the llama.cpp graph node and its value, `[rows][channels]`.
///
/// Per-block nodes carry the block index as llama.cpp formats it, e.g.
/// `layer_out-7`. The names [`encode_tapped`] reports are listed there.
pub type InklingVisionTap<'a> = &'a mut dyn FnMut(&str, &[f32]);

/// `sum_i a[i] * b[i]` in fp32 with a fixed summation order: eight
/// interleaved partial sums over the leading multiple of eight elements,
/// combined pairwise, plus the remaining elements in order.
///
/// The order is part of the reference — it is the same on every machine
/// and at every thread count — and it is not ggml's, whose SIMD width
/// decides its own.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot of unequal lengths");
    let mut lanes = [0.0f32; 8];
    let (a_blocks, a_tail) = a.as_chunks::<8>();
    let (b_blocks, b_tail) = b.as_chunks::<8>();
    for (x, y) in a_blocks.iter().zip(b_blocks) {
        for (lane, (x, y)) in lanes.iter_mut().zip(x.iter().zip(y)) {
            *lane += x * y;
        }
    }
    let tail: f32 = a_tail.iter().zip(b_tail).map(|(x, y)| x * y).sum();
    let [l0, l1, l2, l3, l4, l5, l6, l7] = lanes;
    (((l0 + l4) + (l1 + l5)) + ((l2 + l6) + (l3 + l7))) + tail
}

/// Fill `rows` rows of `width` values, each from its own index, on as many
/// threads as the host offers. Rows are independent, so the result does
/// not depend on the thread count.
fn par_rows(rows: usize, width: usize, fill: impl Fn(usize, &mut [f32]) + Sync) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * width];
    if out.is_empty() {
        return out;
    }
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    let per = rows.div_ceil(threads).max(1);
    let fill = &fill;
    std::thread::scope(|scope| {
        for (chunk, block) in out.chunks_mut(per * width).enumerate() {
            scope.spawn(move || {
                for (i, row) in block.chunks_mut(width).enumerate() {
                    fill(chunk * per + i, row);
                }
            });
        }
    });
    out
}

/// `y = W x + b` for every `n_in`-wide row of `x`; `w` is `[out][in]`.
pub fn linear(w: &[f32], bias: Option<&[f32]>, n_in: usize, x: &[f32]) -> Vec<f32> {
    assert!(w.len().is_multiple_of(n_in) && x.len().is_multiple_of(n_in));
    let n_out = w.len() / n_in;
    par_rows(x.len() / n_in, n_out, |r, y| {
        let row = &x[r * n_in..(r + 1) * n_in];
        for (o, y) in y.iter_mut().enumerate() {
            *y = dot(&w[o * n_in..(o + 1) * n_in], row) + bias.map_or(0.0, |b| b[o]);
        }
    })
}

/// LayerNorm with weight and bias over one row: `(x - mean) / sqrt(var +
/// eps) * w + b`, the variance being the biased (population) one.
///
/// Mean and variance are accumulated in f64.
pub fn layer_norm(x: &[f32], w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let var = x
        .iter()
        .map(|&v| (f64::from(v) - mean).powi(2))
        .sum::<f64>()
        / n;
    let inv = 1.0 / (var + f64::from(eps)).sqrt();
    x.iter()
        .zip(w.iter().zip(b))
        .map(|(&v, (&w, &b))| ((f64::from(v) - mean) * inv * f64::from(w) + f64::from(b)) as f32)
        .collect()
}

/// [`layer_norm`] over every `width`-wide row of `x`.
pub fn layer_norm_rows(x: &[f32], width: usize, w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    x.chunks_exact(width)
        .flat_map(|row| layer_norm(row, w, b, eps))
        .collect()
}

/// The error function, to a relative error of about `1e-15`.
///
/// Rust's standard library has none. This is the all-positive series
/// `erf(x) = 2/sqrt(pi) * exp(-x^2) * sum_n 2^n x^(2n+1) / (2n+1)!!`,
/// which has no cancellation and converges for every `x`; past `|x| = 6`
/// the result is `±1` to f64 precision.
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let a = x.abs();
    if a >= 6.0 {
        return x.signum();
    }
    let mut term = a;
    let mut sum = a;
    let mut n = 0.0f64;
    loop {
        n += 1.0;
        term *= 2.0 * a * a / (2.0 * n + 1.0);
        sum += term;
        if term <= sum * 1e-17 {
            break;
        }
    }
    let magnitude = (std::f64::consts::FRAC_2_SQRT_PI * (-a * a).exp() * sum).min(1.0);
    magnitude.copysign(x)
}

/// The exact GELU, `x/2 * (1 + erf(x / sqrt(2)))` — ggml's `GELU_ERF`, not
/// the tanh approximation its plain `GELU` is. Evaluated in f64.
pub fn gelu_erf(x: f32) -> f32 {
    let x = f64::from(x);
    (0.5 * x * (1.0 + erf(x * std::f64::consts::FRAC_1_SQRT_2))) as f32
}

/// [`gelu_erf`] over a whole buffer.
fn gelu_erf_all(x: &[f32]) -> Vec<f32> {
    const WIDTH: usize = 1024;
    let rows = x.len().div_ceil(WIDTH);
    let mut out = par_rows(rows, WIDTH, |r, y| {
        let src = &x[r * WIDTH..x.len().min((r + 1) * WIDTH)];
        for (y, &v) in y.iter_mut().zip(src) {
            *y = gelu_erf(v);
        }
    });
    out.truncate(x.len());
    out
}

/// The rotary position of sequence row `index` on a `grid_w`-wide patch
/// grid, as `(pos_w, pos_h)`.
///
/// Row 0 is the class token at `(0, 0)`; patch `(x, y)` — row
/// `1 + y * grid_w + x` — is at `(x + 1, y + 1)`. The patches are
/// one-indexed so that none shares the class token's position.
pub fn rope_position(index: usize, grid_w: u32) -> (u32, u32) {
    match index.checked_sub(1) {
        None => (0, 0),
        Some(p) => {
            let gw = grid_w as usize;
            ((p % gw) as u32 + 1, (p / gw) as u32 + 1)
        }
    }
}

/// 2D rotary embedding of one head vector, in place.
///
/// The first half of the head rotates by `pos_w` and the second by
/// `pos_h`. Within a half, adjacent pairs `(2i, 2i + 1)` rotate by
/// `pos * theta^(-i / pairs)` with `pairs = head_dim / 4`, the frequency
/// index restarting at the second half:
/// `x0' = x0 cos - x1 sin`, `x1' = x0 sin + x1 cos`.
///
/// That is `build_rope_2d(..., interleave_freq = false)`: two
/// `ggml_rope_ext` calls of `n_dims = head_dim / 2` in ggml's `NORMAL`
/// pairing, the second offset by half a head. Angles are computed in f64;
/// ggml's are fp32, with the frequency built by repeated multiplication.
pub fn rope_2d(head: &mut [f32], pos_w: u32, pos_h: u32, theta: f32) {
    assert!(
        head.len().is_multiple_of(4),
        "head must split into two halves of pairs"
    );
    let half = head.len() / 2;
    let pairs = half / 2;
    for (section, pos) in [(0, pos_w), (half, pos_h)] {
        for i in 0..pairs {
            let freq = f64::from(theta).powf(-(i as f64) / pairs as f64);
            let (sin, cos) = (f64::from(pos) * freq).sin_cos();
            let (sin, cos) = (sin as f32, cos as f32);
            let at = section + 2 * i;
            let (x0, x1) = (head[at], head[at + 1]);
            head[at] = x0 * cos - x1 * sin;
            head[at + 1] = x0 * sin + x1 * cos;
        }
    }
}

/// [`rope_2d`] on each of the `heads` head vectors of every `[hidden]` row
/// of `x`, row `r` at [`rope_position`]`(r, grid_w)`.
pub fn rope_rows(x: &[f32], hidden: usize, heads: usize, grid_w: u32, theta: f32) -> Vec<f32> {
    let d_head = hidden / heads;
    par_rows(x.len() / hidden, hidden, |r, row| {
        row.copy_from_slice(&x[r * hidden..(r + 1) * hidden]);
        let (pos_w, pos_h) = rope_position(r, grid_w);
        for head in row.chunks_exact_mut(d_head) {
            rope_2d(head, pos_w, pos_h, theta);
        }
    })
}

/// Unmasked multi-head softmax attention over every row:
/// `softmax(Q K^T / sqrt(d_head)) V` per head. `q`, `k` and `v` are
/// `[rows][heads * d_head]`; so is the result (llama.cpp's `kqv_out`),
/// before the output projection.
///
/// Scores and the weighted sum are fp32; the normalizer is summed in f64.
pub fn attention(q: &[f32], k: &[f32], v: &[f32], heads: usize, d_head: usize) -> Vec<f32> {
    let hidden = heads * d_head;
    let rows = q.len() / hidden;
    assert!(q.len() == rows * hidden && k.len() == q.len() && v.len() == q.len());
    let scale = 1.0 / (d_head as f32).sqrt();
    par_rows(rows, hidden, |i, out| {
        let mut probs = vec![0.0f32; rows];
        for h in 0..heads {
            let head = h * d_head..(h + 1) * d_head;
            let qh = &q[i * hidden..][head.clone()];
            for (j, p) in probs.iter_mut().enumerate() {
                *p = dot(qh, &k[j * hidden..][head.clone()]) * scale;
            }
            let max = probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut total = 0.0f64;
            for p in probs.iter_mut() {
                *p = (*p - max).exp();
                total += f64::from(*p);
            }
            let out = &mut out[head.clone()];
            for (j, &p) in probs.iter().enumerate() {
                let weight = (f64::from(p) / total) as f32;
                for (o, &value) in out.iter_mut().zip(&v[j * hidden..][head.clone()]) {
                    *o += weight * value;
                }
            }
        }
    })
}

/// The learned `edge × edge` position grid resized to `grid_w × grid_h`
/// patches, `[grid_h * grid_w][hidden]` in raster order, ready to add.
///
/// Bilinear with half-pixel centres and clamped edges, no antialiasing —
/// PyTorch's `F.interpolate(mode="bilinear", align_corners=False)`:
/// `src = (i + 0.5) * in / out - 0.5`. In fp32 and in ggml's operation
/// order (`ggml_compute_forward_upscale_f32`, `GGML_SCALE_MODE_BILINEAR`
/// with no flags), as `mtmd_graph::resize_position_embeddings` calls it.
///
/// llama.cpp skips the resize when the patch grid is `edge × edge`; this
/// formula returns the grid unchanged there, so there is no special case.
pub fn resized_position_embedding(
    pos_embed: &[f32],
    hidden: usize,
    edge: usize,
    grid_w: u32,
    grid_h: u32,
) -> Vec<f32> {
    assert_eq!(pos_embed.len(), edge * edge * hidden, "position grid size");
    // The source coordinate, its two clamped neighbours and the weight of
    // the second.
    let axis = |i: u32, out: u32| {
        let scale = out as f32 / edge as f32;
        let x = (i as f32 + 0.5) / scale - 0.5;
        let last = edge as i64 - 1;
        let x0 = (x.floor() as i64).clamp(0, last);
        let x1 = (x0 + 1).clamp(0, last);
        let d = (x - x0 as f32).clamp(0.0, 1.0);
        (x0 as usize, x1 as usize, d)
    };
    let mut out = Vec::with_capacity((grid_w * grid_h) as usize * hidden);
    for y in 0..grid_h {
        let (y0, y1, dy) = axis(y, grid_h);
        for x in 0..grid_w {
            let (x0, x1, dx) = axis(x, grid_w);
            let at = |row: usize, col: usize| &pos_embed[(row * edge + col) * hidden..][..hidden];
            let (a, b, c, d) = (at(y0, x0), at(y0, x1), at(y1, x0), at(y1, x1));
            for ch in 0..hidden {
                out.push(
                    a[ch] * (1.0 - dx) * (1.0 - dy)
                        + b[ch] * dx * (1.0 - dy)
                        + c[ch] * (1.0 - dx) * dy
                        + d[ch] * dx * dy,
                );
            }
        }
    }
    out
}

/// Patch embedding: the bias-free stride-`patch` convolution as one GEMM
/// over the patchified image. Returns `[grid_h * grid_w][hidden]` in raster
/// order (llama.cpp's `patch_embd`).
///
/// Under either llama.cpp numerics the pixels are rounded through F16
/// first, as the F16 `im2col` of `ggml_conv_2d` does.
pub fn patch_embed(
    cfg: &InklingVisionConfig,
    w: &InklingVisionWeights,
    image: &PreprocessedInklingImage,
    numerics: InklingVisionNumerics,
) -> Vec<f32> {
    let patch_len = cfg.patch_elems() as usize;
    assert_eq!(
        image.patches.len(),
        image.n_patches() * patch_len,
        "patch buffer size"
    );
    if numerics == InklingVisionNumerics::Fp32 {
        return linear(&w.patch_embed, None, patch_len, &image.patches);
    }
    let pixels: Vec<f32> = image
        .patches
        .iter()
        .map(|&v| round_through_f16(v))
        .collect();
    linear(&w.patch_embed, None, patch_len, &pixels)
}

/// The sequence entering the tower, `[1 + patches][hidden]` (llama.cpp's
/// `embeddings`): the class row, `class_embed + class_pos_embed`, then each
/// patch embedding plus its resized position embedding.
pub fn embeddings(
    cfg: &InklingVisionConfig,
    w: &InklingVisionWeights,
    patch_embd: &[f32],
    grid_w: u32,
    grid_h: u32,
) -> Vec<f32> {
    let hidden = cfg.hidden_size as usize;
    let pos = resized_position_embedding(
        &w.pos_embed,
        hidden,
        cfg.pos_grid_edge() as usize,
        grid_w,
        grid_h,
    );
    assert_eq!(patch_embd.len(), pos.len(), "patch embedding size");
    let class = w.class_embed.iter().zip(&w.class_pos_embed);
    class
        .map(|(a, b)| a + b)
        .chain(patch_embd.iter().zip(&pos).map(|(a, b)| a + b))
        .collect()
}

/// One block over the whole sequence `x`, `[rows][hidden]`, row 0 being
/// the class token: pre-LN attention with 2D RoPE on queries and keys, a
/// LayerScale and a residual, then a pre-LN erf-GELU MLP with the same.
///
/// `il` only labels the taps.
pub fn block(
    cfg: &InklingVisionConfig,
    w: &InklingVisionBlockWeights,
    x: &[f32],
    grid_w: u32,
    numerics: InklingVisionNumerics,
    il: usize,
    tap: InklingVisionTap<'_>,
) -> Vec<f32> {
    let hidden = cfg.hidden_size as usize;
    let heads = cfg.num_heads as usize;
    let mut tap = |name: &str, value: &[f32]| tap(&format!("{name}-{il}"), value);
    let scaled = |x: &[f32], scale: &[f32]| -> Vec<f32> {
        x.chunks_exact(hidden)
            .flat_map(|row| row.iter().zip(scale).map(|(v, s)| v * s))
            .collect()
    };
    let added =
        |a: &[f32], b: &[f32]| -> Vec<f32> { a.iter().zip(b).map(|(a, b)| a + b).collect() };

    let normed = layer_norm_rows(x, hidden, &w.ln1_w, &w.ln1_b, cfg.ln_eps);
    tap("layer_inp_normed", &normed);
    let q = linear(&w.q_w, Some(&w.q_b), hidden, &normed);
    let k = linear(&w.k_w, Some(&w.k_b), hidden, &normed);
    let mut v = linear(&w.v_w, Some(&w.v_b), hidden, &normed);
    tap("Qcur", &q);
    tap("Kcur", &k);
    tap("Vcur", &v);
    let q = rope_rows(&q, hidden, heads, grid_w, cfg.rope_theta);
    let mut k = rope_rows(&k, hidden, heads, grid_w, cfg.rope_theta);
    tap("Qcur_pos", &q);
    tap("Kcur_pos", &k);
    if numerics == InklingVisionNumerics::LlamaCppFlashAttn {
        for value in k.iter_mut().chain(v.iter_mut()) {
            *value = round_through_f16(*value);
        }
    }
    let mixed = attention(&q, &k, &v, heads, cfg.head_dim() as usize);
    tap("kqv_out", &mixed);
    let attn = linear(&w.out_w, Some(&w.out_b), hidden, &mixed);
    tap("attn_out", &attn);
    let attn = scaled(&attn, &w.ls1);
    tap("attn_out_scaled", &attn);
    let x = added(&attn, x);
    tap("ffn_inp", &x);

    let normed = layer_norm_rows(&x, hidden, &w.ln2_w, &w.ln2_b, cfg.ln_eps);
    tap("ffn_inp_normed", &normed);
    let up = linear(&w.up_w, Some(&w.up_b), hidden, &normed);
    let act = gelu_erf_all(&up);
    let down = linear(&w.down_w, Some(&w.down_b), cfg.ffn_size as usize, &act);
    tap("ffn_out", &down);
    let down = scaled(&down, &w.ls2);
    tap("ffn_out_scaled", &down);
    let x = added(&x, &down);
    tap("layer_out", &x);
    x
}

/// Concatenate each `merge × merge` window of patches into one row.
///
/// `patches` is `[grid_h * grid_w][hidden]` in raster order. Output row
/// `y' * (grid_w / merge) + x'` is the patches of that window in raster
/// order — for the 2×2 merge `[(2x', 2y'), (2x'+1, 2y'), (2x', 2y'+1),
/// (2x'+1, 2y'+1)]` — `[tokens][merge^2 * hidden]`: llama.cpp's
/// `pixel_shuffle` node (`build_patch_merge_permute`).
pub fn merge_patches(
    patches: &[f32],
    hidden: usize,
    grid_w: u32,
    grid_h: u32,
    merge: u32,
) -> Vec<f32> {
    let (gw, gh, m) = (grid_w as usize, grid_h as usize, merge as usize);
    assert_eq!(patches.len(), gw * gh * hidden, "patch sequence size");
    assert!(
        gw.is_multiple_of(m) && gh.is_multiple_of(m),
        "grid must be a multiple of the merge size"
    );
    let mut out = Vec::with_capacity(patches.len());
    for y in 0..gh / m {
        for x in 0..gw / m {
            for dy in 0..m {
                for dx in 0..m {
                    let p = (y * m + dy) * gw + x * m + dx;
                    out.extend_from_slice(&patches[p * hidden..(p + 1) * hidden]);
                }
            }
        }
    }
    out
}

/// The projector over merged rows `[tokens][4 * hidden]`: `mm.1`,
/// erf-GELU, `mm.2`, no biases. Returns `[tokens][projection_dim]`.
pub fn project(cfg: &InklingVisionConfig, w: &InklingVisionWeights, merged: &[f32]) -> Vec<f32> {
    let width = cfg.merger_input_dim() as usize;
    let hidden = linear(&w.mm1_w, None, width, merged);
    linear(&w.mm2_w, None, width, &gelu_erf_all(&hidden))
}

/// Encode one preprocessed image: `[tokens][projection_dim]`, tokens
/// row-major over the merged grid — the rows the language model consumes
/// as input embeddings, one per position, with nothing in between.
///
/// This is [`InklingVisionNumerics::LlamaCpp`]: the rows llama.cpp hands
/// its language model, F16 pixel rounding included, to within fp32
/// summation order.
pub fn encode(
    cfg: &InklingVisionConfig,
    w: &InklingVisionWeights,
    image: &PreprocessedInklingImage,
) -> Vec<f32> {
    encode_tapped(
        cfg,
        w,
        image,
        InklingVisionNumerics::LlamaCpp,
        &mut |_, _| {},
    )
}

/// [`encode`] under a chosen arithmetic, reporting intermediates.
///
/// `tap` sees, in order, the nodes llama.cpp names in its graph, each
/// `[rows][channels]`:
///
/// - `patch_embd` `[patches][hidden]`, `embeddings` and `pre_ln`
///   `[1 + patches][hidden]`;
/// - per block `N`, all `[1 + patches][hidden]`: `layer_inp_normed-N`,
///   `Qcur-N`, `Kcur-N`, `Vcur-N`, `Qcur_pos-N`, `Kcur_pos-N`, `kqv_out-N`,
///   `attn_out-N`, `attn_out_scaled-N`, `ffn_inp-N`, `ffn_inp_normed-N`,
///   `ffn_out-N`, `ffn_out_scaled-N`, `layer_out-N`;
/// - `encoder_out` `[1 + patches][hidden]`, `pixel_shuffle`
///   `[tokens][4 * hidden]` and `projected` `[tokens][projection_dim]`.
///
/// # Panics
///
/// If the image's patch buffer or grid does not fit `cfg`, or a weight has
/// the wrong size.
pub fn encode_tapped(
    cfg: &InklingVisionConfig,
    w: &InklingVisionWeights,
    image: &PreprocessedInklingImage,
    numerics: InklingVisionNumerics,
    tap: InklingVisionTap<'_>,
) -> Vec<f32> {
    let hidden = cfg.hidden_size as usize;
    let (grid_w, grid_h) = (image.grid_w, image.grid_h);
    assert_eq!(w.blocks.len(), cfg.num_layers as usize, "block count");

    let patch_embd = patch_embed(cfg, w, image, numerics);
    tap("patch_embd", &patch_embd);
    let x = embeddings(cfg, w, &patch_embd, grid_w, grid_h);
    tap("embeddings", &x);

    // The residual stream starts from the normalized sequence.
    let mut x = layer_norm_rows(&x, hidden, &w.pre_ln_w, &w.pre_ln_b, cfg.ln_eps);
    tap("pre_ln", &x);
    for (il, blk) in w.blocks.iter().enumerate() {
        x = block(cfg, blk, &x, grid_w, numerics, il, &mut *tap);
    }
    tap("encoder_out", &x);

    // The class row has done its work inside attention; it is not output.
    let merged = merge_patches(&x[hidden..], hidden, grid_w, grid_h, cfg.spatial_merge);
    tap("pixel_shuffle", &merged);
    let out = project(cfg, w, &merged);
    tap("projected", &out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Xorshift64Star as Rng;

    #[test]
    fn erf_matches_tabulated_values_far_below_fp32_resolution() {
        // Abramowitz & Stegun table 7.1 / high-precision references.
        for (x, want) in [
            (0.0, 0.0),
            (1e-8, 1.128_379_167_095_512_6e-8),
            (0.1, 0.112_462_916_018_284_89),
            (0.5, 0.520_499_877_813_046_5),
            (1.0, 0.842_700_792_949_714_9),
            (2.0, 0.995_322_265_018_952_7),
            (3.5, 0.999_999_256_901_627_7),
            (5.0, 0.999_999_999_998_462_5),
            (7.0, 1.0),
        ] {
            let got = erf(x);
            assert!(
                (got - want).abs() <= 2e-15 * want.abs(),
                "erf({x}) = {got:e}, want {want:e}"
            );
            assert_eq!(erf(-x), -got, "erf is odd");
        }
        assert!(erf(f64::NAN).is_nan());
    }

    #[test]
    fn gelu_erf_is_the_exact_gelu_not_the_tanh_one() {
        // Phi(1) = 0.8413447460685429; the tanh approximation gives
        // 0.841192 at x = 1, a difference fp32 resolves.
        assert!((f64::from(gelu_erf(1.0)) - 0.841_344_746_068_542_9).abs() < 3e-8);
        assert_eq!(gelu_erf(0.0), 0.0);
        assert!((f64::from(gelu_erf(-1.0)) + 0.158_655_253_931_457_05).abs() < 1e-8);
        assert_eq!(gelu_erf(12.0), 12.0);
        assert_eq!(gelu_erf(-12.0), 0.0);
    }

    #[test]
    fn dot_matches_an_f64_sum_and_handles_a_ragged_tail() {
        let mut rng = Rng::new(7);
        for len in [0usize, 1, 7, 8, 9, 64, 1027] {
            let a: Vec<f32> = (0..len).map(|_| rng.next_f32() - 0.5).collect();
            let b: Vec<f32> = (0..len).map(|_| rng.next_f32() - 0.5).collect();
            let want: f64 = a
                .iter()
                .zip(&b)
                .map(|(x, y)| f64::from(*x) * f64::from(*y))
                .sum();
            assert!((f64::from(dot(&a, &b)) - want).abs() < 1e-5, "len {len}");
        }
    }

    #[test]
    fn layer_norm_standardizes_then_scales_and_shifts() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        // mean 2.5, population variance 1.25.
        let out = layer_norm(&x, &[2.0; 4], &[0.5; 4], 0.0);
        let s = 1.25f64.sqrt();
        for (o, v) in out.iter().zip(x) {
            let want = (f64::from(v) - 2.5) / s * 2.0 + 0.5;
            assert!((f64::from(*o) - want).abs() < 1e-6);
        }
    }

    #[test]
    fn rope_positions_put_the_class_token_at_the_origin() {
        assert_eq!(rope_position(0, 5), (0, 0));
        assert_eq!(rope_position(1, 5), (1, 1));
        assert_eq!(rope_position(5, 5), (5, 1));
        assert_eq!(rope_position(6, 5), (1, 2));
    }

    #[test]
    fn rope_rotates_the_first_half_by_x_and_the_second_by_y() {
        let mut head: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin()).collect();
        let before = head.clone();

        // The class token's position is the identity.
        rope_2d(&mut head, 0, 0, 10_000.0);
        assert_eq!(head, before);

        // x only: the second half is untouched, and pair i of the first
        // half turns by 3 * 10000^(-i/16).
        rope_2d(&mut head, 3, 0, 10_000.0);
        assert_eq!(head[32..], before[32..]);
        for i in 0..16 {
            let angle = 3.0 * 10_000f64.powf(-(i as f64) / 16.0);
            let (x0, x1) = (f64::from(before[2 * i]), f64::from(before[2 * i + 1]));
            assert!((f64::from(head[2 * i]) - (x0 * angle.cos() - x1 * angle.sin())).abs() < 1e-6);
            assert!(
                (f64::from(head[2 * i + 1]) - (x0 * angle.sin() + x1 * angle.cos())).abs() < 1e-6
            );
        }

        // y only: the mirror image, with the frequency index restarting.
        let mut head = before.clone();
        rope_2d(&mut head, 0, 5, 10_000.0);
        assert_eq!(head[..32], before[..32]);
        let angle = 5.0 * 10_000f64.powf(-1.0 / 16.0);
        let (x0, x1) = (f64::from(before[34]), f64::from(before[35]));
        assert!((f64::from(head[34]) - (x0 * angle.cos() - x1 * angle.sin())).abs() < 1e-6);
    }

    #[test]
    fn rope_preserves_the_norm_of_every_pair() {
        let mut rng = Rng::new(3);
        let mut head: Vec<f32> = (0..64).map(|_| rng.next_f32() - 0.5).collect();
        let before = head.clone();
        rope_2d(&mut head, 17, 29, 10_000.0);
        for (a, b) in head
            .as_chunks::<2>()
            .0
            .iter()
            .zip(before.as_chunks::<2>().0)
        {
            let (na, nb) = (a[0].hypot(a[1]), b[0].hypot(b[1]));
            assert!((na - nb).abs() < 1e-6);
        }
    }

    /// A position grid whose channel 0 is the column and channel 1 the row.
    fn coordinate_grid(edge: usize) -> Vec<f32> {
        let mut grid = Vec::new();
        for row in 0..edge {
            for col in 0..edge {
                grid.extend([col as f32, row as f32]);
            }
        }
        grid
    }

    #[test]
    fn position_grid_at_its_native_size_is_returned_unchanged() {
        let mut rng = Rng::new(11);
        let grid: Vec<f32> = (0..4 * 4 * 3).map(|_| rng.next_f32()).collect();
        assert_eq!(resized_position_embedding(&grid, 3, 4, 4, 4), grid);
    }

    #[test]
    fn position_resize_uses_half_pixel_centres_and_clamps_the_edges() {
        let grid = coordinate_grid(4);
        // 4 -> 8 columns: src = (i + 0.5) / 2 - 0.5 = -0.25, 0.25, 0.75, ...
        // clamped to [0, 3]. Rows unchanged (4 -> 4).
        let out = resized_position_embedding(&grid, 2, 4, 8, 4);
        let cols: Vec<f32> = out
            .as_chunks::<2>()
            .0
            .iter()
            .take(8)
            .map(|v| v[0])
            .collect();
        assert_eq!(cols, [0.0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.0]);
        // Raster order: second output row carries source row 1.
        assert_eq!(out[8 * 2 + 1], 1.0);

        // 4 -> 2 rows: src = (i + 0.5) * 2 - 0.5 = 0.5, 2.5 — plain
        // two-tap averages, no antialiasing.
        let out = resized_position_embedding(&grid, 2, 4, 4, 2);
        let rows: Vec<f32> = out
            .as_chunks::<2>()
            .0
            .iter()
            .step_by(4)
            .map(|v| v[1])
            .collect();
        assert_eq!(rows, [0.5, 2.5]);
    }

    #[test]
    fn attention_with_equal_scores_averages_the_values() {
        // Zero queries: every score ties, so each head's output is the
        // mean of the values.
        let (heads, d_head, rows) = (2, 4, 3);
        let hidden = heads * d_head;
        let q = vec![0.0f32; rows * hidden];
        let mut rng = Rng::new(5);
        let k: Vec<f32> = (0..rows * hidden).map(|_| rng.next_f32()).collect();
        let v: Vec<f32> = (0..rows * hidden).map(|_| rng.next_f32()).collect();
        let out = attention(&q, &k, &v, heads, d_head);
        for c in 0..hidden {
            let mean = (0..rows).map(|j| v[j * hidden + c]).sum::<f32>() / rows as f32;
            for i in 0..rows {
                assert!((out[i * hidden + c] - mean).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn attention_scales_scores_by_the_root_of_the_head_width() {
        // One head of width 4, two rows. q = k_0 = [2, 0, 0, 0], k_1 = 0:
        // scores (4, 0) / sqrt(4) = (2, 0).
        let q = [2.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let k = q;
        let v = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let out = attention(&q, &k, &v, 1, 4);
        let p0 = 2f32.exp() / (2f32.exp() + 1.0);
        assert!((out[0] - p0).abs() < 1e-6);
        assert!((out[1] - (1.0 - p0)).abs() < 1e-6);
        // The second row's query is zero: a tie.
        assert!((out[4] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn merge_concatenates_each_window_in_raster_order() {
        // A 4 x 2 grid of 1-wide patches numbered in raster order.
        let patches: Vec<f32> = (0..8).map(|p| p as f32).collect();
        let merged = merge_patches(&patches, 1, 4, 2, 2);
        assert_eq!(merged, [0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0]);
    }

    #[test]
    fn linear_is_row_major_out_by_in() {
        // [[1, 2, 3], [4, 5, 6]] x [1, 0, -1] + [10, 20], twice.
        let w = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = [1.0, 0.0, -1.0, 0.0, 1.0, 0.0];
        assert_eq!(
            linear(&w, Some(&[10.0, 20.0]), 3, &x),
            [8.0, 18.0, 12.0, 25.0]
        );
        assert_eq!(linear(&w, None, 3, &x[..3]), [-2.0, -2.0]);
    }

    #[test]
    fn only_the_llama_cpp_numerics_round_the_pixels_to_half_precision() {
        // One patch whose every value is 1 + 2^-12 — not an F16 value, it
        // rounds to 1 — against a kernel of ones.
        let (cfg, mut w) = tiny();
        w.patch_embed.fill(1.0);
        let image = PreprocessedInklingImage {
            patches: vec![1.0 + 1.0 / 4096.0; 12],
            grid_w: 1,
            grid_h: 1,
        };
        let fp32 = patch_embed(&cfg, &w, &image, InklingVisionNumerics::Fp32);
        let rounded = patch_embed(&cfg, &w, &image, InklingVisionNumerics::LlamaCpp);
        assert_eq!(fp32[0], 12.0 * (1.0 + 1.0 / 4096.0));
        assert_eq!(rounded[0], 12.0);
    }

    /// A small random tower: 2 blocks, 16 wide, 2 heads of 8, patch 2.
    fn tiny() -> (InklingVisionConfig, InklingVisionWeights) {
        let cfg = InklingVisionConfig {
            num_layers: 2,
            hidden_size: 16,
            num_heads: 2,
            ffn_size: 24,
            image_size: 8,
            patch_size: 2,
            projection_dim: 10,
            ..InklingVisionConfig::inkling2_mini()
        };
        let mut rng = Rng::new(2024);
        let mut t = |n: usize| -> Vec<f32> { (0..n).map(|_| rng.next_f32() - 0.5).collect() };
        let (h, f) = (16, 24);
        let blocks = (0..2)
            .map(|_| InklingVisionBlockWeights {
                ln1_w: t(h),
                ln1_b: t(h),
                q_w: t(h * h),
                q_b: t(h),
                k_w: t(h * h),
                k_b: t(h),
                v_w: t(h * h),
                v_b: t(h),
                out_w: t(h * h),
                out_b: t(h),
                ls1: t(h),
                ln2_w: t(h),
                ln2_b: t(h),
                up_w: t(f * h),
                up_b: t(f),
                down_w: t(h * f),
                down_b: t(h),
                ls2: t(h),
            })
            .collect();
        let w = InklingVisionWeights {
            class_embed: t(h),
            class_pos_embed: t(h),
            patch_embed: t(h * 12),
            pos_embed: t(16 * h),
            pre_ln_w: t(h),
            pre_ln_b: t(h),
            blocks,
            mm1_w: t(4 * h * 4 * h),
            mm2_w: t(10 * 4 * h),
        };
        (cfg, w)
    }

    fn tiny_image(cfg: &InklingVisionConfig, grid_w: u32, grid_h: u32) -> PreprocessedInklingImage {
        let mut rng = Rng::new(99);
        let n = (grid_w * grid_h * cfg.patch_elems()) as usize;
        PreprocessedInklingImage {
            patches: (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect(),
            grid_w,
            grid_h,
        }
    }

    #[test]
    fn encode_returns_one_row_per_merged_window_and_taps_every_node() {
        let (cfg, w) = tiny();
        let image = tiny_image(&cfg, 6, 4);
        let mut seen: Vec<(String, usize)> = Vec::new();
        let out = encode_tapped(
            &cfg,
            &w,
            &image,
            InklingVisionNumerics::Fp32,
            &mut |name, value| seen.push((name.to_string(), value.len())),
        );
        assert_eq!(out.len(), 6 * 10);
        assert!(out.iter().all(|v| v.is_finite()));

        assert_eq!(seen.len(), 3 + 2 * 14 + 3);
        assert_eq!(seen[0], ("patch_embd".into(), 24 * 16));
        assert_eq!(seen[1], ("embeddings".into(), 25 * 16));
        assert_eq!(seen[3].0, "layer_inp_normed-0");
        assert_eq!(seen[3 + 14 + 13].0, "layer_out-1");
        let tail: Vec<&str> = seen[31..].iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(tail, ["encoder_out", "pixel_shuffle", "projected"]);
        assert_eq!(seen[32].1, 6 * 64);
    }

    #[test]
    fn the_class_token_reaches_the_output_only_through_attention() {
        // Changing the class embedding must change the patches' output (it
        // is attended to), and the output must carry no row for it.
        let (cfg, mut w) = tiny();
        let image = tiny_image(&cfg, 4, 4);
        let base = encode(&cfg, &w, &image);
        assert_eq!(base.len(), 4 * 10);
        w.class_embed[0] += 1.0;
        assert_ne!(encode(&cfg, &w, &image), base);
    }

    #[test]
    fn patch_positions_matter_through_the_rotary_embedding_alone() {
        // With a constant position grid the only thing telling two patch
        // layouts apart is RoPE: the same 8 patches as 4x2 and as 2x4 must
        // encode differently before the merge.
        let (cfg, mut w) = tiny();
        let row: Vec<f32> = w.pos_embed[..16].to_vec();
        for chunk in w.pos_embed.as_chunks_mut::<16>().0 {
            chunk.copy_from_slice(&row);
        }
        let wide = tiny_image(&cfg, 4, 2);
        let tall = PreprocessedInklingImage {
            grid_w: 2,
            grid_h: 4,
            ..wide.clone()
        };
        let run = |image: &PreprocessedInklingImage| {
            let mut encoder_out = Vec::new();
            encode_tapped(
                &cfg,
                &w,
                image,
                InklingVisionNumerics::Fp32,
                &mut |name, value| {
                    if name == "encoder_out" {
                        encoder_out = value.to_vec();
                    }
                },
            );
            encoder_out
        };
        let (a, b) = (run(&wide), run(&tall));
        assert_eq!(a.len(), b.len());
        // Patch 0 is at (1, 1) in both; patch 1 is at (2, 1) or (1, 2).
        assert_ne!(a[2 * 16..3 * 16], b[2 * 16..3 * 16]);
    }

    #[test]
    fn the_three_numerics_differ_by_half_precision_rounding_and_no_more() {
        let (cfg, w) = tiny();
        let image = tiny_image(&cfg, 4, 4);
        let run = |numerics| encode_tapped(&cfg, &w, &image, numerics, &mut |_, _| {});
        let fp32 = run(InklingVisionNumerics::Fp32);
        let llama = run(InklingVisionNumerics::LlamaCpp);
        let flash = run(InklingVisionNumerics::LlamaCppFlashAttn);
        assert_eq!(llama, encode(&cfg, &w, &image));
        for (name, a, b) in [("pixels", &llama, &fp32), ("keys/values", &flash, &llama)] {
            assert_ne!(a, b, "{name}");
            let result = crate::compare::compare(a, b);
            assert!(result.max_abs_error < 2e-2, "{name}: {result:?}");
            assert!(result.cosine_similarity > 0.9999, "{name}: {result:?}");
        }
    }
}
