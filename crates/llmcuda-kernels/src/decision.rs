//! Scalar reference for Clef's joint schema head.
//!
//! Ported from `JointSchemaHead.forward` in Cloudflare's
//! `joint_schema_model.py` (the `Cloudflare/clef-flash` model repository) and
//! cross-read against llama.cpp's `llama_model_clef::graph::build_head`
//! (`src/models/clef.cpp`), which is the layout the GGUF tensors follow.
//!
//! The head reads the backbone's final normed hidden state at **every** prompt
//! position and returns one logit per answer option. Questions and options are
//! spans of prompt tokens; the spans, not any generated text, are the output's
//! index.
//!
//! Everything here is f64 internally and f32 at the edges, so it is an oracle
//! for the device head rather than a model of its rounding.

/// `torch.nn.LayerNorm`'s affine transform.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    /// Per-feature scale.
    pub weight: Vec<f32>,
    /// Per-feature shift.
    pub bias: Vec<f32>,
}

/// `y = x W^T + b`, with `weight` stored `[out][in]` row-major.
#[derive(Debug, Clone)]
pub struct Linear {
    /// Row-major `[out][in]`.
    pub weight: Vec<f32>,
    /// `[out]`, absent for the head's bias-free input projections.
    pub bias: Option<Vec<f32>>,
    /// Output features.
    pub out: usize,
    /// Input features.
    pub inp: usize,
}

/// One `torch.nn.MultiheadAttention`: four projections, no mask.
#[derive(Debug, Clone)]
pub struct Attention {
    /// Query projection.
    pub q: Linear,
    /// Key projection.
    pub k: Linear,
    /// Value projection.
    pub v: Linear,
    /// Output projection.
    pub o: Linear,
}

/// One head block. The first `routing_layers` are evidence-routing layers
/// (cross-attention only, with a separate memory norm); the rest are
/// `TransformerDecoderLayer(norm_first=True)` (self, cross, feed-forward).
#[derive(Debug, Clone)]
pub struct HeadLayer {
    /// Self-attention norm and projections; joint layers only.
    pub self_attn: Option<(LayerNorm, Attention)>,
    /// Query-side norm of the cross-attention.
    pub cross_norm: LayerNorm,
    /// Memory-side norm; routing layers only.
    pub cross_norm_kv: Option<LayerNorm>,
    /// Cross-attention over the projected memory.
    pub cross_attn: Attention,
    /// Feed-forward pre-norm.
    pub ffn_norm: LayerNorm,
    /// `width -> feedforward`, followed by exact GELU.
    pub ffn_up: Linear,
    /// `feedforward -> width`.
    pub ffn_down: Linear,
}

/// The whole head, dequantized.
#[derive(Debug, Clone)]
pub struct HeadWeights {
    /// Backbone width.
    pub hidden: usize,
    /// Head width.
    pub width: usize,
    /// Attention heads inside the head.
    pub heads: usize,
    /// `torch.nn.LayerNorm` epsilon.
    pub eps: f32,
    /// LayerNorm over the backbone's final normed hidden states.
    pub hidden_norm: LayerNorm,
    /// `hidden -> width`, every position.
    pub proj_memory: Linear,
    /// `hidden -> width`, question span means.
    pub proj_question: Linear,
    /// `hidden -> width`, a question's mean added to each of its options.
    pub proj_option_question: Linear,
    /// `hidden -> width`, the last position.
    pub proj_global: Linear,
    /// `hidden -> width`, option span means.
    pub proj_option_context: Linear,
    /// `hidden -> width`, option output-embedding means.
    pub proj_option_lexical: Linear,
    /// `[3][width]`: noul, choice, score.
    pub type_embd: Vec<f32>,
    /// Routing layers first, then joint layers.
    pub layers: Vec<HeadLayer>,
    /// Number of leading routing layers in [`Self::layers`].
    pub routing_layers: usize,
    /// Norm of each question's softmax-weighted option summary.
    pub option_summary_norm: LayerNorm,
    /// Final norm of the question vectors.
    pub field_norm: LayerNorm,
    /// Final norm of the routed option vectors.
    pub option_norm: LayerNorm,
    /// `4 * width -> width`, then exact GELU.
    pub scorer: Linear,
    /// `width -> 1`.
    pub scorer_out: Linear,
    /// `[exp(prior), exp(joint), sigmoid(gate)]` as stored by the converter.
    pub scales: [f32; 3],
}

/// Question type ids, in the order the type embedding is indexed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionType {
    /// True/false.
    Noul = 0,
    /// One of named options.
    Choice = 1,
    /// An ordered level.
    Score = 2,
}

/// A question's instruction span, `[start, end)` in prompt positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuestionSpan {
    /// Selects the type embedding.
    pub kind: QuestionType,
    /// First position of the instruction text.
    pub start: usize,
    /// One past its last position.
    pub end: usize,
}

/// An option's span, owned by `question`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OptionSpan {
    /// Index into the question list.
    pub question: usize,
    /// First position of the option text.
    pub start: usize,
    /// One past its last position.
    pub end: usize,
}

fn layer_norm(x: &[f64], n: &LayerNorm, eps: f32) -> Vec<f64> {
    let w = x.len() as f64;
    let mean = x.iter().sum::<f64>() / w;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / w;
    let inv = 1.0 / (var + f64::from(eps)).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, v)| (v - mean) * inv * f64::from(n.weight[i]) + f64::from(n.bias[i]))
        .collect()
}

fn linear(x: &[f64], l: &Linear) -> Vec<f64> {
    assert_eq!(x.len(), l.inp);
    (0..l.out)
        .map(|o| {
            let row = &l.weight[o * l.inp..(o + 1) * l.inp];
            let dot: f64 = row.iter().zip(x).map(|(&w, &v)| f64::from(w) * v).sum();
            dot + l.bias.as_ref().map_or(0.0, |b| f64::from(b[o]))
        })
        .collect()
}

/// `erf` to ~1e-7 relative error (Numerical Recipes' Chebyshev `erfc`).
pub fn erf(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let poly = -z * z - 1.265_512_23
        + t * (1.000_023_68
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_203_98
                                + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))));
    let erfc = t * poly.exp();
    if x >= 0.0 { 1.0 - erfc } else { erfc - 1.0 }
}

/// PyTorch's default `GELU`: `x/2 * (1 + erf(x/sqrt(2)))`.
pub fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + erf(x * std::f64::consts::FRAC_1_SQRT_2))
}

fn attention(q_in: &[Vec<f64>], kv_in: &[Vec<f64>], a: &Attention, heads: usize) -> Vec<Vec<f64>> {
    let q: Vec<_> = q_in.iter().map(|x| linear(x, &a.q)).collect();
    let k: Vec<_> = kv_in.iter().map(|x| linear(x, &a.k)).collect();
    let v: Vec<_> = kv_in.iter().map(|x| linear(x, &a.v)).collect();
    let width = a.q.out;
    let d = width / heads;
    let scale = 1.0 / (d as f64).sqrt();
    q.iter()
        .map(|qr| {
            let mut out = vec![0.0; width];
            for h in 0..heads {
                let r = h * d..(h + 1) * d;
                let scores: Vec<f64> = k
                    .iter()
                    .map(|kr| {
                        qr[r.clone()]
                            .iter()
                            .zip(&kr[r.clone()])
                            .map(|(a, b)| a * b)
                            .sum::<f64>()
                            * scale
                    })
                    .collect();
                let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f64 = e.iter().sum();
                for (j, vr) in v.iter().enumerate() {
                    for (o, x) in out[r.clone()].iter_mut().zip(&vr[r.clone()]) {
                        *o += e[j] / sum * x;
                    }
                }
            }
            linear(&out, &a.o)
        })
        .collect()
}

fn ffn(x: &[f64], l: &HeadLayer, eps: f32) -> Vec<f64> {
    let up: Vec<f64> = linear(&layer_norm(x, &l.ffn_norm, eps), &l.ffn_up)
        .into_iter()
        .map(gelu)
        .collect();
    linear(&up, &l.ffn_down)
}

fn add(a: &mut [f64], b: &[f64]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

fn mean_rows(rows: &[Vec<f64>], start: usize, end: usize) -> Vec<f64> {
    assert!(
        start < end && end <= rows.len(),
        "empty or out-of-range span"
    );
    let mut m = vec![0.0; rows[0].len()];
    for r in &rows[start..end] {
        add(&mut m, r);
    }
    let n = (end - start) as f64;
    m.iter_mut().for_each(|v| *v /= n);
    m
}

fn l2_normalize(x: &[f64]) -> Vec<f64> {
    let n = x.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
    x.iter().map(|v| v / n).collect()
}

/// Scores, one per option in `options` order.
///
/// `hidden` is `[positions][hidden]`: the backbone's final RMS-normed state.
/// `tokens` are the prompt's ids, and `output_row(id)` returns the
/// dequantized row of the untied output embedding for that id.
pub fn joint_schema_head(
    w: &HeadWeights,
    hidden: &[f32],
    tokens: &[i32],
    questions: &[QuestionSpan],
    options: &[OptionSpan],
    output_row: impl Fn(i32) -> Vec<f32>,
) -> Vec<f32> {
    let positions = tokens.len();
    assert_eq!(hidden.len(), positions * w.hidden);
    assert!(!questions.is_empty() && !options.is_empty());
    let eps = w.eps;
    let normed: Vec<Vec<f64>> = hidden
        .chunks_exact(w.hidden)
        .map(|r| {
            let r: Vec<f64> = r.iter().map(|&v| f64::from(v)).collect();
            layer_norm(&r, &w.hidden_norm, eps)
        })
        .collect();
    let memory: Vec<Vec<f64>> = normed.iter().map(|r| linear(r, &w.proj_memory)).collect();
    let global = normed[positions - 1].clone();
    let question_vec: Vec<Vec<f64>> = questions
        .iter()
        .map(|q| mean_rows(&normed, q.start, q.end))
        .collect();
    let context: Vec<Vec<f64>> = options
        .iter()
        .map(|o| mean_rows(&normed, o.start, o.end))
        .collect();
    let lexical: Vec<Vec<f64>> = options
        .iter()
        .map(|o| {
            let rows: Vec<Vec<f64>> = tokens[o.start..o.end]
                .iter()
                .map(|&t| output_row(t).into_iter().map(f64::from).collect())
                .collect();
            mean_rows(&rows, 0, rows.len())
        })
        .collect();

    let mut opts: Vec<Vec<f64>> = options
        .iter()
        .enumerate()
        .map(|(i, o)| {
            let mut v = linear(&context[i], &w.proj_option_context);
            add(&mut v, &linear(&lexical[i], &w.proj_option_lexical));
            add(
                &mut v,
                &linear(&question_vec[o.question], &w.proj_option_question),
            );
            v
        })
        .collect();

    for l in &w.layers[..w.routing_layers] {
        let kv_norm = l
            .cross_norm_kv
            .as_ref()
            .expect("routing layer has a memory norm");
        let mem: Vec<Vec<f64>> = memory.iter().map(|m| layer_norm(m, kv_norm, eps)).collect();
        let q: Vec<Vec<f64>> = opts
            .iter()
            .map(|o| layer_norm(o, &l.cross_norm, eps))
            .collect();
        let routed = attention(&q, &mem, &l.cross_attn, w.heads);
        for (o, r) in opts.iter_mut().zip(&routed) {
            add(o, r);
        }
        for o in opts.iter_mut() {
            let f = ffn(o, l, eps);
            add(o, &f);
        }
    }

    let scale = 1.0 / (w.width as f64).sqrt();
    let global_proj = linear(&global, &w.proj_global);
    let mut fields: Vec<Vec<f64>> = questions
        .iter()
        .enumerate()
        .map(|(qi, q)| {
            let base = linear(&question_vec[qi], &w.proj_question);
            let own: Vec<usize> = (0..options.len())
                .filter(|&i| options[i].question == qi)
                .collect();
            assert!(!own.is_empty(), "every question needs an option");
            let logits: Vec<f64> = own
                .iter()
                .map(|&i| opts[i].iter().zip(&base).map(|(a, b)| a * b).sum::<f64>() * scale)
                .collect();
            let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = logits.iter().map(|s| (s - max).exp()).collect();
            let sum: f64 = e.iter().sum();
            let mut summary = vec![0.0; w.width];
            for (k, &i) in own.iter().enumerate() {
                for (s, x) in summary.iter_mut().zip(&opts[i]) {
                    *s += e[k] / sum * x;
                }
            }
            let mut f = base;
            add(&mut f, &layer_norm(&summary, &w.option_summary_norm, eps));
            add(&mut f, &global_proj);
            let t = q.kind as usize;
            let ty: Vec<f64> = w.type_embd[t * w.width..(t + 1) * w.width]
                .iter()
                .map(|&v| f64::from(v))
                .collect();
            add(&mut f, &ty);
            f
        })
        .collect();

    for l in &w.layers[w.routing_layers..] {
        let (self_norm, self_attn) = l
            .self_attn
            .as_ref()
            .expect("joint layer has self-attention");
        let cur: Vec<Vec<f64>> = fields
            .iter()
            .map(|f| layer_norm(f, self_norm, eps))
            .collect();
        let s = attention(&cur, &cur, self_attn, w.heads);
        for (f, r) in fields.iter_mut().zip(&s) {
            add(f, r);
        }
        let cur: Vec<Vec<f64>> = fields
            .iter()
            .map(|f| layer_norm(f, &l.cross_norm, eps))
            .collect();
        let c = attention(&cur, &memory, &l.cross_attn, w.heads);
        for (f, r) in fields.iter_mut().zip(&c) {
            add(f, r);
        }
        for f in fields.iter_mut() {
            let x = ffn(f, l, eps);
            add(f, &x);
        }
    }
    let fields: Vec<Vec<f64>> = fields
        .iter()
        .map(|f| layer_norm(f, &w.field_norm, eps))
        .collect();

    options
        .iter()
        .enumerate()
        .map(|(i, o)| {
            let mut anchor = question_vec[o.question].clone();
            add(&mut anchor, &global);
            let anchor = l2_normalize(&anchor);
            let lex = l2_normalize(&lexical[i]);
            let prior: f64 = lex.iter().zip(&anchor).map(|(a, b)| a * b).sum();

            let opt = layer_norm(&opts[i], &w.option_norm, eps);
            let field = &fields[o.question];
            let cosine: f64 = l2_normalize(field)
                .iter()
                .zip(&l2_normalize(&opt))
                .map(|(a, b)| a * b)
                .sum();
            let mut features = Vec::with_capacity(4 * w.width);
            features.extend_from_slice(field);
            features.extend_from_slice(&opt);
            features.extend(field.iter().zip(&opt).map(|(a, b)| a * b));
            features.extend(field.iter().zip(&opt).map(|(a, b)| (a - b).abs()));
            let hidden: Vec<f64> = linear(&features, &w.scorer).into_iter().map(gelu).collect();
            let residual = linear(&hidden, &w.scorer_out)[0];
            let [prior_scale, joint_scale, gate] = w.scales.map(f64::from);
            let joint = joint_scale * cosine + residual;
            (prior_scale * prior + gate * joint) as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_matches_known_values() {
        for (x, want) in [
            (0.0, 0.0),
            (0.5, 0.520_499_877_813_046_5),
            (1.0, 0.842_700_792_949_714_9),
            (-1.5, -0.966_105_146_475_310_7),
            (3.0, 0.999_977_909_503_001_4),
        ] {
            assert!(
                (erf(x) - want).abs() < 2e-7,
                "erf({x}) = {} want {want}",
                erf(x)
            );
        }
    }

    #[test]
    fn gelu_is_the_exact_form_not_tanh() {
        // torch.nn.functional.gelu(torch.tensor(1.0, dtype=torch.float64))
        assert!((gelu(1.0) - 0.841_344_746_068_542_9).abs() < 1e-7);
        assert!((gelu(-2.0) - -0.045_500_263_896_358_42).abs() < 1e-7);
    }
}
