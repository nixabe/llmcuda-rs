//! Scalar K2-Horizon reference, ported from IFM llama.cpp's
//! `k2_horizon_group_rms_norm` and `graph::build_routed_value` in
//! `src/models/k2-horizon.cpp`, and its softplus attention gate.

/// Normalize contiguous groups, then apply the learned full-width vector.
pub fn grouped_rms_norm(x: &[f32], weight: &[f32], groups: usize, eps: f32) -> Vec<f32> {
    let width = weight.len();
    assert!(groups > 0 && width.is_multiple_of(groups) && x.len().is_multiple_of(width));
    let part = width / groups;
    let mut out = vec![0.0; x.len()];
    for (token, row) in x.chunks_exact(width).enumerate() {
        for g in 0..groups {
            let start = g * part;
            let sum: f32 = row[start..start + part].iter().map(|v| v * v).sum();
            let inv = 1.0 / (sum / part as f32 + eps).sqrt();
            for j in start..start + part {
                out[token * width + j] = (row[j] * inv) * weight[j];
            }
        }
    }
    out
}
/// Select by sigmoid(logit)+bias, normalize unbiased scores, then scale.
pub fn route(logits: &[f32], bias: &[f32], topk: usize, scale: f32) -> (Vec<i32>, Vec<f32>) {
    assert_eq!(logits.len(), bias.len());
    assert!(topk > 0 && topk <= logits.len());
    let probs: Vec<_> = logits.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
    let mut ids: Vec<_> = (0..logits.len()).collect();
    ids.sort_by(|&a, &b| {
        (probs[b] + bias[b])
            .total_cmp(&(probs[a] + bias[a]))
            .then(a.cmp(&b))
    });
    ids.truncate(topk);
    let sum = ids
        .iter()
        .map(|&i| probs[i])
        .sum::<f32>()
        .max(6.103_515_6e-5);
    let weights = ids.iter().map(|&i| (probs[i] / sum) * scale).collect();
    (ids.into_iter().map(|v| v as i32).collect(), weights)
}
/// SiLU each selected expert's projected values before combining them.
pub fn values(projections: &[f32], weights: &[f32], width: usize) -> Vec<f32> {
    assert_eq!(projections.len(), weights.len() * width);
    let mut out = vec![0.0; width];
    for (row, &w) in projections.chunks_exact(width).zip(weights) {
        for (y, &v) in out.iter_mut().zip(row) {
            *y += (v / (1.0 + (-v).exp())) * w;
        }
    }
    out
}
/// K2's softplus gate has beta ln(2), making a zero gate the identity.
pub fn attention_gate(x: &[f32], gate: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), gate.len());
    x.iter()
        .zip(gate)
        .map(|(&x, &g)| {
            let v = g * std::f32::consts::LN_2;
            x * ((v.max(0.0) + (-v.abs()).exp().ln_1p()) * std::f32::consts::LOG2_E)
        })
        .collect()
}
/// Symmetric 16-bit activation quantization with an fp32 scale for each
/// 32-element group. Two signed byte digits have an exact 256:1 scale ratio.
pub fn quantize_activation(x: &[f32]) -> Vec<f32> {
    quantize_activation_with_bits(x, 16)
}
/// Quantization reference for the explicit projection precision.
pub fn quantize_activation_with_bits(x: &[f32], bits: usize) -> Vec<f32> {
    quantize_activation_grouped(x, bits, 32)
}
/// Activation reference with explicit scale group.
pub fn quantize_activation_grouped(x: &[f32], bits: usize, group: usize) -> Vec<f32> {
    assert!([8, 12, 16].contains(&bits));
    let limit = if bits == 8 {
        127.0
    } else if bits == 12 {
        2039.0
    } else {
        32639.0
    };
    assert!(x.len().is_multiple_of(32));
    let mut out = Vec::with_capacity(x.len());
    for block in x.chunks_exact(group) {
        let a = block.iter().map(|v| v.abs()).fold(0f32, f32::max);
        let low = block.iter().copied().fold(f32::INFINITY, f32::min);
        let high = block.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = if bits == 8 {
            if a > 0.0 {
                ((high * 0.5 - low * 0.5) / 127.0)
                    .max(a / 32512.0)
                    .max(f32::MIN_POSITIVE)
            } else {
                1.0
            }
        } else if a > 0.0 {
            a / limit
        } else {
            1.0
        };
        let offset = if bits == 8 {
            ((low * 0.5 + high * 0.5) / scale)
                .round_ties_even()
                .clamp(-32512.0, 32512.0)
        } else {
            0.0
        };
        let inverse = if a > 0.0 { limit / a } else { 0.0 };
        out.extend(block.iter().map(|v| {
            let code = if bits == 8 {
                (v / scale).round_ties_even() - offset
            } else {
                (v * inverse).round_ties_even()
            };
            (code.clamp(-limit, limit) + offset) * scale
        }));
    }
    out
}
/// Affine offset correction: sum of quantized values minus the publisher's
/// ordered sum of the original values, per 32-element group.
pub fn activation_sum_correction(x: &[f32]) -> Vec<f32> {
    activation_sum_correction_with_bits(x, 16)
}
fn activation_sum_correction_with_bits(x: &[f32], bits: usize) -> Vec<f32> {
    let q = quantize_activation_with_bits(x, bits);
    x.as_chunks::<32>()
        .0
        .iter()
        .zip(q.as_chunks::<32>().0.iter())
        .map(|(x, q)| {
            let mut tree = x.to_vec();
            for off in [16, 8, 4, 2, 1] {
                let prev = tree.clone();
                for l in 0..32 {
                    tree[l] = prev[l] + prev[l ^ off];
                }
            }
            q.iter().sum::<f32>() - tree[0]
        })
        .collect()
}
/// Projection with explicit activation quantization and optional Q4 affine minima.
#[allow(clippy::too_many_arguments)]
pub fn quantized_project(
    w: &[f32],
    x: &[f32],
    ids: Option<&[i32]>,
    inner: usize,
    rows: usize,
    tokens: usize,
    topk: usize,
    flat_input: bool,
    minima: Option<&[f32]>,
) -> Vec<f32> {
    quantized_project_with_bits(w, x, ids, inner, rows, tokens, topk, flat_input, minima, 16)
}
/// CPU projection reference for 8- or 16-bit activations.
#[allow(clippy::too_many_arguments)]
pub fn quantized_project_with_bits(
    w: &[f32],
    x: &[f32],
    ids: Option<&[i32]>,
    inner: usize,
    rows: usize,
    tokens: usize,
    topk: usize,
    flat_input: bool,
    minima: Option<&[f32]>,
    bits: usize,
) -> Vec<f32> {
    quantized_project_grouped(
        w, x, ids, inner, rows, tokens, topk, flat_input, minima, bits, 32,
    )
}
/// Projection reference with explicit activation scale group.
#[allow(clippy::too_many_arguments)]
pub fn quantized_project_grouped(
    w: &[f32],
    x: &[f32],
    ids: Option<&[i32]>,
    inner: usize,
    rows: usize,
    tokens: usize,
    topk: usize,
    flat_input: bool,
    minima: Option<&[f32]>,
    bits: usize,
    group: usize,
) -> Vec<f32> {
    let quantized = quantize_activation_grouped(x, bits, group);
    let mut out = project(w, &quantized, ids, inner, rows, tokens, topk, flat_input);
    if let Some(minima) = minima {
        let correction: Vec<f32> = x
            .as_chunks::<32>()
            .0
            .iter()
            .zip(quantized.as_chunks::<32>().0.iter())
            .map(|(x, q)| {
                let mut tree = x.to_vec();
                for off in [16, 8, 4, 2, 1] {
                    let prev = tree.clone();
                    for l in 0..32 {
                        tree[l] = prev[l] + prev[l ^ off];
                    }
                }
                q.iter().sum::<f32>() - tree[0]
            })
            .collect();
        for f in 0..tokens * topk {
            let t = if flat_input { f } else { f / topk };
            let e = ids.map_or(0, |ids| ids[f] as usize);
            for r in 0..rows {
                for g in 0..inner / 32 {
                    out[f * rows + r] += minima[(e * rows + r) * (inner / 32) + g]
                        * correction[t * (inner / 32) + g];
                }
            }
        }
    }
    out
}
/// Dense or selected expert projection with fp32 operands. FFN down uses
/// one input per flat routed slot; value and gate/up use one per token.
#[allow(clippy::too_many_arguments)]
pub fn project(
    w: &[f32],
    x: &[f32],
    ids: Option<&[i32]>,
    inner: usize,
    rows: usize,
    tokens: usize,
    topk: usize,
    flat_input: bool,
) -> Vec<f32> {
    assert!(inner > 0 && rows > 0 && topk > 0);
    assert_eq!(x.len(), tokens * inner * if flat_input { topk } else { 1 });
    if let Some(ids) = ids {
        assert_eq!(ids.len(), tokens * topk);
    } else {
        assert_eq!(topk, 1);
        assert_eq!(w.len(), inner * rows);
    }
    let mut out = vec![0.0; tokens * topk * rows];
    for f in 0..tokens * topk {
        let t = if flat_input { f } else { f / topk };
        let e = ids.map_or(0, |ids| {
            usize::try_from(ids[f]).expect("nonnegative expert")
        });
        for r in 0..rows {
            let mut sum = 0f32;
            for j in 0..inner {
                sum = w[(e * rows + r) * inner + j].mul_add(x[t * inner + j], sum);
            }
            out[f * rows + r] = sum;
        }
    }
    out
}
/// Ordered weighted sum of the selected FFN projections for one token.
pub fn combine(projections: &[f32], weights: &[f32], width: usize) -> Vec<f32> {
    assert_eq!(projections.len(), weights.len() * width);
    let mut out = vec![0.0; width];
    for (row, &w) in projections.chunks_exact(width).zip(weights) {
        for (y, &v) in out.iter_mut().zip(row) {
            *y += v * w;
        }
    }
    out
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bias_selects_but_does_not_weight() {
        let (ids, w) = route(&[0.0, 2.0, -2.0], &[0.0, 0.0, 2.0], 2, 2.5);
        assert_eq!(ids, [2, 1]);
        let p0 = 1.0 / (1.0 + 2.0f32.exp());
        let p1 = 1.0 / (1.0 + (-2.0f32).exp());
        assert!((w[0] - 2.5 * p0 / (p0 + p1)).abs() < 1e-6);
        assert!((w.iter().sum::<f32>() - 2.5).abs() < 1e-6);
    }
    #[test]
    fn groups_have_independent_scales_and_weights() {
        let y = grouped_rms_norm(&[1.0, 1.0, 100.0, 100.0], &[1.0, 2.0, 3.0, 4.0], 2, 0.0);
        assert_eq!(y, [1.0, 2.0, 3.0, 4.0]);
    }
    #[test]
    fn zero_attention_gate_is_identity() {
        assert_eq!(attention_gate(&[2.0, -3.0], &[0.0, 0.0]), [2.0, -3.0]);
    }
    #[test]
    fn silu_is_applied_before_the_value_sum() {
        let y = values(&[2.0, -2.0], &[0.5, 0.5], 1)[0];
        assert!(y > 0.7);
    }
}
