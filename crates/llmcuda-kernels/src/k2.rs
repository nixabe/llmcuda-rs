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
