//! The `inkling2_loop` CPU reference against a llama.cpp capture.
//!
//! This gates the oracle itself: every device path for the architecture is
//! compared with `llmcuda_kernels::inkling`, so that reference has to be
//! shown to compute what llama.cpp computes on the real weights first.
//!
//! Needs the model (`LLMCUDA_INKLING_MODEL`, default
//! `models/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_XL.gguf`) and a capture
//! from that exact file (`LLMCUDA_INKLING_GOLDEN`, default
//! `.golden/inkling2-mini-q4-golden.bin`); skips without either. No device.
#[allow(dead_code)]
mod inkling_common;

use llmcuda_kernels::compare::compare;
use llmcuda_kernels::inkling;

/// Floor on a layer output's cosine against llama.cpp.
///
/// The two agree to ~1e-5 in cosine through the prelude. In the looped
/// layers llama.cpp's own arithmetic (reduced-precision expert products)
/// moves router logits by ~2e-2, which flips the last-ranked expert for a
/// token or two out of twelve; the worst layer output observed is 0.99883.
/// A wrong operator shows up as 0.3-0.8, as the unscaled-bias bug did.
const L_OUT_COSINE: f32 = 0.998;

#[test]
fn cpu_reference_matches_llama_cpp_at_every_layer_application() {
    let Some((file, golden)) = inkling_common::model_and_golden() else {
        return;
    };
    let h = inkling_common::hparams(&file);
    let weights = inkling_common::FileWeights::new(&file);
    let tokens = golden.tokens();
    println!("{} tokens, {} cache slots", tokens.len(), h.slots());

    let embeddings: Vec<f32> = tokens
        .iter()
        .flat_map(|t| {
            let row = inkling::Weights::part(&weights, "token_embd.weight", *t as usize, h.n_embd);
            row.as_ref().clone()
        })
        .collect();

    let mut worst_cos = 1.0f32;
    let mut failures = Vec::new();
    let mut seen = 0usize;
    let mut tap = |name: &str, slot: usize, got: &[f32]| {
        let Some(want) = golden.get(&format!("{name}-{slot}")) else {
            return;
        };
        if want.f32_data.len() != got.len() {
            failures.push(format!(
                "{name}-{slot}: {} elements, capture has {:?}",
                got.len(),
                want.shape()
            ));
            return;
        }
        if name == "ffn_moe_topk" {
            let diff = got
                .chunks(8)
                .zip(want.f32_data.chunks(8))
                .filter(|(a, b)| {
                    let mut a = a.to_vec();
                    let mut b = b.to_vec();
                    a.sort_by(f32::total_cmp);
                    b.sort_by(f32::total_cmp);
                    a != b
                })
                .count();
            println!("ffn_moe_topk-{slot}: {diff} tokens route to a different expert set");
            return;
        }
        let r = compare(got, &want.f32_data);
        seen += 1;
        worst_cos = worst_cos.min(r.cosine_similarity);
        println!(
            "{name:>8}-{slot:<2} max_abs {:.3e} cos {:.8}",
            r.max_abs_error, r.cosine_similarity
        );
        if name == "l_out" && r.cosine_similarity < L_OUT_COSINE {
            failures.push(format!("{name}-{slot}: cosine {}", r.cosine_similarity));
        }
    };
    let hidden = inkling::hidden_states(&h, &weights, &embeddings, &mut tap);
    assert!(
        seen >= h.slots(),
        "only {seen} captured nodes were compared"
    );
    assert!(failures.is_empty(), "{failures:#?}");

    let last = &hidden[(tokens.len() - 1) * h.n_embd..];
    let norm = compare(last, golden.f32("result_norm"));
    println!(
        "result_norm max_abs {:.3e} cos {:.8}",
        norm.max_abs_error, norm.cosine_similarity
    );
    let logits = inkling::logits(&h, &weights, last);
    let want = golden.logits();
    assert_eq!(logits.len(), want.len());
    let r = compare(&logits, want);
    let argmax = |x: &[f32]| {
        x.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    println!(
        "logits max_abs {:.3e} cos {:.8} argmax {} vs {}",
        r.max_abs_error,
        r.cosine_similarity,
        argmax(&logits),
        argmax(want)
    );
    assert_eq!(argmax(&logits), argmax(want));
    assert!(
        r.cosine_similarity > 0.9999,
        "logit cosine {}",
        r.cosine_similarity
    );
    assert!(r.max_abs_error < 0.5, "logit max-abs {}", r.max_abs_error);
}
