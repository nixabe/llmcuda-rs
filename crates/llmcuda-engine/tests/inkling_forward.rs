//! The `inkling2_loop` device forward against a llama.cpp capture, and
//! against itself across the ways a prompt can be cut into passes.
//!
//! Needs a device, the model and its capture (see `inkling_reference.rs`,
//! which gates the CPU reference this path follows); skips without them.
#[allow(dead_code)]
mod inkling_common;

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};
use llmcuda_cuda::device::driver_available;
use llmcuda_engine::inkling::{InklingDecode, InklingModel, InklingPass, InklingState, Segment};
use llmcuda_kernels::compare::compare;

/// Floor on a layer output's cosine against llama.cpp; see
/// `inkling_reference.rs` for why the looped layers sit where they do.
const L_OUT_COSINE: f32 = 0.998;

fn argmax(x: &[f32]) -> usize {
    x.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}

/// Run `tokens` through passes of the given row counts on a fresh sequence
/// and return the last row's normalized hidden state.
fn last_hidden(
    stream: &Arc<CudaStream>,
    model: &Arc<InklingModel>,
    tokens: &[i32],
    cuts: &[usize],
) -> Vec<f32> {
    let n = model.config().hidden_size as usize;
    let mut state = InklingState::new(stream, model, 64).expect("state");
    let mut at = 0;
    let mut last = Vec::new();
    for &rows in cuts {
        let mut pass = InklingPass::new(model, stream, rows).expect("pass");
        pass.run(
            stream,
            &mut [Segment {
                state: &mut state,
                rows,
            }],
            &tokens[at..at + rows],
            None,
        )
        .expect("the pass runs");
        at += rows;
        let hidden = stream.clone_dtoh(pass.hidden()).expect("read back");
        last = hidden[(rows - 1) * n..rows * n].to_vec();
    }
    assert_eq!(at, tokens.len());
    last
}

/// Floor on the decode path's hidden state against the prefill path's.
///
/// The two are the same arithmetic in a different summation order and agree
/// to ~1e-7 in cosine until they first route differently. They do: the
/// router's sigmoid top-8 has near-ties, and on the capture prompt two to
/// five of the twelve positions (it moves with the summation order) send
/// one to four of their 48 layer applications to a different eighth expert.
/// The test prints which. The worst such position measured 0.99822, with
/// three applications flipped; llama.cpp against the CPU reference flips
/// the same way (`inkling_reference.rs`). A wrong operator reads 0.3-0.8.
const DECODE_COSINE: f32 = 0.995;

#[test]
fn device_forward_matches_llama_cpp_and_its_own_chunkings() {
    let Some((file, golden)) = inkling_common::model_and_golden() else {
        return;
    };
    if !driver_available() {
        println!("SKIPPED: no CUDA driver");
        return;
    }
    let ctx = CudaContext::new(0).expect("device 0");
    let stream = ctx.default_stream();
    let model = Arc::new(InklingModel::load(&ctx, &stream, &file).expect("the model loads"));
    model.use_reference_kernels(std::env::var_os("LLMCUDA_INK_REFERENCE").is_some());
    println!(
        "weights resident: {:.2} GiB",
        model.weight_bytes() as f64 / (1u64 << 30) as f64
    );
    let tokens = golden.tokens().to_vec();
    let rows = tokens.len();
    let n = model.config().hidden_size as usize;

    // One pass over the whole prompt, every layer application against the
    // capture.
    let mut state = InklingState::new(&stream, &model, 64).expect("state");
    let mut pass = InklingPass::new(&model, &stream, rows).expect("pass");
    let mut failures = Vec::new();
    let mut seen = 0;
    let mut tap = |slot: usize, got: Vec<f32>| {
        let want = golden.f32(&format!("l_out-{slot}"));
        let r = compare(&got, want);
        seen += 1;
        println!(
            "l_out-{slot:<2} max_abs {:.3e} cos {:.8}",
            r.max_abs_error, r.cosine_similarity
        );
        if r.cosine_similarity < L_OUT_COSINE {
            failures.push(format!("l_out-{slot}: cosine {}", r.cosine_similarity));
        }
    };
    pass.run(
        &stream,
        &mut [Segment {
            state: &mut state,
            rows,
        }],
        &tokens,
        Some(&mut tap),
    )
    .expect("the pass runs");
    assert_eq!(seen, 48);
    assert!(failures.is_empty(), "{failures:#?}");
    assert_eq!(state.position(), rows);

    let every_row = stream.clone_dtoh(pass.hidden()).expect("read back");
    let whole = every_row[(rows - 1) * n..].to_vec();
    let norm = compare(&whole, golden.f32("result_norm"));
    println!(
        "result_norm max_abs {:.3e} cos {:.8}",
        norm.max_abs_error, norm.cosine_similarity
    );
    pass.logits(&stream, rows - 1, 1).expect("the head runs");
    let vocab = model.config().vocab_size as usize;
    let logits = stream.clone_dtoh(pass.head_logits()).expect("read back")[..vocab].to_vec();
    let want = golden.logits();
    let r = compare(&logits, want);
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
    drop(pass);

    // The same prompt cut into chunks, and one token at a time, lands on
    // the same final state to the bit: a row's contractions, its scan of
    // the cache and its reductions do not depend on what shares the pass.
    for cuts in [vec![5, 7], vec![1; rows], vec![3, 1, 4, 1, 3]] {
        let got = last_hidden(&stream, &model, &tokens, &cuts);
        let r = compare(&got, &whole);
        println!(
            "cuts {cuts:?}: max_abs {:.3e} cos {:.8}",
            r.max_abs_error, r.cosine_similarity
        );
        assert_eq!(r.max_abs_error, 0.0, "cuts {cuts:?} changed the result");
    }

    // Two sequences stepped together, one row each: both must equal the
    // sequence stepped alone.
    let alone = last_hidden(&stream, &model, &tokens, &vec![1; rows]);
    let mut a = InklingState::new(&stream, &model, 64).expect("state");
    let mut b = InklingState::new(&stream, &model, 64).expect("state");
    let mut pass = InklingPass::new(&model, &stream, 2).expect("pass");
    for &t in &tokens {
        pass.run(
            &stream,
            &mut [
                Segment {
                    state: &mut a,
                    rows: 1,
                },
                Segment {
                    state: &mut b,
                    rows: 1,
                },
            ],
            &[t, t],
            None,
        )
        .expect("the batched step runs");
    }
    let hidden = stream.clone_dtoh(pass.hidden()).expect("read back");
    for (which, row) in hidden.chunks(n).enumerate() {
        let r = compare(row, &alone);
        println!(
            "batched row {which}: max_abs {:.3e} cos {:.8}",
            r.max_abs_error, r.cosine_similarity
        );
        assert_eq!(r.max_abs_error, 0.0, "batching changed row {which}");
    }

    // The decode path: fused kernels and the decode GEMM. Same arithmetic
    // class as the passes above, different summation order, so it agrees
    // with them to rounding rather than to the bit.
    drop(pass);
    let step_alone = |prefill: usize| -> (Vec<f32>, Vec<f32>) {
        let mut state = InklingState::new(&stream, &model, 64).expect("state");
        if prefill > 0 {
            let mut pass = InklingPass::new(&model, &stream, prefill).expect("pass");
            pass.run(
                &stream,
                &mut [Segment {
                    state: &mut state,
                    rows: prefill,
                }],
                &tokens[..prefill],
                None,
            )
            .expect("the prefill runs");
        }
        let mut decode = InklingDecode::new(&model, &stream, 1).expect("decode");
        for &t in &tokens[prefill..] {
            decode
                .step(&stream, &mut [&mut state], &[t])
                .expect("the step runs");
        }
        decode.logits(&stream).expect("the head runs");
        (
            stream.clone_dtoh(decode.hidden()).expect("read back"),
            stream.clone_dtoh(decode.head_logits()).expect("read back"),
        )
    };
    // Position by position against a pass of one row, with the experts
    // each routed to. The two paths agree to rounding until a router
    // near-tie resolves differently, and then by one expert of eight.
    {
        let (slots, topk) = (48usize, 8usize);
        let mut state = InklingState::new(&stream, &model, 64).expect("state");
        let mut decode = InklingDecode::new(&model, &stream, 1).expect("decode");
        decode.log_routes();
        let mut ref_state = InklingState::new(&stream, &model, 64).expect("state");
        let mut single = InklingPass::new(&model, &stream, 1).expect("pass");
        single.log_routes();
        // Until a routing flips, nothing separates the paths but rounding.
        // After one, later positions attend over a cache that differs.
        let mut clean_worst = 1.0f32;
        let mut flipped_positions = 0usize;
        for (at, &t) in tokens.iter().enumerate() {
            decode
                .step(&stream, &mut [&mut state], &[t])
                .expect("the step runs");
            single
                .run(
                    &stream,
                    &mut [Segment {
                        state: &mut ref_state,
                        rows: 1,
                    }],
                    &[t],
                    None,
                )
                .expect("the pass runs");
            let got = stream.clone_dtoh(decode.hidden()).expect("read back");
            let r = compare(&got, &every_row[at * n..][..n]);
            let sets = |ids: &[i32]| -> Vec<Vec<i32>> {
                ids[at * slots * topk..][..slots * topk]
                    .chunks(topk)
                    .map(|c| {
                        let mut c = c.to_vec();
                        c.sort_unstable();
                        c
                    })
                    .collect()
            };
            let (a, b) = (sets(decode.routes()), sets(single.routes()));
            let differing: Vec<usize> = (0..slots).filter(|&i| a[i] != b[i]).collect();
            println!(
                "decode path position {at}: max_abs {:.3e} cos {:.8}, slots routed differently {differing:?}",
                r.max_abs_error, r.cosine_similarity
            );
            assert!(
                r.cosine_similarity > DECODE_COSINE,
                "decode position {at}: {}",
                r.cosine_similarity
            );
            if differing.is_empty() && flipped_positions == 0 {
                clean_worst = clean_worst.min(r.cosine_similarity);
            } else if !differing.is_empty() {
                flipped_positions += 1;
            }
        }
        println!(
            "{flipped_positions} of {} positions flip a routing; worst cosine before the first {clean_worst:.8}",
            tokens.len()
        );
        assert!(
            clean_worst > 0.99999,
            "the paths drifted before any routing flipped: {clean_worst}"
        );
    }
    let (decoded, decoded_logits) = step_alone(0);
    let r = compare(&decoded, &whole);
    println!(
        "decode path, 12 steps vs one pass: max_abs {:.3e} cos {:.8}",
        r.max_abs_error, r.cosine_similarity
    );
    assert!(
        r.cosine_similarity > DECODE_COSINE,
        "decode path: {}",
        r.cosine_similarity
    );
    let r = compare(&decoded_logits, want);
    println!(
        "decode path logits vs llama.cpp: max_abs {:.3e} cos {:.8} argmax {}",
        r.max_abs_error,
        r.cosine_similarity,
        argmax(&decoded_logits)
    );
    assert_eq!(argmax(&decoded_logits), argmax(want));
    assert!(
        r.cosine_similarity > 0.9995,
        "decode logit cosine {}",
        r.cosine_similarity
    );
    assert!(
        r.max_abs_error < 0.5,
        "decode logit max-abs {}",
        r.max_abs_error
    );

    // A prefill pass handing over to decode steps mid-prompt.
    let (handed, _) = step_alone(8);
    let r = compare(&handed, &whole);
    println!(
        "prefill 8 then decode 4: max_abs {:.3e} cos {:.8}",
        r.max_abs_error, r.cosine_similarity
    );
    assert!(
        r.cosine_similarity > DECODE_COSINE,
        "handover: {}",
        r.cosine_similarity
    );

    // Three sequences decoded together are three sequences decoded alone:
    // a row's column of the contraction does not depend on its neighbours.
    let mut states: Vec<InklingState> = (0..3)
        .map(|_| InklingState::new(&stream, &model, 64).expect("state"))
        .collect();
    let mut decode = InklingDecode::new(&model, &stream, 3).expect("decode");
    for &t in &tokens {
        let mut refs: Vec<&mut InklingState> = states.iter_mut().collect();
        decode
            .step(&stream, &mut refs, &[t, t, t])
            .expect("the batched step runs");
    }
    let hidden = stream.clone_dtoh(decode.hidden()).expect("read back");
    for (which, row) in hidden.chunks(n).enumerate() {
        let r = compare(row, &decoded);
        println!(
            "decode path row {which} of 3: max_abs {:.3e} cos {:.8}",
            r.max_abs_error, r.cosine_similarity
        );
        assert_eq!(r.max_abs_error, 0.0, "batching changed decode row {which}");
    }
}
