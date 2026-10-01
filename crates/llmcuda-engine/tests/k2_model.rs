//! Real K2 weights: oracle logits, chunking, decode, batching and graph replay.
//! Set LLMCUDA_K2_MODEL and optionally LLMCUDA_K2_GOLDEN (tools/oracle/capture.cpp).
#[path = "golden.rs"]
mod golden;

use cudarc::driver::CudaContext;
use llmcuda_cuda::device::driver_available;
use llmcuda_engine::{
    DeviceWeights,
    forward::{Forward, arena_holds_model_entry},
};
use llmcuda_gguf::GgufFile;
use llmcuda_kernels::compare::compare;
use llmcuda_model::{ModelConfig, WeightSchema};

// Six-token publisher captures: worst block max-abs/RMS was 0.2172 (Q4_K_M),
// worst cosine 0.999729 (Q6_K), and logit max-abs 0.5535 (Q6_K). These gates
// allow independent activation-quantization drift across 48 layers; the
// scalar/device primitive tests and execution-mode comparisons are tighter.
const BLOCK_ABS_OVER_RMS: f32 = 0.3;
const ORACLE_MIN_COSINE: f32 = 1.0 - 5e-4;
const ORACLE_LOGIT_MAX_ABS: f32 = 0.75;

fn agreement(a: &[f32], b: &[f32], label: &str, max_abs: f32, cosine: f32) {
    assert!(
        a.iter().chain(b).all(|v| v.is_finite()),
        "{label}: nonfinite"
    );
    let c = compare(a, b);
    println!(
        "{label}: max_abs={} cosine={}",
        c.max_abs_error, c.cosine_similarity
    );
    assert!(
        c.max_abs_error <= max_abs,
        "{label}: {} > {max_abs}",
        c.max_abs_error
    );
    assert!(
        c.cosine_similarity >= cosine,
        "{label}: {} < {cosine}",
        c.cosine_similarity
    );
}

#[test]
fn k2_full_model() {
    let Some(path) = std::env::var_os("LLMCUDA_K2_MODEL") else {
        println!("SKIPPED: set LLMCUDA_K2_MODEL to a Q4_K_M or Q6_K file");
        return;
    };
    assert!(driver_available(), "K2 validation requires a CUDA driver");
    let oracle =
        std::env::var_os("LLMCUDA_K2_GOLDEN").map(|p| golden::Golden::load(p).expect("K2 capture"));
    let tokens = oracle.as_ref().map_or_else(
        || vec![0, 1000, 2000, 3000, 4000, 5000],
        |g| g.tokens().to_vec(),
    );
    assert!(tokens.len() >= 2);
    let ctx = CudaContext::new(0).unwrap();
    // SAFETY: all uploads, passes and readbacks use this one stream, as in
    // the worker runtime. Automatic cross-stream events cannot be captured.
    unsafe { ctx.disable_event_tracking() };
    let stream = ctx.new_stream().unwrap();
    let file = GgufFile::open(path).unwrap();
    let config = ModelConfig::from_gguf(&file).unwrap();
    assert_eq!(config.architecture, "k2-horizon");
    let schema = WeightSchema::new(&config);
    let directory = schema.resolve(&file).unwrap();
    let (weights, _) = DeviceWeights::load_where_entry(&ctx, &stream, &file, &directory, |r, t| {
        arena_holds_model_entry(&config, r, t)
    })
    .unwrap();
    println!("building {}-token K2 pass", tokens.len());
    let mut pass = Forward::new(
        &ctx,
        &stream,
        &file,
        &directory,
        &weights,
        config.clone(),
        tokens.len(),
    )
    .unwrap();
    assert_eq!(
        DeviceWeights::k2_required_bytes(&directory),
        pass.report().weight_bytes(),
        "K2 preflight must match resident weight allocations"
    );
    let mut state = pass.new_state(&stream, 64).unwrap();
    let mut block_checks = Vec::new();
    pass.run(&stream, &mut state, &tokens, |layer, buf| {
        if let (Some(layer), Some(g)) = (layer, &oracle) {
            let actual = stream.clone_dtoh(buf).unwrap();
            if let Some(r) = g.get(&format!("l_out-{layer}")) {
                let actual = if r.f32_data.len() == config.hidden_size as usize {
                    &actual[actual.len() - config.hidden_size as usize..]
                } else {
                    &actual[..]
                };
                // Independent quantized accumulation may grow across 48 layers.
                // Residual magnitude changes greatly with depth, so express
                // the absolute bound in units of this tensor's RMS.
                let rms = (r
                    .f32_data
                    .iter()
                    .map(|v| f64::from(*v).powi(2))
                    .sum::<f64>()
                    / r.f32_data.len() as f64)
                    .sqrt() as f32;
                assert!(actual.iter().chain(&r.f32_data).all(|v| v.is_finite()));
                let c = compare(actual, &r.f32_data);
                println!(
                    "oracle block {layer}: max_abs={} abs/rms={} cosine={}",
                    c.max_abs_error,
                    c.max_abs_error / rms,
                    c.cosine_similarity
                );
                block_checks.push((layer, c, rms));
            }
        }
    })
    .unwrap();
    let cold = stream.clone_dtoh(pass.logits()).unwrap();
    assert!(cold.iter().all(|v| v.is_finite()));
    println!("cold argmax={}", pass.sample_argmax(&stream).unwrap());
    if let Some(g) = &oracle {
        agreement(
            &cold,
            g.logits(),
            "oracle logits",
            ORACLE_LOGIT_MAX_ABS,
            ORACLE_MIN_COSINE,
        );
        let norm = stream.clone_dtoh(pass.final_norm()).unwrap();
        let reference = g.f32("result_norm");
        let rms = (reference.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>()
            / reference.len() as f64)
            .sqrt() as f32;
        agreement(
            &norm[norm.len() - reference.len()..],
            reference,
            "oracle final grouped norm",
            BLOCK_ABS_OVER_RMS * rms,
            ORACLE_MIN_COSINE,
        );
        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        };
        assert_eq!(argmax(&cold), argmax(g.logits()), "oracle argmax");
    } else {
        println!("SKIPPED: oracle comparison; set LLMCUDA_K2_GOLDEN");
    }
    let mut single = pass
        .reshape(&ctx, &stream, &file, &directory, &weights, 1)
        .unwrap();
    let mut chunked = pass.new_state(&stream, 64).unwrap();
    for &token in &tokens {
        single
            .run(&stream, &mut chunked, &[token], |_, _| {})
            .unwrap();
    }
    agreement(
        &stream.clone_dtoh(single.logits()).unwrap(),
        &cold,
        "single-token chunking",
        0.02,
        0.99999,
    );
    let next = pass.sample_argmax(&stream).unwrap();
    let mut decode_stages = Vec::new();
    single
        .run_with_stage_waypoints(&stream, &mut state, &[next], |layer, stage, buf| {
            decode_stages.push((layer, stage, stream.clone_dtoh(buf).unwrap()));
        })
        .unwrap();
    let decoded = stream.clone_dtoh(single.logits()).unwrap();
    // Graph and ordinary decode must start from identical prefill arithmetic.
    // The chunking comparison above permits fp16 KV rounding differences.
    chunked.reset(&stream).unwrap();
    pass.run(&stream, &mut chunked, &tokens, |_, _| {}).unwrap();
    let graph = single.capture_step(&stream, &mut chunked).unwrap();
    single
        .replay_step(&stream, &mut chunked, &graph, &[next])
        .unwrap();
    agreement(
        &stream.clone_dtoh(single.logits()).unwrap(),
        &decoded,
        "decode graph",
        0.0,
        0.999999,
    );

    let mut batch = pass
        .reshape(&ctx, &stream, &file, &directory, &weights, 3 * tokens.len())
        .unwrap();
    batch.enable_batch_prefill(&ctx, &stream, 3).unwrap();
    let mut states = vec![
        pass.new_state(&stream, 64).unwrap(),
        pass.new_state(&stream, 64).unwrap(),
        pass.new_state(&stream, 64).unwrap(),
    ];
    let mut both = tokens.clone();
    both.extend(&tokens);
    both.extend(&tokens);
    batch
        .run_batch_prefill(&stream, &mut states, &both)
        .unwrap();
    let logits = stream.clone_dtoh(batch.batch_logits().unwrap()).unwrap();
    for row in logits.chunks(cold.len()) {
        agreement(row, &cold, "batch prefill", 0.02, 0.99999);
    }
    let mut bd = pass
        .reshape(&ctx, &stream, &file, &directory, &weights, 3)
        .unwrap();
    bd.enable_batch_decode(&ctx, &stream).unwrap();
    bd.run_batch_decode_with_stage_waypoints(
        &stream,
        &mut states,
        &[next, next, next],
        |layer, stage, buf| {
            let actual = stream.clone_dtoh(buf).unwrap();
            let reference = &decode_stages
                .iter()
                .find(|(l, s, _)| *l == layer && *s == stage)
                .unwrap()
                .2;
            let c = compare(&actual[..reference.len()], reference);
            println!(
                "batch stage {layer:?}/{stage:?}: max_abs={} cosine={}",
                c.max_abs_error, c.cosine_similarity
            );
        },
    )
    .unwrap();
    let logits = stream.clone_dtoh(bd.batch_logits().unwrap()).unwrap();
    for row in logits.chunks(decoded.len()) {
        agreement(row, &decoded, "batch decode", 0.02, 0.99999);
    }
    // Exercise the batched split attention path beyond the shallow oracle
    // prompt. Both physical widths start from the same chunked prefixes.
    let repeats = 40 / tokens.len() + 1;
    if repeats * tokens.len() < 64 {
        let mut deep = pass.new_state(&stream, 64).unwrap();
        for state in &mut states {
            state.reset(&stream).unwrap();
        }
        for _ in 0..repeats {
            pass.run(&stream, &mut deep, &tokens, |_, _| {}).unwrap();
            batch
                .run_batch_prefill(&stream, &mut states, &both)
                .unwrap();
        }
        // This width exercises compensated tensor-core attention in the
        // real model, while the six-token oracle covers masked short tiles.
        let chunked_logits = stream.clone_dtoh(pass.logits()).unwrap();
        let prefix = tokens.repeat(repeats);
        let mut wide = pass
            .reshape(&ctx, &stream, &file, &directory, &weights, prefix.len())
            .unwrap();
        let mut wide_state = wide.new_state(&stream, 64).unwrap();
        wide.run(&stream, &mut wide_state, &prefix, |_, _| {})
            .unwrap();
        agreement(
            &stream.clone_dtoh(wide.logits()).unwrap(),
            &chunked_logits,
            "wide prefill versus chunks",
            0.02,
            0.99999,
        );
        single.run(&stream, &mut deep, &[next], |_, _| {}).unwrap();
        let reference = stream.clone_dtoh(single.logits()).unwrap();
        let graph = bd.capture_batch_step(&stream, &mut states).unwrap();
        bd.replay_batch_step(&stream, &mut states, &graph, &[next, next, next])
            .unwrap();
        for row in stream
            .clone_dtoh(bd.batch_logits().unwrap())
            .unwrap()
            .chunks(reference.len())
        {
            agreement(row, &reference, "deep batch decode graph", 0.02, 0.99999);
        }
    }
    // Defer gates until the whole curve is printed, making a failed audit
    // useful without dropping any layer's numerical checks.
    for (layer, c, rms) in block_checks {
        assert!(
            c.max_abs_error <= BLOCK_ABS_OVER_RMS * rms,
            "block {layer}: absolute error {} / rms {rms}",
            c.max_abs_error
        );
        assert!(
            c.cosine_similarity >= ORACLE_MIN_COSINE,
            "block {layer}: cosine {}",
            c.cosine_similarity
        );
    }
}
