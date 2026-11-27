//! `inkling2_loop` through the serving path: the engine, its scheduler and
//! pager, and the architecture's own device runtime behind the worker.
//!
//! The forward is gated elsewhere (`inkling_forward.rs`). This is the layer
//! above it: a request placed through `Engine::place_tokens` prefills, emits
//! its first token with the last prompt row, decodes in steps shared with
//! other requests, stops at its output cap, and frees its place — and what
//! it generates does not depend on who it shared a step with.
//!
//! Needs a device, the model and its capture; SKIPS without them.
#[allow(dead_code)]
mod inkling_common;

use std::collections::BTreeMap;

use llmcuda_cache::CacheConfig;
use llmcuda_cuda::device::driver_available;
use llmcuda_engine::{Engine, RouterConfig, SamplingParams, ServingConfig};
use llmcuda_model::ModelConfig;
use llmcuda_sched::SchedulerConfig;
use llmcuda_sched::request::{NewRequest, RequestId};

const OUTPUT: u32 = 6;

fn place(engine: &mut Engine, id: u64, prompt: &[i32], sampling: SamplingParams) {
    engine
        .place_tokens(
            NewRequest {
                id: RequestId(id),
                prompt_tokens: prompt.len() as u32,
                max_output_tokens: OUTPUT,
            },
            prompt.to_vec(),
            Vec::new(),
            sampling,
            None,
        )
        .unwrap_or_else(|failure| panic!("request {id} was not placed: {failure}"));
}

/// Step until every request in `ids` has completed; tokens by request.
fn run(engine: &Engine, ids: &[u64]) -> BTreeMap<u64, Vec<i32>> {
    let mut generated: BTreeMap<u64, Vec<i32>> = ids.iter().map(|id| (*id, Vec::new())).collect();
    let mut done = 0;
    let mut steps = 0;
    while done < ids.len() {
        steps += 1;
        assert!(steps < 200, "the requests did not complete");
        for (_, step) in engine.step_devices().expect("the step runs") {
            assert!(step.retained.is_empty(), "this runtime keeps no snapshots");
            for (id, token) in step.generated {
                generated
                    .get_mut(&id.0)
                    .expect("a known request")
                    .push(token);
            }
            done += step.completed.len();
        }
    }
    generated
}

#[test]
fn requests_prefill_decode_and_complete_through_the_engine() {
    let Some((file, golden)) = inkling_common::model_and_golden() else {
        return;
    };
    if !driver_available() {
        println!("SKIPPED: no CUDA driver present");
        return;
    }
    let model = ModelConfig::from_gguf(&file).expect("the configuration reads");
    drop(file);
    let cache = CacheConfig::with_defaults(model.clone()).expect("cache configuration");
    let block = cache.attention_block_size();
    let sched = SchedulerConfig::with_defaults(4096, block, 3).expect("scheduler configuration");
    let mut engine = Engine::new(
        &[0],
        cache,
        sched,
        16_384 / block,
        3,
        RouterConfig::balanced(),
    );
    engine
        .bind_devices(
            &inkling_common::model_path(),
            model.clone(),
            ServingConfig::new(256),
        )
        .unwrap_or_else(|(worker, failure)| panic!("{worker} failed to bind: {failure}"));

    let capture = golden.tokens().to_vec();
    let want_first = golden
        .logits()
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as i32;
    // A second prompt of another length: a ragged prefill beside the first,
    // and a different neighbour in every decode step.
    let other: Vec<i32> = (0..300usize)
        .map(|i| ((i * 7919 + 1234) % model.vocab_size as usize) as i32)
        .collect();

    // Two copies of the capture prompt and a stranger, served together.
    place(&mut engine, 1, &capture, SamplingParams::GREEDY);
    place(&mut engine, 2, &capture, SamplingParams::GREEDY);
    place(&mut engine, 3, &other, SamplingParams::GREEDY);
    let together = run(&engine, &[1, 2, 3]);
    println!("served together: {together:?}");
    assert_eq!(
        together[&1][0], want_first,
        "the first token is llama.cpp's argmax for the capture prompt"
    );
    assert_eq!(
        together[&1], together[&2],
        "two copies of one prompt diverged"
    );
    for (id, tokens) in &together {
        assert!(
            !tokens.is_empty() && tokens.len() <= OUTPUT as usize,
            "request {id}"
        );
    }

    // The same prompt again, alone: places freed, and the same tokens.
    place(&mut engine, 4, &capture, SamplingParams::GREEDY);
    let alone = run(&engine, &[4]);
    assert_eq!(
        alone[&4], together[&1],
        "a neighbour changed a sequence's tokens"
    );

    // Sampled: a seed is a sequence of tokens, and another seed another.
    let sampled = |seed: u64| SamplingParams {
        temperature: 1.5,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        seed,
    };
    place(&mut engine, 5, &capture, sampled(7));
    place(&mut engine, 6, &capture, sampled(7));
    place(&mut engine, 7, &capture, sampled(8));
    let drawn = run(&engine, &[5, 6, 7]);
    println!("sampled: {drawn:?}");
    assert_eq!(drawn[&5], drawn[&6], "one seed drew two sequences");
    assert_ne!(drawn[&5], drawn[&7], "two seeds drew one sequence");
}
