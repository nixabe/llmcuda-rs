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
use llmcuda_engine::{Engine, RouterConfig, SamplingParams, ServingConfig, SnapshotArenaSize};
use llmcuda_model::ModelConfig;
use llmcuda_sched::SchedulerConfig;
use llmcuda_sched::request::{NewRequest, RequestId};

const OUTPUT: u32 = 6;

fn place(engine: &mut Engine, id: u64, prompt: &[i32], sampling: SamplingParams) {
    place_for(engine, id, prompt, sampling, OUTPUT);
}

/// Place a request; the prefix tokens it resumes from.
fn place_for(
    engine: &mut Engine,
    id: u64,
    prompt: &[i32],
    sampling: SamplingParams,
    output: u32,
) -> u32 {
    engine
        .place_tokens(
            NewRequest {
                id: RequestId(id),
                prompt_tokens: prompt.len() as u32,
                max_output_tokens: output,
            },
            prompt.to_vec(),
            Vec::new(),
            sampling,
            None,
        )
        .unwrap_or_else(|failure| panic!("request {id} was not placed: {failure}"))
        .reusable_prefix_tokens
}

/// One card serving this model with `arena` snapshot slots.
fn engine_with(model: &ModelConfig, arena: SnapshotArenaSize) -> Engine {
    let cache = CacheConfig::with_defaults(model.clone()).expect("cache configuration");
    let block = cache.attention_block_size();
    let sched = SchedulerConfig::with_defaults(4096, block, 3).expect("scheduler configuration");
    let engine = Engine::new(
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
            ServingConfig {
                snapshot_slots: arena,
                ..ServingConfig::new(2048)
            },
        )
        .unwrap_or_else(|(worker, failure)| panic!("{worker} failed to bind: {failure}"));
    engine
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
            // Nothing here reaches a retention boundary; the default arena
            // would pin gigabytes of host memory for nothing.
            ServingConfig {
                snapshot_slots: SnapshotArenaSize::NONE,
                ..ServingConfig::new(256)
            },
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

    // A request reserved for far more positions than the others decodes
    // beside them: its caches and theirs are the same rings.
    place(&mut engine, 8, &capture, SamplingParams::GREEDY);
    place_for(&mut engine, 9, &other, SamplingParams::GREEDY, 6000);
    let mixed = {
        let mut generated: BTreeMap<u64, Vec<i32>> = BTreeMap::new();
        let mut done = false;
        let mut steps = 0;
        while !done {
            steps += 1;
            assert!(steps < 100, "request 8 did not complete");
            for (_, step) in engine
                .step_devices()
                .expect("a long and a short request step together")
            {
                for (id, token) in step.generated {
                    generated.entry(id.0).or_default().push(token);
                }
                done |= step.completed.iter().any(|id| id.0 == 8);
            }
        }
        generated
    };
    assert_eq!(
        mixed[&8], together[&1],
        "a long neighbour changed a sequence's tokens"
    );
    assert_eq!(
        mixed[&9][..OUTPUT as usize],
        together[&3][..],
        "a long reservation changed its own tokens"
    );
    assert!(
        engine.cancel(RequestId(9)),
        "the long request was not resident"
    );

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

/// A prefix served once is not prefilled again, and resuming from its
/// snapshot changes no token.
///
/// Every prompt is run cold first, on an engine with no snapshot arena,
/// then on one that retains. The retention interval is 2,048 positions:
///
/// - a 2,300-token prompt leaves a snapshot at 2,048, taken mid-prefill;
/// - a prompt sharing its first 2,100 tokens resumes at 2,048 and prefills
///   the rest;
/// - the first 2,048 tokens alone are a whole-prompt resume: nothing to
///   prefill, and the first token is the one the snapshot recorded;
/// - a 2,040-token prompt crosses 2,048 while decoding and leaves a
///   snapshot there; that prompt plus its own first eight tokens then
///   resumes with nothing to prefill, and continues with the ninth.
#[test]
fn a_served_prefix_is_resumed_and_changes_no_token() {
    let Some((file, _)) = inkling_common::model_and_golden() else {
        return;
    };
    if !driver_available() {
        println!("SKIPPED: no CUDA driver present");
        return;
    }
    let model = ModelConfig::from_gguf(&file).expect("the configuration reads");
    drop(file);
    let token = |seed: usize, i: usize| {
        ((seed * 104_729 + i * 7919 + 1234) % (model.vocab_size as usize - 1000)) as i32
    };
    let long: Vec<i32> = (0..2300).map(|i| token(1, i)).collect();
    let mut sibling = long[..2100].to_vec();
    sibling.extend((0..150).map(|i| token(2, i)));
    let whole = long[..2048].to_vec();
    let short: Vec<i32> = (0..2040).map(|i| token(3, i)).collect();
    const LONG_OUTPUT: u32 = 12;

    let mut cold = engine_with(&model, SnapshotArenaSize::NONE);
    let mut cold_run = |id: u64, prompt: &[i32]| {
        assert_eq!(
            place_for(&mut cold, id, prompt, SamplingParams::GREEDY, LONG_OUTPUT),
            0
        );
        run(&cold, &[id])[&id].clone()
    };
    let want_long = cold_run(1, &long);
    let want_sibling = cold_run(2, &sibling);
    let want_whole = cold_run(3, &whole);
    let want_short = cold_run(4, &short);
    assert_eq!(
        want_short.len(),
        LONG_OUTPUT as usize,
        "the short prompt stopped early"
    );
    let mut continued = short.clone();
    continued.extend(&want_short[..8]);
    let want_continued = cold_run(5, &continued);
    assert_eq!(
        want_continued[0], want_short[8],
        "the model disagrees with itself cold"
    );
    drop(cold_run);
    drop(cold);

    let mut warm = engine_with(&model, SnapshotArenaSize::dense(4));
    let mut warm_run = |id: u64, prompt: &[i32], resumed: u32| {
        let got = place_for(&mut warm, id, prompt, SamplingParams::GREEDY, LONG_OUTPUT);
        assert_eq!(got, resumed, "request {id} resumed from the wrong place");
        run(&warm, &[id])[&id].clone()
    };
    // The engine takes a step's snapshots before the step is returned, so
    // what shows a boundary was retained is the next request resuming there.
    assert_eq!(warm_run(11, &long, 0), want_long);
    assert_eq!(
        warm_run(12, &sibling, 2048),
        want_sibling,
        "resuming mid-prompt changed the tokens"
    );
    assert_eq!(
        warm_run(13, &whole, 2048),
        want_whole,
        "resuming a whole prompt changed the tokens"
    );
    assert_eq!(warm_run(14, &short, 0), want_short);
    assert_eq!(
        warm_run(15, &continued, 2048),
        want_continued,
        "resuming a decode-time snapshot changed the tokens"
    );
    println!(
        "resumed at 2048 mid-prompt, whole-prompt and from a decode-time snapshot: tokens unchanged"
    );
}
