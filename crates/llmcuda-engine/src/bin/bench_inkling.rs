//! Inkling 2 Mini prefill and decode throughput, shaped like
//! `llama-batched-bench`: `N` sequences prefilled with `PP` tokens each in
//! one pass, then `TG` decode steps of one row a sequence.
//!
//! `bench_inkling <pp> <tg> <n>[,<n>...]`, model from `LLMCUDA_MODEL`.
//! `LLMCUDA_BENCH_CHUNK` cuts each sequence's prefill into chunks of that
//! many tokens (default: the whole prompt in one pass). Wall-clock around
//! whole passes with the stream synchronized; the first row of each width
//! is a warm-up and is reported as such; `LLMCUDA_BENCH_ROUNDS` (default 3)
//! counts it. `LLMCUDA_INK_DEC_ATTN=scalar|full` puts decode attention back
//! on the scalar kernel, everywhere or on the sliding layers only.
//! `LLMCUDA_PROFILE=1` adds a
//! per-stage breakdown, synchronizing at every stage boundary; its totals
//! are not the throughput.
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::CudaContext;
use llmcuda_engine::inkling::{InklingDecode, InklingModel, InklingPass, InklingState, Segment};
use tracing::{error, info};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn run(args: &[String]) -> Result<()> {
    let pp: usize = args.first().map_or(Ok(512), |s| s.parse())?;
    let tg: usize = args.get(1).map_or(Ok(64), |s| s.parse())?;
    let widths: Vec<usize> = args
        .get(2)
        .map_or("1,3", String::as_str)
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    let chunk = std::env::var("LLMCUDA_BENCH_CHUNK")
        .ok()
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(pp)
        .min(pp);
    let path = std::env::var("LLMCUDA_MODEL")?;
    let file = llmcuda_gguf::GgufFile::open(&path)?;
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let started = Instant::now();
    let model = Arc::new(InklingModel::load(&ctx, &stream, &file)?);
    info!(
        "loaded {:.2} GiB in {:.1} s",
        model.weight_bytes() as f64 / (1u64 << 30) as f64,
        started.elapsed().as_secs_f64()
    );
    let vocab = model.config().vocab_size as usize;
    // A fixed pseudo-random prompt: throughput does not depend on the text,
    // and routing over random tokens is the unfavourable case for grouping.
    let token = |i: usize| (((i as u64 * 2_654_435_761) >> 7) % (vocab as u64 - 1000)) as i32;

    for &n in &widths {
        // One pass per distinct chunk length, reused: a pass owns its
        // workspace, and one per chunk would not fit at depth.
        let mut prefill: Vec<(usize, InklingPass)> = Vec::new();
        let mut plan = Vec::new();
        let mut at = 0;
        while at < pp {
            let rows = chunk.min(pp - at);
            let index = match prefill.iter().position(|(have, _)| *have == rows) {
                Some(index) => index,
                None => {
                    let pass = InklingPass::with_logit_rows(&model, &stream, n * rows, n)?;
                    prefill.push((rows, pass));
                    prefill.len() - 1
                }
            };
            plan.push(index);
            at += rows;
        }
        let mut decode = InklingDecode::new(&model, &stream, n)?;
        let mut states = Vec::new();
        for _ in 0..n {
            // Bounded sliding caches, as a server would hold them.
            states.push(InklingState::bounded(
                &stream,
                &model,
                pp + tg + 1,
                chunk.min(pp),
            )?);
        }
        info!(
            "caches: {:.1} MiB a sequence of {} positions",
            states[0].cache_bytes() as f64 / (1u64 << 20) as f64,
            pp + tg + 1
        );
        let profiling = std::env::var_os("LLMCUDA_PROFILE").is_some();
        model.use_reference_kernels(std::env::var_os("LLMCUDA_INK_REFERENCE").is_some());
        {
            use llmcuda_cuda::kernels::inkling::DecodeAttention;
            match std::env::var("LLMCUDA_INK_DEC_ATTN").as_deref() {
                Ok("scalar") => model.use_decode_attention(DecodeAttention::Scalar),
                Ok("full") => model.use_decode_attention(DecodeAttention::MmaFull),
                _ => model.use_decode_attention(DecodeAttention::Mma),
            }
        }
        if profiling {
            decode.enable_profiling();
            for (_, pass) in &mut prefill {
                pass.enable_profiling();
            }
        }
        let rounds: usize = std::env::var("LLMCUDA_BENCH_ROUNDS")
            .ok()
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(3);
        for round in 0..rounds {
            for s in &mut states {
                s.reset(&stream)?;
            }
            stream.synchronize()?;
            let t0 = Instant::now();
            let mut at = 0;
            for &index in &plan {
                let (rows, pass) = &mut prefill[index];
                let tokens: Vec<i32> = (0..n)
                    .flat_map(|s| (0..*rows).map(move |i| token(s * 7919 + at + i)))
                    .collect();
                let mut segments: Vec<Segment<'_>> = states
                    .iter_mut()
                    .map(|state| Segment { state, rows: *rows })
                    .collect();
                pass.run(&stream, &mut segments, &tokens, None)?;
                at += *rows;
            }
            stream.synchronize()?;
            let t_pp = t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            for step in 0..tg {
                let tokens: Vec<i32> = (0..n).map(|s| token(s * 104_729 + step)).collect();
                let mut refs: Vec<&mut InklingState> = states.iter_mut().collect();
                decode.step(&stream, &mut refs, &tokens)?;
                decode.logits(&stream)?;
            }
            stream.synchronize()?;
            let t_tg = t1.elapsed().as_secs_f64();
            info!(
                "{} PP {pp} (chunk {chunk}) TG {tg} N {n}: prefill {:.3} s {:.1} tok/s | decode {:.3} s {:.2} tok/s ({:.2} ms/step)",
                if round == 0 { "warm-up" } else { "measured" },
                t_pp,
                (n * pp) as f64 / t_pp,
                t_tg,
                (n * tg) as f64 / t_tg,
                t_tg / tg as f64 * 1e3,
            );
        }
        if profiling {
            for (what, stages) in [
                ("prefill", prefill[0].1.stage_seconds()?.to_vec()),
                ("decode", decode.stage_seconds()?.to_vec()),
            ] {
                let total: f64 = stages.iter().map(|s| s.1).sum();
                for (stage, seconds) in stages {
                    info!(
                        "  {what:7} {:5.1}% {seconds:8.3} s  {stage}",
                        100.0 * seconds / total
                    );
                }
            }
        }
    }
    Ok(())
}

fn main() {
    let args = llmcuda_log::init_from_args();
    if let Err(e) = run(&args) {
        error!("{e}");
        std::process::exit(1);
    }
}
