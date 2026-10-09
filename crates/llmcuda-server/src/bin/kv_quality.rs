//! What a quantized KV cache costs the model, on real text.
//!
//! Teacher-forces one tokenized text through the model one token at a time —
//! every position's attention reads the cache the way decode does — once with
//! a binary16 cache and once per format under test, and reports against the
//! binary16 run: perplexity, the KL divergence of each position's next-token
//! distribution (mean, p99, max) and how often the top-1 token agrees. This
//! is llama.cpp's `llama-perplexity --kl-divergence` measurement, made here
//! because a differential kernel test can only show the kernels read the
//! quantized values correctly, not what quantizing them does.
//!
//! LLMCUDA_MODEL (GGUF), LLMCUDA_KV_QUALITY_TEXT (UTF-8 file),
//! LLMCUDA_KV_QUALITY_TOKENS (default 2048), LLMCUDA_KV_QUALITY_FORMATS
//! (comma-separated K/V pairs, default `q8_0/q8_0,q8_0/f16,f16/q8_0`). The
//! binary16 run's log-probabilities are held in host RAM: tokens x vocab x 4 B.
//!
//! `LLMCUDA_KV_QUALITY_FLOOR_SPLITS=n` adds one more binary16 run with the
//! tensor-core decode's split count changed to `n` (`LLMCUDA_DEC_MMA_SPLITS`):
//! the same cache and the same values, summed in a different order. What
//! that costs is the floor any perturbation of attention reaches on this
//! model — K2 routes every token through top-k expert choices, which turn
//! rounding-level differences into different experts.
#[path = "../tokenizer.rs"]
#[allow(dead_code)]
mod tokenizer;

use cudarc::driver::{CudaContext, CudaStream};
use llmcuda_engine::{
    DeviceWeights,
    forward::{Forward, arena_holds_model_entry},
};
use llmcuda_gguf::GgufFile;
use llmcuda_model::{KvCacheType, KvCacheTypes, ModelConfig, WeightSchema};
use std::sync::Arc;
use tracing::{error, info};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn env(name: &str) -> Result<String> {
    std::env::var(name).map_err(|_| format!("set {name}").into())
}

/// Teacher-force `tokens` through a model built for `kv`, calling `each`
/// with every position's logits. Weights are loaded per call and dropped
/// with it, so two formats never hold two copies on the card.
fn teacher_force(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    file: &GgufFile,
    base: &ModelConfig,
    kv: KvCacheTypes,
    tokens: &[i32],
    mut each: impl FnMut(usize, &[f32]),
) -> Result<()> {
    let config = ModelConfig {
        kv_cache: kv,
        ..base.clone()
    };
    let schema = WeightSchema::new(&config);
    let directory = schema
        .resolve(file)
        .map_err(|e| format!("weight schema: {e:?}"))?;
    let (weights, _) = DeviceWeights::load_where_entry(ctx, stream, file, &directory, |r, t| {
        arena_holds_model_entry(&config, r, t)
    })?;
    let mut pass = Forward::new(ctx, stream, file, &directory, &weights, config, 1)?;
    let mut state = pass.new_state(stream, tokens.len())?;
    for (position, &token) in tokens.iter().enumerate() {
        pass.run(stream, &mut state, &[token], |_, _| {})?;
        each(position, &stream.clone_dtoh(pass.logits())?);
    }
    Ok(())
}

/// Natural-log softmax, in f64 so 250K-way sums do not lose the tail.
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|&x| (x as f64 - max).exp()).sum();
    let log_sum = max + sum.ln();
    logits
        .iter()
        .map(|&x| (x as f64 - log_sum) as f32)
        .collect()
}

fn argmax(x: &[f32]) -> usize {
    x.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 { (i, v) } else { best }
        })
        .0
}

fn run() -> Result<()> {
    let model_path = env("LLMCUDA_MODEL")?;
    let text = std::fs::read_to_string(env("LLMCUDA_KV_QUALITY_TEXT")?)?;
    let limit: usize = std::env::var("LLMCUDA_KV_QUALITY_TOKENS")
        .ok()
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(2048);
    let formats: Vec<KvCacheTypes> = std::env::var("LLMCUDA_KV_QUALITY_FORMATS")
        .unwrap_or_else(|_| "q8_0/q8_0,q8_0/f16,f16/q8_0".into())
        .split(',')
        .map(|pair| {
            let (k, v) = pair
                .split_once('/')
                .ok_or_else(|| format!("format pair `{pair}` is not K/V"))?;
            Ok(KvCacheTypes {
                k: k.parse::<KvCacheType>()?,
                v: v.parse::<KvCacheType>()?,
            })
        })
        .collect::<Result<_>>()?;

    let tok = tokenizer::from_gguf(std::path::Path::new(&model_path))?;
    let encoding = tok.encode(text.as_str(), true).map_err(|e| e.to_string())?;
    let mut tokens: Vec<i32> = encoding.get_ids().iter().map(|&t| t as i32).collect();
    // One more token than positions scored: the last position's target.
    tokens.truncate(limit + 1);
    if tokens.len() < 2 {
        return Err("the text tokenizes to fewer than two tokens".into());
    }
    let scored = tokens.len() - 1;

    let ctx = CudaContext::new(0)?;
    // SAFETY: every upload, pass and readback uses this one stream.
    unsafe { ctx.disable_event_tracking() };
    let stream = ctx.new_stream()?;
    let file = GgufFile::open(&model_path)?;
    let base = ModelConfig::from_gguf(&file)?;
    let vocab = base.vocab_size as usize;
    info!(
        "{} positions of {} ({} tokens of text), vocab {vocab}",
        scored,
        model_path,
        encoding.get_ids().len()
    );

    // The reference: binary16 log-probabilities for every scored position.
    let mut reference = vec![0.0f32; scored * vocab];
    let mut reference_nll = 0.0f64;
    teacher_force(
        &ctx,
        &stream,
        &file,
        &base,
        KvCacheTypes::F16,
        &tokens[..scored],
        |p, logits| {
            let lp = log_softmax(logits);
            reference_nll -= lp[tokens[p + 1] as usize] as f64;
            reference[p * vocab..(p + 1) * vocab].copy_from_slice(&lp);
        },
    )?;
    info!(
        "K f16  V f16   perplexity {:.4}",
        (reference_nll / scored as f64).exp()
    );

    let floor = std::env::var("LLMCUDA_KV_QUALITY_FLOOR_SPLITS")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?;
    let runs: Vec<(KvCacheTypes, Option<usize>)> = floor
        .map(|splits| (KvCacheTypes::F16, Some(splits)))
        .into_iter()
        .chain(formats.into_iter().map(|kv| (kv, None)))
        .collect();
    for (kv, splits) in runs {
        if let Some(splits) = splits {
            // SAFETY: single-threaded here; the variable is read when the
            // next pass's kernel sets are built, and removed after it.
            unsafe { std::env::set_var("LLMCUDA_DEC_MMA_SPLITS", splits.to_string()) };
        }
        let mut nll = 0.0f64;
        let mut kl = Vec::with_capacity(scored);
        let mut agree = 0usize;
        teacher_force(
            &ctx,
            &stream,
            &file,
            &base,
            kv,
            &tokens[..scored],
            |p, logits| {
                let lp = log_softmax(logits);
                let r = &reference[p * vocab..(p + 1) * vocab];
                nll -= lp[tokens[p + 1] as usize] as f64;
                kl.push(
                    r.iter()
                        .zip(&lp)
                        .map(|(&a, &b)| (a as f64).exp() * (a as f64 - b as f64))
                        .sum::<f64>()
                        .max(0.0),
                );
                agree += usize::from(argmax(r) == argmax(&lp));
            },
        )?;
        if splits.is_some() {
            // SAFETY: as above.
            unsafe { std::env::remove_var("LLMCUDA_DEC_MMA_SPLITS") };
        }
        let mean = kl.iter().sum::<f64>() / scored as f64;
        let mut sorted = kl.clone();
        sorted.sort_by(f64::total_cmp);
        let p99 = sorted[(scored * 99 / 100).min(scored - 1)];
        let label = match splits {
            Some(n) => format!("K f16   V f16   decode splits {n}"),
            None => format!("K {:<5} V {:<5}", kv.k.name(), kv.v.name()),
        };
        info!(
            "{label} perplexity {:.4}  KL vs f16 mean {:.6} p99 {:.6} max {:.6}  top-1 agreement {:.2}% ({agree}/{scored})",
            (nll / scored as f64).exp(),
            mean,
            p99,
            sorted[scored - 1],
            100.0 * agree as f64 / scored as f64,
        );
    }
    Ok(())
}

fn main() {
    llmcuda_log::init_from_args();
    if let Err(e) = run() {
        error!("{e}");
        std::process::exit(1);
    }
}
