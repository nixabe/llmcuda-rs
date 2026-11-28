//! The MoE router projection alone, register-tiled against tiled-GEMM, at
//! the widths a pass actually launches it at.
//!
//! `logits[t][e] = dot(ffn_gate_inp[e], normed[t])` is the smallest matmul in
//! the layer, and register-tiled it costs a byte of L2 traffic per
//! multiply-add. `moe_block_router_logits_wide` is the same product through
//! shared-memory tiles, at one warp or eight a block; this measures the three
//! against each other and finds the widths where each stops paying, which is
//! what `ROUTER_WIDE_MIN_TOKENS` and `ROUTER_WIDE_FULL_MIN_TOKENS` are set
//! from.
//!
//! Synthetic weights and activations: the kernels' cost does not depend on
//! the values, so this needs no model file and runs in seconds. Each width
//! also checks a sample of the wide kernel's logits against the host's
//! `f32::mul_add` chain, bit for bit — the order the kernel promises.
//!
//! ```text
//! CUDA_VISIBLE_DEVICES=0 cargo run --release -p llmcuda-engine --bin bench_router
//! ```
//!
//! `LLMCUDA_BENCH_WIDTHS` overrides the comma-separated token counts.

use std::process::ExitCode;
use std::time::Instant;

use cudarc::driver::CudaContext;
use llmcuda_cuda::device::{DeviceInfo, driver_available};
use llmcuda_engine::block::moe::{DEFAULT_RMS_EPS, MoeBlock};
use llmcuda_model::config::ModelConfig;
use tracing::{error, info, warn};

const DEFAULT_WIDTHS: [usize; 9] = [128, 512, 768, 1024, 1536, 2048, 3072, 4096, 6138];

const WARMUP: usize = 5;
const REPS: usize = 40;
const ROUNDS: usize = 3;

/// Xorshift64.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f32(&mut self, lo: f32, hi: f32) -> f32 {
        let u = (self.next() >> 40) as f32 / 16_777_216.0;
        lo + u * (hi - lo)
    }
}

fn main() -> ExitCode {
    llmcuda_log::init_from_args();

    if !driver_available() {
        warn!("SKIPPED: no CUDA driver present");
        return ExitCode::SUCCESS;
    }
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            error!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn widths() -> Vec<usize> {
    std::env::var("LLMCUDA_BENCH_WIDTHS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|w| w.trim().parse().ok())
                .collect::<Vec<usize>>()
        })
        .filter(|w| !w.is_empty())
        .unwrap_or_else(|| DEFAULT_WIDTHS.to_vec())
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let info = DeviceInfo::from_context(0, &ctx)?;
    if !info.is_supported() {
        warn!("SKIPPED: device 0 is below the sm_75 minimum");
        return Ok(());
    }
    let stream = ctx.default_stream();
    let config = ModelConfig::qwen3_6_35b_a3b();
    let hidden = config.hidden_size as usize;

    let mut rng = Rng(0x5EED_0707_2026);
    info!("device 0: {}", info.name);
    info!(
        "{:>7}  {:>18}  {:>18}  {:>18}",
        "tokens", "register-tiled ms", "one warp ms", "eight warps ms"
    );

    for tokens in widths() {
        let g = MoeBlock::geometry_for(
            &config,
            llmcuda_engine::forward::moe_block_size(tokens),
            tokens,
        )
        .ok_or("the routed model has a MoE geometry")?;
        let mut block = MoeBlock::new(&ctx, &stream, g, DEFAULT_RMS_EPS)?;
        if !block.tensor_cores_enabled() {
            warn!("SKIPPED: integer tensor cores unavailable on this device");
            return Ok(());
        }
        block.publish_tokens(&stream, tokens)?;

        let w: Vec<f32> = (0..g.num_experts * hidden)
            .map(|_| rng.f32(-0.05, 0.05))
            .collect();
        let x: Vec<f32> = (0..tokens * hidden).map(|_| rng.f32(-4.0, 4.0)).collect();
        let d_w = stream.clone_htod(&w)?;
        let d_x = stream.clone_htod(&x)?;

        // 0 is the register-tiled kernel, 1 the one-warp GEMM, 2 the
        // eight-warp one.
        let time = |block: &mut MoeBlock, arm: usize| -> Result<f64, Box<dyn std::error::Error>> {
            match arm {
                0 => block.set_router_wide_min_tokens(usize::MAX, usize::MAX),
                1 => block.set_router_wide_min_tokens(0, usize::MAX),
                _ => block.set_router_wide_min_tokens(0, 0),
            }
            assert_eq!(block.router_is_wide(), arm != 0);
            for _ in 0..WARMUP {
                block.route_logits(&stream, &d_w, &d_x)?;
            }
            stream.synchronize()?;
            let t0 = Instant::now();
            for _ in 0..REPS {
                block.route_logits(&stream, &d_w, &d_x)?;
            }
            stream.synchronize()?;
            Ok(t0.elapsed().as_secs_f64() * 1e3 / REPS as f64)
        };

        let mut ms: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for round in 0..ROUNDS {
            // Rotate which arm goes first.
            for i in 0..3 {
                let arm = (i + round) % 3;
                ms[arm].push(time(&mut block, arm)?);
            }
        }

        // Check a sample of both wide kernels against the host, bit for bit.
        let mut mismatches = 0usize;
        let mut checked = 0usize;
        for full in [usize::MAX, 0] {
            block.set_router_wide_min_tokens(0, full);
            block.route_logits(&stream, &d_w, &d_x)?;
            stream.synchronize()?;
            let logits = stream.clone_dtoh(block.router_logits())?;
            stream.synchronize()?;
            for sample in 0..256 {
                let t = (rng.next() as usize) % tokens;
                let e = if sample < 8 {
                    sample * 37 % g.num_experts
                } else {
                    (rng.next() as usize) % g.num_experts
                };
                let mut acc = 0.0f32;
                for j in 0..hidden {
                    acc = w[e * hidden + j].mul_add(x[t * hidden + j], acc);
                }
                checked += 1;
                if acc.to_bits() != logits[t * g.num_experts + e].to_bits() {
                    mismatches += 1;
                }
            }
        }

        let span = |v: &[f64]| {
            let lo = v.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = v.iter().copied().fold(0.0f64, f64::max);
            (lo, hi)
        };
        let cell = |v: &[f64]| {
            let (lo, hi) = span(v);
            format!("{lo:.4} - {hi:.4}")
        };
        info!(
            "{tokens:>7}  {:>18}  {:>18}  {:>18}",
            cell(&ms[0]),
            cell(&ms[1]),
            cell(&ms[2]),
        );
        if mismatches != 0 {
            return Err(format!(
                "{tokens} tokens: {mismatches} of {checked} sampled wide logits differ from the \
                 host's mul_add chain"
            )
            .into());
        }
    }
    info!("every sampled logit of both wide kernels matched the host's mul_add chain bit for bit");
    Ok(())
}
