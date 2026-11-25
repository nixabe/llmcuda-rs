//! The GDN prefill scan and what feeds it: `gdn_scan_prefill` against
//! `gdn_scan_prefill_fold`, and the convolution, SiLU, split and
//! normalization as separate launches against `gdn_scan_prelude`, at the
//! shape a batched prefill launches them at.
//!
//! The scan is the largest GDN kernel in a prefill pass (7.8 ms of one at
//! three 2,046-token sequences) and a whole-forward A/B needs a model load to
//! see it. This runs both kernels on synthetic inputs in seconds, and checks
//! that the folded kernel's outputs and final states are the unfolded
//! kernel's bit for bit, which is what it promises.
//!
//! Synthetic q, k, v, decay and beta: the kernels' cost does not depend on
//! the values.
//!
//! ```text
//! CUDA_VISIBLE_DEVICES=0 cargo run --release -p llmcuda-engine --bin bench_gdn_scan
//! ```
//!
//! `LLMCUDA_BENCH_SEQS` (default 3) and `LLMCUDA_BENCH_TOKENS` (default
//! 2046) set the batch and the tokens per sequence.

use std::process::ExitCode;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaSlice};
use llmcuda_cuda::device::{DeviceInfo, driver_available};
use llmcuda_cuda::kernels::gdn_chunked::GdnChunkedKernels;
use llmcuda_engine::block::gdn::{GdnBlock, GdnGeometry};
use llmcuda_model::config::ModelConfig;
use tracing::{error, info, warn};

const WARMUP: usize = 2;
const REPS: usize = 10;
const ROUNDS: usize = 4;

/// Xorshift64*, inline: `llmcuda_kernels::rng` is a dev-dependency and not
/// reachable from a binary.
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

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .filter(|&v| v > 0)
        .unwrap_or(default)
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let info = DeviceInfo::from_context(0, &ctx)?;
    if !info.is_supported() {
        warn!("SKIPPED: device 0 is below the sm_75 minimum");
        return Ok(());
    }
    let stream = ctx.default_stream();
    let g = ModelConfig::qwen3_6_35b_a3b().gdn;
    let head_dim = g.head_dim as usize;
    let value_heads = g.value_heads as usize;
    let qk_heads = g.qk_heads as usize;

    let seqs = env_usize("LLMCUDA_BENCH_SEQS", 3);
    let tokens = env_usize("LLMCUDA_BENCH_TOKENS", 2046);
    let total = seqs * tokens;
    info!(
        "device 0: {}; {seqs} sequence(s) of {tokens} tokens, head_dim {head_dim}, \
         {value_heads} value heads, {qk_heads} query/key heads",
        info.name
    );

    let mut kernels =
        GdnChunkedKernels::new(&ctx, head_dim, value_heads, qk_heads, g.chunk_len as usize)?;
    let mut scratch = kernels.scratch(&stream, total)?;

    let mut rng = Rng(0x5EED_6D17_2026);
    let mut gen_vec =
        |n: usize, lo: f32, hi: f32| -> Vec<f32> { (0..n).map(|_| rng.f32(lo, hi)).collect() };
    let d_q = stream.clone_htod(&gen_vec(total * qk_heads * head_dim, -1.0, 1.0))?;
    let d_k = stream.clone_htod(&gen_vec(total * qk_heads * head_dim, -1.0, 1.0))?;
    let d_v = stream.clone_htod(&gen_vec(total * value_heads * head_dim, -1.0, 1.0))?;
    let d_g = stream.clone_htod(&gen_vec(total * value_heads, -0.2, 0.0))?;
    let d_b = stream.clone_htod(&gen_vec(total * value_heads, 0.0, 1.0))?;
    let state0 = gen_vec(value_heads * head_dim * head_dim, -0.1, 0.1);
    let mut d_out = stream.alloc_zeros::<f32>(total * value_heads * head_dim)?;
    let mut states: Vec<CudaSlice<f32>> = (0..seqs)
        .map(|_| stream.clone_htod(&state0))
        .collect::<Result<_, _>>()?;

    // One launch of the batched scan from the same initial states.
    let mut launch = |kernels: &GdnChunkedKernels,
                      states: &mut [CudaSlice<f32>],
                      out: &mut CudaSlice<f32>|
     -> Result<(), Box<dyn std::error::Error>> {
        let ptrs: Vec<u64> = states
            .iter_mut()
            .map(|s| {
                use cudarc::driver::DevicePtrMut;
                s.device_ptr_mut(&stream).0
            })
            .collect();
        // SAFETY: every pointer is a live, distinct
        // `[value_heads][head_dim][head_dim]` state held mutably for the
        // call, and each flattened buffer was allocated above at
        // `[seqs][tokens][width]`.
        unsafe {
            kernels.scan_batch_raw(
                &stream,
                &mut scratch,
                &ptrs,
                &d_q,
                &d_k,
                &d_v,
                &d_g,
                &d_b,
                out,
                tokens,
            )?;
        }
        Ok(())
    };

    // Bits first: one launch each from the same state.
    let mut reference: Option<(Vec<f32>, Vec<f32>)> = None;
    for fold in [false, true] {
        kernels.set_scan_fold(fold);
        for s in states.iter_mut() {
            stream.memcpy_htod(&state0, s)?;
        }
        launch(&kernels, &mut states, &mut d_out)?;
        stream.synchronize()?;
        let out = stream.clone_dtoh(&d_out)?;
        let state = stream.clone_dtoh(&states[seqs - 1])?;
        stream.synchronize()?;
        match &reference {
            None => reference = Some((out, state)),
            Some((ref_out, ref_state)) => {
                let differ = |a: &[f32], b: &[f32]| {
                    a.iter()
                        .zip(b)
                        .filter(|(x, y)| x.to_bits() != y.to_bits())
                        .count()
                };
                let (o, s) = (differ(&out, ref_out), differ(&state, ref_state));
                if o != 0 || s != 0 {
                    return Err(format!(
                        "the folded scan differs from the unfolded one: {o} of {} outputs, \
                         {s} of {} final-state elements",
                        out.len(),
                        state.len()
                    )
                    .into());
                }
                info!(
                    "folded scan: {} outputs and {} final-state elements equal the unfolded \
                     scan's bit for bit",
                    out.len(),
                    state.len()
                );
            }
        }
    }

    let mut ms: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    for round in 0..ROUNDS {
        for i in 0..2 {
            let arm = (i + round) % 2;
            kernels.set_scan_fold(arm == 1);
            for _ in 0..WARMUP {
                launch(&kernels, &mut states, &mut d_out)?;
            }
            stream.synchronize()?;
            let t0 = Instant::now();
            for _ in 0..REPS {
                launch(&kernels, &mut states, &mut d_out)?;
            }
            stream.synchronize()?;
            ms[arm].push(t0.elapsed().as_secs_f64() * 1e3 / REPS as f64);
        }
    }
    let cell = |v: &[f64]| {
        let lo = v.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = v.iter().copied().fold(0.0f64, f64::max);
        format!("{lo:.3} - {hi:.3}")
    };
    info!(
        "normalize + scan, ms a launch over {ROUNDS} rounds: unfolded {}, folded {}",
        cell(&ms[0]),
        cell(&ms[1])
    );

    // Everything from the q/k/v projection's output to the scan's: the
    // convolution a sequence, the SiLU with its split, the normalization and
    // the scan as separate launches, against the fused prelude and the scan.
    let geometry = GdnGeometry::from_config(&ModelConfig::qwen3_6_35b_a3b(), total, 1e-6);
    let block = GdnBlock::new(&ctx, geometry)?;
    let conv_dim = geometry.conv_dim();
    let conv_kernel = geometry.conv_kernel;
    let seq_conv = tokens * conv_dim;
    let d_x = stream.clone_htod(&gen_vec(total * conv_dim, -2.0, 2.0))?;
    let d_w = stream.clone_htod(&gen_vec(conv_dim * conv_kernel, -0.5, 0.5))?;
    let mut caches: Vec<CudaSlice<f32>> = (0..seqs)
        .map(|_| stream.alloc_zeros::<f32>(geometry.conv_state_len()))
        .collect::<Result<_, _>>()?;
    let xs: Vec<CudaSlice<f32>> = (0..seqs)
        .map(|seq| {
            let mut own = stream.alloc_zeros::<f32>(seq_conv)?;
            stream.memcpy_dtod(&d_x.slice(seq * seq_conv..(seq + 1) * seq_conv), &mut own)?;
            Ok::<_, cudarc::driver::DriverError>(own)
        })
        .collect::<Result<_, _>>()?;
    let mut conv_raw: Vec<CudaSlice<f32>> = (0..seqs)
        .map(|_| stream.alloc_zeros::<f32>(seq_conv))
        .collect::<Result<_, _>>()?;
    // The pass convolves every sequence into one flattened buffer, which the
    // SiLU then reads. Here each sequence convolves into its own buffer and
    // the SiLU reads a flattened one of the same size holding other values:
    // the same launches over the same bytes, which is all the timing needs.
    let conv_all = stream.clone_htod(&gen_vec(total * conv_dim, -2.0, 2.0))?;
    let mut silu = stream.alloc_zeros::<f32>(total * conv_dim)?;
    let mut q = stream.alloc_zeros::<f32>(total * geometry.key_dim())?;
    let mut k = stream.alloc_zeros::<f32>(total * geometry.key_dim())?;
    let mut v = stream.alloc_zeros::<f32>(total * geometry.value_dim())?;
    kernels.set_scan_fold(true);

    let mut whole = |fused: bool| -> Result<(), Box<dyn std::error::Error>> {
        use cudarc::driver::DevicePtrMut;
        let state_ptrs: Vec<u64> = states
            .iter_mut()
            .map(|s| s.device_ptr_mut(&stream).0)
            .collect();
        if fused {
            let cache_ptrs: Vec<u64> = caches
                .iter_mut()
                .map(|c| c.device_ptr_mut(&stream).0)
                .collect();
            // SAFETY: live, distinct states and caches of the geometry's
            // lengths, held for the call; buffers allocated above at
            // `[seqs][tokens][width]`.
            unsafe {
                kernels.prelude_scan_batch_raw(
                    &stream,
                    &mut scratch,
                    &state_ptrs,
                    &cache_ptrs,
                    &d_x,
                    &d_w,
                    conv_kernel,
                    &mut v,
                    &d_g,
                    &d_b,
                    &mut d_out,
                    tokens,
                )?;
            }
            for (seq, cache) in caches.iter_mut().enumerate() {
                block.layer_ops().conv1d_advance_state(
                    &stream,
                    &xs[seq],
                    cache,
                    tokens,
                    conv_dim,
                    conv_kernel,
                )?;
            }
        } else {
            for (seq, cache) in caches.iter_mut().enumerate() {
                block.layer_ops().conv1d(
                    &stream,
                    &xs[seq],
                    &d_w,
                    cache,
                    &mut conv_raw[seq],
                    tokens,
                    conv_dim,
                    conv_kernel,
                )?;
            }
            block.silu_split_qkv(&stream, &conv_all, &mut silu, &mut q, &mut k, &mut v, total)?;
            // SAFETY: as above.
            unsafe {
                kernels.scan_batch_raw(
                    &stream,
                    &mut scratch,
                    &state_ptrs,
                    &q,
                    &k,
                    &v,
                    &d_g,
                    &d_b,
                    &mut d_out,
                    tokens,
                )?;
            }
        }
        Ok(())
    };
    let mut ms: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    for round in 0..ROUNDS {
        for i in 0..2 {
            let arm = (i + round) % 2;
            for _ in 0..WARMUP {
                whole(arm == 1)?;
            }
            stream.synchronize()?;
            let t0 = Instant::now();
            for _ in 0..REPS {
                whole(arm == 1)?;
            }
            stream.synchronize()?;
            ms[arm].push(t0.elapsed().as_secs_f64() * 1e3 / REPS as f64);
        }
    }
    info!(
        "convolution to scan, ms a layer over {ROUNDS} rounds: separate launches {}, fused \
         prelude {} (both with the folded scan)",
        cell(&ms[0]),
        cell(&ms[1])
    );
    Ok(())
}
