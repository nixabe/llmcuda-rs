//! The fp16 weight-only GEMM (`hgemm`) against the int8 split path, at the
//! dense models' prefill shapes, with synthetic Q8_0 weights, and cuBLASLt's
//! fp16 GEMM over the same shape as the card's practical ceiling.
//!
//! The `hgemm` arm is the production call: [`HgemmKernels::gemm`] with the
//! plan it picks and a split-K workspace sized the way the blocks size it.
//! Both candidate arms include their activation narrowing (`quantize_rows` to
//! int8 for the integer path, `to_half` for fp16), because each is a cost the
//! other does not pay. CUDA events; each shape alternates the arms over three
//! rounds so drift lands on all of them. Run it on an idle card after a
//! moment of load: this card's sustained clocks sit ~20% under its burst.
//!
//! `LLMCUDA_HGEMM_TOKENS` sets the comma-separated token counts (default
//! 512,2048).

use std::sync::Arc;

use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};
use cudarc::driver::{CudaContext, CudaStream, sys::CUevent_flags};
use half::f16;
use llmcuda_cuda::kernels::hgemm::{HalfWeight, HgemmKernels};
use llmcuda_cuda::kernels::mma::MmaKernels;
use tracing::{error, info};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// `(name, output rows, contraction)`: Qwen3.8-27B then Clef-Flash.
const SHAPES: [(&str, usize, usize); 9] = [
    ("27B ffn gate/up", 17_408, 5_120),
    ("27B ffn down", 5_120, 17_408),
    ("27B gdn qkv", 10_240, 5_120),
    ("27B gdn out", 5_120, 6_144),
    ("27B attn q", 12_288, 5_120),
    ("27B attn k/v", 1_024, 5_120),
    ("clef ffn gate/up", 12_288, 4_096),
    ("clef ffn down", 4_096, 12_288),
    ("clef gdn qkv", 8_192, 4_096),
];

fn measure(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    mut body: impl FnMut() -> Result<()>,
) -> Result<f32> {
    for _ in 0..3 {
        body()?;
    }
    stream.synchronize()?;
    let start = ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    let stop = ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    start.record(stream)?;
    for _ in 0..10 {
        body()?;
    }
    stop.record(stream)?;
    stop.synchronize()?;
    Ok(start.elapsed_ms(&stop)? / 10.0)
}

fn q8_0_bytes(rows: usize, k: usize) -> Vec<u8> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut bytes = Vec::with_capacity(rows * k / 32 * 34);
    for _ in 0..rows * k / 32 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let bits: u16 = 0x2000 | (state % 0x800) as u16;
        bytes.extend_from_slice(&bits.to_le_bytes());
        for j in 0..32 {
            bytes.push(((state >> (j % 56)) as u8).wrapping_add(j as u8));
        }
    }
    bytes
}

fn run() -> Result<()> {
    let tokens: Vec<usize> = std::env::var("LLMCUDA_HGEMM_TOKENS")
        .ok()
        .map(|v| v.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![512, 2048]);
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let mma = MmaKernels::new(&ctx)?;
    let hg = HgemmKernels::new(&ctx)?;
    let blas = CudaBlasLT::new(Arc::clone(&stream))?;
    let max_t = *tokens.iter().max().unwrap_or(&512);
    for (name, rows, k) in SHAPES {
        let bytes = q8_0_bytes(rows, k);
        let src = stream.clone_htod(&bytes)?;
        let mut wq = stream.alloc_zeros::<i8>(rows * k)?;
        let mut ws = stream.alloc_zeros::<u16>(rows * k / 32)?;
        mma.repack_q8_0_half(&stream, &src, &mut wq, &mut ws, rows * k)?;
        drop(src);
        let x: Vec<f32> = (0..max_t * k)
            .map(|i| ((i % 97) as f32 - 48.0) / 37.0)
            .collect();
        let xd = stream.clone_htod(&x)?;
        let mut xq = stream.alloc_zeros::<i8>(max_t * k)?;
        let mut xs = stream.alloc_zeros::<f32>(max_t * k / 32)?;
        let mut xh = stream.alloc_zeros::<u16>(max_t * k)?;
        let mut workspace = stream.alloc_zeros::<f32>(hg.workspace_elems(max_t, rows, k).max(1))?;
        // cuBLASLt's fp16 GEMM over the same shape: the card's practical
        // tensor-core ceiling, not a candidate (its output is fp16).
        let x16 = stream.clone_htod(&x.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>())?;
        let w16 = stream.clone_htod(
            &(0..rows * k)
                .map(|i| f16::from_f32(((i % 89) as f32 - 44.0) / 512.0))
                .collect::<Vec<_>>(),
        )?;
        for &t in &tokens {
            let mut out = stream.alloc_zeros::<f32>(t * rows)?;
            let mut out16 = stream.alloc_zeros::<f16>(t * rows)?;
            let mut int8 = Vec::new();
            let mut half = Vec::new();
            let mut lt = Vec::new();
            for _ in 0..3 {
                int8.push(measure(&ctx, &stream, || {
                    mma.quantize_rows(&stream, &xd, &mut xq, &mut xs, t, k)?;
                    mma.q8_0_proj_split_half(&stream, &wq, &ws, &xq, &xs, &mut out, k, rows, t)?;
                    Ok(())
                })?);
                half.push(measure(&ctx, &stream, || {
                    hg.to_half(&stream, &xd, &mut xh, t * k)?;
                    hg.gemm(
                        &stream,
                        HalfWeight::Q8_0 {
                            q: &wq,
                            scales: &ws,
                        },
                        &xh,
                        &mut out,
                        Some(&mut workspace),
                        t,
                        rows,
                        k,
                        rows,
                        false,
                    )?;
                    Ok(())
                })?);
                lt.push(measure(&ctx, &stream, || {
                    let cfg = MatmulConfig {
                        transa: true,
                        transb: false,
                        transc: false,
                        m: rows as u64,
                        n: t as u64,
                        k: k as u64,
                        alpha: 1.0,
                        lda: k as i64,
                        ldb: k as i64,
                        beta: 0.0,
                        ldc: rows as i64,
                        stride_a: None,
                        stride_b: None,
                        stride_c: None,
                        stride_bias: None,
                        batch_size: None,
                    };
                    // SAFETY: `w16` is rows x k, `x16` at least t x k, `out16`
                    // t x rows, matching the column-major shapes above.
                    unsafe { blas.matmul(cfg, &w16, &x16, &mut out16, None, None) }?;
                    Ok(())
                })?);
            }
            let tflops = |ms: f32| 2.0 * (t * rows * k) as f64 / (f64::from(ms) * 1e9);
            let best = |v: &[f32]| v.iter().cloned().fold(f32::INFINITY, f32::min);
            let plan = hg.plan(t, rows, k);
            info!(
                "{name:18} t={t:5}  int8 {:7.3} ms ({:5.1} TOP/s)  hgemm {:7.3} ms ({:5.1} TFLOP/s, {} x{})  x{:.2}  cublasLt f16 {:7.3} ms ({:5.1})   rounds int8 {:?} hgemm {:?}",
                best(&int8),
                tflops(best(&int8)),
                best(&half),
                tflops(best(&half)),
                if plan.wide { "256" } else { "128" },
                plan.splits,
                best(&int8) / best(&half),
                best(&lt),
                tflops(best(&lt)),
                int8,
                half,
            );
        }
    }
    Ok(())
}

fn main() {
    llmcuda_log::init_from_args();
    if let Err(e) = run() {
        error!("hgemm benchmark failed: {e}");
        std::process::exit(1);
    }
}
