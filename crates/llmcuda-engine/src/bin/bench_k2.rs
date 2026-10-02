//! K2 dense and routed value projections in isolation, with real GGUF weights.
//! Use LLMCUDA_MODEL and optionally LLMCUDA_K2_N (default 1,3,128,512).
//! LLMCUDA_K2_SHARED_ROUTES selects identical experts across token rows.
//! LLMCUDA_K2_INTEGER_ONLY skips the scalar controls; LLMCUDA_K2_TENSORS
//! selects a comma-separated list of GGUF tensor names.
//! CUDA events include dispatch for the grouped value path. Each round
//! alternates scalar and tiled kernels; full-model benches remain the gate.
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, sys::CUevent_flags};
use llmcuda_cuda::kernels::{
    k2::{K2Dispatch, K2Kernels},
    k2_gemm::K2Gemm,
    moe::{ExpertQuant, to_device_layout},
};
use llmcuda_gguf::{GgmlType, GgufFile};
use tracing::{error, info};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

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
    for _ in 0..20 {
        body()?;
    }
    stop.record(stream)?;
    stop.synchronize()?;
    Ok(start.elapsed_ms(&stop)? / 20.0)
}

fn run() -> Result<()> {
    let path = std::env::var("LLMCUDA_MODEL")?;
    let file = GgufFile::open(&path)?;
    let ctx = CudaContext::new(0)?;
    // SAFETY: this harness uses one stream for every allocation and launch.
    unsafe {
        ctx.disable_event_tracking();
    }
    let stream = ctx.default_stream();
    let ops = K2Kernels::new(&ctx)?;
    let shapes: Vec<usize> = std::env::var("LLMCUDA_K2_N")
        .unwrap_or_else(|_| "1,3,128,512".into())
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    let names = std::env::var("LLMCUDA_K2_TENSORS")
        .unwrap_or_else(|_| "blk.3.attn_q.weight,blk.3.attn_v_exps.weight".into());
    let integer_only = std::env::var_os("LLMCUDA_K2_INTEGER_ONLY").is_some();
    for name in names.split(',') {
        let tensor = file
            .tensors()
            .iter()
            .find(|t| t.name == name)
            .ok_or("missing K2 projection")?;
        let (inner, rows, experts) = (
            tensor.dims[0] as usize,
            tensor.dims[1] as usize,
            tensor.dims.get(2).copied().unwrap_or(1) as usize,
        );
        let quant = match tensor.ggml_type {
            GgmlType::Q4K => ExpertQuant::Q4K,
            GgmlType::Q6K => ExpertQuant::Q6K,
            other => return Err(format!("unsupported benchmark format {other:?}").into()),
        };
        let dw = stream.clone_htod(&*to_device_layout(
            quant,
            file.tensor_bytes(name).ok_or("unreadable projection")?,
        ))?;
        let packed = K2Gemm::repack(&ctx, &stream, &dw, quant, inner, rows, experts)?;
        info!("{name}: {quant:?}, inner {inner}, rows {rows}, experts {experts}");
        for &tokens in &shapes {
            let topk = if experts == 1 { 1 } else { 4 };
            let input: Vec<f32> = (0..tokens * inner)
                .map(|i| (i as f32 * 0.013).sin())
                .collect();
            let stride = if std::env::var_os("LLMCUDA_K2_SHARED_ROUTES").is_some() {
                0
            } else {
                13
            };
            let ids: Vec<i32> = (0..tokens)
                .flat_map(|t| (0..topk).map(move |k| ((t * stride + k * 7) % experts) as i32))
                .collect();
            let dx = stream.clone_htod(&input)?;
            let di = stream.clone_htod(&ids)?;
            let mut out = stream.alloc_zeros::<f32>(tokens * topk * rows)?;
            let mut dispatch = K2Dispatch::new(&stream, tokens, topk, experts)?;
            let mut bd = K2Dispatch::new_gemm(&stream, tokens, topk, experts)?;
            let mut gemm = K2Gemm::new_with_activation_bits(
                &ctx,
                &stream,
                tokens,
                topk,
                experts,
                inner * rows * experts,
                inner,
                rows,
                std::env::var("LLMCUDA_K2_BITS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(16),
            )?;
            for round in 1..=3 {
                let mut tiles = if integer_only {
                    vec![0]
                } else if experts == 1 {
                    vec![1, 4, 8, 0]
                } else {
                    vec![1, 8, 0]
                };
                if round % 2 == 0 {
                    tiles.reverse();
                }
                for tile in &tiles {
                    let ms = measure(&ctx, &stream, || {
                        if *tile == 0 {
                            if experts > 1 {
                                ops.dispatch_gemm(&stream, &di, &mut bd)?;
                            }
                            gemm.project(
                                &stream,
                                &packed,
                                quant,
                                &dx,
                                &mut out,
                                inner,
                                rows,
                                experts,
                                if experts > 1 { Some(&bd) } else { None },
                                false,
                            )?;
                        } else if experts > 1 && *tile == 8 {
                            ops.project_grouped(
                                &stream,
                                &dw,
                                quant,
                                &dx,
                                &di,
                                &mut out,
                                inner,
                                rows,
                                &mut dispatch,
                            )?;
                        } else {
                            ops.project_with_tile(
                                &stream,
                                &dw,
                                Some(quant),
                                &dx,
                                &di,
                                &mut out,
                                inner,
                                rows,
                                experts,
                                topk,
                                experts > 1,
                                *tile,
                            )?;
                        }
                        Ok(())
                    })?;
                    let path = if *tile == 0 {
                        "integer".to_owned()
                    } else {
                        format!("scalar-tile-{tile}")
                    };
                    info!("T={tokens} round={round} path={path}: {ms:.4} ms");
                }
            }
        }
    }
    Ok(())
}
fn main() {
    llmcuda_log::init_from_args();
    if let Err(e) = run() {
        error!("K2 benchmark failed: {e}");
        std::process::exit(1);
    }
}
