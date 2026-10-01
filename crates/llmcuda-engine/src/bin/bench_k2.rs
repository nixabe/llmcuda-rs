//! K2 dense and routed value projections in isolation, with real GGUF weights.
//! Use LLMCUDA_MODEL and optionally LLMCUDA_K2_N (default 1,3,128,512).
//! CUDA events include dispatch for the grouped value path. Each round
//! alternates scalar and tiled kernels; full-model benches remain the gate.
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, sys::CUevent_flags};
use llmcuda_cuda::kernels::{
    k2::{K2Dispatch, K2Kernels},
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
    for name in ["blk.3.attn_q.weight", "blk.3.attn_v_exps.weight"] {
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
        info!("{name}: {quant:?}, inner {inner}, rows {rows}, experts {experts}");
        for &tokens in &shapes {
            let topk = if experts == 1 { 1 } else { 4 };
            let input: Vec<f32> = (0..tokens * inner)
                .map(|i| (i as f32 * 0.013).sin())
                .collect();
            let ids: Vec<i32> = (0..tokens)
                .flat_map(|t| (0..topk).map(move |k| ((t * 13 + k * 7) % experts) as i32))
                .collect();
            let dx = stream.clone_htod(&input)?;
            let di = stream.clone_htod(&ids)?;
            let mut out = stream.alloc_zeros::<f32>(tokens * topk * rows)?;
            let mut dispatch = K2Dispatch::new(&stream, tokens, topk, experts)?;
            for round in 1..=3 {
                for tile in if experts == 1 {
                    &[1, 4, 8][..]
                } else {
                    &[1, 8][..]
                } {
                    let ms = measure(&ctx, &stream, || {
                        if experts > 1 && *tile == 8 {
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
                    info!("T={tokens} round={round} tile={tile}: {ms:.4} ms");
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
