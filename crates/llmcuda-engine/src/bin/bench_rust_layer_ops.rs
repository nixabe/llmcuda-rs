//! Interleaved CUDA-event timing of the Rust layer ops against their CUDA
//! C++ originals, through the launchers the model uses.
//!
//! `rust_kernels_differential` is the correctness gate; this only times.
//! Each arm is 100 launches captured in a CUDA graph, so the figure is the
//! kernel and not the launch path, and the arms alternate order every pair.
use std::sync::Arc;

use cudarc::driver::sys::{CUevent_flags, CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr};
use llmcuda_cuda::kernels::layer_ops::{LayerOpsError, LayerOpsKernels};
use llmcuda_kernels::conv::gdn_conv_channels;
use llmcuda_model::config::ModelConfig;
use tracing::info;

type Error = Box<dyn std::error::Error>;
const LAUNCHES: usize = 100;
const PAIRS: usize = 6;

fn time_pairs(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    arms: &[LayerOpsKernels; 2],
    case: &str,
    mut launch: impl FnMut(&LayerOpsKernels) -> Result<(), LayerOpsError>,
) -> Result<(), Error> {
    stream.synchronize()?;
    let mut graphs = Vec::new();
    for arm in arms {
        stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        for _ in 0..LAUNCHES {
            launch(arm)?;
        }
        graphs.push(
            stream
                .end_capture(
                    CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                )?
                .ok_or("empty graph")?,
        );
    }
    for _ in 0..10 {
        for graph in &graphs {
            graph.launch()?;
        }
    }
    stream.synchronize()?;
    let start = ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    let stop = ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    for pair in 0..PAIRS {
        let mut us = [0.0; 2];
        for arm in if pair % 2 == 0 { [0, 1] } else { [1, 0] } {
            start.record(stream)?;
            graphs[arm].launch()?;
            stop.record(stream)?;
            stream.synchronize()?;
            us[arm] = f64::from(start.elapsed_ms(&stop)?) * 1000.0 / LAUNCHES as f64;
        }
        info!(
            case,
            pair,
            cpp_us = us[0],
            rust_us = us[1],
            delta_pct = 100.0 * (us[0] / us[1] - 1.0),
            "CUDA event time per launch"
        );
    }
    Ok(())
}

fn ramp(n: usize, period: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i % period) as f32 - period as f32 / 2.0) / (period as f32 / 6.0))
        .collect()
}

fn main() -> Result<(), Error> {
    llmcuda_log::init_from_args();
    let ctx = CudaContext::new(0)?;
    // SAFETY: one stream, allocations retained until synchronized.
    unsafe { ctx.disable_event_tracking() };
    let stream = ctx.new_stream()?;
    let arms = [
        LayerOpsKernels::cuda_cpp(&ctx)?,
        LayerOpsKernels::new(&ctx)?,
    ];
    let up = |v: &[f32]| -> Result<CudaSlice<f32>, Error> { Ok(stream.clone_htod(v)?) };

    let g = ModelConfig::qwen3_6_35b_a3b().gdn;
    let conv_kernel = g.conv_kernel as usize;
    let channels = gdn_conv_channels(
        g.qk_heads as usize,
        g.value_heads as usize,
        g.head_dim as usize,
    );
    let carry = conv_kernel - 1;
    let weight = up(&ramp(channels * conv_kernel, 89))?;

    // The convolution at a decode step, a three-sequence decode step and a
    // 2,048-token prefill chunk.
    for seq_len in [1usize, 2048] {
        let x = up(&ramp(seq_len * channels, 997))?;
        let mut state = up(&ramp(channels * carry, 61))?;
        let mut out = stream.alloc_zeros::<f32>(seq_len * channels)?;
        time_pairs(
            &ctx,
            &stream,
            &arms,
            &format!("conv1d seq={seq_len}"),
            |k| {
                k.conv1d(
                    &stream,
                    &x,
                    &weight,
                    &mut state,
                    &mut out,
                    seq_len,
                    channels,
                    conv_kernel,
                )
            },
        )?;
        let mut state = up(&ramp(channels * carry, 61))?;
        time_pairs(
            &ctx,
            &stream,
            &arms,
            &format!("conv1d cache advance seq={seq_len}"),
            |k| k.conv1d_advance_state(&stream, &x, &mut state, seq_len, channels, conv_kernel),
        )?;
    }
    {
        let n = 3;
        let x = up(&ramp(n * channels, 997))?;
        let states = (0..n)
            .map(|_| up(&ramp(channels * carry, 61)))
            .collect::<Result<Vec<_>, _>>()?;
        let ptrs: Vec<u64> = states.iter().map(|s| s.device_ptr(&stream).0).collect();
        let mut out = stream.alloc_zeros::<f32>(n * channels)?;
        time_pairs(&ctx, &stream, &arms, "conv1d batched step n=3", |k| {
            // SAFETY: `states` are three distinct live caches of
            // `channels * carry` floats that outlive every graph launch.
            unsafe {
                k.conv1d_step_batch_raw(
                    &stream,
                    &x,
                    &weight,
                    &ptrs,
                    &mut out,
                    channels,
                    conv_kernel,
                )
            }
        })?;
    }

    // RMSNorm in its three forms: hidden rows and Gated DeltaNet head rows,
    // at decode and at a 2,048-token prefill chunk.
    for (rows, width) in [
        (3, 2048),
        (2048, 2048),
        (96, 128),
        (65536, 128),
        (48, 256),
        (32768, 256),
    ] {
        let n = rows * width;
        let x = up(&ramp(n, 997))?;
        let gate = up(&ramp(n, 991))?;
        let w = up(&ramp(width, 71))?;
        let mut normed = stream.alloc_zeros::<f32>(n)?;
        let mut out = stream.alloc_zeros::<f32>(n)?;
        let mut q = stream.alloc_zeros::<i8>(n)?;
        let mut scales = stream.alloc_zeros::<f32>(n / 32)?;
        time_pairs(
            &ctx,
            &stream,
            &arms,
            &format!("rms_norm {rows}x{width}"),
            |k| k.rms_norm(&stream, &x, &w, &mut out, rows, width, 1e-6),
        )?;
        time_pairs(
            &ctx,
            &stream,
            &arms,
            &format!("rms_norm_swiglu {rows}x{width}"),
            |k| {
                k.rms_norm_swiglu(
                    &stream,
                    &x,
                    &w,
                    &gate,
                    &mut normed,
                    &mut out,
                    rows,
                    width,
                    1e-6,
                )
            },
        )?;
        if width <= 1024 {
            time_pairs(
                &ctx,
                &stream,
                &arms,
                &format!("rms_norm_swiglu_q8 {rows}x{width}"),
                |k| {
                    k.rms_norm_swiglu_quantize(
                        &stream,
                        &x,
                        &w,
                        &gate,
                        &mut q,
                        &mut scales,
                        rows,
                        width,
                        1e-6,
                    )
                    .map(|_| ())
                },
            )?;
        }
    }
    Ok(())
}
