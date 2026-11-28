//! Inkling 2 Mini's decode attention alone, CUDA-event timed over random
//! caches: one layer's launch for `rows` sequences each `depth` positions
//! deep, a sliding layer (window 1,024) and a full-attention one, on the
//! scalar kernel and the tensor-core one. Prints microseconds a launch and,
//! for the full layer, the cache bytes streamed per second.
//!
//! `bench_inkling_attn <depth> <rows>`.
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, sys::CUevent_flags};
use llmcuda_cuda::kernels::inkling::{
    DEC_ATTN_PARTS, DEC_ROW_POINTERS, DecodeAttention, InklingDecodeKernels, ProjOffsets,
};
use tracing::{error, info};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn time(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    reps: usize,
    mut f: impl FnMut() -> Result<()>,
) -> Result<f64> {
    for _ in 0..3 {
        f()?;
    }
    stream.synchronize()?;
    let start = ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    let end = ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    start.record(stream)?;
    for _ in 0..reps {
        f()?;
    }
    end.record(stream)?;
    end.synchronize()?;
    Ok(f64::from(start.elapsed_ms(&end)?) * 1e3 / reps as f64)
}

fn run(args: &[String]) -> Result<()> {
    let depth: usize = args.first().map_or(Ok(126_976), |s| s.parse())?;
    let rows: usize = args.get(1).map_or(Ok(3), |s| s.parse())?;
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let kernels = InklingDecodeKernels::new(&ctx)?;
    let (heads, kv_heads, dim, n_rel) = (24usize, 8usize, 128usize, 16usize);
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut unit = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / (1u64 << 24) as f32
    };
    let o = ProjOffsets {
        ldo: (2 * heads * dim + heads * n_rel + heads) as i32,
        q: 0,
        rel: (heads * dim) as i32,
        sink: (heads * dim + heads * n_rel) as i32,
        gate: (heads * dim + heads * n_rel + heads) as i32,
        k: 0,
        v: 0,
        k_taps: -1,
        v_taps: -1,
    };
    let proj: Vec<f32> = (0..rows * o.ldo as usize)
        .map(|_| unit() * 2.0 - 1.0)
        .collect();
    let q_clip = vec![1.0f32; heads * dim];
    let rel_clip = vec![1.0f32; heads * n_rel];
    let rel_bias: Vec<f32> = (0..1024 * n_rel).map(|_| unit() * 0.2 - 0.1).collect();
    // Keys and values of magnitude one in binary16, signs and low bits
    // random: one host plane, uploaded once a cache.
    let plane: Vec<u16> = (0..depth * kv_heads * dim)
        .map(|_| {
            let u = unit();
            0x3800 | (((u * 65536.0) as u16) & 0x83FF)
        })
        .collect();
    let mut caches: Vec<CudaSlice<u16>> = Vec::new();
    let mut ptrs = vec![0u64; rows * DEC_ROW_POINTERS];
    for t in 0..rows {
        for which in 0..2 {
            let cache = stream.clone_htod(&plane)?;
            ptrs[t * DEC_ROW_POINTERS + which] = cache.device_ptr(&stream).0;
            caches.push(cache);
        }
    }
    let (d_proj, d_ptrs) = (stream.clone_htod(&proj)?, stream.clone_htod(&ptrs)?);
    let d_pos = stream.clone_htod(&vec![depth as i32 - 1; rows])?;
    let (d_q_clip, d_rel_clip, d_rel_bias) = (
        stream.clone_htod(&q_clip)?,
        stream.clone_htod(&rel_clip)?,
        stream.clone_htod(&rel_bias)?,
    );
    let mut out = stream.alloc_zeros::<u16>(rows * heads * dim)?;
    let mut part = stream.alloc_zeros::<f32>(rows * heads * DEC_ATTN_PARTS * (dim + 2))?;
    for round in 0..3 {
        for (name, mode) in [
            ("scalar", DecodeAttention::Scalar),
            ("mma", DecodeAttention::Mma),
        ] {
            kernels.use_decode_attention(mode);
            for (layer, window, log) in [
                ("sliding", 1024usize, None),
                ("full", 0usize, Some((0.1f32, 8192.0f32))),
            ] {
                let reps = if window > 0 { 200 } else { 20 };
                let us = time(&ctx, &stream, reps, || {
                    kernels.attn(
                        &stream,
                        &d_proj,
                        &d_ptrs,
                        &d_pos,
                        &d_q_clip,
                        &d_rel_clip,
                        &d_rel_bias,
                        &mut part,
                        &mut out,
                        o,
                        heads,
                        kv_heads,
                        dim,
                        n_rel,
                        window,
                        1024,
                        22.627,
                        8.0,
                        log,
                        None,
                        rows,
                    )?;
                    Ok(())
                })?;
                let keys = if window > 0 { window.min(depth) } else { depth };
                let bytes = (rows * keys * kv_heads * dim * 4) as f64;
                info!(
                    "round {round} {name:6} {layer:7} depth {depth} rows {rows}: {us:9.1} us, {:6.1} GB/s of cache",
                    bytes / us / 1e3
                );
            }
        }
    }
    drop(caches);
    Ok(())
}

fn main() {
    let args = llmcuda_log::init_from_args();
    if let Err(e) = run(&args) {
        error!("{e}");
        std::process::exit(1);
    }
}
