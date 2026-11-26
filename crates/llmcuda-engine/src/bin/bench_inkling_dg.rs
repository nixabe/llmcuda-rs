//! The decode GEMM alone at Inkling 2 Mini's shapes, CUDA-event timed over
//! random weights: the attention projection pack (Q8_0), and the expert
//! gate/up and down projections (NVFP4) for one token's eight experts.
//! Prints microseconds a launch and the weight bytes streamed per second.
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, sys::CUevent_flags};
use llmcuda_cuda::kernels::hgemm::HalfWeight;
use llmcuda_cuda::kernels::inkling::DecodeGemm;
use tracing::{error, info};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn time(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    mut f: impl FnMut() -> Result<()>,
) -> Result<f64> {
    const REPS: usize = 200;
    for _ in 0..10 {
        f()?;
    }
    stream.synchronize()?;
    let start = ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    let end = ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    start.record(stream)?;
    for _ in 0..REPS {
        f()?;
    }
    end.record(stream)?;
    end.synchronize()?;
    Ok(f64::from(start.elapsed_ms(&end)?) * 1e3 / REPS as f64)
}

fn run() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let dg = DecodeGemm::new(&ctx)?;
    let noise = |n: usize, seed: u64| -> Vec<u8> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 24) as u8
            })
            .collect()
    };
    let (n, ff, experts, topk) = (2304usize, 1152usize, 128usize, 8usize);
    let half_one = 0x3C00u16;

    // The projection pack: 8608 rows of split Q8_0.
    let rows = 8608usize;
    let q: Vec<i8> = noise(rows * n, 1).into_iter().map(|b| b as i8).collect();
    let d_q = stream.clone_htod(&q)?;
    let d_s = stream.clone_htod(&vec![0x2000u16; rows * n / 32])?;
    for tokens in [1usize, 3, 8] {
        let x = stream.clone_htod(&vec![half_one; 3 * tokens * n])?;
        let mut out = stream.alloc_zeros::<f32>(tokens * rows)?;
        let us = time(&ctx, &stream, || {
            dg.dense(
                &stream,
                HalfWeight::Q8_0 {
                    q: &d_q,
                    scales: &d_s,
                },
                &x,
                &mut out,
                rows,
                n,
                tokens,
                rows,
                tokens,
                6560,
                7584,
            )?;
            Ok(())
        })?;
        let bytes = (rows * n) as f64 * 1.0625;
        info!(
            "dense q8_0 {rows} x {n}, {tokens} rows: {us:8.1} us, {:6.1} GB/s",
            bytes / us / 1e3
        );
    }

    // One token's experts, NVFP4.
    let ids: Vec<i32> = (0..topk as i32)
        .map(|j| (j * 17 + 3) % experts as i32)
        .collect();
    let d_ids = stream.clone_htod(&ids)?;
    let d_w = stream.clone_htod(&vec![1.0f32; topk])?;
    let scales = stream.clone_htod(&vec![1.0f32; experts])?;
    // Valid scale bytes: keep the top bit clear and away from 0x7F.
    let mut gu = noise(experts * 2 * ff * n / 64 * 36, 2);
    let mut dn = noise(experts * n * ff / 64 * 36, 3);
    for w in [&mut gu, &mut dn] {
        for block in w.as_chunks_mut::<36>().0 {
            for s in &mut block[..4] {
                *s = 40 + (*s & 15);
            }
        }
    }
    let (d_gu, d_dn) = (stream.clone_htod(&gu)?, stream.clone_htod(&dn)?);
    let x = stream.clone_htod(&vec![half_one; n])?;
    let mut hidden = stream.alloc_zeros::<u16>(topk * ff)?;
    let us = time(&ctx, &stream, || {
        dg.gate_up(
            &stream,
            HalfWeight::Nvfp4 {
                bytes: &d_gu,
                scale: 1.0,
            },
            experts,
            &d_ids,
            Some(&scales),
            &x,
            &mut hidden,
            2 * ff,
            n,
            topk,
            topk,
            true,
            4.0,
            25.0,
        )?;
        Ok(())
    })?;
    let bytes = (topk * 2 * ff * n) as f64 * 0.5625;
    info!(
        "gate/up nvfp4, {topk} experts: {us:8.1} us, {:6.1} GB/s",
        bytes / us / 1e3
    );
    let mut out = stream.alloc_zeros::<f32>(topk * n)?;
    let us = time(&ctx, &stream, || {
        dg.down(
            &stream,
            HalfWeight::Nvfp4 {
                bytes: &d_dn,
                scale: 1.0,
            },
            experts,
            &d_ids,
            Some(&scales),
            &hidden,
            &d_w,
            &mut out,
            n,
            ff,
            topk,
        )?;
        Ok(())
    })?;
    let bytes = (topk * n * ff) as f64 * 0.5625;
    info!(
        "down nvfp4, {topk} experts:    {us:8.1} us, {:6.1} GB/s",
        bytes / us / 1e3
    );
    Ok(())
}

fn main() {
    llmcuda_log::init_from_args();
    if let Err(e) = run() {
        error!("{e}");
        std::process::exit(1);
    }
}
