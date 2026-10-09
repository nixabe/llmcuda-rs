//! K2 attention at independent decode windows, CUDA-event timings.
//! LLMCUDA_DEC_MMA_SPLITS selects fixed tensor-core split geometry;
//! LLMCUDA_CACHE_TYPE_K / LLMCUDA_CACHE_TYPE_V (f16, q8_0) the cache formats.
use cudarc::driver::{CudaContext, CudaStream, DevicePtr, sys::CUevent_flags};
use llmcuda_cuda::kernels::attention::{AttentionKernels, AttnDecodeScratch, KvFormat, KvFormats};
use std::sync::Arc;
use tracing::{error, info};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn time(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    mut f: impl FnMut() -> Result<()>,
) -> Result<f32> {
    for _ in 0..5 {
        f()?;
    }
    stream.synchronize()?;
    let start = ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    let end = ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    start.record(stream)?;
    for _ in 0..100 {
        f()?;
    }
    end.record(stream)?;
    end.synchronize()?;
    Ok(start.elapsed_ms(&end)? * 10.0)
}
/// `LLMCUDA_CACHE_TYPE_{half}`, binary16 when unset.
fn format(half: &str) -> Result<KvFormat> {
    match std::env::var(format!("LLMCUDA_CACHE_TYPE_{half}")).as_deref() {
        Err(_) | Ok("f16") => Ok(KvFormat::F16),
        Ok("q8_0") => Ok(KvFormat::Q8_0),
        Ok(other) => Err(format!("LLMCUDA_CACHE_TYPE_{half}={other}: expected f16 or q8_0").into()),
    }
}
/// `rows` positions of 1024 elements in `format`: binary16 sines, or q8_0
/// codes spanning the int8 range under a finite scale.
fn cache_words(format: KvFormat, rows: usize, salt: usize) -> Vec<u16> {
    match format {
        KvFormat::F16 => (0..rows * 1024)
            .map(|j| half::f16::from_f32(((j + salt) as f32 * 0.017).sin()).to_bits())
            .collect(),
        KvFormat::Q8_0 => {
            let scale = half::f16::from_f32(1.0 / 127.0).to_bits();
            let mut words = Vec::with_capacity(rows * 544);
            for r in 0..rows {
                for pair in 0..512 {
                    let code = |j: usize| (((r * 1024 + 2 * pair + j + salt) * 37) % 255) as u16;
                    words.push(code(0) | (code(1) << 8));
                }
                words.extend(std::iter::repeat_n(scale, 32));
            }
            words
        }
    }
}
fn run() -> Result<()> {
    let formats = KvFormats {
        k: format("K")?,
        v: format("V")?,
    };
    info!("KV cache K={} V={}", formats.k.name(), formats.v.name());
    let ctx = CudaContext::new(0)?;
    // SAFETY: all buffers and launches use this single stream.
    unsafe {
        ctx.disable_event_tracking();
    }
    let stream = ctx.default_stream();
    let ops = AttentionKernels::with_kv_formats(&ctx, 32, 8, 128, formats)?;
    for depth in [512, 2048, 8192, 32768] {
        for n in [1, 3] {
            let mut p = Vec::new();
            let mut k = Vec::new();
            let mut v = Vec::new();
            let mut qs = Vec::new();
            let mut os = Vec::new();
            let mut ds = Vec::new();
            let q: Vec<f32> = (0..n * 32 * 128)
                .map(|i| (i as f32 * 0.013).sin())
                .collect();
            let dq = stream.clone_htod(&q)?;
            let mut out = stream.alloc_zeros::<f32>(q.len())?;
            let mut dec = AttnDecodeScratch::new_batch(&stream, 32, 128, n)?;
            for i in 0..n {
                p.push(stream.clone_htod(&[depth as i32])?);
                k.push(stream.clone_htod(&cache_words(formats.k, depth + 1, i * 101))?);
                v.push(stream.clone_htod(&cache_words(formats.v, depth + 1, i * 101))?);
                qs.push(stream.clone_htod(&q[i * 4096..(i + 1) * 4096])?);
                os.push(stream.alloc_zeros::<f32>(4096)?);
                ds.push(AttnDecodeScratch::new(&stream, 32, 128)?);
            }
            let ptr = |s: &Vec<_>| {
                s.iter()
                    .map(|b: &cudarc::driver::CudaSlice<u16>| b.device_ptr(&stream).0)
                    .collect::<Vec<_>>()
            };
            let pp: Vec<_> = p.iter().map(|b| b.device_ptr(&stream).0).collect();
            let kp = ptr(&k);
            let vp = ptr(&v);
            for round in 1..=3 {
                for lever in if round % 2 == 1 {
                    [0, 2, 4, 12, 14]
                } else {
                    [14, 12, 4, 2, 0]
                } {
                    let us = time(&ctx, &stream, || {
                        if lever == 0 {
                            // SAFETY: all independent cache and position allocations
                            // remain live, with depth+1 initialized keys each.
                            unsafe {
                                ops.decode_batch_k2_raw(
                                    &stream, &mut dec, &dq, &mut out, &pp, &kp, &vp,
                                )?;
                            }
                        } else if lever >= 10 {
                            // SAFETY: same independent live cache slots as above.
                            unsafe {
                                ops.decode_batch_k2_mma_raw(
                                    &stream,
                                    &mut dec,
                                    &dq,
                                    &mut out,
                                    &pp,
                                    &kp,
                                    &vp,
                                    lever - 10,
                                )?;
                            }
                        } else {
                            ops.set_decode_mma_wpo(lever);
                            for i in 0..n {
                                ops.forward_k2(
                                    &stream,
                                    &mut ds[i],
                                    &qs[i],
                                    &k[i],
                                    &v[i],
                                    &mut os[i],
                                    1,
                                    depth + 1,
                                    depth + 1,
                                    &p[i],
                                )?;
                            }
                        }
                        Ok(())
                    })?;
                    let path = match lever {
                        0 => "warp-batch",
                        2 => "mma-serial-2",
                        4 => "mma-serial-4",
                        12 => "mma-batch-2",
                        14 => "mma-batch-4",
                        _ => unreachable!(),
                    };
                    info!("depth={depth} N={n} round={round} path={path}: {us:.3} us");
                }
            }
        }
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
