//! Rust ports against the CUDA C++ kernels they are about to replace, bit
//! for bit and timed, at the launch shapes the engine uses.
//!
//! Each case lives here only while both sources exist: when a kernel's C++
//! is deleted its case goes with it.

use super::{RUST_KERNELS_PTX, compile};
use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use std::sync::Arc;

struct Gpu {
    stream: Arc<CudaStream>,
    ctx: Arc<CudaContext>,
    rust: Arc<CudaModule>,
}

fn gpu() -> Option<Gpu> {
    if !crate::device::driver_available() {
        println!("SKIPPED: no CUDA driver present");
        return None;
    }
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            println!("SKIPPED: could not create a context on device 0: {e}");
            return None;
        }
    };
    let rust = ctx
        .load_module(cudarc::nvrtc::Ptx::from_src(RUST_KERNELS_PTX))
        .expect("Rust PTX loads");
    Some(Gpu {
        stream: ctx.default_stream(),
        ctx,
        rust,
    })
}

impl Gpu {
    /// `[CUDA C++, Rust]` for one entry point.
    fn pair(&self, src: &str, name: &str) -> [CudaFunction; 2] {
        let cpp = self
            .ctx
            .load_module(compile(src, "rust_parity").expect("C++ compiles"))
            .expect("C++ loads");
        [
            cpp.load_function(name).expect("C++ entry"),
            self.rust.load_function(name).expect("Rust entry"),
        ]
    }
}

/// Sentinel every output starts as: 7.0f32, an implausible integer.
const SENTINEL: u32 = 0x40E0_0000;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn vec(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n)
            .map(|_| lo + (hi - lo) * ((self.next() >> 40) as f32 / (1u64 << 24) as f32))
            .collect()
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.next() >> 56) as u8).collect()
    }
}

fn blank(words: usize) -> Vec<u32> {
    vec![SENTINEL; words]
}

fn cfg(grid: (u32, u32), block: (u32, u32), shared: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (grid.0, grid.1, 1),
        block_dim: (block.0, block.1, 1),
        shared_mem_bytes: shared,
    }
}

/// Launch both arms into fresh sentinel-filled outputs and require equal
/// bits, then time them: six alternating pairs of 100 launches between CUDA
/// events. `outs` are the outputs' initial contents as 32-bit words (an
/// in-place kernel's input, otherwise sentinels); the closure launches arm 0
/// (CUDA C++) or arm 1 (Rust).
fn check(
    ctx: &Arc<CudaContext>,
    s: &Arc<CudaStream>,
    label: &str,
    outs: &[Vec<u32>],
    launch: impl Fn(usize, &mut [CudaSlice<u32>]),
) {
    let fresh = || -> Vec<CudaSlice<u32>> {
        outs.iter()
            .map(|init| s.clone_htod(init).expect("alloc"))
            .collect()
    };
    let result = [0, 1].map(|arm| {
        let mut out = fresh();
        launch(arm, &mut out);
        out.iter()
            .map(|d| s.clone_dtoh(d).expect("read"))
            .collect::<Vec<_>>()
    });
    assert!(result[0] == result[1], "{label}: Rust and CUDA C++ differ");
    assert!(result[0][..] != outs[..], "{label}: nothing was written");

    let mut out = fresh();
    for arm in [0, 1] {
        for _ in 0..20 {
            launch(arm, &mut out);
        }
    }
    s.synchronize().expect("sync");
    let start = ctx
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .expect("event");
    let stop = ctx
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .expect("event");
    let mut us = [Vec::new(), Vec::new()];
    for p in 0..6 {
        for arm in if p % 2 == 0 { [0, 1] } else { [1, 0] } {
            start.record(s).expect("record");
            for _ in 0..100 {
                launch(arm, &mut out);
            }
            stop.record(s).expect("record");
            s.synchronize().expect("sync");
            us[arm].push(f64::from(start.elapsed_ms(&stop).expect("elapsed")) * 10.0);
        }
    }
    let span = |v: &Vec<f64>| {
        let lo = v.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = v.iter().copied().fold(0.0, f64::max);
        format!("{lo:.2}-{hi:.2}")
    };
    println!(
        "TIMING {label}: bit-identical; us/launch C++ {} Rust {}",
        span(&us[0]),
        span(&us[1])
    );
}

#[test]
fn the_batched_combine_and_dequant_ports_match_cuda_cpp() {
    let Some(g) = gpu() else { return };
    let s = &g.stream;
    let mut rng = Rng(0x0014_C0DE);

    let combine = g.pair(
        super::attention::ATTENTION_SRC,
        "attn_flash_decode_combine_k2_batch",
    );
    for (seqs, q_heads, head_dim, splits) in [
        (1usize, 16usize, 128usize, 4usize),
        (3, 16, 128, 32),
        (8, 16, 128, 32),
        (2, 4, 64, 5),
    ] {
        let slots = seqs * splits * q_heads;
        let acc = s.clone_htod(&rng.vec(slots * head_dim, -3.0, 3.0)).unwrap();
        let mut m = rng.vec(slots, -30.0, 30.0);
        // Empty splits carry -inf; split 0 of a sequence never does.
        for (i, slot) in m.iter_mut().enumerate() {
            if (i / q_heads) % splits != 0 && i % 5 == 0 {
                *slot = f32::NEG_INFINITY;
            }
        }
        let m = s.clone_htod(&m).unwrap();
        let l = s.clone_htod(&rng.vec(slots, 0.5, 300.0)).unwrap();
        let (qh, hd, ns) = (q_heads as i32, head_dim as i32, splits as i32);
        let label = format!("decode_combine_k2_batch {seqs}x{q_heads}x{head_dim}x{splits}");
        check(
            &g.ctx,
            s,
            &label,
            &[blank(seqs * q_heads * head_dim)],
            |arm, out| {
                let mut b = s.launch_builder(&combine[arm]);
                b.arg(&acc)
                    .arg(&m)
                    .arg(&l)
                    .arg(&mut out[0])
                    .arg(&qh)
                    .arg(&hd)
                    .arg(&ns);
                // SAFETY: one block per head and sequence.
                unsafe { b.launch(cfg((q_heads as u32, seqs as u32), (head_dim as u32, 1), 0)) }
                    .unwrap();
            },
        );
    }

    let dequant = g.pair(super::dequant::DEQUANT_SRC, "dequantize_q8_0");
    for n in [2048usize, 3 * 2048, 2048 * 2048] {
        let src = s.clone_htod(&rng.bytes(n / 32 * 34)).unwrap();
        let n_arg = n as i64;
        check(
            &g.ctx,
            s,
            &format!("dequantize_q8_0 n={n}"),
            &[blank(n)],
            |arm, out| {
                let mut b = s.launch_builder(&dequant[arm]);
                b.arg(&src).arg(&mut out[0]).arg(&n_arg);
                // SAFETY: src holds n / 32 blocks.
                unsafe { b.launch(cfg(((n as u32).div_ceil(256), 1), (256, 1), 0)) }.unwrap();
            },
        );
    }
}

#[test]
fn the_normalization_ports_match_cuda_cpp() {
    let Some(g) = gpu() else { return };
    let s = &g.stream;
    let mut rng = Rng(0x0015_C0DE);
    let step = g.pair(super::gdn::GDN_SRC, "gdn_normalize_qk");
    let chunk = g.pair(
        super::gdn_chunked::GDN_CHUNKED_SRC,
        "gdn_chunk_normalize_qk",
    );
    for (name, pair, rows, head_dim) in [
        ("gdn_normalize_qk", &step, 16usize, 128usize),
        ("gdn_normalize_qk", &step, 5, 64),
        ("gdn_chunk_normalize_qk", &chunk, 16, 128),
        ("gdn_chunk_normalize_qk", &chunk, 2048 * 16, 128),
        ("gdn_chunk_normalize_qk", &chunk, 7, 96),
    ] {
        let n = rows * head_dim;
        let mut q = rng.vec(n, -4.0, 4.0);
        // One all-zero row: the epsilon is all that keeps it finite.
        q[..head_dim].fill(0.0);
        let q = s.clone_htod(&q).unwrap();
        let k = s.clone_htod(&rng.vec(n, -4.0, 4.0)).unwrap();
        let (hd, eps, scale) = (head_dim as i32, 1e-6f32, 1.0 / (head_dim as f32).sqrt());
        let label = format!("{name} {rows}x{head_dim}");
        check(&g.ctx, s, &label, &[blank(n), blank(n)], |arm, out| {
            let [q_out, k_out] = out else { unreachable!() };
            let mut b = s.launch_builder(&pair[arm]);
            b.arg(&q)
                .arg(&k)
                .arg(q_out)
                .arg(k_out)
                .arg(&hd)
                .arg(&eps)
                .arg(&scale);
            // SAFETY: one block per row, a float of shared memory per warp.
            unsafe { b.launch(cfg((rows as u32, 1), (head_dim as u32, 1), 128)) }.unwrap();
        });
    }

    let norm = g.pair(super::decision::DECISION_SRC, "dec_layer_norm");
    for (rows, cols, mode) in [
        (4usize, 2048usize, 0),
        (700, 2048, 0),
        (700, 2048, 1),
        (700, 2048, 2),
        (9, 300, 0),
    ] {
        let (ldx, ldy) = (cols + 8, cols + 16);
        let x = s.clone_htod(&rng.vec(rows * ldx, -6.0, 6.0)).unwrap();
        let gain = s.clone_htod(&rng.vec(cols, 0.2, 2.0)).unwrap();
        let bias = s.clone_htod(&rng.vec(cols, -1.0, 1.0)).unwrap();
        let (ldx_arg, ldy_arg, rows_arg, cols_arg, eps) =
            (ldx as i64, ldy as i64, rows as i32, cols as i32, 1e-5f32);
        let null = 0u64;
        let label = format!("dec_layer_norm {rows}x{cols} mode {mode}");
        check(
            &g.ctx,
            s,
            &label,
            &[blank(rows * 2), blank(rows * ldy)],
            |arm, out| {
                let [stats, y] = out else { unreachable!() };
                let mut b = s.launch_builder(&norm[arm]);
                b.arg(&x)
                    .arg(&ldx_arg)
                    .arg(&rows_arg)
                    .arg(&cols_arg)
                    .arg(&eps)
                    .arg(&gain)
                    .arg(&bias);
                // Mode 1 records the statistics only, mode 2 the rows only.
                if mode == 2 {
                    b.arg(&null)
                } else {
                    b.arg(stats)
                };
                if mode == 1 {
                    b.arg(&null)
                } else {
                    b.arg(y)
                };
                b.arg(&ldy_arg);
                // SAFETY: one block per row; a null output is skipped by the kernel.
                unsafe { b.launch(cfg((rows as u32, 1), (256, 1), 0)) }.unwrap();
            },
        );
    }
}
