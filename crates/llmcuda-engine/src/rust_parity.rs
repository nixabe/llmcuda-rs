//! Rust ports against the CUDA C++ kernels they are about to replace, bit
//! for bit and timed. Lives only while both sources exist.
#![allow(dead_code, unused_imports)]

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig,
    PushKernelArg,
};
use llmcuda_cuda::device::driver_available;
use llmcuda_cuda::kernels::{RUST_KERNELS_PTX, compile};
use std::sync::Arc;

struct Gpu {
    stream: Arc<CudaStream>,
    ctx: Arc<CudaContext>,
    rust: Arc<CudaModule>,
}

fn gpu() -> Option<Gpu> {
    if !driver_available() {
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
    fn ints(&mut self, n: usize, below: i32) -> Vec<i32> {
        (0..n)
            .map(|_| ((self.next() >> 33) % below as u64) as i32)
            .collect()
    }
}

fn blank(words: usize) -> Vec<u32> {
    vec![SENTINEL; words]
}

/// Every binary16 bit pattern once, in a scrambled order, `copies` times.
fn all_halves(rng: &mut Rng, copies: usize) -> Vec<u16> {
    let mut v: Vec<u16> = (0..copies).flat_map(|_| 0..=u16::MAX).collect();
    for i in (1..v.len()).rev() {
        v.swap(i, (rng.next() % (i as u64 + 1)) as usize);
    }
    v
}

fn words(halves: &[u16]) -> Vec<u32> {
    halves
        .chunks(2)
        .map(|p| u32::from(p[0]) | (u32::from(p[1]) << 16))
        .collect()
}

/// As `Rng::vec`, with the values that stress a conversion or an exponential.
fn spiked(rng: &mut Rng, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut v = rng.vec(n, lo, hi);
    let specials = [
        0.0,
        -0.0,
        65504.0,
        65520.0,
        -65520.0,
        1e-8,
        -1e-8,
        6.0e-5,
        5.96e-8,
        2.98e-8,
        89.0,
        -89.0,
        f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    for (slot, special) in v.iter_mut().step_by(37).zip(specials) {
        *slot = special;
    }
    v
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
fn the_fp32_projection_port_matches_cuda_cpp() {
    let Some(g) = gpu() else { return };
    let s = &g.stream;
    let mut rng = Rng(0x0020_C0DE);
    let proj = g.pair(crate::block::gdn::GDN_BLOCK_SRC, "gdn_proj_f32");
    for (tokens, k_dim, n_rows) in [
        (1usize, 2048usize, 32usize),
        (3, 2048, 32),
        (2048, 2048, 32),
        (5, 300, 13),
    ] {
        let weight = s.clone_htod(&rng.vec(n_rows * k_dim, -1.0, 1.0)).unwrap();
        let x = s.clone_htod(&rng.vec(tokens * k_dim, -4.0, 4.0)).unwrap();
        let (k_arg, rows_arg) = (k_dim as i32, n_rows as i32);
        let label = format!("gdn_proj_f32 {tokens}x{k_dim}->{n_rows}");
        check(&g.ctx, s, &label, &[blank(tokens * n_rows)], |arm, out| {
            let mut b = s.launch_builder(&proj[arm]);
            b.arg(&weight)
                .arg(&x)
                .arg(&mut out[0])
                .arg(&k_arg)
                .arg(&rows_arg);
            // SAFETY: one warp per output row, eight rows a block.
            unsafe {
                b.launch(cfg(
                    ((n_rows as u32).div_ceil(8), tokens as u32),
                    (32, 8),
                    0,
                ))
            }
            .unwrap();
        });
    }
}

#[test]
fn the_gate_prelude_port_matches_cuda_cpp() {
    let Some(g) = gpu() else { return };
    let s = &g.stream;
    let mut rng = Rng(0x0022_C0DE);
    let pair = g.pair(
        crate::block::gdn::GDN_BLOCK_SRC,
        "gdn_norm_alpha_beta_gates",
    );
    for (tokens, k_dim, heads) in [
        (1usize, 2048usize, 32usize),
        (3, 2048, 32),
        (2048, 2048, 32),
        (5, 1024, 7),
    ] {
        let w_alpha = s.clone_htod(&rng.vec(heads * k_dim, -1.0, 1.0)).unwrap();
        let w_beta = s.clone_htod(&rng.vec(heads * k_dim, -1.0, 1.0)).unwrap();
        let x = s.clone_htod(&rng.vec(tokens * k_dim, -6.0, 6.0)).unwrap();
        let dt_bias = s.clone_htod(&rng.vec(heads, -2.0, 2.0)).unwrap();
        let ssm_a = s.clone_htod(&rng.vec(heads, -3.0, -0.01)).unwrap();
        let norm_weight = s.clone_htod(&rng.vec(k_dim, 0.2, 2.0)).unwrap();
        let (k_arg, h_arg, t_arg, eps) = (k_dim as i32, heads as i32, tokens as i32, 1e-6f32);
        let n = tokens * heads;
        let label = format!("gdn_norm_alpha_beta_gates {tokens}x{k_dim} heads={heads}");
        let lens = [
            blank(n),
            blank(n),
            blank(n),
            blank(n),
            blank(n),
            blank(tokens * k_dim),
        ];
        check(&g.ctx, s, &label, &lens, |arm, out| {
            let [alpha, beta_raw, a_softplus, log_decay, beta, normed] = out else {
                unreachable!()
            };
            let mut b = s.launch_builder(&pair[arm]);
            b.arg(&w_alpha)
                .arg(&w_beta)
                .arg(&x)
                .arg(&dt_bias)
                .arg(&ssm_a)
                .arg(alpha)
                .arg(beta_raw)
                .arg(a_softplus)
                .arg(log_decay)
                .arg(beta)
                .arg(&k_arg)
                .arg(&h_arg)
                .arg(&t_arg)
                .arg(&norm_weight)
                .arg(normed)
                .arg(&eps);
            // SAFETY: one block per head and token; shared holds the row.
            unsafe {
                b.launch(cfg(
                    (heads as u32, tokens as u32),
                    (256, 1),
                    ((k_dim + 32) * 4) as u32,
                ))
            }
            .unwrap();
        });
    }
}
