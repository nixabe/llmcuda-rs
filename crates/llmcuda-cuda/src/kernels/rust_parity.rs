//! Rust ports against the CUDA C++ kernels they are about to replace, bit
//! for bit and timed. Lives only while both sources exist.
#![allow(dead_code, unused_imports)]

use super::{RUST_KERNELS_PTX, compile};
use crate::device::driver_available;
use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig,
    PushKernelArg,
};
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
fn the_k2_rotary_append_ports_match_cuda_cpp() {
    let Some(g) = gpu() else { return };
    let s = &g.stream;
    let mut rng = Rng(0x0C2A_99E7);
    for (kq8, vq8) in [(0i32, 0i32), (1, 1), (1, 0), (0, 1)] {
        let quantized = kq8 + vq8 > 0;
        let src = if quantized {
            format!(
                "#define KV_KQ8 {kq8}\n#define KV_VQ8 {vq8}\n{}{}",
                super::attention::KV_Q8_HELPERS,
                super::k2_rope::SRC
            )
        } else {
            super::k2_rope::SRC.to_string()
        };
        let cpp = g
            .ctx
            .load_module(compile(&src, "rust_parity").expect("C++ compiles"))
            .expect("C++ loads");
        let pair = [
            cpp.load_function("k2_rotary_append").unwrap(),
            g.rust
                .load_function(if quantized {
                    "k2_rotary_append_kv8"
                } else {
                    "k2_rotary_append"
                })
                .unwrap(),
        ];
        for (n, qh, kh, dim, rope_dim) in [
            (1usize, 16usize, 2usize, 128usize, 64usize),
            (3, 16, 2, 128, 64),
            (8, 16, 2, 128, 64),
            (3, 8, 4, 256, 256),
            (2, 4, 2, 64, 32),
        ] {
            let (qn, kn) = (n * qh * dim, n * kh * dim);
            let q = s.clone_htod(&spiked(&mut rng, qn, -4.0, 4.0)).unwrap();
            let k = s.clone_htod(&spiked(&mut rng, kn, -4.0, 4.0)).unwrap();
            let mut vv = spiked(&mut rng, kn, -40.0, 40.0);
            vv[..32].fill(0.0);
            let v = s.clone_htod(&vv).unwrap();
            let coefficients = s.clone_htod(&rng.vec(n * rope_dim, -1.0, 1.0)).unwrap();
            let positions: Vec<i32> = (0..n).map(|i| 1 + 2 * i as i32).collect();
            let app: Vec<CudaSlice<i32>> = positions
                .iter()
                .map(|&p| s.clone_htod(&[p]).unwrap())
                .collect();
            let row = kh * dim;
            let mut lens = vec![blank(qn), blank(kn)];
            for &p in &positions {
                lens.push(blank((p as usize + 1) * row / 2));
                lens.push(blank((p as usize + 1) * row / 2));
            }
            let (qh_i, kh_i, d_i, rd_i) = (qh as i32, kh as i32, dim as i32, rope_dim as i32);
            let label =
                format!("k2_rotary_append k8={kq8} v8={vq8} n={n} {qh}+{kh}x{dim} rope={rope_dim}");
            check(&g.ctx, s, &label, &lens, |arm, out| {
                let slots = |f: &dyn Fn(usize) -> u64| -> [u64; 8] {
                    core::array::from_fn(|i| f(i.min(n - 1)))
                };
                let ap = slots(&|i| app[i].device_ptr(s).0);
                let kc = slots(&|i| out[2 + 2 * i].device_ptr(s).0);
                let vc = slots(&|i| out[3 + 2 * i].device_ptr(s).0);
                let (q_out, rest) = out.split_at_mut(1);
                let mut b = s.launch_builder(&pair[arm]);
                b.arg(&q)
                    .arg(&mut q_out[0])
                    .arg(&k)
                    .arg(&mut rest[0])
                    .arg(&v)
                    .arg(&coefficients);
                for slot in ap.iter().chain(&kc).chain(&vc) {
                    b.arg(slot);
                }
                b.arg(&qh_i).arg(&kh_i).arg(&d_i).arg(&rd_i);
                if arm == 1 && quantized {
                    b.arg(&kq8).arg(&vq8);
                }
                // SAFETY: grid.y = n bounds the slots to live pointers.
                unsafe { b.launch(cfg(((qh + kh) as u32, n as u32), (dim as u32, 1), 0)) }.unwrap();
            });
        }
    }
}
