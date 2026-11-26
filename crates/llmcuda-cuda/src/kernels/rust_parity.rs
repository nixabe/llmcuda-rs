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

/// Finite binary16 values between 2^-5 and 2^3 in magnitude, with a few zeros.
fn finite_halves(rng: &mut Rng, n: usize) -> Vec<u16> {
    (0..n)
        .map(|i| {
            let r = rng.next();
            if i % 97 == 0 {
                return 0;
            }
            (((r >> 20) & 1) as u16) << 15
                | ((10 + (r >> 24) % 9) as u16) << 10
                | ((r >> 40) & 0x3ff) as u16
        })
        .collect()
}

#[test]
fn the_vision_hgemm_and_decision_ports_match_cuda_cpp() {
    let Some(g) = gpu() else { return };
    let s = &g.stream;
    let mut rng = Rng(0x00DE_C151);
    let layer_norm = g.pair(super::vision::VISION_SRC, "vision_layer_norm");
    let to_half = g.pair(super::hgemm::HGEMM_SRC, "hg_q8_to_half_w");
    let dsrc = super::decision::DECISION_SRC;
    let lexical = g.pair(dsrc, "dec_lexical");
    let fields = g.pair(dsrc, "dec_question_fields");
    let features = g.pair(dsrc, "dec_features");

    for (rows, width) in [
        (1usize, 1152usize),
        (729, 1152),
        (5, 100),
        (3, 1024),
        (2916, 1152),
    ] {
        let x = s
            .clone_htod(&finite_halves(&mut rng, rows * width))
            .unwrap();
        let gamma = s.clone_htod(&finite_halves(&mut rng, width)).unwrap();
        let beta = s.clone_htod(&finite_halves(&mut rng, width)).unwrap();
        let (w, eps) = (width as i32, 1e-6f32);
        let label = format!("vision_layer_norm {rows}x{width}");
        check(&g.ctx, s, &label, &[blank(rows * width / 2)], |arm, out| {
            let mut b = s.launch_builder(&layer_norm[arm]);
            b.arg(&x)
                .arg(&mut out[0])
                .arg(&gamma)
                .arg(&beta)
                .arg(&w)
                .arg(&eps);
            // SAFETY: one block per row.
            unsafe { b.launch(cfg((rows as u32, 1), (256, 1), 0)) }.unwrap();
        });
    }

    for n in [4096usize, 1 << 16, 1 << 22, 1 << 24] {
        let wq = s.clone_htod(&rng.bytes(n)).unwrap();
        let ws = s.clone_htod(&finite_halves(&mut rng, n / 32)).unwrap();
        let n_i = n as i64;
        let label = format!("hg_q8_to_half_w n={n}");
        check(&g.ctx, s, &label, &[blank(n / 2)], |arm, out| {
            let mut b = s.launch_builder(&to_half[arm]);
            b.arg(&wq).arg(&ws).arg(&mut out[0]).arg(&n_i);
            // SAFETY: one thread per four codes.
            unsafe { b.launch(cfg(((n / 4).div_ceil(256) as u32, 1), (256, 1), 0)) }.unwrap();
        });
    }

    let cols = 1024usize;
    for (fmt, row_bytes) in [
        (0i32, cols / 32 * 34),
        (1, 2 * cols),
        (2, cols / 256 * 210),
        (3, 2 * cols),
        (4, cols / 32 * 18),
        (5, cols / 256 * 144),
        (6, cols / 256 * 176),
    ] {
        let vocab = 50usize;
        let table = s.clone_htod(&rng.bytes(vocab * row_bytes)).unwrap();
        let tokens = s.clone_htod(&rng.ints(240, vocab as i32)).unwrap();
        let (stride, nspans) = (3i32, 9i32);
        let mut spans = Vec::new();
        let mut at = 0;
        for i in 0..nspans {
            let len = 1 + (i * 7) % 40;
            spans.extend([at, at + len, 0]);
            at += len / 2 + 1;
        }
        assert!(
            spans
                .iter()
                .step_by(3)
                .zip(spans.iter().skip(1).step_by(3))
                .all(|(_, &e)| e <= 240)
        );
        let spans = s.clone_htod(&spans).unwrap();
        let (rb, cols_i, ldo) = (row_bytes as i64, cols as i32, cols as i64 + 3);
        let label = format!("dec_lexical fmt={fmt}");
        check(
            &g.ctx,
            s,
            &label,
            &[blank(nspans as usize * ldo as usize)],
            |arm, out| {
                let mut b = s.launch_builder(&lexical[arm]);
                b.arg(&table)
                    .arg(&fmt)
                    .arg(&rb)
                    .arg(&cols_i)
                    .arg(&tokens)
                    .arg(&spans)
                    .arg(&stride)
                    .arg(&nspans)
                    .arg(&mut out[0])
                    .arg(&ldo);
                // SAFETY: threads past cols return.
                unsafe { b.launch(cfg((cols.div_ceil(256) as u32, nspans as u32), (256, 1), 0)) }
                    .unwrap();
            },
        );
    }

    for (nq, n_opts, width, hidden) in [
        (4usize, 37usize, 768usize, 1024usize),
        (1, 4, 256, 256),
        (9, 300, 768, 2048),
    ] {
        let (ldo, ldb, ldf) = (width + 4, width, width + 8);
        let opts = s.clone_htod(&rng.vec(n_opts * ldo, -1.5, 1.5)).unwrap();
        let mut owners = rng.ints(n_opts, nq as i32);
        if nq > 2 {
            // One question owns nothing.
            for o in owners.iter_mut() {
                if *o == 1 {
                    *o = 0;
                }
            }
        }
        let ospans: Vec<i32> = owners.iter().flat_map(|&o| [0, 0, o]).collect();
        let ospans = s.clone_htod(&ospans).unwrap();
        let base = s.clone_htod(&rng.vec(nq * ldb, -1.5, 1.5)).unwrap();
        let gproj = s.clone_htod(&rng.vec(width, -1.0, 1.0)).unwrap();
        let type_embd = s.clone_htod(&rng.vec(3 * width, -1.0, 1.0)).unwrap();
        let qspans: Vec<i32> = (0..nq as i32).flat_map(|q| [0, 0, q % 3]).collect();
        let qspans = s.clone_htod(&qspans).unwrap();
        let sg = s.clone_htod(&rng.vec(width, 0.5, 1.5)).unwrap();
        let sb = s.clone_htod(&rng.vec(width, -0.5, 0.5)).unwrap();
        let (n_i, stride, w_i, eps) = (n_opts as i32, 3i32, width as i32, 1e-5f32);
        let (ldo_i, ldb_i, ldf_i) = (ldo as i64, ldb as i64, ldf as i64);
        let scale = 1.0 / (width as f32).sqrt();
        let label = format!("dec_question_fields q={nq} opts={n_opts} width={width}");
        check(
            &g.ctx,
            s,
            &label,
            &[blank(n_opts), blank(nq * ldf)],
            |arm, out| {
                let [weights, fq] = out else { unreachable!() };
                let mut b = s.launch_builder(&fields[arm]);
                b.arg(&opts)
                    .arg(&ldo_i)
                    .arg(&n_i)
                    .arg(&ospans)
                    .arg(&stride)
                    .arg(&base)
                    .arg(&ldb_i)
                    .arg(&gproj)
                    .arg(&type_embd)
                    .arg(&qspans)
                    .arg(&stride)
                    .arg(&sg)
                    .arg(&sb)
                    .arg(&eps)
                    .arg(weights)
                    .arg(fq)
                    .arg(&ldf_i)
                    .arg(&w_i)
                    .arg(&scale);
                // SAFETY: one block per question; summary is width floats.
                unsafe { b.launch(cfg((nq as u32, 1), (256, 1), width as u32 * 4)) }.unwrap();
            },
        );

        let (ldq, ldl, ldfeat) = (hidden + 2, hidden, 4 * width + 5);
        let fieldn = s.clone_htod(&rng.vec(nq * ldf, -1.5, 1.5)).unwrap();
        let qmean = s.clone_htod(&rng.vec(nq * ldq, -1.5, 1.5)).unwrap();
        let global = s.clone_htod(&rng.vec(hidden, -1.5, 1.5)).unwrap();
        let mut lex = rng.vec(n_opts * ldl, -1.5, 1.5);
        lex[..ldl].fill(0.0);
        let lex = s.clone_htod(&lex).unwrap();
        let (ldq_i, ldl_i, ldfeat_i, h_i) = (ldq as i64, ldl as i64, ldfeat as i64, hidden as i32);
        let label = format!("dec_features opts={n_opts} width={width} hidden={hidden}");
        check(
            &g.ctx,
            s,
            &label,
            &[blank(n_opts * ldfeat), blank(2 * n_opts)],
            |arm, out| {
                let [feat, aux] = out else { unreachable!() };
                let mut b = s.launch_builder(&features[arm]);
                b.arg(&fieldn)
                    .arg(&ldf_i)
                    .arg(&opts)
                    .arg(&ldo_i)
                    .arg(&ospans)
                    .arg(&stride)
                    .arg(&n_i)
                    .arg(&qmean)
                    .arg(&ldq_i)
                    .arg(&global)
                    .arg(&lex)
                    .arg(&ldl_i)
                    .arg(&h_i)
                    .arg(&w_i)
                    .arg(feat)
                    .arg(&ldfeat_i)
                    .arg(aux);
                // SAFETY: one block per option.
                unsafe { b.launch(cfg((n_opts as u32, 1), (256, 1), 0)) }.unwrap();
            },
        );
    }
}
