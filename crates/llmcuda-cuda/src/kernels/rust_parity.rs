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
fn the_recurrent_step_and_scorer_ports_match_cuda_cpp() {
    let Some(g) = gpu() else { return };
    let s = &g.stream;
    let mut rng = Rng(0x0021_C0DE);
    let step = g.pair(super::gdn::GDN_SRC, "gdn_recurrent_step");
    for (seqs, value_heads, qk_heads, head_dim) in [
        (1usize, 32usize, 16usize, 128usize),
        (3, 32, 16, 128),
        (8, 32, 16, 128),
        (2, 4, 2, 256),
        (1, 2, 2, 512),
    ] {
        let per_seq = value_heads * head_dim * head_dim;
        let state: Vec<u32> = rng
            .vec(seqs * per_seq, -1.0, 1.0)
            .into_iter()
            .map(f32::to_bits)
            .collect();
        let q = s
            .clone_htod(&rng.vec(seqs * qk_heads * head_dim, -1.0, 1.0))
            .unwrap();
        let k = s
            .clone_htod(&rng.vec(seqs * qk_heads * head_dim, -1.0, 1.0))
            .unwrap();
        let v = s
            .clone_htod(&rng.vec(seqs * value_heads * head_dim, -4.0, 4.0))
            .unwrap();
        let log_decay = s
            .clone_htod(&rng.vec(seqs * value_heads, -3.0, 0.0))
            .unwrap();
        let beta = s
            .clone_htod(&rng.vec(seqs * value_heads, 0.0, 1.0))
            .unwrap();
        let (hd, qh) = (head_dim as i32, qk_heads as i32);
        let label = format!("gdn_recurrent_step {seqs}x{value_heads}x{head_dim}");
        check(
            &g.ctx,
            s,
            &label,
            &[state, blank(seqs * value_heads * head_dim)],
            |arm, out| {
                let [state, o] = out else { unreachable!() };
                let base = state.device_ptr(s).0;
                let p: [u64; 8] =
                    std::array::from_fn(|i| base + (i.min(seqs - 1) * per_seq * 4) as u64);
                let mut b = s.launch_builder(&step[arm]);
                for slot in &p {
                    b.arg(slot);
                }
                b.arg(&q)
                    .arg(&k)
                    .arg(&v)
                    .arg(&log_decay)
                    .arg(&beta)
                    .arg(o)
                    .arg(&hd)
                    .arg(&qh);
                // SAFETY: each sequence's state is its own slice.
                unsafe {
                    b.launch(LaunchConfig {
                        grid_dim: (
                            (head_dim as u32).div_ceil(8),
                            value_heads as u32,
                            seqs as u32,
                        ),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .unwrap();
            },
        );
    }

    let scores = g.pair(super::decision::DECISION_SRC, "dec_final_scores");
    for (n_opts, width) in [(2i32, 2048i32), (4, 2048), (26, 2048), (5, 300)] {
        let ldh = (width + 8) as i64;
        let h = s
            .clone_htod(&rng.vec(n_opts as usize * ldh as usize, -4.0, 4.0))
            .unwrap();
        let w_out = s.clone_htod(&rng.vec(width as usize, -1.0, 1.0)).unwrap();
        let aux = s
            .clone_htod(&rng.vec(n_opts as usize * 2, -1.0, 1.0))
            .unwrap();
        let (b_out, prior_scale, joint_scale, gate) = (0.37f32, 1.3f32, 7.1f32, 0.61f32);
        let label = format!("dec_final_scores {n_opts} options x {width}");
        check(&g.ctx, s, &label, &[blank(n_opts as usize)], |arm, out| {
            let mut b = s.launch_builder(&scores[arm]);
            b.arg(&h)
                .arg(&ldh)
                .arg(&n_opts)
                .arg(&width)
                .arg(&w_out)
                .arg(&b_out)
                .arg(&aux)
                .arg(&prior_scale)
                .arg(&joint_scale)
                .arg(&gate)
                .arg(&mut out[0]);
            // SAFETY: one warp per option.
            unsafe { b.launch(cfg((((n_opts * 32) as u32).div_ceil(256), 1), (256, 1), 0)) }
                .unwrap();
        });
    }
}

#[test]
fn the_argmax_prob_port_matches_cuda_cpp() {
    let Some(g) = gpu() else { return };
    let s = &g.stream;
    let mut rng = Rng(0x0017_C0DE);
    let prob = g.pair(super::lm_head::LM_HEAD_SRC, "argmax_prob");
    for (n, spread) in [
        (5usize, 4.0f32),
        (1000, 20.0),
        (248_320, 20.0),
        (248_320, 0.01),
        (2049, 3.0),
    ] {
        let mut x = rng.vec(n, -spread, spread);
        x[n / 2] = spread + 3.0;
        let x = s.clone_htod(&x).unwrap();
        let n_arg = n as i32;
        let label = format!("argmax_prob n={n} spread {spread}");
        check(&g.ctx, s, &label, &[blank(1)], |arm, out| {
            let mut b = s.launch_builder(&prob[arm]);
            b.arg(&x).arg(&n_arg).arg(&mut out[0]);
            // SAFETY: one 256-thread block over n floats.
            unsafe { b.launch(cfg((1, 1), (256, 1), 0)) }.unwrap();
        });
    }
}

/// A Rust kernel that was rewritten after its C++ was deleted is held to its
/// own previous PTX (`OLD_PTX` names a file holding it).
#[test]
fn the_rewritten_moe_tail_matches_its_previous_ptx() {
    let Ok(path) = std::env::var("OLD_PTX") else {
        return;
    };
    let Some(g) = gpu() else { return };
    let s = &g.stream;
    let old = g
        .ctx
        .load_module(cudarc::nvrtc::Ptx::from_src(
            std::fs::read_to_string(path).unwrap(),
        ))
        .unwrap();
    let name = "moe_block_gate_and_combine";
    let pair = [
        old.load_function(name).unwrap(),
        g.rust.load_function(name).unwrap(),
    ];
    let mut rng = Rng(0x0019_C0DE);
    for (max_tokens, valid, top_k, hidden, gy) in [
        (1usize, 1i32, 8i32, 2048usize, 8u32),
        (3, 3, 8, 2048, 8),
        (2048, 2041, 8, 2048, 1),
        (2048, 2041, 8, 2048, 8),
        (5, 4, 3, 1000, 2),
        (2, 2, 6, 4096, 16),
    ] {
        let rows = max_tokens * hidden;
        let w = s.clone_htod(&rng.vec(hidden, -0.2, 0.2)).unwrap();
        let normed = s.clone_htod(&rng.vec(rows, -4.0, 4.0)).unwrap();
        let partial = s
            .clone_htod(&rng.vec(rows * top_k as usize, -4.0, 4.0))
            .unwrap();
        let shexp = s.clone_htod(&rng.vec(rows, -4.0, 4.0)).unwrap();
        let residual = s.clone_htod(&rng.vec(rows, -4.0, 4.0)).unwrap();
        let valid_tokens = s.clone_htod(&[valid]).unwrap();
        let hidden_arg = hidden as i32;
        let label = format!("{name} (old Rust vs new) {max_tokens}x{top_k}x{hidden} gy={gy}");
        let lens = [blank(max_tokens), blank(rows), blank(rows), blank(rows)];
        check(&g.ctx, s, &label, &lens, |arm, out| {
            let [gate, routed, ffn_out, l_out] = out else {
                unreachable!()
            };
            let mut b = s.launch_builder(&pair[arm]);
            b.arg(&w)
                .arg(&normed)
                .arg(&partial)
                .arg(&shexp)
                .arg(&residual)
                .arg(&valid_tokens)
                .arg(&top_k)
                .arg(&hidden_arg)
                .arg(gate)
                .arg(routed)
                .arg(ffn_out)
                .arg(l_out);
            // SAFETY: every buffer is sized for max_tokens.
            unsafe { b.launch(cfg((max_tokens as u32, gy), (256, 1), 128)) }.unwrap();
        });
    }
}
