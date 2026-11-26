//! Rust ports against the CUDA C++ kernels they are about to replace, bit
//! for bit and timed, at the launch shapes the engine uses.
//!
//! Each case lives here only while both sources exist: when a kernel's C++
//! is deleted its case goes with it.

use super::{RUST_KERNELS_PTX, compile};
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
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.next() >> 56) as u8).collect()
    }
    /// Finite binary16 bit patterns, as 32-bit words holding two each.
    fn finite_halves(&mut self, n: usize) -> Vec<u32> {
        let mut half = || loop {
            let h = (self.next() >> 48) as u32;
            if h & 0x7C00 != 0x7C00 {
                break h;
            }
        };
        (0..n / 2).map(|_| half() | (half() << 16)).collect()
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
fn the_q6k_and_rotary_prepare_ports_match_cuda_cpp() {
    let Some(g) = gpu() else { return };
    let s = &g.stream;
    let mut rng = Rng(0x0017_C0DE);

    let q6k = g.pair(super::dequant::DEQUANT_SRC, "dequantize_q6_k");
    for superblocks in [8usize, 3 * 8, 2048 * 8] {
        let src = s.clone_htod(&rng.bytes(superblocks * 210)).unwrap();
        let n_arg = superblocks as i64;
        let label = format!("dequantize_q6_k {superblocks} superblocks");
        check(
            &g.ctx,
            s,
            &label,
            &[blank(superblocks * 256)],
            |arm, out| {
                let mut b = s.launch_builder(&q6k[arm]);
                b.arg(&src).arg(&mut out[0]).arg(&n_arg);
                // SAFETY: one thread per (superblock, half, l).
                unsafe {
                    b.launch(cfg(
                        (((superblocks * 64) as u32).div_ceil(256), 1),
                        (256, 1),
                        0,
                    ))
                }
                .unwrap();
            },
        );
    }

    let prepare = g.pair(super::k2_rope::SRC, "k2_rotary_prepare");
    for (seqs, chunk, rope_dim, offset) in [
        (1usize, 1usize, 64usize, 0i32),
        (3, 1, 64, 5),
        (1, 2048, 64, 0),
        (8, 13, 128, 2),
    ] {
        let positions: Vec<i32> = (0..seqs as i32).map(|i| 100_000 * i + 7).collect();
        let positions = s.clone_htod(&positions).unwrap();
        let base = positions.device_ptr(s).0;
        let p: [u64; 8] = std::array::from_fn(|i| base + 4 * i.min(seqs - 1) as u64);
        let (chunk_arg, rope_arg, theta) = (chunk as i32, rope_dim as i32, 10_000_000.0f32);
        let words = (offset as usize + seqs * chunk) * rope_dim;
        let label = format!("k2_rotary_prepare {seqs}x{chunk} rope {rope_dim}");
        check(&g.ctx, s, &label, &[blank(words)], |arm, out| {
            let mut b = s.launch_builder(&prepare[arm]);
            b.arg(&mut out[0])
                .arg(&chunk_arg)
                .arg(&rope_arg)
                .arg(&theta)
                .arg(&offset);
            for slot in &p {
                b.arg(slot);
            }
            // SAFETY: eight tokens a block; each position pointer is live.
            unsafe {
                b.launch(cfg(
                    ((chunk as u32).div_ceil(8), seqs as u32),
                    ((rope_dim / 2) as u32, 8),
                    0,
                ))
            }
            .unwrap();
        });
    }
}

#[test]
fn the_vision_softmax_and_rope_ports_match_cuda_cpp() {
    let Some(g) = gpu() else { return };
    let s = &g.stream;
    let mut rng = Rng(0x0018_C0DE);
    let src = super::vision::VISION_SRC;
    let softmax = g.pair(src, "vision_softmax_rows");
    let rope = g.pair(src, "vision_rope_qk");

    for (rows, row_len) in [(16usize, 730usize), (16 * 730, 730), (5, 100), (3, 2048)] {
        let scores = rng.finite_halves(rows * row_len);
        let row_arg = row_len as i32;
        let label = format!("vision_softmax_rows {rows}x{row_len}");
        check(&g.ctx, s, &label, &[scores], |arm, out| {
            let mut b = s.launch_builder(&softmax[arm]);
            b.arg(&mut out[0]).arg(&row_arg);
            // SAFETY: one block per row.
            unsafe { b.launch(cfg((rows as u32, 1), (256, 1), 0)) }.unwrap();
        });
    }

    for (tokens, heads, head_dim) in [(1usize, 16usize, 72usize), (729, 16, 72), (5, 2, 128)] {
        let hidden = heads * head_dim;
        let qkv = rng.finite_halves(tokens * 3 * hidden);
        let positions = s.clone_htod(&rng.ints(tokens * 2, 64)).unwrap();
        let (hidden_arg, hd, theta) = (hidden as i32, head_dim as i32, 10_000.0f32);
        let label = format!("vision_rope_qk {tokens}x{heads}x{head_dim}");
        check(&g.ctx, s, &label, &[qkv], |arm, out| {
            let mut b = s.launch_builder(&rope[arm]);
            b.arg(&mut out[0])
                .arg(&positions)
                .arg(&hidden_arg)
                .arg(&hd)
                .arg(&theta);
            // SAFETY: one block per token, slice and head.
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (tokens as u32, 2, heads as u32),
                    block_dim: ((head_dim / 2) as u32, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .unwrap();
        });
    }
}
