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
fn the_moe_routing_ports_match_cuda_cpp() {
    let Some(g) = gpu() else { return };
    let s = &g.stream;
    let mut rng = Rng(0x0031_C0DE);
    let src = super::moe::MOE_SRC;
    let route = g.pair(src, "moe_route");
    let dispatch = g.pair(src, "moe_dispatch_t1");
    let both = g.pair(src, "moe_route_dispatch_t1");
    let align = g.pair(src, "moe_align_block_size");
    const SHARED: u32 = 4 * (512 + 2 * 256 + 513 + 256 + 64);

    for (max_tokens, valid, experts, top_k, ties) in [
        (1usize, 1i32, 256i32, 8i32, false),
        (3, 3, 256, 8, false),
        (2048, 2041, 256, 8, false),
        (64, 64, 256, 8, true),
        (5, 4, 60, 3, true),
        (2, 2, 512, 8, false),
    ] {
        let mut logits = rng.vec(max_tokens * experts as usize, -6.0, 6.0);
        if ties {
            // Coarse values: many exact ties, at the maximum and in the tail.
            for v in logits.iter_mut() {
                *v = (*v * 2.0).round() / 2.0;
            }
        }
        let logits = s.clone_htod(&logits).unwrap();
        let valid_tokens = s.clone_htod(&[valid]).unwrap();
        let n = max_tokens * top_k as usize;
        let label = format!("moe_route {max_tokens}x{experts} top{top_k} ties={ties}");
        check(&g.ctx, s, &label, &[blank(n), blank(n)], |arm, out| {
            let [ids, weights] = out else { unreachable!() };
            let mut b = s.launch_builder(&route[arm]);
            b.arg(&logits)
                .arg(&valid_tokens)
                .arg(&experts)
                .arg(&top_k)
                .arg(ids)
                .arg(weights);
            // SAFETY: one block per token slot, gated on the device count.
            unsafe { b.launch(cfg((max_tokens as u32, 1), (256, 1), SHARED)) }.unwrap();
        });
    }

    for (experts, top_k, block_size, ties) in [
        (256i32, 8i32, 4i32, false),
        (256, 8, 8, true),
        (60, 3, 4, true),
        (512, 8, 4, false),
    ] {
        let mut logits = rng.vec(experts as usize, -6.0, 6.0);
        if ties {
            for v in logits.iter_mut() {
                *v = (*v * 2.0).round() / 2.0;
            }
        }
        let logits = s.clone_htod(&logits).unwrap();
        let one = s.clone_htod(&[1i32]).unwrap();
        let sorted_capacity = top_k * block_size + 8;
        let expert_capacity = sorted_capacity / block_size + 3;
        let ids = s.clone_htod(&rng.ints(top_k as usize, experts)).unwrap();
        let label = format!("moe_dispatch_t1 {experts} top{top_k} bs={block_size}");
        let lens = [
            blank(sorted_capacity as usize),
            blank(expert_capacity as usize),
            blank(1),
        ];
        check(&g.ctx, s, &label, &lens, |arm, out| {
            let [sorted, owners, post] = out else {
                unreachable!()
            };
            let mut b = s.launch_builder(&dispatch[arm]);
            b.arg(&ids)
                .arg(&one)
                .arg(&top_k)
                .arg(&experts)
                .arg(&block_size)
                .arg(&sorted_capacity)
                .arg(&expert_capacity)
                .arg(sorted)
                .arg(owners)
                .arg(post);
            // SAFETY: one block; the tables are sized as declared.
            unsafe { b.launch(cfg((1, 1), (256, 1), SHARED)) }.unwrap();
        });
        let label =
            format!("moe_route_dispatch_t1 {experts} top{top_k} bs={block_size} ties={ties}");
        let k = top_k as usize;
        let lens = [
            blank(k),
            blank(k),
            blank(sorted_capacity as usize),
            blank(expert_capacity as usize),
            blank(1),
        ];
        check(&g.ctx, s, &label, &lens, |arm, out| {
            let [ids, weights, sorted, owners, post] = out else {
                unreachable!()
            };
            let mut b = s.launch_builder(&both[arm]);
            b.arg(&logits)
                .arg(&one)
                .arg(&experts)
                .arg(&top_k)
                .arg(&block_size)
                .arg(&sorted_capacity)
                .arg(&expert_capacity)
                .arg(ids)
                .arg(weights)
                .arg(sorted)
                .arg(owners)
                .arg(post);
            // SAFETY: one block; the tables are sized as declared.
            unsafe { b.launch(cfg((1, 1), (256, 1), SHARED)) }.unwrap();
        });
    }

    for (valid, experts, top_k, block_size) in [
        (3i32, 256i32, 8i32, 4i32),
        (64, 256, 8, 8),
        (2048, 256, 8, 16),
        (41, 60, 3, 4),
    ] {
        let numel = (valid * top_k) as usize;
        let ids = rng.ints(numel, experts);
        let mut counts = vec![0i32; experts as usize];
        for &e in &ids {
            counts[e as usize] += 1;
        }
        let padded: i32 = counts
            .iter()
            .map(|c| (c + block_size - 1) / block_size * block_size)
            .sum();
        let sorted_capacity = padded + 16;
        let expert_capacity = padded / block_size + 5;
        let ids = s.clone_htod(&ids).unwrap();
        let counts = s.clone_htod(&counts).unwrap();
        let valid_tokens = s.clone_htod(&[valid]).unwrap();
        let label =
            format!("moe_align_block_size {valid}x{top_k} experts={experts} bs={block_size}");
        let lens = [
            blank(sorted_capacity as usize),
            blank(expert_capacity as usize),
            blank(1),
            blank(expert_capacity as usize),
        ];
        check(&g.ctx, s, &label, &lens, |arm, out| {
            let [sorted, owners, post, live] = out else {
                unreachable!()
            };
            let mut b = s.launch_builder(&align[arm]);
            b.arg(&ids)
                .arg(&valid_tokens)
                .arg(&counts)
                .arg(&top_k)
                .arg(&experts)
                .arg(&block_size)
                .arg(&sorted_capacity)
                .arg(&expert_capacity)
                .arg(sorted)
                .arg(owners)
                .arg(post)
                .arg(live);
            // SAFETY: one block per expert.
            unsafe { b.launch(cfg((experts as u32, 1), (256, 1), SHARED)) }.unwrap();
        });
    }
}
