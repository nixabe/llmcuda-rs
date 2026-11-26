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
    fn ints(&mut self, n: usize, below: i32) -> Vec<i32> {
        (0..n)
            .map(|_| ((self.next() >> 33) % below as u64) as i32)
            .collect()
    }
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
/// events. `outs` are the output lengths in 32-bit words; the closure
/// launches arm 0 (CUDA C++) or arm 1 (Rust).
fn check(
    ctx: &Arc<CudaContext>,
    s: &Arc<CudaStream>,
    label: &str,
    outs: &[usize],
    launch: impl Fn(usize, &mut [CudaSlice<u32>]),
) {
    let fresh = || -> Vec<CudaSlice<u32>> {
        outs.iter()
            .map(|&n| s.clone_htod(&vec![SENTINEL; n]).expect("alloc"))
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
    assert!(
        result[0].iter().flatten().any(|&w| w != SENTINEL),
        "{label}: nothing was written"
    );

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
fn the_argmax_ports_match_cuda_cpp() {
    let Some(g) = gpu() else { return };
    let s = &g.stream;
    let mut rng = Rng(0x000A_C0DE);
    let partial = g.pair(super::lm_head::LM_HEAD_SRC, "argmax_partial");
    let last = g.pair(super::lm_head::LM_HEAD_SRC, "argmax_final");

    for (n, flavour) in [
        (5usize, 0),
        (1000, 1),
        (248_320, 1),
        (248_320, 2),
        (248_320, 3),
    ] {
        let mut x = rng.vec(n, -20.0, 20.0);
        match flavour {
            // Ties for the maximum, the lower index far from the higher.
            1 => {
                x[n / 3] = 25.0;
                x[n - 1] = 25.0;
                x[n / 2] = f32::NAN;
            }
            // Element 0 is NaN: it is the seed and nothing compares above it.
            2 => x[0] = f32::NAN,
            // Entirely NaN, and an infinity that must still win elsewhere.
            3 => x.fill(f32::NAN),
            _ => x[2] = f32::INFINITY,
        }
        let blocks = n.div_ceil(256).min(512);
        let x = s.clone_htod(&x).unwrap();
        let n_arg = n as i32;
        let label = format!("argmax_partial n={n} case {flavour}");
        check(&g.ctx, s, &label, &[blocks, blocks], |arm, out| {
            let [pv, pi] = out else { unreachable!() };
            let mut b = s.launch_builder(&partial[arm]);
            b.arg(&x).arg(&n_arg).arg(pv).arg(pi);
            // SAFETY: x covers n floats; pv and pi cover one entry a block.
            unsafe { b.launch(cfg((blocks as u32, 1), (256, 1), 0)) }.unwrap();
        });
    }

    for n in [1usize, 20, 512] {
        let mut pv = rng.vec(n, -20.0, 20.0);
        let pi: Vec<i32> = (0..n as i32).map(|i| 1000 - i).collect();
        if n > 4 {
            // A tie whose lower *index* sits in the later slot.
            pv[1] = 30.0;
            pv[n - 1] = 30.0;
        }
        let pv = s.clone_htod(&pv).unwrap();
        let pi = s.clone_htod(&pi).unwrap();
        let n_arg = n as i32;
        check(
            &g.ctx,
            s,
            &format!("argmax_final n={n}"),
            &[1],
            |arm, out| {
                let mut b = s.launch_builder(&last[arm]);
                b.arg(&pv).arg(&pi).arg(&n_arg).arg(&mut out[0]);
                // SAFETY: pv and pi cover n entries; one block writes one int.
                unsafe { b.launch(cfg((1, 1), (256, 1), 0)) }.unwrap();
            },
        );
    }
}

#[test]
fn the_splitk_reduce_port_matches_cuda_cpp() {
    let Some(g) = gpu() else { return };
    let s = &g.stream;
    let mut rng = Rng(0x000B_C0DE);
    let reduce = g.pair(super::hgemm::HGEMM_SRC, "hg_splitk_reduce");
    for (m, n, ldo, splits, accumulate) in [
        (1usize, 2048usize, 2048usize, 4i32, 0i32),
        (3, 2048, 2048, 8, 1),
        (2048, 2048, 2048, 4, 0),
        (2048, 512, 600, 2, 1),
        (7, 34, 40, 1, 1),
    ] {
        let part = s
            .clone_htod(&rng.vec(splits as usize * m * n, -4.0, 4.0))
            .unwrap();
        let (m_arg, n_arg, ldo_arg) = (m as i32, n as i32, ldo as i32);
        let label = format!("hg_splitk_reduce {m}x{n} splits={splits} acc={accumulate}");
        check(&g.ctx, s, &label, &[m * ldo], |arm, out| {
            let mut b = s.launch_builder(&reduce[arm]);
            b.arg(&part)
                .arg(&mut out[0])
                .arg(&m_arg)
                .arg(&n_arg)
                .arg(&ldo_arg)
                .arg(&splits)
                .arg(&accumulate);
            // SAFETY: part covers every split; out covers m rows of ldo.
            unsafe { b.launch(cfg((((m * n / 2) as u32).div_ceil(256), 1), (256, 1), 0)) }.unwrap();
        });
    }
}

#[test]
fn the_k2_dispatch_and_norm_ports_match_cuda_cpp() {
    let Some(g) = gpu() else { return };
    let s = &g.stream;
    let mut rng = Rng(0x000C_C0DE);
    let src = super::k2::source();
    let count = g.pair(&src, "k2_dispatch_count");
    let prefix = g.pair(&src, "k2_dispatch_prefix");
    let gather = g.pair(&src, "k2_dispatch_gather");
    let norm = g.pair(&src, "k2_norm");

    for (tokens, topk, experts, tile) in [
        (1usize, 8usize, 128i32, 4i32),
        (3, 8, 128, 4),
        (2048, 8, 128, 8),
        (300, 3, 37, 8),
    ] {
        let pairs = tokens * topk;
        let ids = s.clone_htod(&rng.ints(pairs, experts)).unwrap();
        let sorted_len = pairs + experts as usize * tile as usize;
        let capacity = (sorted_len / tile as usize) as i32;
        let (pairs_arg, grid) = (pairs as i32, (experts as u32, 1));
        let label =
            format!("k2_dispatch (count, prefix, gather) {tokens}x{topk} experts={experts}");
        let lens = [
            experts as usize * 128,
            experts as usize,
            capacity as usize,
            sorted_len,
        ];
        check(&g.ctx, s, &label, &lens, |arm, out| {
            let [counts, starts, owners, sorted] = out else {
                unreachable!()
            };
            // SAFETY: the buffers are sized for `experts` blocks of 128 lanes
            // and for every padded run.
            unsafe {
                let mut b = s.launch_builder(&count[arm]);
                b.arg(&ids).arg(&mut *counts).arg(&pairs_arg);
                b.launch(cfg(grid, (128, 1), 0)).unwrap();
                let mut b = s.launch_builder(&prefix[arm]);
                b.arg(&*counts)
                    .arg(&mut *starts)
                    .arg(owners)
                    .arg(&experts)
                    .arg(&capacity)
                    .arg(&tile);
                b.launch(cfg((1, 1), (128, 1), 0)).unwrap();
                let mut b = s.launch_builder(&gather[arm]);
                b.arg(&ids)
                    .arg(&*counts)
                    .arg(&*starts)
                    .arg(sorted)
                    .arg(&pairs_arg)
                    .arg(&tile);
                b.launch(cfg(grid, (128, 1), 0)).unwrap();
            }
        });
    }

    for (tokens, width, groups) in [
        (1usize, 2048usize, 1i32),
        (3, 2048, 4),
        (2048, 2048, 4),
        (5, 768, 3),
    ] {
        let x = s.clone_htod(&rng.vec(tokens * width, -6.0, 6.0)).unwrap();
        let w = s.clone_htod(&rng.vec(width, 0.2, 2.0)).unwrap();
        let (width_arg, eps) = (width as i32, 1e-6f32);
        let label = format!("k2_norm {tokens}x{width} groups={groups}");
        check(&g.ctx, s, &label, &[tokens * width], |arm, out| {
            let mut b = s.launch_builder(&norm[arm]);
            b.arg(&x)
                .arg(&w)
                .arg(&mut out[0])
                .arg(&width_arg)
                .arg(&groups)
                .arg(&eps);
            // SAFETY: one 256-thread block per token and group.
            unsafe { b.launch(cfg(((tokens as i32 * groups) as u32, 1), (256, 1), 0)) }.unwrap();
        });
    }
}

#[test]
fn the_moe_align_count_port_matches_cuda_cpp() {
    let Some(g) = gpu() else { return };
    let s = &g.stream;
    let mut rng = Rng(0x000D_C0DE);
    let count = g.pair(super::moe::MOE_SRC, "moe_align_count");
    for (valid, top_k, experts) in [
        (1i32, 8i32, 256i32),
        (3, 8, 256),
        (2048, 8, 256),
        (41, 3, 60),
    ] {
        let numel = (valid * top_k) as usize;
        let ids = s.clone_htod(&rng.ints(numel, experts)).unwrap();
        let valid_tokens = s.clone_htod(&[valid]).unwrap();
        let sorted_capacity = (numel + experts as usize * 8) as i32;
        let expert_capacity = sorted_capacity / 4;
        let label = format!("moe_align_count {valid}x{top_k} experts={experts}");
        let lens = [
            sorted_capacity as usize,
            expert_capacity as usize,
            experts as usize,
        ];
        check(&g.ctx, s, &label, &lens, |arm, out| {
            let [sorted, owners, counts] = out else {
                unreachable!()
            };
            let mut b = s.launch_builder(&count[arm]);
            b.arg(&ids)
                .arg(&valid_tokens)
                .arg(&top_k)
                .arg(&sorted_capacity)
                .arg(&expert_capacity)
                .arg(sorted)
                .arg(owners)
                .arg(counts);
            // SAFETY: one block per expert, an int of shared memory a thread.
            unsafe { b.launch(cfg((experts as u32, 1), (256, 1), 256 * 4)) }.unwrap();
        });
    }
}
