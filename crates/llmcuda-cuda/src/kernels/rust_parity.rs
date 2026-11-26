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
fn the_half_conversion_ports_match_cuda_cpp() {
    let Some(g) = gpu() else { return };
    let s = &g.stream;
    let mut rng = Rng(0x0010_C0DE);

    let to_half = g.pair(super::hgemm::HGEMM_SRC, "hg_to_half");
    for n in [1024usize, 3 * 2048, 2048 * 2048] {
        let x = s
            .clone_htod(&spiked(&mut rng, n, -70000.0, 70000.0))
            .unwrap();
        let n_arg = n as i64;
        check(
            &g.ctx,
            s,
            &format!("hg_to_half n={n}"),
            &[blank(n / 2)],
            |arm, out| {
                let mut b = s.launch_builder(&to_half[arm]);
                b.arg(&x).arg(&mut out[0]).arg(&n_arg);
                // SAFETY: x covers n floats, out n halves.
                unsafe { b.launch(cfg(((n as u32).div_ceil(256), 1), (256, 1), 0)) }.unwrap();
            },
        );
    }

    let src = super::vision::VISION_SRC;
    let to_float = g.pair(src, "vision_half_to_float");
    let add = g.pair(src, "vision_add_f16");
    let gelu = g.pair(src, "vision_gelu");
    for copies in [1usize, 16] {
        let a = all_halves(&mut rng, copies);
        let n = a.len();
        let (n_arg, grid) = (n as i64, ((n as u32).div_ceil(256), 1));
        let da = s.clone_htod(&a).unwrap();
        let db = s.clone_htod(&all_halves(&mut rng, copies)).unwrap();
        check(
            &g.ctx,
            s,
            &format!("vision_half_to_float n={n}"),
            &[blank(n)],
            |arm, out| {
                let mut b = s.launch_builder(&to_float[arm]);
                b.arg(&da).arg(&mut out[0]).arg(&n_arg);
                // SAFETY: src covers n halves, dst n floats.
                unsafe { b.launch(cfg(grid, (256, 1), 0)) }.unwrap();
            },
        );
        check(
            &g.ctx,
            s,
            &format!("vision_add_f16 n={n}"),
            &[words(&a)],
            |arm, out| {
                let mut b = s.launch_builder(&add[arm]);
                b.arg(&mut out[0]).arg(&db).arg(&n_arg);
                // SAFETY: both cover n halves.
                unsafe { b.launch(cfg(grid, (256, 1), 0)) }.unwrap();
            },
        );
        check(
            &g.ctx,
            s,
            &format!("vision_gelu n={n}"),
            &[words(&a)],
            |arm, out| {
                let mut b = s.launch_builder(&gelu[arm]);
                b.arg(&mut out[0]).arg(&n_arg);
                // SAFETY: x covers n halves.
                unsafe { b.launch(cfg(grid, (256, 1), 0)) }.unwrap();
            },
        );
    }
}

#[test]
fn the_kv_append_and_rotary_ports_match_cuda_cpp() {
    let Some(g) = gpu() else { return };
    let s = &g.stream;
    let mut rng = Rng(0x0011_C0DE);

    let append = g.pair(super::attention::ATTENTION_SRC, "attn_kv_append");
    for (tokens, row, position) in [
        (1usize, 512usize, 5i32),
        (3, 512, 0),
        (256, 512, 1000),
        (2, 96, 3),
    ] {
        let span = tokens * row;
        let key = s.clone_htod(&spiked(&mut rng, span, -8.0, 8.0)).unwrap();
        let value = s
            .clone_htod(&spiked(&mut rng, span, -70000.0, 70000.0))
            .unwrap();
        let pos = s.clone_htod(&[position]).unwrap();
        let cache_words = (position as usize + tokens + 1) * row / 2;
        let (span_arg, row_arg) = (span as i32, row as i32);
        let label = format!("attn_kv_append {tokens}x{row} at {position}");
        check(
            &g.ctx,
            s,
            &label,
            &[blank(cache_words), blank(cache_words)],
            |arm, out| {
                let [k_cache, v_cache] = out else {
                    unreachable!()
                };
                let mut b = s.launch_builder(&append[arm]);
                b.arg(&key)
                    .arg(&value)
                    .arg(k_cache)
                    .arg(v_cache)
                    .arg(&pos)
                    .arg(&span_arg)
                    .arg(&row_arg);
                // SAFETY: both caches hold position + tokens rows.
                unsafe { b.launch(cfg(((2 * span as u32).div_ceil(256), 1), (256, 1), 0)) }
                    .unwrap();
            },
        );
    }

    let apply = g.pair(super::k2_rope::SRC, "k2_rotary_apply");
    for (tokens, heads, dim, rope_dim, offset) in [
        (1usize, 16usize, 256usize, 64usize, 0i32),
        (3, 16, 256, 64, 7),
        (2048, 16, 256, 64, 0),
        (13, 2, 128, 128, 3),
    ] {
        let n = tokens * heads * dim;
        let input = s.clone_htod(&rng.vec(n, -6.0, 6.0)).unwrap();
        let coefficients = s
            .clone_htod(&rng.vec((offset as usize + tokens) * rope_dim, -1.0, 1.0))
            .unwrap();
        let (heads_arg, dim_arg, rope_arg, tokens_arg) =
            (heads as i32, dim as i32, rope_dim as i32, tokens as i32);
        let label = format!("k2_rotary_apply {tokens}x{heads}x{dim} rope {rope_dim}");
        check(&g.ctx, s, &label, &[blank(n)], |arm, out| {
            let mut b = s.launch_builder(&apply[arm]);
            b.arg(&input)
                .arg(&mut out[0])
                .arg(&coefficients)
                .arg(&heads_arg)
                .arg(&dim_arg)
                .arg(&rope_arg)
                .arg(&offset)
                .arg(&tokens_arg);
            // SAFETY: eight tokens a block, one thread per dimension.
            unsafe {
                b.launch(cfg(
                    ((tokens as u32).div_ceil(8), heads as u32),
                    (dim as u32, 1),
                    0,
                ))
            }
            .unwrap();
        });
    }
}

#[test]
fn the_decision_head_ports_match_cuda_cpp() {
    let Some(g) = gpu() else { return };
    let s = &g.stream;
    let mut rng = Rng(0x0012_C0DE);
    let src = super::decision::DECISION_SRC;
    let add = g.pair(src, "dec_add_owner_rows");
    let mean = g.pair(src, "dec_span_mean");
    let combine = g.pair(src, "dec_attention_combine");

    for (rows, cols, nspans) in [(40usize, 2048usize, 4usize), (700, 2048, 64), (30, 300, 5)] {
        let (ld, stride) = (cols + 8, 3i32);
        // (start, end, owner) triples covering uneven, overlapping spans.
        let spans: Vec<i32> = (0..nspans)
            .flat_map(|i| {
                let start = (i * 7) % (rows - 9);
                let end = start + 1 + (i * 5) % 8;
                [start as i32, end as i32, ((i * 3) % rows) as i32]
            })
            .collect();
        let spans = s.clone_htod(&spans).unwrap();
        let x = s.clone_htod(&rng.vec(rows * ld, -6.0, 6.0)).unwrap();
        let stats = s.clone_htod(&rng.vec(rows * 2, 0.1, 3.0)).unwrap();
        let gain = s.clone_htod(&rng.vec(cols, 0.2, 2.0)).unwrap();
        let bias = s.clone_htod(&rng.vec(cols, -1.0, 1.0)).unwrap();
        let (ld_arg, cols_arg, n_arg) = (ld as i64, cols as i32, nspans as i32);
        let grid = ((cols as u32).div_ceil(256), nspans as u32);

        let label = format!("dec_span_mean {nspans} spans x {cols}");
        check(&g.ctx, s, &label, &[blank(nspans * ld)], |arm, out| {
            let mut b = s.launch_builder(&mean[arm]);
            b.arg(&x)
                .arg(&ld_arg)
                .arg(&cols_arg)
                .arg(&stats)
                .arg(&gain)
                .arg(&bias)
                .arg(&spans)
                .arg(&stride)
                .arg(&n_arg)
                .arg(&mut out[0])
                .arg(&ld_arg);
            // SAFETY: every span row is inside x; out has a row per span.
            unsafe { b.launch(cfg(grid, (256, 1), 0)) }.unwrap();
        });

        let y: Vec<u32> = rng
            .vec(nspans * ld, -6.0, 6.0)
            .into_iter()
            .map(f32::to_bits)
            .collect();
        let label = format!("dec_add_owner_rows {nspans} x {cols}");
        check(&g.ctx, s, &label, &[y], |arm, out| {
            let mut b = s.launch_builder(&add[arm]);
            b.arg(&mut out[0])
                .arg(&ld_arg)
                .arg(&x)
                .arg(&ld_arg)
                .arg(&spans)
                .arg(&stride)
                .arg(&n_arg)
                .arg(&cols_arg);
            // SAFETY: every owner row is inside x.
            unsafe { b.launch(cfg(grid, (256, 1), 0)) }.unwrap();
        });
    }

    for (splits, nq, heads, dh) in [
        (1i32, 4i32, 16i32, 64i32),
        (4, 64, 16, 64),
        (8, 300, 16, 128),
        (3, 5, 2, 20),
    ] {
        let slots = (splits * nq * heads) as usize;
        let mut part = rng.vec(slots * dh as usize, -3.0, 3.0);
        // (running max, normalizer) pairs follow the weighted sums.
        for _ in 0..slots {
            part.extend([rng.vec(1, -30.0, 30.0)[0], rng.vec(1, 0.5, 300.0)[0]]);
        }
        let part = s.clone_htod(&part).unwrap();
        let ldo = (heads * dh + 8) as i64;
        let label = format!("dec_attention_combine {splits}x{nq}x{heads}x{dh}");
        check(
            &g.ctx,
            s,
            &label,
            &[blank(nq as usize * ldo as usize)],
            |arm, out| {
                let mut b = s.launch_builder(&combine[arm]);
                b.arg(&part)
                    .arg(&splits)
                    .arg(&nq)
                    .arg(&heads)
                    .arg(&dh)
                    .arg(&mut out[0])
                    .arg(&ldo);
                // SAFETY: part holds every split's sums and pairs.
                unsafe { b.launch(cfg((nq as u32, heads as u32), (128, 1), 0)) }.unwrap();
            },
        );
    }
}
