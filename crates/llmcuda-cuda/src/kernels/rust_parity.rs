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

struct Rng(u64);
impl Rng {
    fn vec(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                self.0 ^= self.0 >> 12;
                self.0 ^= self.0 << 25;
                self.0 ^= self.0 >> 27;
                let u =
                    (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32;
                lo + (hi - lo) * u
            })
            .collect()
    }
    /// As `vec`, with the values that break an exponential planted in it.
    fn spiked(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        let mut v = self.vec(n, lo, hi);
        let specials = [
            0.0,
            -0.0,
            20.0,
            20.000002,
            19.999998,
            89.0,
            -89.0,
            104.0,
            -104.0,
            1e-30,
            -1e-30,
            f32::MAX,
            f32::MIN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1e-42,
        ];
        for (slot, special) in v.iter_mut().step_by(37).zip(specials) {
            *slot = special;
        }
        v
    }
}

fn cfg(grid: (u32, u32), block: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (grid.0, grid.1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Launch both kernels into fresh sentinel-filled outputs and require equal
/// bits, then time them: six alternating pairs of 100 launches between CUDA
/// events. `outs` are the output lengths; the closure launches one kernel.
fn check(
    ctx: &Arc<CudaContext>,
    s: &Arc<CudaStream>,
    label: &str,
    pair: &[CudaFunction; 2],
    outs: &[usize],
    launch: impl Fn(&CudaFunction, &mut [CudaSlice<f32>]),
) {
    let fresh = || -> Vec<CudaSlice<f32>> {
        outs.iter()
            .map(|&n| s.clone_htod(&vec![7.0f32; n]).expect("alloc"))
            .collect()
    };
    let result = pair.each_ref().map(|f| {
        let mut out = fresh();
        launch(f, &mut out);
        out.iter()
            .map(|d| {
                let v = s.clone_dtoh(d).expect("read");
                v.into_iter().map(f32::to_bits).collect::<Vec<u32>>()
            })
            .collect::<Vec<_>>()
    });
    assert!(result[0] == result[1], "{label}: Rust and CUDA C++ differ");

    let mut out = fresh();
    for f in pair {
        for _ in 0..20 {
            launch(f, &mut out);
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
                launch(&pair[arm], &mut out);
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
fn the_moe_swiglu_and_reduce_ports_match_cuda_cpp() {
    let Some(g) = gpu() else { return };
    let s = &g.stream;
    let mut rng = Rng(0x0005_C0DE);

    let swiglu = g.pair(super::moe::MOE_SRC, "moe_swiglu");
    for n in [1000usize, 3 * 512, 2048 * 512] {
        let gate = s.clone_htod(&rng.spiked(n, -8.0, 8.0)).unwrap();
        let up = s.clone_htod(&rng.spiked(n, -8.0, 8.0)).unwrap();
        let n_arg = n as i64;
        check(
            &g.ctx,
            s,
            &format!("moe_swiglu n={n}"),
            &swiglu,
            &[n + 3],
            |f, out| {
                let mut b = s.launch_builder(f);
                b.arg(&gate).arg(&up).arg(&n_arg).arg(&mut out[0]);
                // SAFETY: every buffer covers n floats and the kernel bounds its index.
                unsafe { b.launch(cfg(((n as u32).div_ceil(256), 1), 256)) }.unwrap();
            },
        );
    }

    let reduce = g.pair(super::moe::MOE_SRC, "moe_reduce");
    for (max_tokens, valid, top_k, hidden) in [
        (1usize, 1i32, 8i32, 2048usize),
        (3, 3, 8, 2048),
        (2048, 2041, 8, 2048),
        (5, 4, 3, 1000),
    ] {
        let partial = s
            .clone_htod(&rng.vec(max_tokens * top_k as usize * hidden, -4.0, 4.0))
            .unwrap();
        let valid_tokens = s.clone_htod(&[valid]).unwrap();
        let hidden_arg = hidden as i32;
        let label = format!("moe_reduce {max_tokens}x{top_k}x{hidden}");
        check(
            &g.ctx,
            s,
            &label,
            &reduce,
            &[max_tokens * hidden],
            |f, out| {
                let mut b = s.launch_builder(f);
                b.arg(&partial)
                    .arg(&valid_tokens)
                    .arg(&top_k)
                    .arg(&hidden_arg)
                    .arg(&mut out[0]);
                // SAFETY: partial and out are sized for max_tokens; the kernel
                // gates on valid_tokens and bounds its column.
                unsafe { b.launch(cfg((max_tokens as u32, (hidden as u32).div_ceil(256)), 256)) }
                    .unwrap();
            },
        );
    }
}

#[test]
fn the_scan_decay_port_matches_cuda_cpp() {
    let Some(g) = gpu() else { return };
    let s = &g.stream;
    let mut rng = Rng(0x0006_C0DE);
    let decay = g.pair(super::gdn_chunked::GDN_CHUNKED_SRC, "gdn_scan_decay");
    for (n, blocks) in [(32usize, 1u32), (3 * 32, 1), (2048 * 32, 64), (100_003, 16)] {
        let log_decay = s.clone_htod(&rng.spiked(n, -12.0, 0.0)).unwrap();
        let n_arg = n as i64;
        check(
            &g.ctx,
            s,
            &format!("gdn_scan_decay n={n}"),
            &decay,
            &[n + 3],
            |f, out| {
                let mut b = s.launch_builder(f);
                b.arg(&log_decay).arg(&mut out[0]).arg(&n_arg);
                // SAFETY: both buffers cover n floats; the loop is grid-stride.
                unsafe { b.launch(cfg((blocks, 1), 256)) }.unwrap();
            },
        );
    }
}

#[test]
fn the_k2_combine_ports_match_cuda_cpp() {
    let Some(g) = gpu() else { return };
    let s = &g.stream;
    let mut rng = Rng(0x0007_C0DE);
    let src = super::k2::source();
    let combine = g.pair(&src, "k2_combine");
    let residual = g.pair(&src, "k2_combine_residual");
    let values = g.pair(&src, "k2_values");
    let gate = g.pair(&src, "k2_gate");

    for (tokens, topk, rows) in [
        (1usize, 8i32, 2048usize),
        (3, 8, 2048),
        (2048, 8, 2048),
        (5, 3, 300),
    ] {
        let p = s
            .clone_htod(&rng.spiked(tokens * topk as usize * rows, -6.0, 6.0))
            .unwrap();
        let w = s
            .clone_htod(&rng.vec(tokens * topk as usize, 0.0, 1.0))
            .unwrap();
        let shared = s.clone_htod(&rng.vec(tokens * rows, -4.0, 4.0)).unwrap();
        let x = s.clone_htod(&rng.vec(tokens * rows, -4.0, 4.0)).unwrap();
        let rows_arg = rows as i32;
        let grid = ((rows as u32).div_ceil(256), tokens as u32);
        let shape = format!("{tokens}x{topk}x{rows}");

        for (name, pair) in [("k2_combine", &combine), ("k2_values", &values)] {
            check(
                &g.ctx,
                s,
                &format!("{name} {shape}"),
                pair,
                &[tokens * rows],
                |f, out| {
                    let mut b = s.launch_builder(f);
                    b.arg(&p).arg(&w).arg(&mut out[0]).arg(&rows_arg).arg(&topk);
                    // SAFETY: p, w and out are sized for this grid.
                    unsafe { b.launch(cfg(grid, 256)) }.unwrap();
                },
            );
        }

        let label = format!("k2_combine_residual {shape}");
        check(
            &g.ctx,
            s,
            &label,
            &residual,
            &[tokens * rows; 2],
            |f, out| {
                let [ffn, out] = out else { unreachable!() };
                let mut b = s.launch_builder(f);
                b.arg(&p)
                    .arg(&w)
                    .arg(&shared)
                    .arg(&x)
                    .arg(ffn)
                    .arg(out)
                    .arg(&rows_arg)
                    .arg(&topk);
                // SAFETY: every buffer is sized for this grid; outputs are disjoint.
                unsafe { b.launch(cfg(grid, 256)) }.unwrap();
            },
        );
    }

    for n in [1000usize, 3 * 4096, 2048 * 4096] {
        let x = s.clone_htod(&rng.vec(n, -6.0, 6.0)).unwrap();
        let gates = s.clone_htod(&rng.spiked(n, -40.0, 40.0)).unwrap();
        let n_arg = n as i64;
        check(
            &g.ctx,
            s,
            &format!("k2_gate n={n}"),
            &gate,
            &[n + 3],
            |f, out| {
                let mut b = s.launch_builder(f);
                b.arg(&x).arg(&gates).arg(&mut out[0]).arg(&n_arg);
                // SAFETY: every buffer covers n floats and the kernel bounds its index.
                unsafe { b.launch(cfg(((n as u32).div_ceil(256), 1), 256)) }.unwrap();
            },
        );
    }
}

#[test]
fn the_decode_combine_port_matches_cuda_cpp() {
    let Some(g) = gpu() else { return };
    let s = &g.stream;
    let mut rng = Rng(0x0008_C0DE);
    let combine = g.pair(super::attention::ATTENTION_SRC, "attn_flash_decode_combine");
    for (q_heads, head_dim, splits) in [
        (16usize, 256usize, 1usize),
        (16, 256, 4),
        (16, 256, 32),
        (8, 128, 7),
        (2, 64, 3),
    ] {
        let acc = s
            .clone_htod(&rng.vec(splits * q_heads * head_dim, -3.0, 3.0))
            .unwrap();
        let mut m = rng.vec(splits * q_heads, -30.0, 30.0);
        // Empty splits carry -inf; split 0 never does.
        for slot in m.iter_mut().skip(q_heads).step_by(5) {
            *slot = f32::NEG_INFINITY;
        }
        let m = s.clone_htod(&m).unwrap();
        let l = s
            .clone_htod(&rng.vec(splits * q_heads, 0.5, 300.0))
            .unwrap();
        let (qh, hd, ns) = (q_heads as i32, head_dim as i32, splits as i32);
        let label = format!("attn_flash_decode_combine {q_heads}x{head_dim}x{splits}");
        check(
            &g.ctx,
            s,
            &label,
            &combine,
            &[q_heads * head_dim],
            |f, out| {
                let mut b = s.launch_builder(f);
                b.arg(&acc)
                    .arg(&m)
                    .arg(&l)
                    .arg(&mut out[0])
                    .arg(&qh)
                    .arg(&hd)
                    .arg(&ns);
                // SAFETY: one block per head, one thread per dimension; the
                // partials are sized for `splits`.
                unsafe { b.launch(cfg((q_heads as u32, 1), head_dim as u32)) }.unwrap();
            },
        );
    }
}
