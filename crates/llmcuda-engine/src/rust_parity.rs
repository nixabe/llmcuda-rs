//! Rust ports against the CUDA C++ kernels they are about to replace, bit
//! for bit and timed. Lives only while both sources exist.

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use llmcuda_cuda::device::driver_available;
use llmcuda_cuda::kernels::{RUST_KERNELS_PTX, compile};
use std::sync::Arc;

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
fn the_moe_gate_and_combine_port_matches_cuda_cpp() {
    if !driver_available() {
        println!("SKIPPED: no CUDA driver present");
        return;
    }
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            println!("SKIPPED: could not create a context on device 0: {e}");
            return;
        }
    };
    let s = &ctx.default_stream();
    let rust = ctx
        .load_module(cudarc::nvrtc::Ptx::from_src(RUST_KERNELS_PTX))
        .expect("Rust PTX loads");
    let cpp = ctx
        .load_module(compile(crate::block::moe::GLUE_SRC, "rust_parity").expect("C++ compiles"))
        .expect("C++ loads");
    let name = "moe_block_gate_and_combine";
    let pair = [
        cpp.load_function(name).expect("C++ entry"),
        rust.load_function(name).expect("Rust entry"),
    ];
    let mut rng = Rng(0x0019_C0DE);
    for (max_tokens, valid, top_k, hidden, gy) in [
        (1usize, 1i32, 8i32, 2048usize, 8u32),
        (3, 3, 8, 2048, 8),
        (2048, 2041, 8, 2048, 1),
        (2048, 2041, 8, 2048, 8),
        (5, 4, 3, 1000, 2),
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
        let label = format!("{name} {max_tokens}x{top_k}x{hidden} gy={gy}");
        let lens = [blank(max_tokens), blank(rows), blank(rows), blank(rows)];
        check(&ctx, s, &label, &lens, |arm, out| {
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
            // SAFETY: every buffer is sized for max_tokens; the kernel gates
            // on the device-side count.
            unsafe { b.launch(cfg((max_tokens as u32, gy), (256, 1), 128)) }.unwrap();
        });
    }
}
