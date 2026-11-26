//! Rust ports against the CUDA C++ kernels they are about to replace, bit
//! for bit and timed. Lives only while both sources exist.

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
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
fn the_engine_glue_ports_match_cuda_cpp() {
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
    let pair = |src: &str, name: &str| -> [CudaFunction; 2] {
        let cpp = ctx
            .load_module(compile(src, "rust_parity").expect("C++ compiles"))
            .expect("C++ loads");
        [
            cpp.load_function(name).expect("C++ entry"),
            rust.load_function(name).expect("Rust entry"),
        ]
    };
    let mut rng = Rng(0x000E_C0DE);

    let add = pair(crate::block::dense_ffn::GLUE_SRC, "dense_ffn_add_residual");
    for (tokens, valid, hidden, gy) in [
        (1usize, 1i32, 5120usize, 20u32),
        (3, 3, 5120, 20),
        (2048, 2041, 5120, 20),
        (5, 4, 1000, 2),
    ] {
        let ffn = s.clone_htod(&rng.vec(tokens * hidden, -4.0, 4.0)).unwrap();
        let residual = s.clone_htod(&rng.vec(tokens * hidden, -4.0, 4.0)).unwrap();
        let valid_tokens = s.clone_htod(&[valid]).unwrap();
        let hidden_arg = hidden as i32;
        let label = format!("dense_ffn_add_residual {tokens}x{hidden}");
        check(&ctx, s, &label, &[tokens * hidden], |arm, out| {
            let mut b = s.launch_builder(&add[arm]);
            b.arg(&ffn)
                .arg(&residual)
                .arg(&valid_tokens)
                .arg(&hidden_arg)
                .arg(&mut out[0]);
            // SAFETY: every buffer covers the grid's tokens.
            unsafe { b.launch(cfg((tokens as u32, gy), (256, 1), 0)) }.unwrap();
        });
    }

    let concat = pair(crate::block::mtp::MTP_GLUE_SRC, "mtp_concat_eh");
    for (tokens, hidden) in [(1usize, 2048usize), (3, 2048), (2048, 2048), (5, 300)] {
        let e = s.clone_htod(&rng.vec(tokens * hidden, -4.0, 4.0)).unwrap();
        let h = s.clone_htod(&rng.vec(tokens * hidden, -4.0, 4.0)).unwrap();
        let (hidden_arg, tokens_arg) = (hidden as i32, tokens as i32);
        let label = format!("mtp_concat_eh {tokens}x{hidden}");
        check(&ctx, s, &label, &[tokens * 2 * hidden], |arm, out| {
            let mut b = s.launch_builder(&concat[arm]);
            b.arg(&e)
                .arg(&h)
                .arg(&mut out[0])
                .arg(&hidden_arg)
                .arg(&tokens_arg);
            // SAFETY: out is twice the width of either input.
            unsafe { b.launch(cfg((tokens as u32, 1), (256, 1), 0)) }.unwrap();
        });
    }

    let tap = pair(crate::dflash::DFLASH_GLUE_SRC, "dflash_tap");
    for (rows, hidden, taps, which) in [
        (1usize, 2048usize, 4i32, 0i32),
        (3, 2048, 4, 3),
        (2048, 2048, 4, 2),
        (5, 300, 3, 1),
    ] {
        let src = s.clone_htod(&rng.vec(rows * hidden, -4.0, 4.0)).unwrap();
        let (hidden_arg, rows_arg) = (hidden as i32, rows as i32);
        let label = format!("dflash_tap {rows}x{hidden} tap {which}/{taps}");
        check(
            &ctx,
            s,
            &label,
            &[rows * taps as usize * hidden],
            |arm, out| {
                let mut b = s.launch_builder(&tap[arm]);
                b.arg(&src)
                    .arg(&mut out[0])
                    .arg(&hidden_arg)
                    .arg(&taps)
                    .arg(&which)
                    .arg(&rows_arg);
                // SAFETY: aux covers `taps` rows of hidden per source row.
                unsafe { b.launch(cfg((rows as u32, 1), (256, 1), 0)) }.unwrap();
            },
        );
    }
}
