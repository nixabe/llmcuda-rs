//! The Rust ports of `gdn_silu`, `gdn_split_qkv` and `gdn_gates` against the
//! CUDA C++ kernels they are about to replace, bit for bit and timed.
//!
//! Lives only while both sources exist.

use super::GDN_BLOCK_SRC;
use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use llmcuda_cuda::device::driver_available;
use llmcuda_cuda::kernels::{RUST_KERNELS_PTX, compile};
use llmcuda_kernels::rng::Xorshift64Star;
use std::sync::Arc;

fn spiked(rng: &mut Xorshift64Star, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut v = rng.vec_f32(n, lo, hi);
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
fn the_silu_split_and_gate_ports_match_cuda_cpp() {
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
    let cpp = ctx
        .load_module(compile(GDN_BLOCK_SRC, "gdn_block").expect("C++ compiles"))
        .expect("C++ loads");
    let rust = ctx
        .load_module(cudarc::nvrtc::Ptx::from_src(RUST_KERNELS_PTX))
        .expect("Rust PTX loads");
    let pair = |name: &str| -> [CudaFunction; 2] {
        [
            cpp.load_function(name).expect("C++ entry"),
            rust.load_function(name).expect("Rust entry"),
        ]
    };
    let cfg = |grid: (u32, u32), block: u32| LaunchConfig {
        grid_dim: (grid.0, grid.1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut rng = Xorshift64Star::new(0x0009_C0DE);

    let silu = pair("gdn_silu");
    for (n, blocks) in [(1000usize, 4u32), (3 * 8192, 96), (2048 * 8192, 1024)] {
        let x = s.clone_htod(&spiked(&mut rng, n, -8.0, 8.0)).unwrap();
        let n_arg = n as i64;
        check(
            &ctx,
            s,
            &format!("gdn_silu n={n}"),
            &silu,
            &[n + 3],
            |f, out| {
                let mut b = s.launch_builder(f);
                b.arg(&x).arg(&mut out[0]).arg(&n_arg);
                // SAFETY: both buffers cover n floats; the loop is grid-stride.
                unsafe { b.launch(cfg((blocks, 1), 256)) }.unwrap();
            },
        );
    }

    let split = pair("gdn_split_qkv");
    for (tokens, head_dim, qk_heads, value_heads) in [
        (1usize, 128usize, 16usize, 32usize),
        (3, 128, 16, 32),
        (2048, 128, 16, 32),
        (5, 64, 4, 4),
    ] {
        let conv_dim = (2 * qk_heads + value_heads) * head_dim;
        let conv = s
            .clone_htod(&rng.vec_f32(tokens * conv_dim, -4.0, 4.0))
            .unwrap();
        let (hd, qh, vh) = (head_dim as i32, qk_heads as i32, value_heads as i32);
        let narrow = tokens * qk_heads * head_dim;
        let lens = [narrow, narrow, tokens * value_heads * head_dim];
        let label = format!("gdn_split_qkv {tokens}x{head_dim}");
        check(&ctx, s, &label, &split, &lens, |f, out| {
            let [q, k, v] = out else { unreachable!() };
            let mut b = s.launch_builder(f);
            b.arg(&conv).arg(q).arg(k).arg(v).arg(&hd).arg(&qh).arg(&vh);
            // SAFETY: one block per value head and token, one thread per
            // head dimension; the buffers are sized for that grid.
            unsafe { b.launch(cfg((value_heads as u32, tokens as u32), head_dim as u32)) }.unwrap();
        });
    }

    let gates = pair("gdn_gates");
    for (tokens, heads, blocks) in [
        (1usize, 32usize, 1u32),
        (3, 32, 1),
        (2048, 32, 64),
        (7, 5, 1),
    ] {
        let n = tokens * heads;
        let alpha = s.clone_htod(&spiked(&mut rng, n, -30.0, 30.0)).unwrap();
        let beta_raw = s.clone_htod(&spiked(&mut rng, n, -30.0, 30.0)).unwrap();
        let dt_bias = s.clone_htod(&rng.vec_f32(heads, -2.0, 2.0)).unwrap();
        let ssm_a = s.clone_htod(&rng.vec_f32(heads, -3.0, -0.01)).unwrap();
        let (heads_arg, n_arg) = (heads as i32, n as i64);
        let label = format!("gdn_gates {tokens}x{heads}");
        check(&ctx, s, &label, &gates, &[n + 3; 3], |f, out| {
            let [a_softplus, log_decay, beta] = out else {
                unreachable!()
            };
            let mut b = s.launch_builder(f);
            b.arg(&alpha)
                .arg(&beta_raw)
                .arg(&dt_bias)
                .arg(&ssm_a)
                .arg(a_softplus)
                .arg(log_decay)
                .arg(beta)
                .arg(&heads_arg)
                .arg(&n_arg);
            // SAFETY: per-element buffers cover n floats, per-head ones
            // cover heads; the loop is grid-stride.
            unsafe { b.launch(cfg((blocks, 1), 256)) }.unwrap();
        });
    }
}
