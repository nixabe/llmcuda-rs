//! Rust ports against the CUDA C++ kernels they are about to replace, bit
//! for bit and timed. Lives only while both sources exist.

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
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
    fn ints(&mut self, n: usize, below: i32) -> Vec<i32> {
        (0..n)
            .map(|_| ((self.next() >> 33) % below as u64) as i32)
            .collect()
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.next() >> 56) as u8).collect()
    }
}

fn blank(words: usize) -> Vec<u32> {
    vec![SENTINEL; words]
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
fn the_embedding_prelude_and_dense_norm_ports_match_cuda_cpp() {
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
    let pair = |src: &str, cpp_name: &str, rust_name: &str| -> [CudaFunction; 2] {
        let cpp = ctx
            .load_module(compile(src, "rust_parity").expect("C++ compiles"))
            .expect("C++ loads");
        [
            cpp.load_function(cpp_name).expect("C++ entry"),
            rust.load_function(rust_name).expect("Rust entry"),
        ]
    };
    let mut rng = Rng(0x0016_C0DE);

    // Three identical C++ copies, one Rust kernel.
    for (src, cpp_name) in [
        (crate::forward::EMBED_SRC, "fwd_embed_q8_0"),
        (crate::block::mtp::MTP_GLUE_SRC, "mtp_embed_q8_0"),
        (crate::dflash::DFLASH_GLUE_SRC, "dflash_embed_q8_0"),
    ] {
        let embed = pair(src, cpp_name, "embed_q8_0");
        for (tokens, hidden, vocab) in [
            (1usize, 2048usize, 1000usize),
            (3, 2048, 1000),
            (2048, 2048, 1000),
            (5, 96, 40),
        ] {
            let table = s.clone_htod(&rng.bytes(vocab * hidden / 32 * 34)).unwrap();
            let ids = s.clone_htod(&rng.ints(tokens, vocab as i32)).unwrap();
            let (hidden_arg, tokens_arg) = (hidden as i32, tokens as i32);
            let label = format!("{cpp_name} {tokens}x{hidden}");
            check(&ctx, s, &label, &[blank(tokens * hidden)], |arm, out| {
                let mut b = s.launch_builder(&embed[arm]);
                b.arg(&table)
                    .arg(&ids)
                    .arg(&mut out[0])
                    .arg(&hidden_arg)
                    .arg(&tokens_arg);
                // SAFETY: every id names a row of the table.
                unsafe { b.launch(cfg((tokens as u32, 1), (256, 1), 0)) }.unwrap();
            });
        }
    }

    let gdn = crate::block::gdn::GDN_BLOCK_SRC;
    let split = pair(gdn, "gdn_silu_split_qkv", "gdn_silu_split_qkv");
    let step = pair(
        gdn,
        "gdn_conv_silu_split_step_batch",
        "gdn_conv_silu_split_step_batch",
    );
    for (tokens, head_dim, qk_heads, value_heads) in [
        (1usize, 128usize, 16usize, 32usize),
        (3, 128, 16, 32),
        (8, 128, 16, 32),
        (2048, 128, 16, 32),
        (5, 64, 4, 4),
    ] {
        let conv_dim = (2 * qk_heads + value_heads) * head_dim;
        let conv = s
            .clone_htod(&spiked(&mut rng, tokens * conv_dim, -8.0, 8.0))
            .unwrap();
        let (hd, qh, vh) = (head_dim as i32, qk_heads as i32, value_heads as i32);
        let narrow = tokens * qk_heads * head_dim;
        let wide = tokens * value_heads * head_dim;
        let grid = (value_heads as u32, tokens as u32);
        let label = format!("gdn_silu_split_qkv {tokens}x{head_dim}");
        let lens = [
            blank(tokens * conv_dim),
            blank(narrow),
            blank(narrow),
            blank(wide),
        ];
        check(&ctx, s, &label, &lens, |arm, out| {
            let [silu, q, k, v] = out else { unreachable!() };
            let mut b = s.launch_builder(&split[arm]);
            b.arg(&conv)
                .arg(silu)
                .arg(q)
                .arg(k)
                .arg(v)
                .arg(&hd)
                .arg(&qh)
                .arg(&vh);
            // SAFETY: one block per value head and token.
            unsafe { b.launch(cfg(grid, (head_dim as u32, 1), 0)) }.unwrap();
        });

        if tokens > 8 {
            continue;
        }
        for conv_kernel in [1i32, 2, 4, 8] {
            let carry = (conv_kernel - 1) as usize;
            let weight = s
                .clone_htod(&rng.vec(conv_dim * conv_kernel as usize, -1.0, 1.0))
                .unwrap();
            let state: Vec<u32> = rng
                .vec((tokens * conv_dim * carry).max(1), -6.0, 6.0)
                .into_iter()
                .map(f32::to_bits)
                .collect();
            let label =
                format!("gdn_conv_silu_split_step_batch {tokens}x{head_dim} k={conv_kernel}");
            let lens = [
                state,
                blank(tokens * conv_dim),
                blank(tokens * conv_dim),
                blank(narrow),
                blank(narrow),
                blank(wide),
            ];
            check(&ctx, s, &label, &lens, |arm, out| {
                let [state, raw, silu, q, k, v] = out else {
                    unreachable!()
                };
                let base = state.device_ptr(s).0;
                let ptr = |t: usize| base + (t.min(tokens - 1) * conv_dim * carry * 4) as u64;
                let p: [u64; 8] = std::array::from_fn(ptr);
                let mut b = s.launch_builder(&step[arm]);
                b.arg(&conv).arg(&weight);
                for slot in &p {
                    b.arg(slot);
                }
                b.arg(raw)
                    .arg(silu)
                    .arg(q)
                    .arg(k)
                    .arg(v)
                    .arg(&hd)
                    .arg(&qh)
                    .arg(&vh)
                    .arg(&conv_kernel);
                // SAFETY: each sequence's cache is its own slice of `state`;
                // the unused pointers are never selected.
                unsafe { b.launch(cfg(grid, (head_dim as u32, 1), 0)) }.unwrap();
            });
        }
    }

    let norm = pair(
        crate::block::dense_ffn::GLUE_SRC,
        "dense_rms_norm_narrow",
        "dense_rms_norm_narrow",
    );
    for (rows, width) in [(1usize, 5120usize), (4, 5120), (2048, 5120), (5, 300)] {
        let x = s
            .clone_htod(&spiked(&mut rng, rows * width, -300.0, 300.0))
            .unwrap();
        let weight = s.clone_htod(&rng.vec(width, 0.2, 2.0)).unwrap();
        let (width_arg, eps) = (width as i32, 1e-6f32);
        let label = format!("dense_rms_norm_narrow {rows}x{width}");
        check(
            &ctx,
            s,
            &label,
            &[blank(rows * width), blank(rows * width / 2)],
            |arm, out| {
                let [wide, half] = out else { unreachable!() };
                let mut b = s.launch_builder(&norm[arm]);
                b.arg(&x)
                    .arg(&weight)
                    .arg(wide)
                    .arg(half)
                    .arg(&width_arg)
                    .arg(&eps);
                // SAFETY: one block per row, a float of shared memory per warp.
                unsafe { b.launch(cfg((rows as u32, 1), (256, 1), 128)) }.unwrap();
            },
        );
    }
}
