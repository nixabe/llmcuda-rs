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
fn the_attention_tail_ports_match_cuda_cpp() {
    let Some(g) = gpu() else { return };
    let s = &g.stream;
    let mut rng = Rng(0x00A7_7E11);
    let src = super::attention::ATTENTION_SRC;
    let imrope = g.pair(src, "attn_rope_partial_imrope");
    let kv8 = g.pair(src, "attn_kv_append_kv8");
    let batch = g.pair(src, "attn_decode_rope_append_batch");

    for (tokens, heads, head_dim, rope_dim, hb, wb) in [
        (3usize, 16usize, 256usize, 64i32, 22i32, 44i32),
        (512, 16, 256, 64, 22, 44),
        (512, 2, 256, 64, 22, 44),
        (37, 4, 128, 128, 40, 80),
        (5, 2, 128, 0, 0, 0),
    ] {
        let n = tokens * heads * head_dim;
        let x = s.clone_htod(&rng.vec(n, -4.0, 4.0)).unwrap();
        let pos = s.clone_htod(&rng.ints(3 * tokens, 70_000)).unwrap();
        let (h, d, theta, nt) = (heads as i32, head_dim as i32, 1.0e7f32, tokens as i32);
        let label = format!("attn_rope_partial_imrope {tokens}x{heads}x{head_dim} rope={rope_dim}");
        check(&g.ctx, s, &label, &[blank(n)], |arm, out| {
            let mut b = s.launch_builder(&imrope[arm]);
            b.arg(&x)
                .arg(&mut out[0])
                .arg(&h)
                .arg(&d)
                .arg(&rope_dim)
                .arg(&pos)
                .arg(&hb)
                .arg(&wb)
                .arg(&theta)
                .arg(&nt);
            // SAFETY: one block per (token, head), one thread per dimension.
            unsafe { b.launch(cfg((tokens as u32, heads as u32), (head_dim as u32, 1), 0)) }
                .unwrap();
        });
    }

    for (tokens, row, position, kq8, vq8) in [
        (1usize, 512usize, 5i32, 1i32, 1i32),
        (1, 512, 0, 1, 0),
        (1, 512, 3, 0, 1),
        (64, 512, 2, 1, 1),
        (2048, 512, 1, 1, 1),
        (7, 256, 4, 1, 0),
    ] {
        let span = tokens * row;
        let mut key = spiked(&mut rng, span, -3.0, 3.0);
        let mut value = spiked(&mut rng, span, -40.0, 40.0);
        // An all-zero block (scale 0) and exact .5 ties after scaling.
        for v in key.iter_mut().take(32) {
            *v = 0.0;
        }
        for (i, v) in value.iter_mut().skip(32).take(32).enumerate() {
            *v = (i as f32 - 16.0) * 0.5 + if i == 0 { 127.0 } else { 0.0 };
        }
        let key = s.clone_htod(&key).unwrap();
        let value = s.clone_htod(&value).unwrap();
        let pos = s.clone_htod(&[position]).unwrap();
        let rows = position as usize + tokens;
        // Either format fits in row u16s a position.
        let cache = blank(rows * row / 2);
        let (span_i, row_i) = (span as i32, row as i32);
        let label = format!("attn_kv_append_kv8 {tokens}x{row} k8={kq8} v8={vq8}");
        check(&g.ctx, s, &label, &[cache.clone(), cache], |arm, out| {
            let [k, v] = out else { unreachable!() };
            let mut b = s.launch_builder(&kv8[arm]);
            b.arg(&key)
                .arg(&value)
                .arg(k)
                .arg(v)
                .arg(&pos)
                .arg(&span_i)
                .arg(&row_i)
                .arg(&kq8)
                .arg(&vq8);
            // SAFETY: threads past 2 * span return; caches cover every row.
            unsafe { b.launch(cfg(((2 * span).div_ceil(256) as u32, 1), (256, 1), 0)) }.unwrap();
        });
    }

    for (n, q_heads, kv_heads, head_dim, rope_dim) in [
        (1usize, 16usize, 2usize, 256usize, 64i32),
        (3, 16, 2, 256, 64),
        (8, 16, 2, 256, 64),
        (3, 8, 4, 128, 128),
        (2, 4, 2, 128, 0),
    ] {
        let (qn, kn) = (n * q_heads * head_dim, n * kv_heads * head_dim);
        let q_in = s.clone_htod(&spiked(&mut rng, qn, -4.0, 4.0)).unwrap();
        let k_in = s.clone_htod(&spiked(&mut rng, kn, -4.0, 4.0)).unwrap();
        let v_in = s.clone_htod(&spiked(&mut rng, kn, -4.0, 4.0)).unwrap();
        let positions: Vec<i32> = (0..n).map(|i| 1 + 2 * i as i32).collect();
        let rope: Vec<CudaSlice<i32>> = (0..n)
            .map(|i| s.clone_htod(&[100_000 - 7919 * i as i32]).unwrap())
            .collect();
        let app: Vec<CudaSlice<i32>> = positions
            .iter()
            .map(|&p| s.clone_htod(&[p]).unwrap())
            .collect();
        let row = kv_heads * head_dim;
        let mut lens = vec![blank(qn), blank(kn)];
        for &p in &positions {
            lens.push(blank((p as usize + 1) * row / 2));
            lens.push(blank((p as usize + 1) * row / 2));
        }
        let (qh, kh, hd, theta) = (q_heads as i32, kv_heads as i32, head_dim as i32, 1.0e7f32);
        let label = format!(
            "attn_decode_rope_append_batch n={n} {q_heads}+{kv_heads}x{head_dim} rope={rope_dim}"
        );
        check(&g.ctx, s, &label, &lens, |arm, out| {
            let ptr = |d: &CudaSlice<u32>| d.device_ptr(s).0;
            let slots = |f: &dyn Fn(usize) -> u64| -> [u64; 8] {
                core::array::from_fn(|i| f(i.min(n - 1)))
            };
            let rp = slots(&|i| rope[i].device_ptr(s).0);
            let ap = slots(&|i| app[i].device_ptr(s).0);
            let kc = slots(&|i| ptr(&out[2 + 2 * i]));
            let vc = slots(&|i| ptr(&out[3 + 2 * i]));
            let (q_out, rest) = out.split_at_mut(1);
            let mut b = s.launch_builder(&batch[arm]);
            b.arg(&q_in)
                .arg(&mut q_out[0])
                .arg(&k_in)
                .arg(&mut rest[0])
                .arg(&v_in);
            for slot in rp.iter().chain(&ap).chain(&kc).chain(&vc) {
                b.arg(slot);
            }
            b.arg(&qh).arg(&kh).arg(&hd).arg(&rope_dim).arg(&theta);
            // SAFETY: grid.y = n bounds the slots to live pointers.
            unsafe {
                b.launch(cfg(
                    ((q_heads + kv_heads) as u32, n as u32),
                    (head_dim as u32, 1),
                    0,
                ))
            }
            .unwrap();
        });
    }
}
