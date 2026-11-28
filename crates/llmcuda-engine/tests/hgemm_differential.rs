//! The fp16 weight-only GEMM against a scalar oracle that rounds exactly where
//! the kernel says it rounds.
//!
//! The kernel's contract is `sum_k half(x) * half(w)` in fp32: activations
//! rounded to fp16 once, each weight converted to the correctly rounded fp16
//! of its stored value (`q * d` for Q8_0; the bf16 value at a power-of-two
//! scale for bf16), and nothing rounded inside the contraction except the
//! fp32 accumulator itself. The oracle computes exactly those operands and
//! sums their products in f64, so the only freedom left to the device is
//! the order (and the tensor core's internal width) of the fp32 sum — which
//! is what the tolerance below has to absorb, and all it has to absorb.
//!
//! Shapes cover one row and one token, a tile and a tile plus a tail on both
//! axes, a contraction of one stage and of many, an output stride wider than
//! the row count, and accumulation into an existing residual — each through
//! both the 128-row and the 256-row tile, unsplit and split-K two to four ways
//! (a split is a different place to break the fp32 sum, so it gets the same
//! bound and its own coverage).
//!
//! SKIPS — reporting that it skipped — without a driver or a supported device.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};
use llmcuda_cuda::device::{DeviceInfo, driver_available};
use llmcuda_cuda::kernels::hgemm::{
    HalfWeight, HgemmKernels, HgemmPlan, bf16_half_exponent, nv4_tile_major, nv4_tiles_fit,
};
use llmcuda_cuda::kernels::mma::MmaKernels;
use llmcuda_kernels::f16::round_through_f16;
use llmcuda_kernels::quant::quantize_q8_0;
use llmcuda_kernels::rng::Xorshift64Star;

fn device() -> Option<Arc<CudaContext>> {
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
    let info = DeviceInfo::from_context(0, &ctx).expect("device properties readable");
    if !info.is_supported() {
        println!("SKIPPED: device 0 is below the sm_75 minimum");
        return None;
    }
    Some(ctx)
}

/// Q8_0 bytes for `rows x k` weights, plus the dequantized fp16-rounded copy
/// the oracle multiplies with.
fn q8_0_weights(rows: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let mut bytes = Vec::with_capacity(rows * k / 32 * 34);
    let mut deq = Vec::with_capacity(rows * k);
    for r in 0..rows {
        for _ in 0..k / 32 {
            // Vary block magnitude by row so scales span several binades.
            let scale = 0.01 + (r % 7) as f32 * 0.03;
            let mut block = [0.0f32; 32];
            for v in &mut block {
                *v = rng.next_f32_range(-scale, scale);
            }
            let q = quantize_q8_0(&block);
            bytes.extend_from_slice(&q.d.to_bits().to_le_bytes());
            bytes.extend(q.qs.iter().map(|&v| v as u8));
            let d = q.d.to_f32();
            deq.extend(q.qs.iter().map(|&v| round_through_f16(f32::from(v) * d)));
        }
    }
    (bytes, deq)
}

/// bf16 bytes for `rows x k` weights spanning many binades (the kind of range
/// a real projection has, tiny values included), plus the exact values.
fn bf16_weights(rows: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let mut bytes = Vec::with_capacity(rows * k * 2);
    let mut vals = Vec::with_capacity(rows * k);
    for i in 0..rows * k {
        let mag = 2f32.powi(-((i % 23) as i32)) * 0.4;
        let v = rng.next_f32_range(-mag, mag);
        let bits = (v.to_bits() >> 16) as u16;
        bytes.extend_from_slice(&bits.to_le_bytes());
        vals.push(f32::from_bits(u32::from(bits) << 16));
    }
    (bytes, vals)
}

/// NVFP4 bytes for `rows x k` weights and the values they store. Scale bytes
/// cover the subnormal exponent, the two encodings of zero and a set sign
/// bit; nibbles cover all sixteen codes.
fn nvfp4_weights(rows: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let mut bytes = Vec::with_capacity(rows * k / 64 * 36);
    for block in 0..rows * k / 64 {
        for sub in 0..4 {
            let pick = (rng.next_f32_range(0.0, 1.0) * 97.0) as usize;
            let byte = match (block + sub) % 29 {
                0 => 0u8,
                1 => 0x7F,
                2 => 0x80 | ((pick % 80) as u8 + 8),
                _ => (pick % 88) as u8,
            };
            bytes.push(byte);
        }
        for _ in 0..32 {
            bytes.push((rng.next_f32_range(0.0, 1.0) * 255.99) as u8);
        }
    }
    let vals = llmcuda_kernels::quant::dequantize_row_nvfp4(&bytes).unwrap();
    (bytes, vals)
}

/// The per-matrix factor the NVFP4 cases pass, as an expert's scale would be.
const NVFP4_SCALE: f32 = 0.371;

enum Format {
    Q8_0,
    Bf16,
    Nvfp4,
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    stream: &Arc<CudaStream>,
    mma: &MmaKernels,
    hg: &HgemmKernels,
    format: &Format,
    plan: HgemmPlan,
    tokens: usize,
    rows: usize,
    k: usize,
    ldo: usize,
    accumulate: bool,
    seed: u64,
) {
    // An NVFP4 row is whole 64-element blocks.
    let k = if matches!(format, Format::Nvfp4) {
        k.next_multiple_of(64)
    } else {
        k
    };
    let mut rng = Xorshift64Star::new(seed ^ 0x5eed);
    let x = rng.vec_f32(tokens * k, -3.0, 3.0);
    let prior: Vec<f32> = rng.vec_f32(tokens * ldo, -1.0, 1.0);
    let xd = stream.clone_htod(&x).unwrap();
    let mut xh = stream.alloc_zeros::<u16>(tokens * k).unwrap();
    hg.to_half(stream, &xd, &mut xh, tokens * k).unwrap();
    let mut out = stream.clone_htod(&prior).unwrap();
    let mut workspace = stream
        .alloc_zeros::<f32>((plan.splits * tokens * rows).max(1))
        .unwrap();

    let deq = match format {
        Format::Q8_0 => {
            let (bytes, deq) = q8_0_weights(rows, k, seed);
            let src = stream.clone_htod(&bytes).unwrap();
            let mut wq = stream.alloc_zeros::<i8>(rows * k).unwrap();
            let mut ws = stream.alloc_zeros::<u16>(rows * k / 32).unwrap();
            mma.repack_q8_0_half(stream, &src, &mut wq, &mut ws, rows * k)
                .unwrap();
            let w = HalfWeight::Q8_0 {
                q: &wq,
                scales: &ws,
            };
            hg.gemm_with_plan(
                stream,
                w,
                &xh,
                &mut out,
                Some(&mut workspace),
                tokens,
                rows,
                k,
                ldo,
                accumulate,
                plan,
            )
            .unwrap();
            deq
        }
        Format::Bf16 => {
            let (bytes, vals) = bf16_weights(rows, k, seed);
            let exponent = bf16_half_exponent(&bytes);
            // The oracle's weights are the fp16 of each value at that scale,
            // mapped back — which is the bf16 value itself wherever the
            // scaled value is an fp16 normal.
            let scale = 2f32.powi(exponent);
            let deq: Vec<f32> = vals
                .iter()
                .map(|&v| round_through_f16(v * scale) / scale)
                .collect();
            let dev = stream.clone_htod(&bytes).unwrap();
            let w = HalfWeight::Bf16 {
                bytes: &dev,
                exponent,
            };
            hg.gemm_with_plan(
                stream,
                w,
                &xh,
                &mut out,
                Some(&mut workspace),
                tokens,
                rows,
                k,
                ldo,
                accumulate,
                plan,
            )
            .unwrap();
            deq
        }
        Format::Nvfp4 => {
            // The staged weights are the stored values exactly; the factor
            // multiplies the fp32 sum.
            let (bytes, vals) = nvfp4_weights(rows, k, seed);
            assert!(vals.iter().all(|v| round_through_f16(*v) == *v));
            let dev = stream.clone_htod(&bytes).unwrap();
            let w = HalfWeight::Nvfp4 {
                bytes: &dev,
                scale: NVFP4_SCALE,
                partner: 0,
            };
            hg.gemm_with_plan(
                stream,
                w,
                &xh,
                &mut out,
                Some(&mut workspace),
                tokens,
                rows,
                k,
                ldo,
                accumulate,
                plan,
            )
            .unwrap();
            vals.iter().map(|v| v * NVFP4_SCALE).collect()
        }
    };
    let got = stream.clone_dtoh(&out).unwrap();

    let mut want = prior.clone();
    let mut magnitude = 0.0f64;
    for t in 0..tokens {
        for r in 0..rows {
            let mut s = 0.0f64;
            let mut m = 0.0f64;
            for i in 0..k {
                let p = f64::from(round_through_f16(x[t * k + i])) * f64::from(deq[r * k + i]);
                s += p;
                m += p.abs();
            }
            magnitude = magnitude.max(m);
            let base = if accumulate {
                f64::from(prior[t * ldo + r])
            } else {
                0.0
            };
            want[t * ldo + r] = (base + s) as f32;
        }
    }
    // Columns past `rows` in a wide stride must be untouched.
    for t in 0..tokens {
        for c in rows..ldo {
            assert_eq!(
                got[t * ldo + c].to_bits(),
                prior[t * ldo + c].to_bits(),
                "column {c} of token {t} is outside the output and was written"
            );
        }
    }
    let mut worst = 0.0f64;
    for (g, w) in got.iter().zip(&want) {
        assert!(g.is_finite());
        worst = worst.max((f64::from(*g) - f64::from(*w)).abs());
    }
    // fp32 accumulation of `k` products whose absolute sum is `magnitude`:
    // each addition errs by at most half an ulp of the running sum, so the
    // total is bounded by ~k * 2^-24 * magnitude. Allow 2x that bound
    // (tensor-core adds may truncate rather than round) plus the final
    // rounding to fp32; a wrong fragment or a wrong dequant is orders of
    // magnitude above it.
    let bound = 2.0 * (k as f64) * magnitude * 2f64.powi(-24) + 1e-6;
    let name = match format {
        Format::Q8_0 => "q8_0",
        Format::Bf16 => "bf16",
        Format::Nvfp4 => "nvfp4",
    };
    println!(
        "{name} {plan:?} tokens {tokens:4} rows {rows:5} k {k:5} ldo {ldo:5} acc \
         {accumulate:5}: max_abs {worst:.3e} (bound {bound:.3e}, |sum| {magnitude:.2})"
    );
    assert!(worst <= bound, "max_abs {worst} exceeds {bound}");
}

#[test]
fn half_gemm_matches_its_rounding_contract() {
    let Some(ctx) = device() else { return };
    let stream = ctx.default_stream();
    let mma = MmaKernels::new(&ctx).expect("int8 kernels compile");
    let hg = HgemmKernels::new(&ctx).expect("fp16 GEMM kernels compile");
    let shapes = [
        (1, 8, 32, 8, false),
        (1, 1, 64, 2, true),
        (37, 130, 96, 130, false),
        (128, 128, 64, 128, false),
        (129, 257, 128, 260, true),
        (300, 513, 1024, 514, false),
        (513, 384, 320, 384, true),
    ];
    // Split-K needs a whole number of 8-block scale groups and an even row
    // count; these do, with uneven last splits among them (k = 1280 is 40
    // k-blocks: 3 ways is 16 + 16 + 8).
    let split_shapes = [
        (37, 130, 1280, 130, false),
        (129, 258, 1024, 260, true),
        (300, 512, 2048, 512, false),
        (513, 384, 768, 384, true),
    ];
    let mut seed = 17;
    for format in [Format::Q8_0, Format::Bf16, Format::Nvfp4] {
        for wide in [false, true] {
            let whole = HgemmPlan { wide, splits: 1 };
            for &(tokens, rows, k, ldo, accumulate) in &shapes {
                run_case(
                    &stream, &mma, &hg, &format, whole, tokens, rows, k, ldo, accumulate, seed,
                );
                seed += 1;
            }
            for splits in 2..=4 {
                let plan = HgemmPlan { wide, splits };
                for &(tokens, rows, k, ldo, accumulate) in &split_shapes {
                    run_case(
                        &stream, &mma, &hg, &format, plan, tokens, rows, k, ldo, accumulate, seed,
                    );
                    seed += 1;
                }
            }
        }
    }
}

/// The grouped launch: each 128-token tile against the matrix its owner
/// names, scaled by that owner's factor; unowned tiles untouched.
#[test]
fn grouped_gemm_contracts_each_tile_with_its_owner() {
    let Some(ctx) = device() else { return };
    let stream = ctx.default_stream();
    let mma = MmaKernels::new(&ctx).expect("int8 kernels compile");
    let hg = HgemmKernels::new(&ctx).expect("fp16 GEMM kernels compile");
    let (stack, rows, k) = (5usize, 130usize, 192usize);
    let owners: Vec<i32> = vec![3, -1, 0, 4, 3, -1, 1];
    let tiles = owners.len();
    let tokens = tiles * 128;
    let factors: Vec<f32> = (0..stack).map(|e| 0.25 + e as f32 * 0.31).collect();
    let d_owners = stream.clone_htod(&owners).unwrap();
    let d_factors = stream.clone_htod(&factors).unwrap();
    let mut rng = Xorshift64Star::new(0xface);
    let x = rng.vec_f32(tokens * k, -3.0, 3.0);
    let xd = stream.clone_htod(&x).unwrap();
    let mut xh = stream.alloc_zeros::<u16>(tokens * k).unwrap();
    hg.to_half(&stream, &xd, &mut xh, tokens * k).unwrap();
    for (name, format) in [
        ("q8_0", Format::Q8_0),
        ("bf16", Format::Bf16),
        ("nvfp4", Format::Nvfp4),
    ] {
        for wide in [false, true] {
            let prior = rng.vec_f32(tokens * rows, -1.0, 1.0);
            let mut out = stream.clone_htod(&prior).unwrap();
            // What each output is compared with: the staged weight, and the
            // factor the launch applies on top.
            let (deq, scaled) = match format {
                Format::Q8_0 => {
                    let (bytes, deq) = q8_0_weights(stack * rows, k, 91);
                    let src = stream.clone_htod(&bytes).unwrap();
                    let mut wq = stream.alloc_zeros::<i8>(stack * rows * k).unwrap();
                    let mut ws = stream.alloc_zeros::<u16>(stack * rows * k / 32).unwrap();
                    mma.repack_q8_0_half(&stream, &src, &mut wq, &mut ws, stack * rows * k)
                        .unwrap();
                    let w = HalfWeight::Q8_0 {
                        q: &wq,
                        scales: &ws,
                    };
                    hg.gemm_grouped(
                        &stream, w, stack, &d_owners, None, &xh, &mut out, rows, k, wide,
                    )
                    .unwrap();
                    (deq, false)
                }
                Format::Bf16 => {
                    let (bytes, vals) = bf16_weights(stack * rows, k, 92);
                    let exponent = bf16_half_exponent(&bytes);
                    let scale = 2f32.powi(exponent);
                    let dev = stream.clone_htod(&bytes).unwrap();
                    let w = HalfWeight::Bf16 {
                        bytes: &dev,
                        exponent,
                    };
                    hg.gemm_grouped(
                        &stream, w, stack, &d_owners, None, &xh, &mut out, rows, k, wide,
                    )
                    .unwrap();
                    let deq = vals
                        .iter()
                        .map(|&v| round_through_f16(v * scale) / scale)
                        .collect();
                    (deq, false)
                }
                Format::Nvfp4 => {
                    let (bytes, vals) = nvfp4_weights(stack * rows, k, 93);
                    let dev = stream.clone_htod(&bytes).unwrap();
                    let w = HalfWeight::Nvfp4 {
                        bytes: &dev,
                        scale: 1.0,
                        partner: 0,
                    };
                    hg.gemm_grouped(
                        &stream,
                        w,
                        stack,
                        &d_owners,
                        Some(&d_factors),
                        &xh,
                        &mut out,
                        rows,
                        k,
                        wide,
                    )
                    .unwrap();
                    // A stack of whole tiles, as stored and tile-major
                    // under both pairings the decode GEMM uses: a
                    // different address for every byte, the same bits out.
                    let rows = 288usize;
                    let (bytes, _) = nvfp4_weights(stack * rows, k, 94);
                    let per = rows * k / 64 * 36;
                    let seed = rng.vec_f32(tokens * rows, -1.0, 1.0);
                    let run = |bytes: &[u8], partner: usize| {
                        let dev = stream.clone_htod(bytes).unwrap();
                        let mut out = stream.clone_htod(&seed).unwrap();
                        hg.gemm_grouped(
                            &stream,
                            HalfWeight::Nvfp4 {
                                bytes: &dev,
                                scale: 1.0,
                                partner,
                            },
                            stack,
                            &d_owners,
                            Some(&d_factors),
                            &xh,
                            &mut out,
                            rows,
                            k,
                            wide,
                        )
                        .unwrap();
                        stream.clone_dtoh(&out).unwrap()
                    };
                    let stored = run(&bytes, 0);
                    assert!(stored.iter().zip(&seed).any(|(a, b)| a != b));
                    for partner in [8, rows / 2] {
                        assert!(nv4_tiles_fit(rows, partner));
                        let mut packed = vec![0u8; bytes.len()];
                        for (src, dst) in bytes.chunks_exact(per).zip(packed.chunks_exact_mut(per))
                        {
                            nv4_tile_major(src, dst, rows, k, partner);
                        }
                        let got = run(&packed, partner);
                        assert!(
                            got.iter()
                                .zip(&stored)
                                .all(|(a, b)| a.to_bits() == b.to_bits()),
                            "tile-major NVFP4 (partner {partner}, wide {wide}) is not the stored rows' bits"
                        );
                    }
                    (vals, true)
                }
            };
            let got = stream.clone_dtoh(&out).unwrap();
            let mut worst = 0.0f64;
            let mut bound = 0.0f64;
            for (tile, &owner) in owners.iter().enumerate() {
                for t in tile * 128..(tile + 1) * 128 {
                    for r in 0..rows {
                        let g = got[t * rows + r];
                        if owner < 0 {
                            assert_eq!(g.to_bits(), prior[t * rows + r].to_bits());
                            continue;
                        }
                        let e = owner as usize;
                        let factor = if scaled { f64::from(factors[e]) } else { 1.0 };
                        let (mut s, mut m) = (0.0f64, 0.0f64);
                        for i in 0..k {
                            let p = f64::from(round_through_f16(x[t * k + i]))
                                * f64::from(deq[(e * rows + r) * k + i]);
                            s += p;
                            m += p.abs();
                        }
                        worst = worst.max((f64::from(g) - s * factor).abs());
                        bound = bound.max(2.0 * (k as f64) * m * factor * 2f64.powi(-24) + 1e-6);
                    }
                }
            }
            println!("grouped {name} wide {wide}: max_abs {worst:.3e} (bound {bound:.3e})");
            assert!(
                worst <= bound,
                "grouped {name}: max_abs {worst} exceeds {bound}"
            );
        }
    }
}

/// Activations laid out for an NVFP4 contraction: pair `p` of each
/// 16-element group holds elements `p` and `p + 8`.
fn byte_order(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for (g, group) in x.chunks(16).enumerate() {
        for e in 0..16 {
            out[g * 16 + ((e & 7) << 1) + (e >> 3)] = group[e];
        }
    }
    out
}

fn situ(gate: f64, up: f64, beta: f64, lb: f64) -> f64 {
    beta * (gate / beta).tanh() / (1.0 + (-gate).exp()) * lb * (up / lb).tanh()
}

/// The decode-shaped GEMM (`DecodeGemm`): dense with three activation
/// planes, and the expert gate/up and down projections, in every format,
/// against the same rounding contract as the prefill GEMM.
#[test]
fn decode_gemm_matches_its_rounding_contract() {
    use llmcuda_cuda::kernels::inkling::DecodeGemm;
    let Some(ctx) = device() else { return };
    let stream = ctx.default_stream();
    let mma = MmaKernels::new(&ctx).expect("int8 kernels compile");
    let hg = HgemmKernels::new(&ctx).expect("fp16 GEMM kernels compile");
    let dg = DecodeGemm::new(&ctx).expect("decode GEMM compiles");
    let mut rng = Xorshift64Star::new(0xdec0de);
    let half = |x: &[f32]| {
        let d = stream.clone_htod(x).unwrap();
        let mut h = stream.alloc_zeros::<u16>(x.len()).unwrap();
        hg.to_half(&stream, &d, &mut h, x.len()).unwrap();
        h
    };
    // The device operands for a `stack x rows x k` weight in a format, and
    // the values the oracle multiplies with.
    struct Staged {
        q: Option<(
            cudarc::driver::CudaSlice<i8>,
            cudarc::driver::CudaSlice<u16>,
        )>,
        bytes: Option<cudarc::driver::CudaSlice<u8>>,
        exponent: i32,
        deq: Vec<f32>,
        /// NVFP4: 0 as stored, else tile-major with this partner distance.
        partner: usize,
    }
    let stage = |format: &Format, n: usize, k: usize, seed: u64| -> Staged {
        match format {
            Format::Q8_0 => {
                let (bytes, deq) = q8_0_weights(n, k, seed);
                let src = stream.clone_htod(&bytes).unwrap();
                let mut wq = stream.alloc_zeros::<i8>(n * k).unwrap();
                let mut ws = stream.alloc_zeros::<u16>(n * k / 32).unwrap();
                mma.repack_q8_0_half(&stream, &src, &mut wq, &mut ws, n * k)
                    .unwrap();
                Staged {
                    q: Some((wq, ws)),
                    bytes: None,
                    exponent: 0,
                    deq,
                    partner: 0,
                }
            }
            Format::Bf16 => {
                let (bytes, vals) = bf16_weights(n, k, seed);
                let exponent = bf16_half_exponent(&bytes);
                let scale = 2f32.powi(exponent);
                Staged {
                    q: None,
                    bytes: Some(stream.clone_htod(&bytes).unwrap()),
                    exponent,
                    partner: 0,
                    deq: vals
                        .iter()
                        .map(|&v| round_through_f16(v * scale) / scale)
                        .collect(),
                }
            }
            Format::Nvfp4 => {
                let (bytes, vals) = nvfp4_weights(n, k, seed);
                Staged {
                    q: None,
                    bytes: Some(stream.clone_htod(&bytes).unwrap()),
                    exponent: 0,
                    deq: vals,
                    partner: 0,
                }
            }
        }
    };
    fn operand<'a>(format: &Format, s: &'a Staged) -> HalfWeight<'a> {
        match format {
            Format::Q8_0 => {
                let (q, scales) = s.q.as_ref().unwrap();
                HalfWeight::Q8_0 { q, scales }
            }
            Format::Bf16 => HalfWeight::Bf16 {
                bytes: s.bytes.as_ref().unwrap(),
                exponent: s.exponent,
            },
            Format::Nvfp4 => HalfWeight::Nvfp4 {
                bytes: s.bytes.as_ref().unwrap(),
                scale: 1.0,
                partner: s.partner,
            },
        }
    }
    let dot = |x: &[f32], w: &[f32]| -> (f64, f64) {
        x.iter().zip(w).fold((0.0, 0.0), |(s, m), (a, b)| {
            let p = f64::from(round_through_f16(*a)) * f64::from(*b);
            (s + p, m + p.abs())
        })
    };

    // Dense: weight rows [0, 32), [32, 64), [64, 96) read planes 0, 1, 2.
    let (rows, k, plane_rows, ldo) = (96usize, 128usize, 8usize, 100usize);
    for (name, format) in [("q8_0", Format::Q8_0), ("bf16", Format::Bf16)] {
        let w = stage(&format, rows, k, 7);
        for tokens in [1usize, 3, 8] {
            let x = rng.vec_f32(3 * plane_rows * k, -3.0, 3.0);
            let xh = half(&x);
            let prior = rng.vec_f32(tokens * ldo, -1.0, 1.0);
            let mut out = stream.clone_htod(&prior).unwrap();
            dg.dense(
                &stream,
                operand(&format, &w),
                &xh,
                &mut out,
                rows,
                k,
                tokens,
                ldo,
                plane_rows,
                32,
                64,
            )
            .unwrap();
            let got = stream.clone_dtoh(&out).unwrap();
            let (mut worst, mut bound) = (0.0f64, 0.0f64);
            for t in 0..tokens {
                for r in 0..ldo {
                    if r >= rows {
                        assert_eq!(got[t * ldo + r].to_bits(), prior[t * ldo + r].to_bits());
                        continue;
                    }
                    let plane = r / 32;
                    let xr = &x[(plane * plane_rows + t) * k..][..k];
                    let (s, m) = dot(xr, &w.deq[r * k..][..k]);
                    worst = worst.max((f64::from(got[t * ldo + r]) - s).abs());
                    bound = bound.max(2.0 * k as f64 * m * 2f64.powi(-24) + 1e-6);
                }
            }
            println!(
                "decode dense {name} tokens {tokens}: max_abs {worst:.3e} (bound {bound:.3e})"
            );
            assert!(
                worst <= bound,
                "decode dense {name}: {worst} exceeds {bound}"
            );
        }
    }

    // Experts: three tokens routed to two experts each, out of five.
    let (stack, n, ff, topk, tokens) = (5usize, 128usize, 64usize, 2usize, 3usize);
    let pairs = tokens * topk;
    let ids: Vec<i32> = vec![4, 0, 2, 2, 1, 4];
    let factors: Vec<f32> = (0..stack).map(|e| 0.2 + e as f32 * 0.17).collect();
    let pair_w: Vec<f32> = (0..pairs).map(|p| 0.5 + p as f32 * 0.4).collect();
    let (d_ids, d_factors, d_pair_w) = (
        stream.clone_htod(&ids).unwrap(),
        stream.clone_htod(&factors).unwrap(),
        stream.clone_htod(&pair_w).unwrap(),
    );
    let (beta, lb) = (4.0f32, 25.0f32);
    for (name, format) in [
        ("q8_0", Format::Q8_0),
        ("bf16", Format::Bf16),
        ("nvfp4", Format::Nvfp4),
    ] {
        let nv = matches!(format, Format::Nvfp4);
        let mut gate_up = stage(&format, stack * 2 * ff, n, 11);
        let mut down = stage(&format, stack * n, ff, 13);
        // The decode GEMM reads NVFP4 experts tile-major, each projection
        // under its own pairing.
        if nv {
            for (staged, rows, k, partner) in [(&mut gate_up, 2 * ff, n, ff), (&mut down, n, ff, 8)]
            {
                let stored = stream.clone_dtoh(staged.bytes.as_ref().unwrap()).unwrap();
                let per = rows * k / 64 * 36;
                let mut packed = vec![0u8; stored.len()];
                for (src, dst) in stored.chunks_exact(per).zip(packed.chunks_exact_mut(per)) {
                    nv4_tile_major(src, dst, rows, k, partner);
                }
                staged.bytes = Some(stream.clone_htod(&packed).unwrap());
                staged.partner = partner;
            }
        }
        let scale_of = |e: usize| if nv { f64::from(factors[e]) } else { 1.0 };
        // Small activations keep the gates off SiTU's flat ends.
        let x = rng.vec_f32(tokens * n, -0.3, 0.3);
        let xh = half(&if nv { byte_order(&x) } else { x.clone() });
        let mut hidden = stream.alloc_zeros::<u16>(pairs * ff).unwrap();
        dg.gate_up(
            &stream,
            operand(&format, &gate_up),
            stack,
            &d_ids,
            nv.then_some(&d_factors),
            &xh,
            &mut hidden,
            2 * ff,
            n,
            pairs,
            topk,
            nv,
            beta,
            lb,
        )
        .unwrap();
        let got: Vec<f32> = stream
            .clone_dtoh(&hidden)
            .unwrap()
            .iter()
            .map(|h| half::f16::from_bits(*h).to_f32())
            .collect();
        let mut want_hidden = vec![0.0f32; pairs * ff];
        let mut worst = 0.0f64;
        for (pair, &e) in ids.iter().enumerate() {
            let e = e as usize;
            let xr = &x[(pair / topk) * n..][..n];
            for i in 0..ff {
                let (g, _) = dot(xr, &gate_up.deq[(e * 2 * ff + i) * n..][..n]);
                let (u, _) = dot(xr, &gate_up.deq[(e * 2 * ff + ff + i) * n..][..n]);
                let v = situ(
                    g * scale_of(e),
                    u * scale_of(e),
                    f64::from(beta),
                    f64::from(lb),
                );
                want_hidden[pair * ff + i] = v as f32;
                let at = if nv {
                    (i & !15) + ((i & 7) << 1) + ((i & 15) >> 3)
                } else {
                    i
                };
                let err = (f64::from(got[pair * ff + at]) - v).abs() / v.abs().max(1e-2);
                worst = worst.max(err);
            }
        }
        println!("decode gate/up {name}: worst relative error {worst:.3e}");
        // binary16 output: half an ulp is 2^-11 relative, plus the fp32
        // sums' own error through SiTU's slope.
        assert!(worst < 2e-3, "decode gate/up {name}: {worst}");

        // Down, from the oracle's hidden values so the two are separable.
        let hh = half(&if nv {
            byte_order(&want_hidden)
        } else {
            want_hidden.clone()
        });
        let mut out = stream.alloc_zeros::<f32>(pairs * n).unwrap();
        dg.down(
            &stream,
            operand(&format, &down),
            stack,
            &d_ids,
            nv.then_some(&d_factors),
            &hh,
            &d_pair_w,
            &mut out,
            n,
            ff,
            pairs,
        )
        .unwrap();
        let got = stream.clone_dtoh(&out).unwrap();
        let (mut worst, mut bound) = (0.0f64, 0.0f64);
        for (pair, &e) in ids.iter().enumerate() {
            let e = e as usize;
            let f = scale_of(e) * f64::from(pair_w[pair]);
            for r in 0..n {
                let (s, m) = dot(
                    &want_hidden[pair * ff..][..ff],
                    &down.deq[(e * n + r) * ff..][..ff],
                );
                worst = worst.max((f64::from(got[pair * n + r]) - s * f).abs());
                bound = bound.max(2.0 * ff as f64 * m * f * 2f64.powi(-24) + 1e-6);
            }
        }
        println!("decode down {name}: max_abs {worst:.3e} (bound {bound:.3e})");
        assert!(
            worst <= bound,
            "decode down {name}: {worst} exceeds {bound}"
        );
    }
}

#[test]
fn the_plan_splits_only_where_the_grid_underfills_the_card() {
    let Some(ctx) = device() else { return };
    let hg = HgemmKernels::new(&ctx).expect("fp16 GEMM kernels compile");
    // The 27B at a 512-token chunk on a 72-SM card: the 17,408-row FFN gate
    // fills the card unsplit; the 5,120-row down projection does not.
    let gate = hg.plan(512, 17_408, 5_120);
    let down = hg.plan(512, 5_120, 17_408);
    println!("gate {gate:?} down {down:?}");
    assert_eq!(gate.splits, 1);
    assert!(down.splits > 1);
    // A workspace sized by `workspace_elems` covers every width below it.
    let need = hg.workspace_elems(512, 5_120, 17_408);
    for t in 1..=512 {
        let p = hg.plan(t, 5_120, 17_408);
        assert!(
            p.splits == 1 || p.splits * t * 5_120 <= need,
            "t {t}: {p:?}"
        );
    }
}

#[test]
fn the_bf16_exponent_keeps_the_max_under_fp16s_top_binade() {
    let enc = |v: f32| ((v.to_bits() >> 16) as u16).to_le_bytes();
    let bytes: Vec<u8> = [0.3203f32, -0.01, 1e-9]
        .iter()
        .flat_map(|&v| enc(v))
        .collect();
    // 0.32 is in [2^-2, 2^-1): scaled by 2^16 it lands in [2^14, 2^15).
    assert_eq!(bf16_half_exponent(&bytes), 16);
    assert_eq!(bf16_half_exponent(&[0, 0, 0, 0]), 0);
}

/// The opt-in cuBLASLt path ([`HgemmKernels::enable_library`]) under the
/// same contract as the engine's own kernel: the same fp16 operands, an fp32
/// accumulation, an fp32 result.
///
/// Whole launches and launches cut into token chunks, at shapes ragged
/// against every tile, and two weights through one object so the copy made
/// for the first is not served for the second. An overwrite at the natural
/// stride is the only form the library path takes; the accumulate and
/// wide-stride forms fall back to `hg_q8_*`, and the accumulate fallback is
/// checked to add onto the library's result without making a copy.
#[test]
fn the_library_path_meets_the_same_rounding_contract() {
    let Some(ctx) = device() else { return };
    let stream = ctx.default_stream();
    let mma = MmaKernels::new(&ctx).expect("int8 kernels compile");
    for chunk in [0usize, 100] {
        let mut hg = HgemmKernels::new(&ctx).expect("fp16 GEMM kernels compile");
        hg.enable_library(&stream, chunk).expect("cuBLASLt loads");
        let mut expected_bytes = 0;
        let cases = [
            (1usize, 8usize, 32usize),
            (37, 130, 96),
            (128, 128, 64),
            (300, 514, 1024),
            (513, 384, 320),
        ];
        for (case, &(tokens, rows, k)) in cases.iter().enumerate() {
            let seed = 900 + case as u64;
            let mut rng = Xorshift64Star::new(seed ^ 0x5eed);
            let x = rng.vec_f32(tokens * k, -3.0, 3.0);
            let xd = stream.clone_htod(&x).unwrap();
            let mut xh = stream.alloc_zeros::<u16>(tokens * k).unwrap();
            hg.to_half(&stream, &xd, &mut xh, tokens * k).unwrap();
            let (bytes, deq) = q8_0_weights(rows, k, seed);
            let src = stream.clone_htod(&bytes).unwrap();
            let mut wq = stream.alloc_zeros::<i8>(rows * k).unwrap();
            let mut ws = stream.alloc_zeros::<u16>(rows * k / 32).unwrap();
            mma.repack_q8_0_half(&stream, &src, &mut wq, &mut ws, rows * k)
                .unwrap();
            let w = HalfWeight::Q8_0 {
                q: &wq,
                scales: &ws,
            };
            let mut out = stream.alloc_zeros::<f32>(tokens * rows).unwrap();
            hg.gemm(
                &stream, w, &xh, &mut out, None, tokens, rows, k, rows, false,
            )
            .unwrap();
            let got = stream.clone_dtoh(&out).unwrap();
            expected_bytes += rows * k * 2;
            assert_eq!(
                hg.library_resident_bytes(),
                expected_bytes,
                "one fp16 copy a weight"
            );

            let (mut worst, mut magnitude) = (0.0f64, 0.0f64);
            for t in 0..tokens {
                for r in 0..rows {
                    let (mut s, mut m) = (0.0f64, 0.0f64);
                    for i in 0..k {
                        let p =
                            f64::from(round_through_f16(x[t * k + i])) * f64::from(deq[r * k + i]);
                        s += p;
                        m += p.abs();
                    }
                    magnitude = magnitude.max(m);
                    let g = got[t * rows + r];
                    assert!(g.is_finite());
                    worst = worst.max((f64::from(g) - s).abs());
                }
            }
            let bound = 2.0 * (k as f64) * magnitude * 2f64.powi(-24) + 1e-6;
            println!(
                "library chunk {chunk:3} tokens {tokens:4} rows {rows:5} k {k:5}: max_abs \
                 {worst:.3e} (bound {bound:.3e})"
            );
            assert!(worst <= bound, "max_abs {worst} exceeds {bound}");

            // The accumulate form is not the library's; it must still run,
            // through the engine's kernel, and add to what is there.
            let mut acc = stream.clone_htod(&got).unwrap();
            hg.gemm(&stream, w, &xh, &mut acc, None, tokens, rows, k, rows, true)
                .unwrap();
            let doubled = stream.clone_dtoh(&acc).unwrap();
            for (d, g) in doubled.iter().zip(&got) {
                assert!(
                    (f64::from(*d) - 2.0 * f64::from(*g)).abs() <= 2.0 * bound,
                    "the accumulate fallback did not add onto the library's result"
                );
            }
            assert_eq!(
                hg.library_resident_bytes(),
                expected_bytes,
                "the fallback makes no copy"
            );
            // Keep the weights alive until the next iteration allocates, so
            // a reused address cannot alias an earlier copy within the test.
            std::mem::forget((wq, ws));
        }
    }
}
