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
use llmcuda_cuda::kernels::hgemm::{HalfWeight, HgemmKernels, HgemmPlan, bf16_half_exponent};
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

enum Format {
    Q8_0,
    Bf16,
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
    for format in [Format::Q8_0, Format::Bf16] {
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
