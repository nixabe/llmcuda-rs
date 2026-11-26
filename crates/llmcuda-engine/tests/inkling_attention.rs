//! `inkling2_loop` prefill attention, both device kernels, against an f64
//! oracle on the operands the kernels are given: fp32 queries and bias
//! table, the binary16 key and value caches.
//!
//! `p_j = exp(s_j) / (1 + sum exp(s))`, `s_j = q . k_j / sqrt(d) +
//! table[min(p - j, extent)]` over `j <= p` (and `j > p - window`). The
//! shapes put the first row mid-cache, leave a ragged last tile of query
//! rows and a ragged last group of keys, and run full attention with a
//! table shorter than the reach as well as a sliding window.
//!
//! SKIPS — reporting that it skipped — without a driver or a device.

use std::sync::Arc;

use cudarc::driver::CudaContext;
use llmcuda_cuda::device::driver_available;
use llmcuda_cuda::kernels::inkling::InklingKernels;
use llmcuda_kernels::rng::Xorshift64Star;

fn device() -> Option<Arc<CudaContext>> {
    if !driver_available() {
        println!("SKIPPED: no CUDA driver present");
        return None;
    }
    match CudaContext::new(0) {
        Ok(c) => Some(c),
        Err(e) => {
            println!("SKIPPED: could not create a context on device 0: {e}");
            None
        }
    }
}

#[test]
fn prefill_attention_matches_an_f64_oracle_on_both_kernels() {
    let Some(ctx) = device() else { return };
    let stream = ctx.default_stream();
    let kernels = InklingKernels::new(&ctx).expect("kernels load");
    let (heads, kv_heads, dim) = (6usize, 2usize, 128usize);
    let mut rng = Xorshift64Star::new(0x1a77);

    for (tokens, pos0, window, extent) in [
        (37usize, 50usize, 0usize, 30usize),
        (37, 50, 24, 24),
        (16, 0, 0, 64),
        (5, 3, 0, 4),
        (64, 200, 100, 100),
    ] {
        let cache_rows = pos0 + tokens;
        // Queries large enough that the logits are tens wide, as the model's
        // are: that is where a rounded operand shows.
        let q = rng.vec_f32(tokens * heads * dim, -3.0, 3.0);
        let half = |x: Vec<f32>| -> Vec<u16> {
            x.iter()
                .map(|v| half::f16::from_f32(*v).to_bits())
                .collect()
        };
        let k = half(rng.vec_f32(cache_rows * kv_heads * dim, -2.0, 2.0));
        let v = half(rng.vec_f32(cache_rows * kv_heads * dim, -4.0, 4.0));
        // The bias as the model has it: sixteen features a (row, head), a
        // sink logit, and the layer's per-distance vectors. The table is
        // what `ink_rel_table` makes of them, in its arithmetic.
        let n_rel = 16usize;
        let r = rng.vec_f32(tokens * heads * n_rel, -1.5, 1.5);
        let sink = rng.vec_f32(tokens * heads, -2.0, 2.0);
        let rel_bias = rng.vec_f32(extent * n_rel, -1.0, 1.0);
        let mut table = vec![0.0f32; tokens * heads * (extent + 1)];
        for row in 0..tokens * heads {
            for d in 0..=extent {
                let mut acc = 0.0f32;
                if d < extent {
                    for i in 0..n_rel {
                        acc = rel_bias[d * n_rel + i].mul_add(r[row * n_rel + i], acc);
                    }
                }
                table[row * (extent + 1) + d] = acc - sink[row];
            }
        }

        let f = |h: u16| f64::from(half::f16::from_bits(h).to_f32());
        let scale = 1.0 / (dim as f64).sqrt();
        let mut want = vec![0.0f64; tokens * heads * dim];
        for t in 0..tokens {
            let p = pos0 + t;
            let lo = if window > 0 && p + 1 > window {
                p + 1 - window
            } else {
                0
            };
            for h in 0..heads {
                let kvh = h / (heads / kv_heads);
                let qr = &q[(t * heads + h) * dim..][..dim];
                let scores: Vec<f64> = (lo..=p)
                    .map(|j| {
                        let kr = &k[(j * kv_heads + kvh) * dim..][..dim];
                        let dot: f64 = qr.iter().zip(kr).map(|(a, b)| f64::from(*a) * f(*b)).sum();
                        dot * scale
                            + f64::from(table[(t * heads + h) * (extent + 1) + (p - j).min(extent)])
                    })
                    .collect();
                let top = scores.iter().copied().fold(0.0f64, f64::max);
                let denom: f64 = (-top).exp() + scores.iter().map(|s| (s - top).exp()).sum::<f64>();
                for (j, s) in (lo..=p).zip(&scores) {
                    let w = (s - top).exp() / denom;
                    let vr = &v[(j * kv_heads + kvh) * dim..][..dim];
                    for (o, x) in want[(t * heads + h) * dim..][..dim].iter_mut().zip(vr) {
                        *o += w * f(*x);
                    }
                }
            }
        }

        let (d_q, d_k, d_v, d_table) = (
            stream.clone_htod(&q).unwrap(),
            stream.clone_htod(&k).unwrap(),
            stream.clone_htod(&v).unwrap(),
            stream.clone_htod(&table).unwrap(),
        );
        let d_pos = stream.clone_htod(&[pos0 as i32]).unwrap();
        let (d_r, d_sink, d_bias) = (
            stream.clone_htod(&r).unwrap(),
            stream.clone_htod(&sink).unwrap(),
            stream.clone_htod(&rel_bias).unwrap(),
        );
        let mut tabled: Vec<u32> = Vec::new();
        for (name, scalar, fused) in [
            ("scalar", true, false),
            ("tensor-core", false, false),
            ("fused bias", false, true),
        ] {
            kernels.use_reference_kernels(scalar);
            let sentinel = 7.25f32;
            let mut out = stream
                .clone_htod(&vec![sentinel; (tokens + 1) * heads * dim])
                .unwrap();
            if fused {
                assert!(kernels.fuses_the_bias(dim, n_rel));
                kernels
                    .attn_fused(
                        &stream, &d_q, &d_k, &d_v, &d_r, &d_sink, &d_bias, &mut out, &d_pos,
                        window, extent, heads, kv_heads, tokens,
                    )
                    .unwrap();
            } else {
                kernels
                    .attn(
                        &stream, &d_q, &d_k, &d_v, &d_table, &mut out, &d_pos, window, extent,
                        heads, kv_heads, dim, tokens,
                    )
                    .unwrap();
            }
            let got = stream.clone_dtoh(&out).unwrap();
            // Computing the bias in the kernel is the table's arithmetic:
            // the two tensor-core modes agree to the bit.
            let bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            if fused {
                let differing = bits.iter().zip(&tabled).filter(|(a, b)| a != b).count();
                assert_eq!(differing, 0, "fused bias differs from the table");
            } else if !scalar {
                tabled = bits;
            }
            assert!(
                got[tokens * heads * dim..].iter().all(|x| *x == sentinel),
                "{name}: wrote past the last row"
            );
            let (mut worst, mut dot, mut gg, mut ww) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for (g, w) in got.iter().zip(&want) {
                let g = f64::from(*g);
                worst = worst.max((g - w).abs());
                dot += g * w;
                gg += g * g;
                ww += w * w;
            }
            let cos = dot / (gg.sqrt() * ww.sqrt());
            println!(
                "{name:>11} tokens {tokens:3} pos {pos0:3} window {window:3} extent {extent:3}: max_abs {worst:.3e} 1-cos {:.3e}",
                1.0 - cos
            );
            // Values are within +-4 and the weights sum to at most one; fp32
            // sums of a few hundred terms leave ~1e-5.
            assert!(worst < 2e-4, "{name}: max-abs {worst}");
            assert!(1.0 - cos < 1e-9, "{name}: cosine {cos}");
        }
    }
}

/// The tiled fp32 projection is the reference's sums, bit for bit: both are
/// one ascending chain of fused products per output.
#[test]
fn tiled_projection_is_bit_identical_to_the_reference() {
    let Some(ctx) = device() else { return };
    let stream = ctx.default_stream();
    let kernels = InklingKernels::new(&ctx).expect("kernels load");
    let mut rng = Xorshift64Star::new(0x70c7);
    for (rows, k, ldx, tokens) in [
        (128usize, 2304usize, 2304usize, 37usize),
        (130, 200, 260, 8),
        (5, 64, 64, 1),
        (64, 70, 70, 19),
    ] {
        let w = stream
            .clone_htod(&rng.vec_f32(rows * k, -1.0, 1.0))
            .unwrap();
        let x = stream
            .clone_htod(&rng.vec_f32(tokens * ldx, -3.0, 3.0))
            .unwrap();
        let run = |reference: bool| -> Vec<u32> {
            kernels.use_reference_kernels(reference);
            let mut out = stream
                .clone_htod(&vec![7.25f32; (tokens + 1) * rows])
                .unwrap();
            kernels
                .linear(&stream, &w, &x, &mut out, rows, k, ldx, tokens)
                .unwrap();
            stream
                .clone_dtoh(&out)
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        let (want, got) = (run(true), run(false));
        let differing = want.iter().zip(&got).filter(|(a, b)| a != b).count();
        println!(
            "projection {rows} x {k} (stride {ldx}), {tokens} rows: {differing} of {} differ",
            want.len()
        );
        assert_eq!(differing, 0);
        assert!(got[tokens * rows..].iter().all(|b| *b == 7.25f32.to_bits()));
    }
}

/// `ink_clip` with the logarithmic position scale on, at positions either
/// side of the floor: `x * scale`, L2-clipped per group, times
/// `1 + alpha * ln(max(1, (pos + 1) / floor))`. The model's capture is
/// twelve positions against a floor of 8,192, so nothing else takes the
/// branch where the scale is not one.
#[test]
fn prefill_clip_applies_the_position_scale_past_its_floor() {
    let Some(ctx) = device() else { return };
    let stream = ctx.default_stream();
    let kernels = InklingKernels::new(&ctx).expect("kernels load");
    let (groups, width, threshold) = (6usize, 128usize, 22.627f32);
    let mut rng = Xorshift64Star::new(0xc11b);
    // (first position, rows, alpha and floor)
    for (pos0, tokens, log) in [
        (0usize, 5usize, None),
        (8_185, 20, Some((0.1f32, 8_192.0f32))),
        (100_000, 3, Some((0.1, 8_192.0))),
        (2, 12, Some((0.25, 4.0))),
    ] {
        // Half the groups short enough to pass unclipped, half not.
        let mut x = rng.vec_f32(tokens * groups * width, -3.0, 3.0);
        for (g, group) in x.chunks_mut(width).enumerate() {
            if g % 2 == 0 {
                for v in group.iter_mut() {
                    *v *= 0.2;
                }
            }
        }
        let scale = rng.vec_f32(groups * width, 0.5, 1.5);
        let mut want = x.clone();
        llmcuda_kernels::inkling::clip(&mut want, width, Some(&scale), groups, threshold);
        let mut scaled = 0usize;
        for (t, row) in want.chunks_mut(groups * width).enumerate() {
            if let Some((alpha, floor)) = log {
                let tau = 1.0 + alpha * ((pos0 + t + 1) as f32 / floor).max(1.0).ln();
                scaled += usize::from(tau > 1.0);
                for v in row.iter_mut() {
                    *v *= tau;
                }
            }
        }
        let mut d_x = stream.clone_htod(&x).unwrap();
        let d_scale = stream.clone_htod(&scale).unwrap();
        let d_pos = stream.clone_htod(&[pos0 as i32]).unwrap();
        kernels
            .clip(
                &stream,
                &mut d_x,
                Some(&d_scale),
                &d_pos,
                width,
                groups,
                threshold,
                log,
                tokens,
            )
            .unwrap();
        let got = stream.clone_dtoh(&d_x).unwrap();
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!(
            "clip from position {pos0}, {tokens} rows, log {log:?}: {scaled} rows scaled, max_abs {worst:.3e}"
        );
        // fp32 on both sides; the norm is summed in a different order.
        assert!(worst < 2e-5, "max-abs {worst}");
    }
}
