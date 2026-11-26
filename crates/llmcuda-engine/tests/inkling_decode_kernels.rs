//! The fused `inkling2_loop` decode kernels, each against a host oracle on
//! synthetic operands, at the shapes the model's own short capture never
//! reaches: positions past the sliding window, caches longer than one scan
//! chunk, the logarithmic position scale switched on, several rows with
//! different positions in one launch.
//!
//! `inkling_forward.rs` runs these kernels inside the model, twelve positions
//! deep. This is where their branches are taken on purpose.
//!
//! SKIPS — reporting that it skipped — without a driver or a device.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, DevicePtr};
use llmcuda_cuda::device::driver_available;
use llmcuda_cuda::kernels::inkling::{
    ControlWeights, DEC_ROW_POINTERS, InklingDecodeKernels, ProjOffsets,
};
use llmcuda_kernels::inkling as reference;
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

fn to_half(x: &[f32]) -> Vec<u16> {
    x.iter()
        .map(|v| half::f16::from_f32(*v).to_bits())
        .collect()
}

fn from_half(x: &[u16]) -> Vec<f32> {
    x.iter()
        .map(|h| half::f16::from_bits(*h).to_f32())
        .collect()
}

/// Largest absolute difference and `1 - cosine`.
fn distance(got: &[f32], want: &[f64]) -> (f64, f64) {
    let (mut worst, mut dot, mut gg, mut ww) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (g, w) in got.iter().zip(want) {
        let g = f64::from(*g);
        worst = worst.max((g - w).abs());
        dot += g * w;
        gg += g * g;
        ww += w * w;
    }
    (worst, 1.0 - dot / (gg.sqrt() * ww.sqrt()))
}

/// Row positions, window, extent, the logarithmic position scale, and the
/// caches' ring rows.
type AttentionCase = (
    &'static [usize],
    usize,
    usize,
    Option<(f32, f32)>,
    Option<usize>,
);

/// `ink_dec_attn`: clips, the position scale, the bias per distance, the
/// sink, the window, the scan in 256-key chunks and the output gate, for
/// rows at different depths of their own caches.
#[test]
fn decode_attention_matches_an_f64_oracle_past_the_window_and_the_chunk() {
    let Some(ctx) = device() else { return };
    let stream = ctx.default_stream();
    let kernels = InklingDecodeKernels::new(&ctx).expect("kernels load");
    let (heads, kv_heads, dim, n_rel) = (6usize, 2usize, 128usize, 16usize);
    let (qk_clip, rel_bound) = (22.627f32, 8.0f32);
    let mut rng = Xorshift64Star::new(0xdeca77);

    // (positions of the rows, window, extent, log scale)
    let cases: [AttentionCase; 6] = [
        // Sliding: one row before the window fills, one a chunk deep, one
        // past the window with four chunks to scan.
        (&[5, 300, 1500], 1024, 1024, None, None),
        // A window shorter than a chunk, cut mid-chunk.
        (&[0, 255, 256, 700], 100, 100, None, None),
        // Full attention with the position scale on (a floor small enough
        // that tau > 1 here) and a bias table shorter than the reach.
        (&[3, 64, 900], 0, 64, Some((0.1, 8.0)), None),
        // One row, exactly one full chunk.
        (&[255], 0, 1024, Some((0.1, 8192.0)), None),
        // Ring caches: one row not yet round the ring, two that are, one of
        // them with its window straddling the ring's end.
        (&[700, 3000, 5000], 1024, 1024, None, Some(2048)),
        (&[127, 128, 300], 100, 100, None, Some(128)),
    ];
    for (positions, window, extent, log, ring) in cases {
        let rows = positions.len();
        let o = ProjOffsets {
            ldo: (2 * heads * dim + heads * n_rel + heads) as i32,
            q: 0,
            rel: (heads * dim) as i32,
            sink: (heads * dim + heads * n_rel) as i32,
            gate: (heads * dim + heads * n_rel + heads) as i32,
            k: 0,
            v: 0,
            k_taps: -1,
            v_taps: -1,
        };
        let ldo = o.ldo as usize;
        let proj = rng.vec_f32(rows * ldo, -3.0, 3.0);
        let q_clip = rng.vec_f32(heads * dim, 0.5, 1.5);
        let rel_clip = rng.vec_f32(heads * n_rel, 0.5, 4.0);
        let rel_bias = rng.vec_f32(extent * n_rel, -0.4, 0.4);

        // Each row its own caches, as long as its position needs.
        let mut caches: Vec<(Vec<u16>, Vec<u16>)> = Vec::new();
        let mut device_caches: Vec<(CudaSlice<u16>, CudaSlice<u16>)> = Vec::new();
        let mut ptrs = vec![0u64; rows * DEC_ROW_POINTERS];
        for (t, &p) in positions.iter().enumerate() {
            let len = (p + 1) * kv_heads * dim;
            let k = to_half(&rng.vec_f32(len, -2.0, 2.0));
            let v = to_half(&rng.vec_f32(len, -1.0, 1.0));
            // The device sees position j at row j % ring, the last `ring`
            // positions only; the oracle keeps the plain layout.
            let on_device = |cache: &[u16]| -> Vec<u16> {
                let Some(ring) = ring else {
                    return cache.to_vec();
                };
                let row = kv_heads * dim;
                let mut out = vec![0u16; ring * row];
                for j in (p + 1).saturating_sub(ring)..=p {
                    out[(j % ring) * row..][..row].copy_from_slice(&cache[j * row..][..row]);
                }
                out
            };
            let (dk, dv) = (
                stream.clone_htod(&on_device(&k)).unwrap(),
                stream.clone_htod(&on_device(&v)).unwrap(),
            );
            ptrs[t * DEC_ROW_POINTERS] = dk.device_ptr(&stream).0;
            ptrs[t * DEC_ROW_POINTERS + 1] = dv.device_ptr(&stream).0;
            caches.push((k, v));
            device_caches.push((dk, dv));
        }

        let f = |h: u16| f64::from(half::f16::from_bits(h).to_f32());
        let scale = 1.0 / (dim as f64).sqrt();
        let mut want = vec![0.0f64; rows * heads * dim];
        for (t, &p) in positions.iter().enumerate() {
            let row = &proj[t * ldo..][..ldo];
            let tau = match log {
                Some((alpha, floor)) => {
                    1.0 + f64::from(alpha) * ((p + 1) as f64 / f64::from(floor)).max(1.0).ln()
                }
                None => 1.0,
            };
            let lo = if window > 0 && p + 1 > window {
                p + 1 - window
            } else {
                0
            };
            let (k, v) = &caches[t];
            for h in 0..heads {
                let kvh = h / (heads / kv_heads);
                let clipped = |x: &[f32], scales: &[f32], bound: f32| -> Vec<f64> {
                    let v: Vec<f64> = x
                        .iter()
                        .zip(scales)
                        .map(|(a, b)| f64::from(*a) * f64::from(*b))
                        .collect();
                    let norm = v.iter().map(|a| a * a).sum::<f64>().sqrt();
                    let factor = (f64::from(bound) / (norm + 1e-6)).min(1.0) * tau;
                    v.iter().map(|a| a * factor).collect()
                };
                let q = clipped(
                    &row[o.q as usize + h * dim..][..dim],
                    &q_clip[h * dim..][..dim],
                    qk_clip,
                );
                let r = clipped(
                    &row[o.rel as usize + h * n_rel..][..n_rel],
                    &rel_clip[h * n_rel..][..n_rel],
                    rel_bound,
                );
                let sink = f64::from(row[o.sink as usize + h]);
                let scores: Vec<f64> = (lo..=p)
                    .map(|j| {
                        let kr = &k[(j * kv_heads + kvh) * dim..][..dim];
                        let dot: f64 = q.iter().zip(kr).map(|(a, b)| a * f(*b)).sum();
                        let dist = p - j;
                        let bias: f64 = if dist < extent {
                            rel_bias[dist * n_rel..][..n_rel]
                                .iter()
                                .zip(&r)
                                .map(|(b, x)| f64::from(*b) * x)
                                .sum()
                        } else {
                            0.0
                        };
                        dot * scale + bias - sink
                    })
                    .collect();
                let top = scores.iter().copied().fold(0.0f64, f64::max);
                let denom = (-top).exp() + scores.iter().map(|s| (s - top).exp()).sum::<f64>();
                let out = &mut want[(t * heads + h) * dim..][..dim];
                for (j, s) in (lo..=p).zip(&scores) {
                    let w = (s - top).exp() / denom;
                    let vr = &v[(j * kv_heads + kvh) * dim..][..dim];
                    for (acc, x) in out.iter_mut().zip(vr) {
                        *acc += w * f(*x);
                    }
                }
                for (i, acc) in out.iter_mut().enumerate() {
                    let g = f64::from(row[o.gate as usize + h * dim + i]);
                    *acc /= 1.0 + (-g).exp();
                }
            }
        }

        let d_proj = stream.clone_htod(&proj).unwrap();
        let d_ptrs = stream.clone_htod(&ptrs).unwrap();
        let pos: Vec<i32> = positions.iter().map(|p| *p as i32).collect();
        let d_pos = stream.clone_htod(&pos).unwrap();
        let (d_q_clip, d_rel_clip, d_rel_bias) = (
            stream.clone_htod(&q_clip).unwrap(),
            stream.clone_htod(&rel_clip).unwrap(),
            stream.clone_htod(&rel_bias).unwrap(),
        );
        let mut out = stream.alloc_zeros::<u16>(rows * heads * dim).unwrap();
        kernels
            .attn(
                &stream,
                &d_proj,
                &d_ptrs,
                &d_pos,
                &d_q_clip,
                &d_rel_clip,
                &d_rel_bias,
                &mut out,
                o,
                heads,
                kv_heads,
                dim,
                n_rel,
                window,
                extent,
                qk_clip,
                rel_bound,
                log,
                ring,
                rows,
            )
            .unwrap();
        let got = from_half(&stream.clone_dtoh(&out).unwrap());
        drop(device_caches);
        let (worst, one_minus_cos) = distance(&got, &want);
        println!(
            "decode attention positions {positions:?} window {window} extent {extent} log {log:?} ring {ring:?}: max_abs {worst:.3e} 1-cos {one_minus_cos:.3e}"
        );
        // The result is binary16 of a value within +-1: half an ulp is at
        // most 2.4e-4, and the fp32 sums behind it add ~1e-5.
        assert!(worst < 4e-4, "max-abs {worst}");
        assert!(one_minus_cos < 2e-7, "1 - cosine {one_minus_cos}");
    }
}

/// `ink_dec_kv`: a sequence fed a row at a time through the carried
/// convolution, the key clip and the value norm, lands in the cache as the
/// reference's whole-sequence convolution does — static taps and
/// input-dependent ones, two rows with their own carries in one launch.
#[test]
fn decode_keys_and_values_match_the_reference_convolution_step_by_step() {
    let Some(ctx) = device() else { return };
    let stream = ctx.default_stream();
    let kernels = InklingDecodeKernels::new(&ctx).expect("kernels load");
    let (kv_heads, dim, taps, steps, rows) = (4usize, 64usize, 4usize, 9usize, 2usize);
    let channels = kv_heads * dim;
    let (clip, eps) = (6.0f32, 1e-6f32);
    let mut rng = Xorshift64Star::new(0xc0de);

    // A ring shorter than the sequence: the last step lands on row 0.
    for (dynamic, ring) in [(false, None), (true, None), (true, Some(8usize))] {
        let k_conv = rng.vec_f32(channels * taps, -0.5, 0.5);
        let v_conv = rng.vec_f32(channels * taps, -0.5, 0.5);
        let o = ProjOffsets {
            ldo: (2 * channels + 2 * kv_heads * taps) as i32,
            q: 0,
            rel: 0,
            sink: 0,
            gate: 0,
            k: 0,
            v: channels as i32,
            k_taps: if dynamic { 2 * channels as i32 } else { -1 },
            v_taps: if dynamic {
                (2 * channels + kv_heads * taps) as i32
            } else {
                -1
            },
        };
        let ldo = o.ldo as usize;
        // [row][step][ldo]: each row its own sequence of projections.
        let proj: Vec<Vec<f32>> = (0..rows)
            .map(|_| rng.vec_f32(steps * ldo, -1.0, 1.0))
            .collect();

        let mut ptrs = vec![0u64; rows * DEC_ROW_POINTERS];
        let mut state = Vec::new();
        for t in 0..rows {
            let k = stream.alloc_zeros::<u16>(steps * channels).unwrap();
            let v = stream.alloc_zeros::<u16>(steps * channels).unwrap();
            let kc = stream.alloc_zeros::<f32>(channels * (taps - 1)).unwrap();
            let vc = stream.alloc_zeros::<f32>(channels * (taps - 1)).unwrap();
            let at = t * DEC_ROW_POINTERS;
            ptrs[at] = k.device_ptr(&stream).0;
            ptrs[at + 1] = v.device_ptr(&stream).0;
            ptrs[at + 2] = kc.device_ptr(&stream).0;
            ptrs[at + 3] = vc.device_ptr(&stream).0;
            state.push((k, v, kc, vc));
        }
        let d_ptrs = stream.clone_htod(&ptrs).unwrap();
        let (d_k_conv, d_v_conv) = (
            stream.clone_htod(&k_conv).unwrap(),
            stream.clone_htod(&v_conv).unwrap(),
        );
        for step in 0..steps {
            let mut this = Vec::with_capacity(rows * ldo);
            for row in &proj {
                this.extend_from_slice(&row[step * ldo..][..ldo]);
            }
            let d_proj = stream.clone_htod(&this).unwrap();
            let d_pos = stream.clone_htod(&vec![step as i32; rows]).unwrap();
            kernels
                .kv(
                    &stream, &d_proj, &d_ptrs, &d_pos, &d_k_conv, &d_v_conv, o, kv_heads, dim,
                    taps, clip, eps, ring, rows,
                )
                .unwrap();
        }

        for (t, row) in proj.iter().enumerate() {
            let column = |offset: usize, width: usize| -> Vec<f32> {
                (0..steps)
                    .flat_map(|s| row[s * ldo + offset..][..width].to_vec())
                    .collect()
            };
            let (k_raw, v_raw) = (column(0, channels), column(channels, channels));
            let k_taps = column(2 * channels, kv_heads * taps);
            let v_taps = column(2 * channels + kv_heads * taps, kv_heads * taps);
            let mut k_want = reference::convolve(
                &k_raw,
                channels,
                &k_conv,
                taps,
                dynamic.then_some((&k_taps[..], kv_heads)),
            );
            reference::clip(&mut k_want, dim, None, kv_heads, clip);
            let mut v_want = reference::convolve(
                &v_raw,
                channels,
                &v_conv,
                taps,
                dynamic.then_some((&v_taps[..], kv_heads)),
            );
            for group in v_want.chunks_mut(dim) {
                let mean = group.iter().map(|x| x * x).sum::<f32>() / dim as f32;
                let inv = 1.0 / (mean + eps).sqrt();
                for x in group.iter_mut() {
                    *x *= inv;
                }
            }
            for (name, cache, want) in [
                ("keys", &state[t].0, &k_want),
                ("values", &state[t].1, &v_want),
            ] {
                let got = from_half(&stream.clone_dtoh(cache).unwrap());
                // In a ring, row r holds the newest step that maps to it.
                let rows_kept = ring.unwrap_or(steps);
                let mut want64 = vec![0.0f64; rows_kept * channels];
                for step in 0..steps {
                    let at = step % rows_kept;
                    for (o, x) in want64[at * channels..][..channels]
                        .iter_mut()
                        .zip(&want[step * channels..][..channels])
                    {
                        *o = f64::from(*x);
                    }
                }
                let got = &got[..rows_kept * channels];
                let want = want64;
                let (worst, one_minus_cos) = distance(got, &want);
                println!(
                    "decode {name} row {t} dynamic {dynamic} ring {ring:?}: max_abs {worst:.3e} 1-cos {one_minus_cos:.3e}"
                );
                // binary16 of values a few units wide: half an ulp at 4 is
                // 9.8e-4.
                assert!(worst < 1.5e-3, "{name}: max-abs {worst}");
                assert!(one_minus_cos < 2e-7, "{name}: 1 - cosine {one_minus_cos}");
            }
        }
    }
}

fn rms_weighted(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mean = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (mean + eps).sqrt();
    x.iter().zip(w).map(|(v, w)| v * inv * w).collect()
}

/// Three controllers' weights, each tensor the three concatenated as the
/// kernels take them, and the reference's view of each.
struct Controllers {
    down: Vec<f32>,
    gain: Vec<f32>,
    forget: Vec<f32>,
    each: Vec<reference::Controller>,
}

fn controllers(rng: &mut Xorshift64Star, count: usize, n: usize, rank: usize) -> Controllers {
    let mut c = Controllers {
        down: Vec::new(),
        gain: Vec::new(),
        forget: Vec::new(),
        each: Vec::new(),
    };
    for _ in 0..count {
        let down = rng.vec_f32(rank * 2 * n, -0.05, 0.05);
        let gain = rng.vec_f32(n * rank, -0.3, 0.3);
        let forget = rng.vec_f32(n * rank, -0.3, 0.3);
        c.down.extend_from_slice(&down);
        c.gain.extend_from_slice(&gain);
        c.forget.extend_from_slice(&forget);
        c.each.push(reference::Controller {
            down: Arc::new(down),
            gain: Arc::new(gain),
            forget: Arc::new(forget),
        });
    }
    c
}

/// `ink_dec_inputs` with `ink_dec_gate`, `ink_dec_ctrl` and `ink_dec_norm`:
/// a layer's attention inputs in each of the shapes a layer has — no mixing
/// query, one (sliding) or three (full), with and without loop messages —
/// against the reference's mix, control and norm.
#[test]
fn decode_attention_inputs_match_the_reference_mix_control_and_norm() {
    let Some(ctx) = device() else { return };
    let stream = ctx.default_stream();
    let kernels = InklingDecodeKernels::new(&ctx).expect("kernels load");
    let (n, rank, states, rows, eps) = (320usize, 16usize, 3usize, 2usize, 1e-6f32);
    let mut rng = Xorshift64Star::new(0x1217);

    // (mixing queries, norm weights, loop messages)
    for (n_queries, n_norms, looped) in [
        (0usize, 1usize, false),
        (1, 1, false),
        (3, 3, false),
        (1, 1, true),
        (3, 3, true),
    ] {
        let cur = rng.vec_f32(rows * n, -2.0, 2.0);
        let stack = rng.vec_f32(states * rows * n, -2.0, 2.0);
        let queries = rng.vec_f32(n_queries.max(1) * n, -1.0, 1.0);
        let msgs = rng.vec_f32(3 * rows * n, -2.0, 2.0);
        let norms = rng.vec_f32(n_norms * n, 0.5, 1.5);
        let ctrl = controllers(&mut rng, 3, n, rank);

        // What the reference makes of each row.
        let planes = if looped { 3 } else { n_queries.max(1) };
        let mut want = vec![0.0f64; planes * rows * n];
        for t in 0..rows {
            let row = &cur[t * n..][..n];
            let blocks: Vec<&[f32]> = (0..states)
                .map(|s| &stack[(s * rows + t) * n..][..n])
                .collect();
            let mixed: Vec<Vec<f32>> = if n_queries == 0 {
                vec![row.to_vec()]
            } else {
                reference::mix(&blocks, row, &queries[..n_queries * n])
            };
            for j in 0..planes {
                let source = &mixed[j.min(mixed.len() - 1)];
                let x = if looped {
                    reference::control(source, &msgs[(j * rows + t) * n..][..n], &ctrl.each[j])
                } else {
                    source.clone()
                };
                let weight = &norms[j.min(n_norms - 1) * n..][..n];
                for (o, v) in want[(j * rows + t) * n..][..n]
                    .iter_mut()
                    .zip(rms_weighted(&x, weight, eps))
                {
                    *o = f64::from(v);
                }
            }
        }

        let (d_cur, d_stack, d_queries, d_msgs, d_norms) = (
            stream.clone_htod(&cur).unwrap(),
            stream.clone_htod(&stack).unwrap(),
            stream.clone_htod(&queries).unwrap(),
            stream.clone_htod(&msgs).unwrap(),
            stream.clone_htod(&norms).unwrap(),
        );
        let (d_down, d_gain, d_forget) = (
            stream.clone_htod(&ctrl.down).unwrap(),
            stream.clone_htod(&ctrl.gain).unwrap(),
            stream.clone_htod(&ctrl.forget).unwrap(),
        );
        let mut work = stream.alloc_zeros::<f32>(3 * rows * n).unwrap();
        let mut out = stream.alloc_zeros::<u16>(3 * rows * n).unwrap();
        let mut inv = stream.alloc_zeros::<f32>(rows * 6).unwrap();
        let mut gate = stream.alloc_zeros::<f32>(rows * 3 * rank).unwrap();
        let distinct = kernels
            .inputs(
                &stream,
                &d_cur,
                &d_stack,
                (n_queries > 0).then_some((&d_queries, n_queries)),
                looped.then_some(&d_msgs),
                &d_norms,
                &mut work,
                &mut out,
                &mut inv,
                n,
                states,
                eps,
                rows,
            )
            .unwrap();
        assert_eq!(distinct, planes);
        if looped {
            kernels
                .control(
                    &stream,
                    &mut work,
                    &d_msgs,
                    ControlWeights {
                        down: &d_down,
                        gain: &d_gain,
                        forget: &d_forget,
                        rank,
                    },
                    &d_norms,
                    &mut out,
                    &inv,
                    &mut gate,
                    n,
                    3,
                    eps,
                    false,
                    rows,
                )
                .unwrap();
        }
        let got = from_half(&stream.clone_dtoh(&out).unwrap()[..planes * rows * n]);
        let (worst, one_minus_cos) = distance(&got, &want);
        println!(
            "decode inputs queries {n_queries} norms {n_norms} looped {looped}: max_abs {worst:.3e} 1-cos {one_minus_cos:.3e}"
        );
        // binary16 of normalized values times a weight up to 1.5, which
        // reach past 4: half an ulp in [4, 8) is 1.95e-3, and the fp32
        // behind it adds a few 1e-6.
        assert!(worst < 2.2e-3, "max-abs {worst}");
        assert!(one_minus_cos < 2e-7, "1 - cosine {one_minus_cos}");
    }
}

/// `ink_dec_ffn_in`, the feed-forward controller, `ink_dec_router`,
/// `ink_dec_route` and `ink_dec_ffn_out` over three steps of two rows: the
/// carried output convolutions, the residual with and without a new
/// residual block, the mix, the norm in both layouts, the routing, and the
/// layer tail, against the reference.
#[test]
fn decode_feed_forward_glue_matches_the_reference_step_by_step() {
    let Some(ctx) = device() else { return };
    let stream = ctx.default_stream();
    let kernels = InklingDecodeKernels::new(&ctx).expect("kernels load");
    let (n, rank, states, rows, taps, steps) = (320usize, 16usize, 2usize, 2usize, 4usize, 3usize);
    let (experts, topk, eps, weight_scale) = (64usize, 8usize, 1e-6f32, 8.0f32);
    let mut rng = Xorshift64Star::new(0xffe1);

    // (opens a residual block, loop message, byte-order output)
    for (opens, looped, byte_order) in [
        (false, false, false),
        (true, false, true),
        (false, true, true),
        (true, true, false),
    ] {
        let attn_conv = rng.vec_f32(n * taps, -0.5, 0.5);
        let ffn_conv = rng.vec_f32(n * taps, -0.5, 0.5);
        let query = rng.vec_f32(n, -1.0, 1.0);
        let norm_w = rng.vec_f32(n, 0.5, 1.5);
        let router = rng.vec_f32(experts * n, -0.2, 0.2);
        let bias = rng.vec_f32(experts, -0.05, 0.05);
        let ctrl = controllers(&mut rng, 1, n, rank);
        // The residual blocks so far; an opening layer appends its input.
        let mut stack = rng.vec_f32((states + 1) * rows * n, -2.0, 2.0);
        // Per step: the attention output, the layer input, the message, and
        // the weighted expert outputs.
        let o_raw: Vec<Vec<f32>> = (0..steps)
            .map(|_| rng.vec_f32(rows * n, -2.0, 2.0))
            .collect();
        let input: Vec<Vec<f32>> = (0..steps)
            .map(|_| rng.vec_f32(rows * n, -2.0, 2.0))
            .collect();
        let msg: Vec<Vec<f32>> = (0..steps)
            .map(|_| rng.vec_f32(rows * n, -2.0, 2.0))
            .collect();
        let partial: Vec<Vec<f32>> = (0..steps)
            .map(|_| rng.vec_f32(rows * topk * n, -0.5, 0.5))
            .collect();

        let mut ptrs = vec![0u64; rows * DEC_ROW_POINTERS];
        let mut carries = Vec::new();
        for t in 0..rows {
            let a = stream.alloc_zeros::<f32>(n * (taps - 1)).unwrap();
            let f = stream.alloc_zeros::<f32>(n * (taps - 1)).unwrap();
            ptrs[t * DEC_ROW_POINTERS + 4] = a.device_ptr(&stream).0;
            ptrs[t * DEC_ROW_POINTERS + 5] = f.device_ptr(&stream).0;
            carries.push((a, f));
        }
        let d_ptrs = stream.clone_htod(&ptrs).unwrap();
        let (d_attn_conv, d_ffn_conv, d_query, d_norm, d_router, d_bias) = (
            stream.clone_htod(&attn_conv).unwrap(),
            stream.clone_htod(&ffn_conv).unwrap(),
            stream.clone_htod(&query).unwrap(),
            stream.clone_htod(&norm_w).unwrap(),
            stream.clone_htod(&router).unwrap(),
            stream.clone_htod(&bias).unwrap(),
        );
        let (d_down, d_gain, d_forget) = (
            stream.clone_htod(&ctrl.down).unwrap(),
            stream.clone_htod(&ctrl.gain).unwrap(),
            stream.clone_htod(&ctrl.forget).unwrap(),
        );
        let mut d_stack = stream.clone_htod(&stack).unwrap();
        let mut ffn_inp = stream.alloc_zeros::<f32>(rows * n).unwrap();
        let mut work = stream.alloc_zeros::<f32>(rows * n).unwrap();
        let mut out = stream.alloc_zeros::<u16>(rows * n).unwrap();
        let mut inv = stream.alloc_zeros::<f32>(rows * 2).unwrap();
        let mut gate = stream.alloc_zeros::<f32>(rows * rank).unwrap();
        let mut logits = stream.alloc_zeros::<f32>(rows * experts).unwrap();
        let mut ids = stream.alloc_zeros::<i32>(rows * topk).unwrap();
        let mut weights = stream.alloc_zeros::<f32>(rows * topk).unwrap();
        let mut h = stream.alloc_zeros::<f32>(rows * n).unwrap();

        // The reference convolves each row's whole sequence at once.
        let sequence = |per_step: &[Vec<f32>], t: usize, width: usize| -> Vec<f32> {
            per_step
                .iter()
                .flat_map(|s| s[t * width..][..width].to_vec())
                .collect()
        };
        let attn_out: Vec<Vec<f32>> = (0..rows)
            .map(|t| reference::convolve(&sequence(&o_raw, t, n), n, &attn_conv, taps, None))
            .collect();
        let summed: Vec<Vec<f32>> = partial
            .iter()
            .map(|p| {
                let mut sum = vec![0.0f32; rows * n];
                for t in 0..rows {
                    for j in 0..topk {
                        for i in 0..n {
                            sum[t * n + i] += p[(t * topk + j) * n + i];
                        }
                    }
                }
                sum
            })
            .collect();
        let ffn_out: Vec<Vec<f32>> = (0..rows)
            .map(|t| reference::convolve(&sequence(&summed, t, n), n, &ffn_conv, taps, None))
            .collect();

        let mut worst = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for step in 0..steps {
            let (d_o_raw, d_input, d_msg, d_partial) = (
                stream.clone_htod(&o_raw[step]).unwrap(),
                stream.clone_htod(&input[step]).unwrap(),
                stream.clone_htod(&msg[step]).unwrap(),
                stream.clone_htod(&partial[step]).unwrap(),
            );
            kernels
                .ffn_in(
                    &stream,
                    &d_o_raw,
                    &d_input,
                    &d_attn_conv,
                    &d_ptrs,
                    &mut d_stack,
                    &d_query,
                    looped.then_some(&d_msg),
                    &d_norm,
                    &mut ffn_inp,
                    &mut work,
                    &mut out,
                    &mut inv,
                    n,
                    states,
                    opens,
                    taps,
                    byte_order,
                    eps,
                    rows,
                )
                .unwrap();
            if looped {
                kernels
                    .control(
                        &stream,
                        &mut work,
                        &d_msg,
                        ControlWeights {
                            down: &d_down,
                            gain: &d_gain,
                            forget: &d_forget,
                            rank,
                        },
                        &d_norm,
                        &mut out,
                        &inv,
                        &mut gate,
                        n,
                        1,
                        eps,
                        byte_order,
                        rows,
                    )
                    .unwrap();
            }
            kernels
                .route(
                    &stream,
                    &work,
                    &d_router,
                    &d_bias,
                    &mut logits,
                    &mut ids,
                    &mut weights,
                    n,
                    experts,
                    topk,
                    weight_scale,
                    rows,
                )
                .unwrap();
            kernels
                .ffn_out(
                    &stream,
                    &d_partial,
                    &ffn_inp,
                    &d_ffn_conv,
                    &d_ptrs,
                    &mut h,
                    n,
                    topk,
                    taps,
                    rows,
                )
                .unwrap();

            let got_inp = stream.clone_dtoh(&ffn_inp).unwrap();
            let got_x = stream.clone_dtoh(&work).unwrap();
            let got_half = from_half(&stream.clone_dtoh(&out).unwrap());
            let got_ids = stream.clone_dtoh(&ids).unwrap();
            let got_weights = stream.clone_dtoh(&weights).unwrap();
            let got_h = stream.clone_dtoh(&h).unwrap();
            for t in 0..rows {
                let a = &attn_out[t][step * n..][..n];
                let inp = &input[step][t * n..][..n];
                if opens {
                    stack[(states * rows + t) * n..][..n].copy_from_slice(inp);
                }
                let want_inp: Vec<f32> = if opens {
                    a.to_vec()
                } else {
                    inp.iter().zip(a).map(|(x, y)| x + y).collect()
                };
                let live = states + usize::from(opens);
                let blocks: Vec<&[f32]> = (0..live)
                    .map(|s| &stack[(s * rows + t) * n..][..n])
                    .collect();
                let mixed = reference::mix(&blocks, &want_inp, &query).remove(0);
                let x = if looped {
                    reference::control(&mixed, &msg[step][t * n..][..n], &ctrl.each[0])
                } else {
                    mixed
                };
                let x = rms_weighted(&x, &norm_w, eps);
                let as_f64 = |v: &[f32]| v.iter().map(|x| f64::from(*x)).collect::<Vec<f64>>();
                worst.0 = worst
                    .0
                    .max(distance(&got_inp[t * n..][..n], &as_f64(&want_inp)).0);
                worst.1 = worst.1.max(distance(&got_x[t * n..][..n], &as_f64(&x)).0);
                // The binary16 copy, in byte order when asked: element i of a
                // 16-element group at pair i & 7, half i >> 3.
                for (i, v) in x.iter().enumerate() {
                    let at = if byte_order {
                        (i & !15) + ((i & 7) << 1) + ((i & 15) >> 3)
                    } else {
                        i
                    };
                    let err = (f64::from(got_half[t * n + at]) - f64::from(*v)).abs();
                    worst.2 = worst.2.max(err);
                }

                // Routing from the device's own fp32 row, so a near-tie in
                // the oracle's slightly different row cannot flip it.
                let row = &got_x[t * n..][..n];
                let probs: Vec<f32> = (0..experts)
                    .map(|e| {
                        let logit: f32 = router[e * n..][..n]
                            .iter()
                            .zip(row)
                            .map(|(w, v)| w * v)
                            .sum();
                        1.0 / (1.0 + (-logit).exp())
                    })
                    .collect();
                let mut order: Vec<usize> = (0..experts).collect();
                order.sort_by(|a, b| {
                    (probs[*b] + bias[*b])
                        .total_cmp(&(probs[*a] + bias[*a]))
                        .then(a.cmp(b))
                });
                order.truncate(topk);
                let sum: f32 = order.iter().map(|e| probs[*e]).sum::<f32>() + 1e-20;
                let got: Vec<usize> = got_ids[t * topk..][..topk]
                    .iter()
                    .map(|e| *e as usize)
                    .collect();
                assert_eq!(got, order, "step {step} row {t}: routed experts");
                for (j, e) in order.iter().enumerate() {
                    let want = probs[*e] / sum * weight_scale;
                    assert!(
                        (got_weights[t * topk + j] - want).abs() < 1e-4,
                        "step {step} row {t}: weight {j}"
                    );
                }

                let want_h: Vec<f64> = want_inp
                    .iter()
                    .zip(&ffn_out[t][step * n..][..n])
                    .map(|(x, y)| f64::from(*x) + f64::from(*y))
                    .collect();
                worst.3 = worst.3.max(distance(&got_h[t * n..][..n], &want_h).0);
            }
        }
        println!(
            "decode ffn opens {opens} looped {looped} byte-order {byte_order}: ffn input {:.3e}, normed {:.3e}, its binary16 {:.3e}, layer output {:.3e} (max-abs)",
            worst.0, worst.1, worst.2, worst.3
        );
        assert!(worst.0 < 2e-5, "ffn input {}", worst.0);
        assert!(worst.1 < 2e-4, "normed row {}", worst.1);
        // Half an ulp of binary16 in [4, 8), as above.
        assert!(worst.2 < 2.2e-3, "binary16 row {}", worst.2);
        assert!(worst.3 < 2e-5, "layer output {}", worst.3);
    }
}
