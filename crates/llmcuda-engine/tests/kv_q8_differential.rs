//! Differential tests for the q8_0 KV cache (`--cache-type-k/-v q8_0`), at
//! K2-Horizon's attention geometry: 32 query heads, 8 KV heads, head_dim 128.
//!
//! ## What is compared, and against what
//!
//! The writers — `attn_kv_append_kv8` and the q8_0 branch of
//! `k2_rotary_append` — against `llmcuda_kernels::quant::quantize_q8_0`, the
//! port of llama.cpp's `quantize_row_q8_0_ref`, **byte for byte**: codes and
//! binary16 scales are integers and bit patterns, so there is nothing to
//! tolerate. The two writers are also held bit-equal to each other, as their
//! binary16 forms already are.
//!
//! The readers — K2's staged prefill, the warp and tensor-core decode splits
//! and their batched forms — against the same CPU attention the binary16 K2
//! tests use, run on exactly the values the cache holds: `f16(q * d)`, which
//! is what every q8_0 reader converts a code to. That keeps the gate the one
//! the binary16 K2 tests pass at (1e-5): this measures the kernels, not the
//! quantization. What quantization costs the model is a separate measurement,
//! made end to end on real weights.
//!
//! Each format combination runs: both halves q8_0, and each half alone, since
//! the halves are compiled and addressed independently.

use std::sync::Arc;

use cudarc::driver::{CudaContext, DevicePtr};
use llmcuda_cuda::device::{DeviceInfo, driver_available};
use llmcuda_cuda::kernels::attention::{AttentionKernels, AttnDecodeScratch, KvFormat, KvFormats};
use llmcuda_cuda::kernels::k2_rope::K2Rope;
use llmcuda_kernels::attention::{attention_decode, causal_attention_streaming};
use llmcuda_kernels::compare::{Tolerance, assert_matches, compare};
use llmcuda_kernels::f16::{round_through_f16, to_f16_bits};
use llmcuda_kernels::quant::{QK8_0, quantize_q8_0};
use llmcuda_kernels::rng::Xorshift64Star;

const GATE: Tolerance = Tolerance {
    max_abs_error: 1e-5,
    max_rel_error: 1e-5 / 1e-6,
    min_cosine_similarity: 1.0 - 1e-6,
    allow_non_finite: false,
};

const HEADS: usize = 32;
const KV_HEADS: usize = 8;
const HD: usize = 128;
const ROW: usize = KV_HEADS * HD;

/// Every combination with at least one q8_0 half.
const QUANTIZED: [KvFormats; 3] = [
    KvFormats {
        k: KvFormat::Q8_0,
        v: KvFormat::Q8_0,
    },
    KvFormats {
        k: KvFormat::Q8_0,
        v: KvFormat::F16,
    },
    KvFormats {
        k: KvFormat::F16,
        v: KvFormat::Q8_0,
    },
];

fn setup() -> Option<Arc<CudaContext>> {
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

/// A half of the cache built on the host: the `u16` words the device holds,
/// and the f32 value a reader sees at each element.
struct HostHalf {
    words: Vec<u16>,
    seen: Vec<f32>,
}

/// Rows of `ROW` values stored in `format`, the way the writers lay them out:
/// binary16, or per position `ROW` int8 codes followed by `ROW / 32` binary16
/// scales.
fn store(format: KvFormat, x: &[f32]) -> HostHalf {
    match format {
        KvFormat::F16 => HostHalf {
            words: x.iter().copied().map(to_f16_bits).collect(),
            seen: x.iter().copied().map(round_through_f16).collect(),
        },
        KvFormat::Q8_0 => {
            let row_bytes = ROW + ROW / 16;
            let mut bytes = vec![0u8; x.len() / ROW * row_bytes];
            let mut seen = vec![0.0f32; x.len()];
            for (r, row) in x.as_chunks::<ROW>().0.iter().enumerate() {
                let dst = &mut bytes[r * row_bytes..(r + 1) * row_bytes];
                for (b, block) in row.as_chunks::<QK8_0>().0.iter().enumerate() {
                    let q = quantize_q8_0(block);
                    let d = q.d.to_f32();
                    for (j, &code) in q.qs.iter().enumerate() {
                        dst[QK8_0 * b + j] = code as u8;
                        // f16(q * d): the product is exact in f32 (8 x 11
                        // significant bits), so this is mul.rn.f16x2's result.
                        seen[r * ROW + QK8_0 * b + j] = round_through_f16(f32::from(code) * d);
                    }
                    dst[ROW + 2 * b..ROW + 2 * b + 2].copy_from_slice(&q.d.to_bits().to_le_bytes());
                }
            }
            HostHalf {
                words: bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|w| u16::from_le_bytes([w[0], w[1]]))
                    .collect(),
                seen,
            }
        }
    }
}

/// Values that exercise the quantizer's edges as well as its body: an
/// all-zero block (d = 0), exact half-way codes (amax 127 makes d = 1, so
/// ±63.5 must round away from zero), one outlier dominating a block, and
/// subnormal-scale blocks.
fn awkward_rows(rng: &mut Xorshift64Star, rows: usize) -> Vec<f32> {
    let mut x = rng.vec_f32(rows * ROW, -3.0, 3.0);
    for (r, row) in x.as_chunks_mut::<ROW>().0.iter_mut().enumerate() {
        let block = |b: usize| (QK8_0 * ((b + r) % (ROW / QK8_0)))..;
        row[block(0)][..QK8_0].fill(0.0);
        let half = &mut row[block(1)][..QK8_0];
        half[0] = 127.0;
        half[1] = 63.5;
        half[2] = -63.5;
        half[3] = 0.5;
        half[4] = -0.5;
        row[block(2)][..QK8_0][7] = 40.0;
        for v in &mut row[block(3)][..QK8_0] {
            *v *= 1e-7;
        }
    }
    x
}

/// Attention inputs at the scale the binary16 K2 tests use, [-1, 1], so the
/// same absolute gate applies, with the blocks a reader can get wrong: an
/// all-zero block (d = 0), an outlier block whose other codes are near zero,
/// and a block whose binary16 scale is subnormal.
fn reader_rows(rng: &mut Xorshift64Star, rows: usize) -> Vec<f32> {
    let mut x = rng.vec_f32(rows * ROW, -1.0, 1.0);
    for (r, row) in x.as_chunks_mut::<ROW>().0.iter_mut().enumerate() {
        let block = |b: usize| (QK8_0 * ((b + r) % (ROW / QK8_0)))..;
        row[block(0)][..QK8_0].fill(0.0);
        let outlier = &mut row[block(1)][..QK8_0];
        for v in outlier.iter_mut() {
            *v *= 0.01;
        }
        outlier[5] = 1.0;
        for v in &mut row[block(2)][..QK8_0] {
            *v *= 1e-4;
        }
    }
    x
}

#[test]
fn q8_append_writes_the_cpu_quantizers_bytes() {
    let Some(ctx) = setup() else { return };
    let stream = ctx.default_stream();
    let mut rng = Xorshift64Star::new(0x5138_0001);
    for formats in QUANTIZED {
        let kernels =
            AttentionKernels::with_kv_formats(&ctx, HEADS, KV_HEADS, HD, formats).unwrap();
        let (tokens, position, max_keys) = (5, 3, 11);
        let key = awkward_rows(&mut rng, tokens);
        let value = awkward_rows(&mut rng, tokens);
        let mut kc = stream
            .alloc_zeros::<u16>(max_keys * formats.k.row_words(ROW))
            .unwrap();
        let mut vc = stream
            .alloc_zeros::<u16>(max_keys * formats.v.row_words(ROW))
            .unwrap();
        let dp = stream.clone_htod(&[position as i32]).unwrap();
        kernels
            .append_kv(
                &stream,
                &stream.clone_htod(&key).unwrap(),
                &stream.clone_htod(&value).unwrap(),
                &mut kc,
                &mut vc,
                tokens,
                max_keys,
                &dp,
            )
            .unwrap();
        for (half, format, source, device) in [
            ("key", formats.k, &key, &kc),
            ("value", formats.v, &value, &vc),
        ] {
            let words = format.row_words(ROW);
            let mut expected = vec![0u16; max_keys * words];
            expected[position * words..(position + tokens) * words]
                .copy_from_slice(&store(format, source).words);
            assert_eq!(
                stream.clone_dtoh(device).unwrap(),
                expected,
                "{half} half, K={} V={}",
                formats.k.name(),
                formats.v.name()
            );
        }
    }
}

#[test]
fn batched_rotary_append_writes_the_same_q8_rows_as_the_separate_append() {
    let Some(ctx) = setup() else { return };
    let stream = ctx.default_stream();
    // K2's own rotary geometry: all 128 dimensions rotated, four query heads
    // per KV head (the q8_0 kernel set requires the ratio).
    let (qh, kh, theta) = (4, 1, 1_000_000f32);
    let row = kh * HD;
    for formats in QUANTIZED {
        let direct = AttentionKernels::with_kv_formats(&ctx, qh, kh, HD, formats).unwrap();
        for n in [1, 3, 8] {
            let positions: Vec<i32> = (0..n).map(|i| [0, 37, 8192, 524_280][i % 4]).collect();
            let dp: Vec<_> = positions
                .iter()
                .map(|&p| stream.clone_htod(&[p]).unwrap())
                .collect();
            let pointers: Vec<_> = dp.iter().map(|p| p.device_ptr(&stream).0).collect();
            let mut rope = K2Rope::with_kv_formats(&ctx, &stream, n, HD, HD, formats).unwrap();
            // SAFETY: every scalar is owned in dp through all launches.
            unsafe { rope.prepare_raw(&stream, &pointers, 1, 0, theta) }.unwrap();
            let query: Vec<f32> = (0..n * qh * HD).map(|i| (i as f32 * 0.037).sin()).collect();
            let key: Vec<f32> = (0..n * row)
                .map(|i| (i as f32 * 0.021).cos() * 2.0)
                .collect();
            let value: Vec<f32> = (0..n * row)
                .map(|i| (i as f32 * 0.013).sin() * 3.0)
                .collect();
            let (dq, dk, dv) = (
                stream.clone_htod(&query).unwrap(),
                stream.clone_htod(&key).unwrap(),
                stream.clone_htod(&value).unwrap(),
            );
            let mut qr = stream.alloc_zeros::<f32>(query.len()).unwrap();
            let mut kr = stream.alloc_zeros::<f32>(key.len()).unwrap();
            rope.apply(&stream, &dk, &mut kr, kh, 0).unwrap();
            let roped_key = stream.clone_dtoh(&kr).unwrap();
            let append_at: Vec<_> = (0..n)
                .map(|i| stream.clone_htod(&[(i + 3) as i32]).unwrap())
                .collect();
            let append_ptrs: Vec<_> = append_at.iter().map(|p| p.device_ptr(&stream).0).collect();
            let mut kc: Vec<_> = (0..n)
                .map(|_| {
                    stream
                        .alloc_zeros::<u16>(16 * formats.k.row_words(row))
                        .unwrap()
                })
                .collect();
            let mut vc: Vec<_> = (0..n)
                .map(|_| {
                    stream
                        .alloc_zeros::<u16>(16 * formats.v.row_words(row))
                        .unwrap()
                })
                .collect();
            let kcs: Vec<_> = kc.iter_mut().map(|p| p.device_ptr(&stream).0).collect();
            let vcs: Vec<_> = vc.iter_mut().map(|p| p.device_ptr(&stream).0).collect();
            // SAFETY: all scalar and cache slots are live and distinct, and
            // each cache covers positions 3..=10 in this rope's formats.
            unsafe {
                rope.append_raw(
                    &stream,
                    &dq,
                    &mut qr,
                    &dk,
                    &mut kr,
                    &dv,
                    &append_ptrs,
                    &kcs,
                    &vcs,
                    qh,
                    kh,
                )
            }
            .unwrap();
            assert!(
                stream
                    .clone_dtoh(&kr)
                    .unwrap()
                    .iter()
                    .zip(&roped_key)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "batched rotary changed the rotated key"
            );
            for i in 0..n {
                let mut kref = stream
                    .alloc_zeros::<u16>(16 * formats.k.row_words(row))
                    .unwrap();
                let mut vref = stream
                    .alloc_zeros::<u16>(16 * formats.v.row_words(row))
                    .unwrap();
                direct
                    .append_kv(
                        &stream,
                        &stream
                            .clone_htod(&roped_key[i * row..(i + 1) * row])
                            .unwrap(),
                        &stream.clone_htod(&value[i * row..(i + 1) * row]).unwrap(),
                        &mut kref,
                        &mut vref,
                        1,
                        16,
                        &append_at[i],
                    )
                    .unwrap();
                assert_eq!(
                    stream.clone_dtoh(&kc[i]).unwrap(),
                    stream.clone_dtoh(&kref).unwrap(),
                    "key rows, sequence {i} of {n}, K={} V={}",
                    formats.k.name(),
                    formats.v.name()
                );
                assert_eq!(
                    stream.clone_dtoh(&vc[i]).unwrap(),
                    stream.clone_dtoh(&vref).unwrap(),
                    "value rows, sequence {i} of {n}, K={} V={}",
                    formats.k.name(),
                    formats.v.name()
                );
            }
        }
    }
}

/// One query head's rows out of a `[rows][heads][HD]` tensor.
fn head_rows(flat: &[f32], rows: usize, heads: usize, head: usize) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|t| flat[(t * heads + head) * HD..(t * heads + head + 1) * HD].to_vec())
        .collect()
}

#[test]
fn k2_q8_attention_matches_cpu_on_the_values_the_cache_holds() {
    let Some(ctx) = setup() else { return };
    let stream = ctx.default_stream();
    for formats in QUANTIZED {
        let kernels =
            AttentionKernels::with_kv_formats(&ctx, HEADS, KV_HEADS, HD, formats).unwrap();
        // Prefill widths across the 32-query block (the staged kernel), and
        // single-query decode at depths below 32 (also the staged kernel),
        // below 512 (warp splits) and above it (tensor-core splits).
        for (queries, offset) in [
            (16, 0),
            (19, 33),
            (65, 47),
            (1, 20),
            (1, 128),
            (1, 600),
            (1, 2047),
        ] {
            let keys = offset + queries + 7;
            let mut rng = Xorshift64Star::new(0x5138_0002 + keys as u64);
            let q = rng.vec_f32(keys * HEADS * HD, -1.0, 1.0);
            let k = store(formats.k, &reader_rows(&mut rng, keys));
            let v = store(formats.v, &reader_rows(&mut rng, keys));
            let dq = stream
                .clone_htod(&q[offset * HEADS * HD..(offset + queries) * HEADS * HD])
                .unwrap();
            let (dk, dv) = (
                stream.clone_htod(&k.words).unwrap(),
                stream.clone_htod(&v.words).unwrap(),
            );
            let dp = stream.clone_htod(&[offset as i32]).unwrap();
            let mut out = stream.alloc_zeros::<f32>(queries * HEADS * HD).unwrap();
            let mut dec = AttnDecodeScratch::new(&stream, HEADS, HD).unwrap();
            for lever in [0, 2, 4] {
                if queries > 1 && lever != 0 {
                    continue;
                }
                if lever == 0 {
                    kernels.disable_decode_mma();
                } else {
                    kernels.set_decode_mma_wpo(lever);
                }
                kernels
                    .forward_k2(
                        &stream,
                        &mut dec,
                        &dq,
                        &dk,
                        &dv,
                        &mut out,
                        queries,
                        keys,
                        offset + queries,
                        &dp,
                    )
                    .unwrap();
                let device = stream.clone_dtoh(&out).unwrap();
                let mut worst = 0.0f32;
                for h in 0..HEADS {
                    let reference = causal_attention_streaming(
                        &head_rows(&q, keys, HEADS, h),
                        &head_rows(&k.seen, keys, KV_HEADS, h / 4),
                        &head_rows(&v.seen, keys, KV_HEADS, h / 4),
                    );
                    let expected: Vec<f32> = reference[offset..offset + queries]
                        .iter()
                        .flatten()
                        .copied()
                        .collect();
                    let actual: Vec<f32> = (0..queries)
                        .flat_map(|t| {
                            device[(t * HEADS + h) * HD..(t * HEADS + h + 1) * HD]
                                .iter()
                                .copied()
                        })
                        .collect();
                    assert_matches(&actual, &expected, &GATE);
                    worst = worst.max(compare(&actual, &expected).max_abs_error);
                }
                println!(
                    "K={} V={} queries={queries} offset={offset} lever={lever}: max_abs={worst}",
                    formats.k.name(),
                    formats.v.name()
                );
            }
        }
    }
}

#[test]
fn k2_q8_batch_decode_keeps_serial_arithmetic() {
    let Some(ctx) = setup() else { return };
    let stream = ctx.default_stream();
    for formats in QUANTIZED {
        let kernels =
            AttentionKernels::with_kv_formats(&ctx, HEADS, KV_HEADS, HD, formats).unwrap();
        let mut rng = Xorshift64Star::new(0x5138_0003);
        for positions in [
            vec![511],
            vec![33, 61, 511],
            vec![33, 61, 127, 255, 511, 1023, 1535, 2047],
        ] {
            kernels.disable_decode_mma();
            let n = positions.len();
            let queries = rng.vec_f32(n * HEADS * HD, -1.0, 1.0);
            let dq = stream.clone_htod(&queries).unwrap();
            let (mut dk, mut dv, mut dp, mut expected, mut serial) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for (i, &position) in positions.iter().enumerate() {
                let keys = position + 7;
                let k = store(formats.k, &reader_rows(&mut rng, keys));
                let v = store(formats.v, &reader_rows(&mut rng, keys));
                dk.push(stream.clone_htod(&k.words).unwrap());
                dv.push(stream.clone_htod(&v.words).unwrap());
                dp.push(stream.clone_htod(&[position as i32]).unwrap());
                let q = &queries[i * HEADS * HD..(i + 1) * HEADS * HD];
                for h in 0..HEADS {
                    expected.extend(attention_decode(
                        &q[h * HD..(h + 1) * HD],
                        &head_rows(&k.seen, keys, KV_HEADS, h / 4)[..=position],
                        &head_rows(&v.seen, keys, KV_HEADS, h / 4)[..=position],
                    ));
                }
                let mut output = stream.alloc_zeros::<f32>(HEADS * HD).unwrap();
                let mut dec = AttnDecodeScratch::new(&stream, HEADS, HD).unwrap();
                kernels
                    .forward_k2(
                        &stream,
                        &mut dec,
                        &stream.clone_htod(q).unwrap(),
                        &dk[i],
                        &dv[i],
                        &mut output,
                        1,
                        keys,
                        position + 1,
                        &dp[i],
                    )
                    .unwrap();
                serial.extend(stream.clone_dtoh(&output).unwrap());
            }
            let ptrs = |slots: &Vec<cudarc::driver::CudaSlice<u16>>| {
                slots
                    .iter()
                    .map(|x| x.device_ptr(&stream).0)
                    .collect::<Vec<_>>()
            };
            let p: Vec<_> = dp.iter().map(|x| x.device_ptr(&stream).0).collect();
            let (k, v) = (ptrs(&dk), ptrs(&dv));
            let mut output = stream.alloc_zeros::<f32>(n * HEADS * HD).unwrap();
            let mut dec = AttnDecodeScratch::new_batch(&stream, HEADS, HD, n).unwrap();
            // SAFETY: distinct caches hold every visible position and seven
            // future keys in this kernel set's formats; all allocations live
            // through the readback.
            unsafe { kernels.decode_batch_k2_raw(&stream, &mut dec, &dq, &mut output, &p, &k, &v) }
                .unwrap();
            let actual = stream.clone_dtoh(&output).unwrap();
            assert_matches(&actual, &expected, &GATE);
            assert!(
                actual
                    .iter()
                    .zip(&serial)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "q8 batch warp split changed serial arithmetic at {positions:?}"
            );
            for wpo in [2, 4] {
                kernels.set_decode_mma_wpo(wpo);
                let mut serial_mma = Vec::new();
                for (i, &position) in positions.iter().enumerate() {
                    let mut single_out = stream.alloc_zeros::<f32>(HEADS * HD).unwrap();
                    let mut single_dec = AttnDecodeScratch::new(&stream, HEADS, HD).unwrap();
                    kernels
                        .forward_k2(
                            &stream,
                            &mut single_dec,
                            &stream
                                .clone_htod(&queries[i * HEADS * HD..(i + 1) * HEADS * HD])
                                .unwrap(),
                            &dk[i],
                            &dv[i],
                            &mut single_out,
                            1,
                            position + 7,
                            position + 1,
                            &dp[i],
                        )
                        .unwrap();
                    serial_mma.extend(stream.clone_dtoh(&single_out).unwrap());
                }
                // SAFETY: the independent cache slots remain live and initialized.
                unsafe {
                    kernels.decode_batch_k2_mma_raw(
                        &stream,
                        &mut dec,
                        &dq,
                        &mut output,
                        &p,
                        &k,
                        &v,
                        wpo,
                    )
                }
                .unwrap();
                let actual = stream.clone_dtoh(&output).unwrap();
                assert_matches(&actual, &expected, &GATE);
                assert!(
                    actual
                        .iter()
                        .zip(&serial_mma)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "q8 batch MMA wpo={wpo} changed serial arithmetic at {positions:?}"
                );
            }
        }
    }
}

#[test]
fn q8_kernel_sets_refuse_what_they_cannot_read() {
    let Some(ctx) = setup() else { return };
    let q8 = QUANTIZED[0];
    // Qwen3.6's attention geometry: head_dim 256, eight query heads per KV head.
    assert!(AttentionKernels::with_kv_formats(&ctx, 16, 2, 256, q8).is_err());
    let stream = ctx.default_stream();
    let kernels = AttentionKernels::with_kv_formats(&ctx, HEADS, KV_HEADS, HD, q8).unwrap();
    let words = q8.k.row_words(ROW);
    let (dk, dv) = (
        stream.alloc_zeros::<u16>(8 * words).unwrap(),
        stream.alloc_zeros::<u16>(8 * words).unwrap(),
    );
    let dq = stream.alloc_zeros::<f32>(HEADS * HD).unwrap();
    let dp = stream.clone_htod(&[0i32]).unwrap();
    let mut out = stream.alloc_zeros::<f32>(HEADS * HD).unwrap();
    let mut dec = AttnDecodeScratch::new(&stream, HEADS, HD).unwrap();
    // The single-tile oracle kernel reads binary16 only.
    kernels.use_k2_single_tile_prefill(true);
    assert!(
        kernels
            .forward_k2(&stream, &mut dec, &dq, &dk, &dv, &mut out, 1, 8, 1, &dp)
            .is_err()
    );
    kernels.use_k2_single_tile_prefill(false);
    // A binary16-sized check would pass a buffer too short for these rows.
    let short = stream.alloc_zeros::<u16>(7 * words).unwrap();
    assert!(
        kernels
            .forward_k2(&stream, &mut dec, &dq, &short, &dv, &mut out, 1, 8, 1, &dp)
            .is_err()
    );
    kernels
        .forward_k2(&stream, &mut dec, &dq, &dk, &dv, &mut out, 1, 8, 1, &dp)
        .unwrap();
}
