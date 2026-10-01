//! Device K2 primitives against scalar references. Includes Q8_0 as a control,
//! Q4_K/Q6_K selected projections, real-width grouped norms, bias-only routing,
//! value SiLU, softplus gating and dense fp32 router projections.
use cudarc::driver::CudaContext;
use llmcuda_cuda::{
    device::driver_available,
    kernels::{
        k2::{K2Dispatch, K2Kernels},
        moe::{ExpertQuant, to_device_layout},
    },
};
use llmcuda_kernels::{
    compare::{Tolerance, assert_matches, compare},
    k2,
    quant::*,
};
fn check(actual: &[f32], expected: &[f32], label: &str) {
    let c = compare(actual, expected);
    println!(
        "{label}: max_abs={} cosine={}",
        c.max_abs_error, c.cosine_similarity
    );
    assert_matches(
        actual,
        expected,
        &Tolerance {
            max_abs_error: 2e-4,
            max_rel_error: 0.1,
            min_cosine_similarity: 1.0 - 1e-6,
            allow_non_finite: false,
        },
    );
}
fn same_bits(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    for (i, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual.to_bits(), expected.to_bits(), "{label}, element {i}");
    }
}
fn pack(q: ExpertQuant, src: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let mut bytes = Vec::new();
    let mut back = Vec::new();
    match q {
        ExpertQuant::Q4K => {
            for a in src.as_chunks::<256>().0 {
                let b = quantize_q4_k(a);
                bytes.extend(b.to_bytes());
                back.extend(dequantize_q4_k(&b));
            }
        }
        ExpertQuant::Q6K => {
            for a in src.as_chunks::<256>().0 {
                let b = quantize_q6_k(a);
                bytes.extend(b.to_bytes());
                back.extend(dequantize_q6_k(&b));
            }
        }
        ExpertQuant::Q8_0 => {
            for a in src.as_chunks::<32>().0 {
                let b = quantize_q8_0(a);
                bytes.extend(b.to_bytes());
                back.extend(dequantize_q8_0(&b));
            }
        }
        _ => unreachable!(),
    }
    (bytes, back)
}
#[test]
fn k2_operations_match_the_cpu_reference() {
    if !driver_available() {
        println!("SKIPPED: no CUDA driver");
        return;
    }
    let ctx = CudaContext::new(0).expect("GPU 0");
    let stream = ctx.default_stream();
    let ops = K2Kernels::new(&ctx).unwrap();
    let tokens = 3;
    let width = 2560;
    let x: Vec<f32> = (0..tokens * width)
        .map(|i| ((i as f32 * 0.13).sin() + 0.25) * if i % width < width / 2 { 0.02 } else { 10.0 })
        .collect();
    let w: Vec<f32> = (0..width).map(|i| 0.5 + (i as f32 * 0.021).cos()).collect();
    let dx = stream.clone_htod(&x).unwrap();
    let dw = stream.clone_htod(&w).unwrap();
    let mut dy = stream.alloc_zeros::<f32>(x.len()).unwrap();
    ops.norm(&stream, &dx, &dw, &mut dy, width, 2, 1e-6)
        .unwrap();
    check(
        &stream.clone_dtoh(&dy).unwrap(),
        &k2::grouped_rms_norm(&x, &w, 2, 1e-6),
        "group norm",
    );
    for experts in [64, 100] {
        let logits: Vec<f32> = (0..experts * tokens)
            .map(|i| ((i as f32 * 0.137).sin()) * 6.0)
            .collect();
        let bias: Vec<f32> = (0..experts)
            .map(|i| if i % 3 == 0 { 1.2 } else { -0.1 })
            .collect();
        let dl = stream.clone_htod(&logits).unwrap();
        let db = stream.clone_htod(&bias).unwrap();
        let mut ids = stream.alloc_zeros::<i32>(tokens * 4).unwrap();
        let mut weights = stream.alloc_zeros::<f32>(tokens * 4).unwrap();
        ops.route(&stream, &dl, &db, &mut ids, &mut weights, 4, 2.5)
            .unwrap();
        let actual_ids = stream.clone_dtoh(&ids).unwrap();
        let actual_w = stream.clone_dtoh(&weights).unwrap();
        for token in 0..tokens {
            let (ri, rw) = k2::route(
                &logits[token * experts..(token + 1) * experts],
                &bias,
                4,
                2.5,
            );
            assert_eq!(&actual_ids[token * 4..token * 4 + 4], &ri);
            check(&actual_w[token * 4..token * 4 + 4], &rw, "sigmoid route");
        }
    }
    let inner = 512;
    let rows = 128;
    let experts = 5;
    let top = 2;
    let src: Vec<f32> = (0..experts * rows * inner)
        .map(|i| (i as f32 * 0.173).sin() * 0.02 * (1.0 + (i / (rows * inner)) as f32))
        .collect();
    let input: Vec<f32> = (0..tokens * inner)
        .map(|i| (i as f32 * 0.047).cos())
        .collect();
    let ids = vec![4, 1, 0, 3, 2, 4];
    let dids = stream.clone_htod(&ids).unwrap();
    let dinput = stream.clone_htod(&input).unwrap();
    for q in [ExpertQuant::Q8_0, ExpertQuant::Q4K, ExpertQuant::Q6K] {
        let (bytes, back) = pack(q, &src);
        let db = stream.clone_htod(&*to_device_layout(q, &bytes)).unwrap();
        let mut dp = stream.alloc_zeros::<f32>(tokens * top * rows).unwrap();
        ops.project(
            &stream,
            &db,
            Some(q),
            &dinput,
            &dids,
            &mut dp,
            inner,
            rows,
            experts,
            top,
            true,
        )
        .unwrap();
        let mut reference = vec![0.0; tokens * top * rows];
        for t in 0..tokens {
            for k in 0..top {
                for r in 0..rows {
                    let start = (ids[t * top + k] as usize * rows + r) * inner;
                    reference[(t * top + k) * rows + r] = (0..inner)
                        .map(|j| back[start + j] * input[t * inner + j])
                        .sum();
                }
            }
        }
        check(
            &stream.clone_dtoh(&dp).unwrap(),
            &reference,
            &format!("selected {q:?}"),
        );
        let weights = vec![0.75, 1.75, 1.25, 1.25, 2.0, 0.5];
        let dw = stream.clone_htod(&weights).unwrap();
        let mut dv = stream.alloc_zeros::<f32>(tokens * rows).unwrap();
        ops.values(&stream, &dp, &dw, &mut dv, rows, top).unwrap();
        let values: Vec<f32> = (0..tokens)
            .flat_map(|t| {
                k2::values(
                    &reference[t * top * rows..(t + 1) * top * rows],
                    &weights[t * top..t * top + top],
                    rows,
                )
            })
            .collect();
        check(&stream.clone_dtoh(&dv).unwrap(), &values, "routed values");
    }
    let gates: Vec<f32> = (0..x.len())
        .map(|i| {
            if i % 5 == 0 {
                0.0
            } else {
                (i as f32 * 0.19).sin() * 100.0
            }
        })
        .collect();
    let dg = stream.clone_htod(&gates).unwrap();
    ops.gate(&stream, &dx, &dg, &mut dy).unwrap();
    check(
        &stream.clone_dtoh(&dy).unwrap(),
        &k2::attention_gate(&x, &gates),
        "softplus gate",
    );
    let float_rows = 7;
    let fw = &src[..float_rows * inner];
    let bytes: Vec<u8> = fw.iter().flat_map(|x| x.to_le_bytes()).collect();
    let db = stream.clone_htod(&bytes).unwrap();
    let mut dp = stream.alloc_zeros::<f32>(tokens * float_rows).unwrap();
    ops.project(
        &stream, &db, None, &dinput, &dids, &mut dp, inner, float_rows, 1, 1, false,
    )
    .unwrap();
    let mut expected = Vec::new();
    for t in 0..tokens {
        for r in 0..float_rows {
            expected.push(
                (0..inner)
                    .map(|j| fw[r * inner + j] * input[t * inner + j])
                    .sum(),
            );
        }
    }
    check(&stream.clone_dtoh(&dp).unwrap(), &expected, "fp32 router");
}

#[test]
fn tiled_and_grouped_projections_preserve_scalar_contractions() {
    use llmcuda_cuda::kernels::k2::K2Dispatch;
    if !driver_available() {
        println!("SKIPPED: no CUDA driver");
        return;
    }
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let ops = K2Kernels::new(&ctx).unwrap();
    let (inner, rows, experts, topk) = (512, 7, 5, 2);
    let source: Vec<f32> = (0..inner * rows * experts)
        .map(|i| (i as f32 * 0.071).sin() * 0.03)
        .collect();
    for q in [ExpertQuant::Q8_0, ExpertQuant::Q4K, ExpertQuant::Q6K] {
        let (bytes, back) = pack(q, &source);
        let dw = stream.clone_htod(&*to_device_layout(q, &bytes)).unwrap();
        let dense_bytes =
            &to_device_layout(q, &bytes)[..inner * rows / q.block_elements() * q.block_bytes()];
        let dense_w = stream.clone_htod(dense_bytes).unwrap();
        for tokens in [1, 3, 8, 11, 32] {
            let input: Vec<f32> = (0..tokens * inner)
                .map(|i| (i as f32 * 0.043).cos())
                .collect();
            let dx = stream.clone_htod(&input).unwrap();
            let ids: Vec<i32> = (0..tokens)
                .flat_map(|t| [if t % 3 == 0 { 4 } else { 1 }, 0])
                .collect();
            let di = stream.clone_htod(&ids).unwrap();
            let mut scalar = stream.alloc_zeros::<f32>(tokens * rows).unwrap();
            ops.project_with_tile(
                &stream,
                &dense_w,
                Some(q),
                &dx,
                &di,
                &mut scalar,
                inner,
                rows,
                1,
                1,
                false,
                1,
            )
            .unwrap();
            let scalar = stream.clone_dtoh(&scalar).unwrap();
            let reference: Vec<f32> = (0..tokens)
                .flat_map(|t| {
                    let back = &back;
                    let input = &input;
                    (0..rows).map(move |r| {
                        (0..inner)
                            .map(|j| back[r * inner + j] * input[t * inner + j])
                            .sum()
                    })
                })
                .collect();
            check(&scalar, &reference, "dense scalar control");
            for tile in [4, 8] {
                let mut out = stream.alloc_zeros::<f32>(tokens * rows).unwrap();
                ops.project_with_tile(
                    &stream,
                    &dense_w,
                    Some(q),
                    &dx,
                    &di,
                    &mut out,
                    inner,
                    rows,
                    1,
                    1,
                    false,
                    tile,
                )
                .unwrap();
                same_bits(
                    &stream.clone_dtoh(&out).unwrap(),
                    &scalar,
                    &format!("{q:?}, T={tokens}, tile={tile}"),
                );
            }
            let mut scalar = stream.alloc_zeros::<f32>(tokens * topk * rows).unwrap();
            ops.project(
                &stream,
                &dw,
                Some(q),
                &dx,
                &di,
                &mut scalar,
                inner,
                rows,
                experts,
                topk,
                true,
            )
            .unwrap();
            let scalar = stream.clone_dtoh(&scalar).unwrap();
            let mut dispatch = K2Dispatch::new(&stream, tokens, topk, experts).unwrap();
            let mut out = stream.alloc_zeros::<f32>(tokens * topk * rows).unwrap();
            ops.project_grouped(
                &stream,
                &dw,
                q,
                &dx,
                &di,
                &mut out,
                inner,
                rows,
                &mut dispatch,
            )
            .unwrap();
            same_bits(
                &stream.clone_dtoh(&out).unwrap(),
                &scalar,
                &format!("grouped {q:?}, T={tokens}"),
            );
        }
    }
    // F32 router projections also take the dense tile, including a ragged tail.
    let tokens = 11;
    let input: Vec<f32> = (0..tokens * inner)
        .map(|i| (i as f32 * 0.043).cos())
        .collect();
    let dx = stream.clone_htod(&input).unwrap();
    let dw = stream
        .clone_htod(
            &source[..inner * rows]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let ids = stream.alloc_zeros::<i32>(1).unwrap();
    let mut out = stream.alloc_zeros::<f32>(tokens * rows).unwrap();
    ops.project_with_tile(
        &stream, &dw, None, &dx, &ids, &mut out, inner, rows, 1, 1, false, 1,
    )
    .unwrap();
    let scalar = stream.clone_dtoh(&out).unwrap();
    for tile in [4, 8] {
        ops.project_with_tile(
            &stream, &dw, None, &dx, &ids, &mut out, inner, rows, 1, 1, false, tile,
        )
        .unwrap();
        same_bits(&stream.clone_dtoh(&out).unwrap(), &scalar, "F32 dense tile");
    }
}

#[test]
fn parallel_router_preserves_ties_and_normalization_floor() {
    if !driver_available() {
        println!("SKIPPED: no CUDA driver");
        return;
    }
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let ops = K2Kernels::new(&ctx).unwrap();
    for experts in [1, 31, 64, 100, 512] {
        let topk = experts.min(8);
        for logit in [0.0, -100.0, 100.0] {
            let logits = vec![logit; experts];
            let bias: Vec<f32> = (0..experts)
                .map(|i| if i % 3 == 0 { 0.5 } else { 0.0 })
                .collect();
            let dl = stream.clone_htod(&logits).unwrap();
            let db = stream.clone_htod(&bias).unwrap();
            let mut ids = stream.alloc_zeros::<i32>(topk).unwrap();
            let mut weights = stream.alloc_zeros::<f32>(topk).unwrap();
            ops.route(&stream, &dl, &db, &mut ids, &mut weights, topk, 2.5)
                .unwrap();
            let (expected_ids, expected_weights) = k2::route(&logits, &bias, topk, 2.5);
            assert_eq!(stream.clone_dtoh(&ids).unwrap(), expected_ids);
            check(
                &stream.clone_dtoh(&weights).unwrap(),
                &expected_weights,
                "stable route",
            );
        }
    }
}

#[test]
fn integer_projections_match_cpu_across_shapes() {
    if !driver_available() {
        println!("SKIPPED: no CUDA driver");
        return;
    }
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let ops = K2Kernels::new(&ctx).unwrap();
    let (inner, rows, experts, topk) = (512, 135, 5, 2);
    let source: Vec<f32> = (0..inner * rows * experts)
        .map(|i| (i as f32 * 0.071).sin() * 0.03)
        .collect();
    for bits in [8, 12, 16] {
        for q in [ExpertQuant::Q4K, ExpertQuant::Q6K] {
            let (bytes, back) = pack(q, &source);
            let minima = if q == ExpertQuant::Q4K {
                Some(
                    bytes
                        .as_chunks::<144>()
                        .0
                        .iter()
                        .flat_map(|b| q4_k_minima(&BlockQ4K::from_bytes(b).unwrap()))
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            };
            let dw = stream.clone_htod(&*to_device_layout(q, &bytes)).unwrap();
            let mut prefixes: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
            let packed = llmcuda_cuda::kernels::k2_gemm::K2Gemm::repack(
                &ctx, &stream, &dw, q, inner, rows, experts,
            )
            .unwrap();
            for tokens in [1, 3, 8, 11, 32, 65] {
                let ids: Vec<i32> = (0..tokens)
                    .flat_map(|t| [if t % 3 == 0 { 4 } else { 1 }, 0])
                    .collect();
                let di = stream.clone_htod(&ids).unwrap();
                let mut d = K2Dispatch::new_gemm(&stream, tokens, topk, experts).unwrap();
                ops.dispatch_gemm(&stream, &di, &mut d).unwrap();
                let mut blas = llmcuda_cuda::kernels::k2_gemm::K2Gemm::new_with_activation_bits(
                    &ctx,
                    &stream,
                    tokens,
                    topk,
                    experts,
                    inner * rows * experts,
                    inner,
                    rows,
                    bits,
                )
                .unwrap();
                for flat in [false, true] {
                    let input: Vec<f32> = (0..tokens * inner * if flat { topk } else { 1 })
                        .map(|i| (i as f32 * 0.043).cos())
                        .collect();
                    let dx = stream.clone_htod(&input).unwrap();
                    let mut out = stream.alloc_zeros::<f32>(tokens * topk * rows).unwrap();
                    blas.project(
                        &stream,
                        &packed,
                        q,
                        &dx,
                        &mut out,
                        inner,
                        rows,
                        experts,
                        Some(&d),
                        flat,
                    )
                    .unwrap();
                    let reference = k2::quantized_project_grouped(
                        &back,
                        &input,
                        Some(&ids),
                        inner,
                        rows,
                        tokens,
                        topk,
                        flat,
                        minima.as_deref(),
                        bits,
                        32,
                    );
                    let actual = stream.clone_dtoh(&out).unwrap();
                    check(&actual, &reference, "integer projection CPU");
                    let prepared = blas.prepare(&stream, &dx, inner, true).unwrap();
                    for _ in 0..2 {
                        blas.project_prepared(
                            &stream,
                            &packed,
                            q,
                            prepared,
                            &mut out,
                            inner,
                            rows,
                            experts,
                            Some(&d),
                            flat,
                        )
                        .unwrap();
                        same_bits(&stream.clone_dtoh(&out).unwrap(), &actual, "prepared reuse");
                    }
                    blas.prepare(&stream, &dx, inner, true).unwrap();
                    assert!(
                        blas.project_prepared(
                            &stream,
                            &packed,
                            q,
                            prepared,
                            &mut out,
                            inner,
                            rows,
                            experts,
                            Some(&d),
                            flat,
                        )
                        .is_err(),
                        "stale prepared generation must be rejected"
                    );
                    if tokens == 3 {
                        prefixes[usize::from(flat)] = actual.clone();
                    }
                    if tokens >= 8 {
                        same_bits(
                            &actual[..3 * topk * rows],
                            &prefixes[usize::from(flat)],
                            "GEMV/GEMM matching tree",
                        );
                    }
                }
            }
            // Dense routing has no device dispatch and uses a separate wide
            // launch. Exercise its tails and decode token-reuse path as well.
            let dense_elements = inner * rows;
            let dense_src = stream
                .clone_htod(&*to_device_layout(
                    q,
                    &bytes[..dense_elements / q.block_elements()
                        * if q == ExpertQuant::Q4K { 144 } else { 210 }],
                ))
                .unwrap();
            let dense_weights = llmcuda_cuda::kernels::k2_gemm::K2Gemm::repack(
                &ctx, &stream, &dense_src, q, inner, rows, 1,
            )
            .unwrap();
            let mut first = Vec::new();
            for tokens in [1, 3, 8, 65] {
                let input: Vec<f32> = (0..tokens * inner)
                    .map(|i| (i as f32 * 0.043).cos())
                    .collect();
                let dx = stream.clone_htod(&input).unwrap();
                let mut out = stream.alloc_zeros::<f32>(tokens * rows).unwrap();
                let mut gemm = llmcuda_cuda::kernels::k2_gemm::K2Gemm::new_with_activation_bits(
                    &ctx,
                    &stream,
                    tokens,
                    1,
                    1,
                    dense_elements,
                    inner,
                    rows,
                    bits,
                )
                .unwrap();
                gemm.project(
                    &stream,
                    &dense_weights,
                    q,
                    &dx,
                    &mut out,
                    inner,
                    rows,
                    1,
                    None,
                    false,
                )
                .unwrap();
                let reference = k2::quantized_project_grouped(
                    &back[..dense_elements],
                    &input,
                    None,
                    inner,
                    rows,
                    tokens,
                    1,
                    false,
                    minima.as_deref(),
                    bits,
                    32,
                );
                let actual = stream.clone_dtoh(&out).unwrap();
                check(&actual, &reference, "dense integer projection CPU");
                if tokens == 1 {
                    first = actual.clone();
                } else if tokens == 3 {
                    same_bits(&actual[..rows], &first, "dense decode token reuse");
                } else {
                    same_bits(&actual[..rows], &first, "dense GEMV/GEMM matching tree");
                }
            }
            // These dyadic values make the quantization contract independent of
            // division-rounding differences near arbitrary input boundaries.
            let input: Vec<f32> = (0..inner)
                .map(|i| {
                    if i < 32 {
                        0.0
                    } else {
                        (i % 32) as f32 / 16.0 - 1.0
                    }
                })
                .collect();
            let dx = stream.clone_htod(&input).unwrap();
            let mut out = stream.alloc_zeros::<f32>(rows).unwrap();
            let mut gemm = llmcuda_cuda::kernels::k2_gemm::K2Gemm::new_with_activation_bits(
                &ctx,
                &stream,
                1,
                1,
                1,
                dense_elements,
                inner,
                rows,
                bits,
            )
            .unwrap();
            gemm.project(
                &stream,
                &dense_weights,
                q,
                &dx,
                &mut out,
                inner,
                rows,
                1,
                None,
                false,
            )
            .unwrap();
            let (hi, lo, scales, centers) = gemm.activation_parts();
            let hi = stream.clone_dtoh(hi).unwrap();
            let lo = stream.clone_dtoh(lo).unwrap();
            let scales = stream.clone_dtoh(scales).unwrap();
            let centers = stream.clone_dtoh(centers).unwrap();
            let limit = match bits {
                8 => 127.0f32,
                12 => 2039.0,
                16 => 32639.0,
                _ => unreachable!(),
            };
            let radix = match bits {
                8 => 1,
                12 => 16,
                16 => 256,
                _ => unreachable!(),
            };
            let group = 32;
            let expected = k2::quantize_activation_grouped(&input, bits, group);
            for i in 0..inner {
                let (scale, center): (f32, f32) = if i < 32 {
                    (1.0, 0.0)
                } else if bits == 8 {
                    (1.9375 / 254.0, -4.0)
                } else {
                    (1.0 / limit, 0.0)
                };
                assert_eq!(scales[i / 32].to_bits(), scale.to_bits());
                if bits == 8 {
                    assert_eq!(centers[i / 32].to_bits(), center.to_bits());
                }
                let code = i32::from(hi[i]) * radix + i32::from(lo[i]);
                assert_eq!(
                    code,
                    ((expected[i] / scales[i / 32]).round_ties_even()
                        - if bits == 8 { centers[i / 32] } else { 0.0 }) as i32
                );
            }
        }
    }
}

#[test]
fn ffn_combine_matches_cpu_with_padded_rows() {
    if !driver_available() {
        println!("SKIPPED: no CUDA driver");
        return;
    }
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let ops = K2Kernels::new(&ctx).unwrap();
    let (tokens, topk, rows) = (3, 8, 259);
    let p: Vec<f32> = (0..tokens * topk * rows)
        .map(|i| (i as f32 * 0.13).sin())
        .collect();
    let weights: Vec<f32> = (0..tokens * topk)
        .map(|i| (i % topk) as f32 / 28.0)
        .collect();
    let dp = stream.clone_htod(&p).unwrap();
    let dw = stream.clone_htod(&weights).unwrap();
    let mut out = stream.alloc_zeros::<f32>(tokens * rows).unwrap();
    ops.combine(&stream, &dp, &dw, &mut out, rows, topk)
        .unwrap();
    let expected: Vec<f32> = (0..tokens)
        .flat_map(|t| {
            k2::combine(
                &p[t * topk * rows..(t + 1) * topk * rows],
                &weights[t * topk..(t + 1) * topk],
                rows,
            )
        })
        .collect();
    check(
        &stream.clone_dtoh(&out).unwrap(),
        &expected,
        "FFN combine CPU",
    );
}

#[test]
fn rotary_reuse_matches_cpu_direct_and_cache_append() {
    use cudarc::driver::DevicePtr;
    use llmcuda_cuda::kernels::{attention::AttentionKernels, k2_rope::K2Rope};
    if !driver_available() {
        println!("SKIPPED: no CUDA driver");
        return;
    }
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let (dim, rd, qh, kh, theta) = (160, 128, 4, 1, 1_000_000f32);
    let direct = AttentionKernels::new(&ctx, qh, kh, dim).unwrap();
    for n in [1, 3, 8] {
        for chunk in [1, 9] {
            let tokens = n * chunk;
            let positions: Vec<i32> = (0..n).map(|i| [0, 37, 8192, 524280][i % 4]).collect();
            let mut dp: Vec<_> = positions
                .iter()
                .map(|&p| stream.clone_htod(&[p]).unwrap())
                .collect();
            let pointers: Vec<_> = dp.iter().map(|p| p.device_ptr(&stream).0).collect();
            let mut rope = K2Rope::new(&ctx, &stream, tokens, dim, rd).unwrap();
            // SAFETY: every scalar is owned in dp through all launches.
            unsafe {
                rope.prepare_raw(&stream, &pointers, chunk, 0, theta)
                    .unwrap();
            }
            let query: Vec<_> = (0..tokens * qh * dim)
                .map(|i| (i as f32 * 0.037).sin())
                .collect();
            let key: Vec<_> = (0..tokens * kh * dim)
                .map(|i| (i as f32 * 0.021).cos())
                .collect();
            let dq = stream.clone_htod(&query).unwrap();
            let dk = stream.clone_htod(&key).unwrap();
            let mut qr = stream.alloc_zeros::<f32>(query.len()).unwrap();
            let mut kr = stream.alloc_zeros::<f32>(key.len()).unwrap();
            rope.apply(&stream, &dq, &mut qr, qh, 0).unwrap();
            rope.apply(&stream, &dk, &mut kr, kh, 0).unwrap();
            let actual = stream.clone_dtoh(&qr).unwrap();
            let mut reference = Vec::new();
            for (seq, &position) in positions.iter().enumerate() {
                let source = &query[seq * chunk * qh * dim..(seq + 1) * chunk * qh * dim];
                for (head, row) in source.chunks_exact(dim).enumerate() {
                    reference.extend(llmcuda_kernels::rope::apply_rope(
                        row,
                        (position as usize + head / qh) as u32,
                        rd as u32,
                        theta,
                    ));
                }
                let input = stream.clone_htod(source).unwrap();
                let mut output = stream.alloc_zeros::<f32>(source.len()).unwrap();
                direct
                    .rope(&stream, &input, &mut output, chunk, qh, rd, &dp[seq], theta)
                    .unwrap();
                same_bits(
                    &actual[seq * source.len()..(seq + 1) * source.len()],
                    &stream.clone_dtoh(&output).unwrap(),
                    "cached/direct rotary",
                );
            }
            assert_matches(
                &actual,
                &reference,
                &Tolerance {
                    max_abs_error: 1e-5,
                    max_rel_error: 0.01,
                    min_cosine_similarity: 1.0 - 1e-6,
                    allow_non_finite: false,
                },
            );
            if chunk != 1 {
                continue;
            }
            let (expected_q, expected_k) = (actual, stream.clone_dtoh(&kr).unwrap());
            let ap: Vec<_> = (0..n)
                .map(|i| stream.clone_htod(&[(i + 3) as i32]).unwrap())
                .collect();
            let aps: Vec<_> = ap.iter().map(|p| p.device_ptr(&stream).0).collect();
            let mut kc: Vec<_> = (0..n)
                .map(|_| stream.alloc_zeros::<u16>(16 * kh * dim).unwrap())
                .collect();
            let mut vc: Vec<_> = (0..n)
                .map(|_| stream.alloc_zeros::<u16>(16 * kh * dim).unwrap())
                .collect();
            let kcs: Vec<_> = kc.iter_mut().map(|p| p.device_ptr(&stream).0).collect();
            let vcs: Vec<_> = vc.iter_mut().map(|p| p.device_ptr(&stream).0).collect();
            // SAFETY: all scalar/cache slots are live; caches cover positions
            // 3..10, and each sequence owns distinct allocations.
            unsafe {
                rope.append_raw(
                    &stream, &dq, &mut qr, &dk, &mut kr, &dk, &aps, &kcs, &vcs, qh, kh,
                )
                .unwrap();
            }
            same_bits(
                &stream.clone_dtoh(&qr).unwrap(),
                &expected_q,
                "cached batch query",
            );
            same_bits(
                &stream.clone_dtoh(&kr).unwrap(),
                &expected_k,
                "cached batch key",
            );
            for i in 0..n {
                let row = kh * dim;
                let dk = stream
                    .clone_htod(&expected_k[i * row..(i + 1) * row])
                    .unwrap();
                let dv = stream.clone_htod(&key[i * row..(i + 1) * row]).unwrap();
                let mut kref = stream.alloc_zeros::<u16>(16 * row).unwrap();
                let mut vref = stream.alloc_zeros::<u16>(16 * row).unwrap();
                direct
                    .append_kv(&stream, &dk, &dv, &mut kref, &mut vref, 1, 16, &ap[i])
                    .unwrap();
                assert_eq!(
                    stream.clone_dtoh(&kc[i]).unwrap(),
                    stream.clone_dtoh(&kref).unwrap()
                );
                assert_eq!(
                    stream.clone_dtoh(&vc[i]).unwrap(),
                    stream.clone_dtoh(&vref).unwrap()
                );
            }
            // New device positions must refresh the coefficients, including
            // what a graph's first-layer preparation reads on replay.
            stream.memcpy_htod(&[113i32], &mut dp[0]).unwrap();
            // SAFETY: updating a scalar keeps every pointer slot live.
            unsafe {
                rope.prepare_raw(&stream, &pointers, chunk, 0, theta)
                    .unwrap();
            }
            rope.apply(&stream, &dq, &mut qr, qh, 0).unwrap();
            let refreshed = stream.clone_dtoh(&qr).unwrap();
            let updated: Vec<_> = query[..qh * dim]
                .chunks_exact(dim)
                .flat_map(|row| llmcuda_kernels::rope::apply_rope(row, 113, rd as u32, theta))
                .collect();
            assert_matches(
                &refreshed[..qh * dim],
                &updated,
                &Tolerance {
                    max_abs_error: 1e-5,
                    max_rel_error: 0.01,
                    min_cosine_similarity: 1.0 - 1e-6,
                    allow_non_finite: false,
                },
            );
        }
    }
}
