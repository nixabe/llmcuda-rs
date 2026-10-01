//! Device K2 primitives against scalar references. Includes Q8_0 as a control,
//! Q4_K/Q6_K selected projections, real-width grouped norms, bias-only routing,
//! value SiLU, softplus gating and dense fp32 router projections.
use cudarc::driver::CudaContext;
use llmcuda_cuda::{
    device::driver_available,
    kernels::{
        k2::K2Kernels,
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
