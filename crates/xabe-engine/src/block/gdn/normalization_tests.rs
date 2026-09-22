//! Exact fusion checks and composed CPU references for normalized gates.

use super::*;

#[test]
fn normalized_gate_consumer_matches_separate_kernels_and_cpu_references() {
    use xabe_cuda::device::{DeviceInfo, driver_available};
    use xabe_kernels::compare::{Tolerance, assert_matches};
    use xabe_kernels::gemv::gemv;
    use xabe_kernels::norm::{rms_norm, sigmoid, softplus};
    use xabe_kernels::rng::Xorshift64Star;

    if !driver_available() {
        println!("SKIPPED: no CUDA driver present");
        return;
    }
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(error) => {
            println!("SKIPPED: could not create CUDA context: {error}");
            return;
        }
    };
    if !DeviceInfo::from_context(0, &ctx).unwrap().is_supported() {
        println!("SKIPPED: device is below sm_75");
        return;
    }
    let stream = ctx.default_stream();
    let g = GdnGeometry::from_config(&ModelConfig::qwen3_6_35b_a3b(), 4, 1e-6);
    let block = GdnBlock::new(&ctx, g).unwrap();
    let mut rng = Xorshift64Star::new(0x_92EA_2026);
    let norm_weight = rng.vec_f32(g.hidden, 0.5, 1.5);
    let wa = rng.vec_f32(g.hidden * g.value_heads, -0.05, 0.05);
    let wb = rng.vec_f32(g.hidden * g.value_heads, -0.05, 0.05);
    let mut bias = rng.vec_f32(g.value_heads, -2.0, 2.0);
    bias[0] = 30.0;
    bias[1] = -30.0;
    let decay = rng.vec_f32(g.value_heads, -4.0, -0.1);
    let w = GdnLayerWeights {
        input_norm: stream.clone_htod(&norm_weight).unwrap(),
        qkv: None,
        gate: None,
        conv1d: stream.alloc_zeros(g.conv_kernel * g.conv_dim()).unwrap(),
        alpha: GateProjection::F32(stream.clone_htod(&wa).unwrap()),
        beta: GateProjection::F32(stream.clone_htod(&wb).unwrap()),
        dt_bias: stream.clone_htod(&bias).unwrap(),
        a: stream.clone_htod(&decay).unwrap(),
        ssm_norm: stream.alloc_zeros(g.head_dim).unwrap(),
        out: None,
        qkv_fmt: ProjQuant::Q8_0,
        gate_fmt: ProjQuant::Q8_0,
        out_fmt: ProjQuant::Q8_0,
    };
    // Existing anchored GDN elementwise bounds from tests/gdn_block.rs.
    // The separate GPU path below is additionally required to agree in bits.
    let gate = Tolerance {
        max_abs_error: 5e-5,
        max_rel_error: 1e-1,
        min_cosine_similarity: 1.0 - 1e-7,
        allow_non_finite: false,
    };
    // Existing RMSNorm bounds from tests/layer_ops_differential.rs.
    let norm_gate = Tolerance {
        max_abs_error: 1e-5,
        max_rel_error: 1e-4,
        min_cosine_similarity: 1.0 - 1e-7,
        allow_non_finite: false,
    };
    for tokens in 1..=4 {
        let mut input = rng.vec_f32(tokens * g.hidden, -2.0, 2.0);
        if tokens > 1 {
            input[g.hidden..2 * g.hidden].fill(0.0);
        }
        let hidden = stream.clone_htod(&input).unwrap();
        let mut separate = Scratch::new(&stream, &g, tokens).unwrap();
        let mut fused = Scratch::new(&stream, &g, tokens).unwrap();
        let run_gates = |s: &mut Scratch| {
            block
                .alpha_beta_gates(
                    &stream,
                    &w.alpha,
                    &w.beta,
                    &s.normed,
                    &w.dt_bias,
                    &w.a,
                    &mut s.alpha,
                    &mut s.beta_raw,
                    &mut s.a_softplus,
                    &mut s.log_decay,
                    &mut s.beta,
                    tokens,
                )
                .unwrap();
        };
        block
            .layer_ops
            .rms_norm(
                &stream,
                &hidden,
                &w.input_norm,
                &mut separate.normed,
                tokens,
                g.hidden,
                g.rms_eps,
            )
            .unwrap();
        run_gates(&mut separate);
        let ready = block
            .normalize_input(&stream, &hidden, &w, &mut fused, tokens)
            .unwrap();
        assert_eq!(ready, tokens <= 3, "exercise fusion and the wider fallback");
        if !ready {
            run_gates(&mut fused);
        }
        let mut cpu: [Vec<f32>; 6] = std::array::from_fn(|_| Vec::new());
        for row in input.chunks_exact(g.hidden) {
            let normalized = rms_norm(row, &norm_weight, g.rms_eps);
            let alpha = gemv(&wa, g.value_heads, g.hidden, &normalized);
            let beta = gemv(&wb, g.value_heads, g.hidden, &normalized);
            cpu[0].extend_from_slice(&normalized);
            cpu[1].extend_from_slice(&alpha);
            cpu[2].extend_from_slice(&beta);
            for head in 0..g.value_heads {
                let sp = softplus(alpha[head] + bias[head]);
                cpu[3].push(sp);
                cpu[4].push(sp * decay[head]);
                cpu[5].push(sigmoid(beta[head]));
            }
        }
        for (output, (a, b)) in [
            (&separate.normed, &fused.normed),
            (&separate.alpha, &fused.alpha),
            (&separate.beta_raw, &fused.beta_raw),
            (&separate.a_softplus, &fused.a_softplus),
            (&separate.log_decay, &fused.log_decay),
            (&separate.beta, &fused.beta),
        ]
        .into_iter()
        .enumerate()
        {
            let a = stream.clone_dtoh(a).unwrap();
            let b = stream.clone_dtoh(b).unwrap();
            stream.synchronize().unwrap();
            for (index, (av, bv)) in a.iter().zip(&b).enumerate() {
                assert_eq!(
                    av.to_bits(),
                    bv.to_bits(),
                    "tokens={tokens} output={output} index={index}"
                );
            }
            assert_matches(
                &b,
                &cpu[output],
                if output == 0 { &norm_gate } else { &gate },
            );
        }
    }
}
