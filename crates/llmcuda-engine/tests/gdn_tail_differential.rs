//! `rms_norm_swiglu_q8_rows` against the two launches it replaces.
//!
//! The two-launch form runs the Gated DeltaNet's output norm with the SwiGLU
//! (`rms_norm_swiglu_rows`), writing `final_output` as floats, and then
//! quantizes that for the output projection (`mma_quantize_rows_q8`).
//! The fused launch does both with the same arithmetic per element and writes
//! only the int8 operand and its scales, so those must be **bit-identical**
//! to the two-launch form's -- every quant and every scale.
//!
//! Synthetic inputs at the model's own head geometry; no model file. One case
//! plants rows of exact zeros, where the quantizer's scale takes its `1.0`
//! branch, and rows large enough to saturate the SiLU.

use std::sync::Arc;

use cudarc::driver::CudaContext;
use llmcuda_cuda::device::{DeviceInfo, driver_available};
use llmcuda_cuda::kernels::layer_ops::LayerOpsKernels;
use llmcuda_cuda::kernels::mma::MmaKernels;
use llmcuda_kernels::rng::Xorshift64Star;
use llmcuda_model::config::ModelConfig;

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

fn run_case(ctx: &Arc<CudaContext>, seed: u64, tokens: usize, plant: bool) {
    let stream = ctx.default_stream();
    let g = ModelConfig::qwen3_6_35b_a3b().gdn;
    let (head_dim, value_heads) = (g.head_dim as usize, g.value_heads as usize);
    let value_dim = head_dim * value_heads;
    let rows = tokens * value_heads;
    let n = rows * head_dim;
    let eps = 1e-6f32;

    let layer_ops = LayerOpsKernels::new(ctx).expect("layer ops compile");
    let mma = MmaKernels::new(ctx).expect("mma kernels compile");

    let mut rng = Xorshift64Star::new(seed);
    let mut core = rng.vec_f32(n, -3.0, 3.0);
    let mut z = rng.vec_f32(n, -6.0, 6.0);
    if plant {
        // A row of zeros: the norm is 0 / sqrt(eps), every output is zero and
        // each of its four scales takes the quantizer's `1.0` branch.
        core[..head_dim].fill(0.0);
        // A zero gate under a live row: silu(0) = 0, the same branch reached
        // through the other operand.
        z[head_dim..2 * head_dim].fill(0.0);
        // Saturated gates, both signs.
        z[2 * head_dim..3 * head_dim].fill(80.0);
        z[3 * head_dim..4 * head_dim].fill(-80.0);
    }
    let weight = rng.vec_f32(head_dim, 0.5, 1.5);
    let d_core = stream.clone_htod(&core).expect("upload core");
    let d_z = stream.clone_htod(&z).expect("upload z");
    let d_w = stream.clone_htod(&weight).expect("upload weight");

    // The two launches.
    let mut normed = stream.alloc_zeros::<f32>(n).expect("normed");
    let mut out = stream.alloc_zeros::<f32>(n).expect("out");
    layer_ops
        .rms_norm_swiglu(
            &stream,
            &d_core,
            &d_w,
            &d_z,
            &mut normed,
            &mut out,
            rows,
            head_dim,
            eps,
        )
        .expect("norm and swiglu");
    let mut q_ref = stream.alloc_zeros::<i8>(n).expect("q");
    let mut s_ref = stream.alloc_zeros::<f32>(n / 32).expect("scales");
    mma.quantize_rows(&stream, &out, &mut q_ref, &mut s_ref, tokens, value_dim)
        .expect("quantize");

    // The one launch, into buffers longer than it fills and pre-poisoned, so
    // an element it skips or a write past the end shows.
    let pad = 64usize;
    let mut q = stream.clone_htod(&vec![0x55i8; n + pad]).expect("q fused");
    let mut sc = stream
        .clone_htod(&vec![-7.0f32; n / 32 + pad])
        .expect("scales fused");
    let fused = layer_ops
        .rms_norm_swiglu_quantize(
            &stream, &d_core, &d_w, &d_z, &mut q, &mut sc, rows, head_dim, eps,
        )
        .expect("fused launch");
    assert!(fused, "head_dim {head_dim} is a whole number of 32-blocks");

    let q_ref = stream.clone_dtoh(&q_ref).expect("read q");
    let s_ref = stream.clone_dtoh(&s_ref).expect("read scales");
    let q = stream.clone_dtoh(&q).expect("read q fused");
    let sc = stream.clone_dtoh(&sc).expect("read scales fused");
    stream.synchronize().expect("sync");

    let q_wrong = q[..n].iter().zip(&q_ref).filter(|(a, b)| a != b).count();
    let s_wrong = sc[..n / 32]
        .iter()
        .zip(&s_ref)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        (q_wrong, s_wrong),
        (0, 0),
        "{tokens} tokens: {q_wrong} of {n} quants and {s_wrong} of {} scales differ",
        n / 32
    );
    assert!(
        q[n..].iter().all(|&v| v == 0x55),
        "quants written past the end"
    );
    assert!(
        sc[n / 32..].iter().all(|&v| v == -7.0),
        "scales written past the end"
    );
    assert!(
        q_ref.iter().any(|&v| v != 0),
        "the reference quantized nothing"
    );
    if plant {
        assert!(
            s_ref[..head_dim / 32].iter().all(|&v| v == 1.0),
            "the zero row did not take the unit-scale branch"
        );
    }
    println!(
        "{tokens} tokens: {n} quants and {} scales bit-identical",
        n / 32
    );
}

#[test]
fn the_fused_tail_quantizes_to_the_same_bits() {
    let Some(ctx) = setup() else { return };
    run_case(&ctx, 0x_C0DE_5EED_0201, 67, false);
}

#[test]
fn zero_rows_and_saturated_gates_quantize_to_the_same_bits() {
    let Some(ctx) = setup() else { return };
    run_case(&ctx, 0x_C0DE_5EED_0202, 5, true);
}

#[test]
fn a_width_that_is_not_whole_blocks_is_declined_not_launched() {
    let Some(ctx) = setup() else { return };
    let stream = ctx.default_stream();
    let layer_ops = LayerOpsKernels::new(&ctx).expect("layer ops compile");
    let x = stream.alloc_zeros::<f32>(48).expect("x");
    let w = stream.alloc_zeros::<f32>(48).expect("w");
    let mut q = stream.alloc_zeros::<i8>(48).expect("q");
    let mut sc = stream.alloc_zeros::<f32>(2).expect("scales");
    let launched = layer_ops
        .rms_norm_swiglu_quantize(&stream, &x, &w, &x, &mut q, &mut sc, 1, 48, 1e-6)
        .expect("declines cleanly");
    assert!(
        !launched,
        "a 48-wide row is not a whole number of 32-blocks"
    );
}
