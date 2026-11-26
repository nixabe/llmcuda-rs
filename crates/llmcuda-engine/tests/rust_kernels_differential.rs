//! The Rust sigmoid gate against the CUDA C++ one, bit for bit.
//!
//! Sigmoid gating is the one layer op that launches from both languages: the
//! elementwise shape from Rust, the per-row shape from CUDA C++. A per-row
//! gate over rows one element wide is an elementwise gate, so the two kernels
//! can be handed the same problem and must return the same bits.
//!
//! This file used to hold every Rust layer op to its CUDA C++ original the
//! same way. Those originals are deleted; what holds the Rust kernels now is
//! `layer_ops_differential` against the CPU references and the Gated
//! DeltaNet tests that compare them, exactly, with the fused CUDA C++ paths
//! (`gdn_tail_differential`, `gdn_conv_silu_differential`,
//! `gdn_verify_differential`, `gdn_block`).

use cudarc::driver::CudaContext;
use llmcuda_cuda::device::{DeviceInfo, driver_available};
use llmcuda_cuda::kernels::layer_ops::{GateShape, LayerOpsKernels};
use llmcuda_kernels::rng::Xorshift64Star;

#[test]
fn the_elementwise_sigmoid_gate_matches_the_per_row_kernel_at_width_one() {
    if !driver_available() {
        println!("SKIPPED: no CUDA driver present");
        return;
    }
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            println!("SKIPPED: could not create a context on device 0: {e}");
            return;
        }
    };
    let info = DeviceInfo::from_context(0, &ctx).expect("device properties readable");
    if !info.is_supported() {
        println!("SKIPPED: device 0 is below the sm_75 minimum");
        return;
    }
    if cfg!(feature = "rust-kernels") {
        println!("SKIPPED: `rust-kernels` launches both shapes from Rust");
        return;
    }
    let s = ctx.default_stream();
    let kernels = LayerOpsKernels::new(&ctx).expect("kernels must load");
    let mut rng = Xorshift64Star::new(0x0003_C0DE);
    // One token, a decode batch, and past the 1024-block grid cap.
    for n in [2048usize, 3 * 4096, 160 * 2048] {
        let x = rng.vec_f32(n, -6.0, 6.0);
        let mut gate = rng.vec_f32(n, -6.0, 6.0);
        for (slot, special) in gate
            .iter_mut()
            .step_by(37)
            .zip([0.0, -0.0, 89.0, -89.0, 1e-30, -1e-30, 104.0, -104.0])
        {
            *slot = special;
        }
        let dx = s.clone_htod(&x).expect("upload");
        let dg = s.clone_htod(&gate).expect("upload");
        let out = [(GateShape::Elementwise, 1, n), (GateShape::PerRow, n, 1)].map(
            |(shape, rows, width)| {
                let mut sig = s.alloc_zeros::<f32>(n).expect("alloc");
                let mut gated = s.alloc_zeros::<f32>(n).expect("alloc");
                kernels
                    .sigmoid_gate(&s, &dx, &dg, &mut sig, &mut gated, rows, width, shape)
                    .expect("launch");
                [sig, gated].map(|d| {
                    let v = s.clone_dtoh(&d).expect("read");
                    v.iter().map(|f| f.to_bits()).collect::<Vec<_>>()
                })
            },
        );
        assert_eq!(out[0][0], out[1][0], "sigmoid differs at n={n}");
        assert_eq!(out[0][1], out[1][1], "gated output differs at n={n}");
    }
    println!("sigmoid gate: Rust (elementwise) is bit-identical to CUDA C++ (per row, width 1)");
}
