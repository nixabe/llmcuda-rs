//! The Rust layer ops against the CUDA C++ kernels they replace, bit for bit.
//!
//! `layer_ops_differential` holds each kernel to its CPU reference, most
//! within a tolerance. That is not enough to swap a kernel's language: the
//! fused Gated DeltaNet paths are compared to these kernels with no tolerance
//! at all, so a port that reduces in a different order or fuses a different
//! multiply-add is a different kernel even where it is as close to the
//! reference. Here every launcher runs once from
//! [`LayerOpsKernels::new`] and once from [`LayerOpsKernels::cuda_cpp`] on the
//! same inputs, and the outputs must be the same bits.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, DevicePtr};
use llmcuda_cuda::device::{DeviceInfo, driver_available};
use llmcuda_cuda::kernels::layer_ops::{GateShape, LayerOpsKernels};
use llmcuda_kernels::rng::Xorshift64Star;

const EPS: f32 = 1e-6;

struct Arms {
    stream: Arc<CudaStream>,
    rust: LayerOpsKernels,
    cpp: LayerOpsKernels,
}

fn setup() -> Option<Arms> {
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
    Some(Arms {
        stream: ctx.default_stream(),
        rust: LayerOpsKernels::new(&ctx).expect("kernels must load"),
        cpp: LayerOpsKernels::cuda_cpp(&ctx).expect("kernels must compile"),
    })
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Activations with the magnitudes a forward pass carries, plus the values a
/// nonlinearity treats specially.
fn activations(rng: &mut Xorshift64Star, n: usize) -> Vec<f32> {
    let mut v = rng.vec_f32(n, -6.0, 6.0);
    for (i, special) in [0.0, -0.0, 20.0, 20.000002, 89.0, -89.0, 1e-30, -1e-30]
        .into_iter()
        .enumerate()
    {
        if let Some(slot) = v.get_mut(i * 37) {
            *slot = special;
        }
    }
    v
}

#[test]
fn rms_norm_is_bit_identical_at_every_width() {
    let Some(arms) = setup() else { return };
    let s = &arms.stream;
    let mut rng = Xorshift64Star::new(0x0001_C0DE);
    // head_dim, hidden, a non-multiple of 32, and a row wider than one block.
    for (rows, width) in [(7, 128), (5, 256), (3, 2048), (4, 100), (2, 5120), (1, 32)] {
        let x = activations(&mut rng, rows * width);
        let w = rng.vec_f32(width, 0.2, 1.8);
        let dx = s.clone_htod(&x).expect("upload");
        let dw = s.clone_htod(&w).expect("upload");
        let out = [&arms.rust, &arms.cpp].map(|k| {
            let mut o = s.alloc_zeros::<f32>(rows * width).expect("alloc");
            k.rms_norm(s, &dx, &dw, &mut o, rows, width, EPS)
                .expect("launch");
            bits(&s.clone_dtoh(&o).expect("read"))
        });
        assert_eq!(out[0], out[1], "rms_norm differs at {rows}x{width}");
    }
    println!("rms_norm: Rust is bit-identical to CUDA C++ at 6 shapes");
}

#[test]
fn fused_rms_norm_swiglu_is_bit_identical_in_both_forms() {
    let Some(arms) = setup() else { return };
    let s = &arms.stream;
    let mut rng = Xorshift64Star::new(0x0002_C0DE);
    for (rows, width) in [(96, 128), (17, 256), (3, 2048), (4, 100), (6, 1024)] {
        let n = rows * width;
        let x = activations(&mut rng, n);
        let g = activations(&mut rng, n);
        let w = rng.vec_f32(width, 0.2, 1.8);
        let dx = s.clone_htod(&x).expect("upload");
        let dg = s.clone_htod(&g).expect("upload");
        let dw = s.clone_htod(&w).expect("upload");

        let float = [&arms.rust, &arms.cpp].map(|k| {
            let mut normed = s.alloc_zeros::<f32>(n).expect("alloc");
            let mut out = s.alloc_zeros::<f32>(n).expect("alloc");
            k.rms_norm_swiglu(s, &dx, &dw, &dg, &mut normed, &mut out, rows, width, EPS)
                .expect("launch");
            (
                bits(&s.clone_dtoh(&normed).expect("read")),
                bits(&s.clone_dtoh(&out).expect("read")),
            )
        });
        assert_eq!(
            float[0], float[1],
            "rms_norm_swiglu differs at {rows}x{width}"
        );

        let quant = [&arms.rust, &arms.cpp].map(|k| {
            let mut q = s.alloc_zeros::<i8>(n).expect("alloc");
            let mut scales = s.alloc_zeros::<f32>(n.div_ceil(32)).expect("alloc");
            let ran = k
                .rms_norm_swiglu_quantize(s, &dx, &dw, &dg, &mut q, &mut scales, rows, width, EPS)
                .expect("launch");
            (
                ran,
                s.clone_dtoh(&q).expect("read"),
                bits(&s.clone_dtoh(&scales).expect("read")),
            )
        });
        assert_eq!(
            quant[0], quant[1],
            "quantized form differs at {rows}x{width}"
        );
        assert_eq!(quant[0].0, width.is_multiple_of(32) && width <= 1024);
    }
    println!("rms_norm_swiglu (fp32 and q8): Rust is bit-identical to CUDA C++ at 5 shapes");
}

#[test]
fn elementwise_ops_are_bit_identical() {
    let Some(arms) = setup() else { return };
    let s = &arms.stream;
    let mut rng = Xorshift64Star::new(0x0003_C0DE);
    // One token, a decode batch, and past the 1024-block grid cap.
    for (rows, width) in [(1, 2048), (3, 4096), (160, 2048)] {
        let n = rows * width;
        let a = activations(&mut rng, n);
        let b = activations(&mut rng, n);
        let da = s.clone_htod(&a).expect("upload");
        let db = s.clone_htod(&b).expect("upload");
        let out = [&arms.rust, &arms.cpp].map(|k| {
            let mut add = s.alloc_zeros::<f32>(n).expect("alloc");
            let mut soft = s.alloc_zeros::<f32>(n).expect("alloc");
            let mut sig = s.alloc_zeros::<f32>(n).expect("alloc");
            let mut gated = s.alloc_zeros::<f32>(n).expect("alloc");
            k.add(s, &da, &db, &mut add, n).expect("launch");
            k.softplus(s, &da, &mut soft, n).expect("launch");
            k.sigmoid_gate(
                s,
                &da,
                &db,
                &mut sig,
                &mut gated,
                rows,
                width,
                GateShape::Elementwise,
            )
            .expect("launch");
            [add, soft, sig, gated].map(|d| bits(&s.clone_dtoh(&d).expect("read")))
        });
        for (i, name) in ["add", "softplus", "sigmoid", "sigmoid gate"]
            .iter()
            .enumerate()
        {
            assert_eq!(out[0][i], out[1][i], "{name} differs at {rows}x{width}");
        }
    }
    println!("add, softplus, elementwise sigmoid gate: Rust is bit-identical to CUDA C++");
}

#[test]
fn partial_rope_is_bit_identical() {
    let Some(arms) = setup() else { return };
    let s = &arms.stream;
    let mut rng = Xorshift64Star::new(0x0004_C0DE);
    for (seq, heads, head_dim, rope_dim) in [(9, 16, 256, 64), (5, 2, 256, 256), (33, 4, 128, 32)] {
        let n = seq * heads * head_dim;
        let x = activations(&mut rng, n);
        let positions: Vec<u32> = (0..seq as u32).map(|t| t * 4099 + 131_000).collect();
        let dx = s.clone_htod(&x).expect("upload");
        let dp = s.clone_htod(&positions).expect("upload");
        let out = [&arms.rust, &arms.cpp].map(|k| {
            let mut o = s.alloc_zeros::<f32>(n).expect("alloc");
            k.rope(
                s,
                &dx,
                &dp,
                &mut o,
                seq,
                heads,
                head_dim,
                rope_dim,
                10_000_000.0,
            )
            .expect("launch");
            bits(&s.clone_dtoh(&o).expect("read"))
        });
        assert_eq!(
            out[0], out[1],
            "rope differs at {seq}x{heads}x{head_dim}/{rope_dim}"
        );
    }
    println!("rope_partial: Rust is bit-identical to CUDA C++ at 3 shapes");
}

#[test]
fn convolution_is_bit_identical_batched_stepped_and_across_sequences() {
    let Some(arms) = setup() else { return };
    let s = &arms.stream;
    let mut rng = Xorshift64Star::new(0x0005_C0DE);
    for (channels, conv_kernel) in [(6144, 4), (300, 8), (257, 1), (64, 2)] {
        let carry = conv_kernel - 1;
        let w = rng.vec_f32(channels * conv_kernel, -0.6, 0.6);
        let state0 = rng.vec_f32(channels * carry, -2.0, 2.0);
        let dw = s.clone_htod(&w).expect("upload");

        // A batch, then single-token steps through the same cache, then a
        // cache advance on its own: the three ways the model drives it.
        for seq_len in [1usize, 2, 3, 40] {
            let x = activations(&mut rng, seq_len * channels);
            let dx = s.clone_htod(&x).expect("upload");
            let out = [&arms.rust, &arms.cpp].map(|k| {
                let mut state = s.clone_htod(&state0).expect("upload");
                let mut advanced = s.clone_htod(&state0).expect("upload");
                let mut o = s.alloc_zeros::<f32>(seq_len * channels).expect("alloc");
                k.conv1d(
                    s,
                    &dx,
                    &dw,
                    &mut state,
                    &mut o,
                    seq_len,
                    channels,
                    conv_kernel,
                )
                .expect("launch");
                k.conv1d_advance_state(s, &dx, &mut advanced, seq_len, channels, conv_kernel)
                    .expect("launch");
                [o, state, advanced].map(|d| bits(&s.clone_dtoh(&d).expect("read")))
            });
            assert_eq!(
                out[0], out[1],
                "conv1d differs at {seq_len}x{channels} k={conv_kernel}"
            );
        }

        // The batched decode step: one grid slice and one cache per sequence.
        for n in [1usize, 3, 8] {
            let x = activations(&mut rng, n * channels);
            let dx = s.clone_htod(&x).expect("upload");
            let out = [&arms.rust, &arms.cpp].map(|k| {
                let states: Vec<_> = (0..n)
                    .map(|z| {
                        let mut st = state0.clone();
                        st.iter_mut().for_each(|v| *v += z as f32);
                        s.clone_htod(&st).expect("upload")
                    })
                    .collect();
                let ptrs: Vec<u64> = states.iter().map(|d| d.device_ptr(s).0).collect();
                let mut o = s.alloc_zeros::<f32>(n * channels).expect("alloc");
                // SAFETY: `states` are `n` distinct live allocations of
                // `channels * carry` floats, held here for the whole call.
                unsafe {
                    k.conv1d_step_batch_raw(s, &dx, &dw, &ptrs, &mut o, channels, conv_kernel)
                }
                .expect("launch");
                s.synchronize().expect("sync");
                let mut all = vec![bits(&s.clone_dtoh(&o).expect("read"))];
                all.extend(states.iter().map(|d| bits(&s.clone_dtoh(d).expect("read"))));
                all
            });
            assert_eq!(
                out[0], out[1],
                "batched step differs at n={n} k={conv_kernel}"
            );
        }
    }
    println!(
        "conv1d (batch, step, batched step, cache advance): Rust is bit-identical to CUDA C++"
    );
}
