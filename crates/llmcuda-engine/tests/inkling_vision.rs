//! The `inkling2v` device tower against its CPU reference, on the real
//! projector file.
//!
//! The reference (`llmcuda_kernels::inkling_vision`) is itself held to
//! llama.cpp's tensors by `llmcuda-kernels/tests/inkling_vision_golden.rs`;
//! this is the device's distance from it. The device keeps binary16
//! activations where the reference keeps fp32, so the bound is a rounding
//! bound, measured, not equality.
//!
//! Needs a device and the projector (`LLMCUDA_INKLING_MMPROJ`, default
//! `models/Inkling-2-Mini-GGUF/mmproj-F16.gguf`); SKIPS without either.

use std::path::PathBuf;

use cudarc::driver::CudaContext;
use llmcuda_cuda::device::driver_available;
use llmcuda_engine::inkling::vision::InklingVisionForward;
use llmcuda_gguf::GgufFile;
use llmcuda_kernels::compare::compare;
use llmcuda_kernels::inkling_vision::{InklingVisionWeights, encode, preprocess};
use llmcuda_model::InklingVisionConfig;

/// Bounds on the device rows against the reference rows of one image.
/// Measured on the three images here: cosine 0.999989 to 0.999991 and
/// max-abs 4.2e-3 to 5.0e-3, on rows whose elements reach 2.0 - binary16
/// activations against fp32 ones, as for the `qwen3vl_merger` tower.
const COSINE: f32 = 0.99995;
const MAX_ABS: f32 = 0.02;

fn mmproj() -> Option<GgufFile> {
    let path = std::env::var_os("LLMCUDA_INKLING_MMPROJ")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../models/Inkling-2-Mini-GGUF/mmproj-F16.gguf")
        });
    if !path.exists() {
        println!("SKIPPED: {} is not on this host", path.display());
        return None;
    }
    Some(GgufFile::open(&path).expect("the projector opens"))
}

/// A deterministic RGB image with structure at several scales.
fn image(width: u32, height: u32, seed: u32) -> Vec<u8> {
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let checker = ((x / 16 + y / 16 + seed) % 2) * 90;
            rgb.push(((x * 255 / width.max(1)) / 2 + checker) as u8);
            rgb.push(((y * 255 / height.max(1)) / 2 + checker) as u8);
            rgb.push((((x * 7 + y * 13 + seed * 31) % 251) / 2 + checker) as u8);
        }
    }
    rgb
}

#[test]
fn device_tower_matches_the_reference_on_the_real_projector() {
    let Some(file) = mmproj() else {
        return;
    };
    if !driver_available() {
        println!("SKIPPED: no CUDA driver present");
        return;
    }
    let cfg = InklingVisionConfig::inkling2_mini();
    let weights = InklingVisionWeights::from_gguf(&file, &cfg).expect("the weights load");
    let ctx = CudaContext::new(0).expect("device 0");
    let stream = ctx.default_stream();
    let mut tower =
        InklingVisionForward::new(&ctx, stream, &cfg, &weights, 1024).expect("the tower builds");

    // A size that pads on both edges, a tiny one (one 2x2 cell), and a
    // wide one that is already whole cells.
    for (width, height, seed) in [(333u32, 207u32, 1u32), (64, 48, 2), (480, 160, 3)] {
        let rgb = image(width, height, seed);
        let pre = preprocess(&cfg, &rgb, width, height, cfg.max_image_tokens)
            .expect("the image preprocesses");
        let want = encode(&cfg, &weights, &pre);
        let got = tower.encode_to_host(&pre).expect("the device encodes");
        assert_eq!(got.len(), want.len(), "{width}x{height}: row count");
        let r = compare(&got, &want);
        let peak = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        println!(
            "{width}x{height} -> {}x{} patches, {} rows: max_abs {:.3e} (of |row| <= {peak:.2}) cosine {:.6}",
            pre.grid_w,
            pre.grid_h,
            got.len() / cfg.projection_dim as usize,
            r.max_abs_error,
            r.cosine_similarity
        );
        assert!(
            r.cosine_similarity > COSINE,
            "{width}x{height}: cosine {}",
            r.cosine_similarity
        );
        assert!(
            r.max_abs_error < MAX_ABS,
            "{width}x{height}: max-abs {}",
            r.max_abs_error
        );
    }
}
