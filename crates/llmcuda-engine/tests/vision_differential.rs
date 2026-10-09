//! Differential test: the device vision tower against the scalar
//! reference, at a small synthetic geometry and at the real mmproj.
//!
//! The reference (`llmcuda_kernels::vision::encode`) is itself validated
//! against llama.cpp executing the same mmproj file
//! (`llmcuda-kernels/tests/vision_golden.rs`), so agreement here chains the
//! device path back to upstream.
//!
//! ## Tolerances
//!
//! The device path stores every activation as f16 (matching llama.cpp's
//! CUDA clip path) while the reference is scalar f32, so this is *not* a
//! same-formula fp32 comparison: 27 blocks of f16 rounding compound.
//! The gates below were set from measurement — see each assertion — and
//! cosine similarity is the primary gate, per `docs/TESTING.md`, with
//! max-abs quoted against the observed value on the real weights.
//!
//! SKIPS without a driver, an sm_75 device, or (for the real-weights case)
//! the mmproj file.

use std::path::PathBuf;
use std::sync::Arc;

use cudarc::driver::CudaContext;
use llmcuda_engine::vision::{VisionForward, load_vision_weights};
use llmcuda_kernels::compare::compare;
use llmcuda_kernels::rng::Xorshift64Star;
use llmcuda_kernels::vision::tower::{VisionBlockWeights, VisionWeights, encode};
use llmcuda_kernels::vision::{PreprocessedImage, preprocess};
use llmcuda_model::VisionConfig;

const DEFAULT_MMPROJ_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/Qwen3.6-35B-A3B-GGUF/mmproj-F16.gguf"
);

fn setup() -> Option<Arc<CudaContext>> {
    if !llmcuda_cuda::device::driver_available() {
        println!("SKIPPED: no CUDA driver present");
        return None;
    }
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("SKIPPED: could not create a context on device 0: {e}");
            return None;
        }
    };
    match llmcuda_cuda::device::DeviceInfo::from_context(0, &ctx) {
        Ok(info) if info.is_supported() => Some(ctx),
        Ok(_) => {
            println!("SKIPPED: device 0 is below the sm_75 minimum");
            None
        }
        Err(e) => {
            println!("SKIPPED: could not probe device 0: {e}");
            None
        }
    }
}

fn tiny_cfg() -> VisionConfig {
    VisionConfig {
        num_layers: 3,
        hidden_size: 64,
        num_heads: 2,
        ffn_size: 128,
        image_size: 64, // 4x4 position grid
        patch_size: 16,
        temporal_patch_size: 2,
        spatial_merge: 2,
        projection_dim: 32,
        ln_eps: 1e-6,
        rope_theta: 10_000.0,
        image_mean: [0.5; 3],
        image_std: [0.5; 3],
    }
}

fn tiny_weights(cfg: &VisionConfig, seed: u64) -> VisionWeights {
    let mut rng = Xorshift64Star::new(seed);
    let h = cfg.hidden_size as usize;
    let ffn = cfg.ffn_size as usize;
    let patch = (cfg.patch_elems() / cfg.temporal_patch_size) as usize;
    let edge = cfg.pos_grid_edge() as usize;
    let merged = cfg.merger_input_dim() as usize;
    let proj = cfg.projection_dim as usize;
    let mut v = |n: usize| rng.vec_f32(n, -0.08, 0.08);
    VisionWeights {
        patch_embed: v(patch * h),
        patch_bias: v(h),
        pos_embed: v(edge * edge * h),
        blocks: (0..cfg.num_layers)
            .map(|_| VisionBlockWeights {
                ln1_w: v(h).iter().map(|x| 1.0 + x).collect(),
                ln1_b: v(h),
                qkv_w: v(h * 3 * h),
                qkv_b: v(3 * h),
                out_w: v(h * h),
                out_b: v(h),
                ln2_w: v(h).iter().map(|x| 1.0 + x).collect(),
                ln2_b: v(h),
                up_w: v(h * ffn),
                up_b: v(ffn),
                down_w: v(ffn * h),
                down_b: v(h),
            })
            .collect(),
        post_ln_w: v(h).iter().map(|x| 1.0 + x).collect(),
        post_ln_b: v(h),
        fc1_w: v(merged * merged),
        fc1_b: v(merged),
        fc2_w: v(merged * proj),
        fc2_b: v(proj),
    }
}

/// The shared qwen3vl tower with the file's one model-dependent width
/// (see `VisionConfig::for_model`): 2048 for Qwen3.6-35B-A3B, 4096 for
/// Clef-Flash.
fn file_cfg(file: &llmcuda_gguf::GgufFile) -> VisionConfig {
    VisionConfig {
        projection_dim: file
            .get_u32("clip.vision.projection_dim")
            .expect("mmproj declares its projection width"),
        ..VisionConfig::qwen3_6_35b_a3b()
    }
}

/// A 96x96 gradient image through the real preprocessing: 6x6 patch
/// grid, 9 output tokens.
fn gradient_image(cfg: &VisionConfig) -> PreprocessedImage {
    let mut rgb = vec![0u8; 96 * 96 * 3];
    for y in 0..96u32 {
        for x in 0..96u32 {
            let i = ((y * 96 + x) * 3) as usize;
            rgb[i] = (x * 255 / 95) as u8;
            rgb[i + 1] = (y * 255 / 95) as u8;
            rgb[i + 2] = ((x + y) * 255 / 190) as u8;
        }
    }
    let image = preprocess(cfg, &rgb, 96, 96);
    assert_eq!((image.grid_h, image.grid_w), (6, 6));
    image
}

#[test]
fn device_tower_matches_the_reference_at_a_synthetic_geometry() {
    let Some(ctx) = setup() else { return };
    let stream = ctx.default_stream();
    let cfg = tiny_cfg();
    let weights = tiny_weights(&cfg, 11);
    let mut fwd =
        VisionForward::new(&ctx, stream, &cfg, &weights, 64).expect("device tower builds");

    // A non-square grid exercises the position-embedding resize, the rope
    // row/col split, and the cell ordering asymmetrically.
    let patch_len = (cfg.patch_elems() / cfg.temporal_patch_size) as usize;
    let (gh, gw) = (4u32, 8u32);
    let mut rng = Xorshift64Star::new(23);
    let image = PreprocessedImage {
        patches: rng.vec_f32((gh * gw) as usize * patch_len, -1.0, 1.0),
        grid_h: gh,
        grid_w: gw,
    };

    let device = fwd.encode_to_host(&image).expect("device encode");
    let reference = encode(&cfg, &weights, &image.patches, gh, gw);

    let result = compare(&device, &reference);
    println!(
        "tiny: max_abs {:.3e}  max_rel {:.3e}  cosine {:.6}",
        result.max_abs_error, result.max_rel_error, result.cosine_similarity
    );
    // Measured on this geometry: max_abs 1.1e-3, cosine 1.000000. The
    // gates leave ~4x headroom without admitting a transposed weight or a
    // head-offset bug, both of which push cosine far below 0.99.
    assert!(result.cosine_similarity > 0.9999, "cosine degraded");
    assert!(result.max_abs_error < 8e-3, "max_abs degraded");
}

#[test]
fn device_tower_matches_the_reference_on_the_real_mmproj() {
    let Some(ctx) = setup() else { return };
    let path = std::env::var_os("LLMCUDA_MMPROJ")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MMPROJ_PATH));
    if !path.exists() {
        println!("SKIPPED: mmproj not found at {}", path.display());
        return;
    }
    let file = llmcuda_gguf::GgufFile::open(&path).expect("mmproj parses");
    let cfg = file_cfg(&file);
    let weights = load_vision_weights(&file, &cfg).expect("mmproj loads");

    let stream = ctx.default_stream();
    let mut fwd =
        VisionForward::new(&ctx, stream, &cfg, &weights, 256).expect("device tower builds");
    let image = gradient_image(&cfg);

    let device = fwd.encode_to_host(&image).expect("device encode");
    let reference = encode(&cfg, &weights, &image.patches, image.grid_h, image.grid_w);

    let result = compare(&device, &reference);
    println!(
        "real: max_abs {:.3e}  max_rel {:.3e}  cosine {:.6}",
        result.max_abs_error, result.max_rel_error, result.cosine_similarity
    );
    // 27 blocks of f16 activation rounding against scalar f32; measured
    // max_abs 1.02e-2, cosine 0.999990 (max_rel is uninformative here —
    // near-zero elements, the gdn_differential.rs trap). Gates leave ~5x
    // headroom on max_abs without admitting a transposed weight or a
    // head-offset bug, which push cosine far below 0.99.
    assert!(result.cosine_similarity > 0.999, "cosine degraded");
    assert!(result.max_abs_error < 5e-2, "max_abs degraded");
}

/// The tower's matrices by name — every tensor a Q8_0 export may quantize.
fn matrices(w: &VisionWeights) -> Vec<(String, &[f32])> {
    let mut out: Vec<(String, &[f32])> = w
        .blocks
        .iter()
        .enumerate()
        .flat_map(|(l, b)| {
            [
                ("attn_qkv", &b.qkv_w),
                ("attn_out", &b.out_w),
                ("ffn_up", &b.up_w),
                ("ffn_down", &b.down_w),
            ]
            .map(|(name, t)| (format!("v.blk.{l}.{name}"), t.as_slice()))
        })
        .collect();
    out.push(("mm.0".into(), &w.fc1_w));
    out.push(("mm.2".into(), &w.fc2_w));
    out
}

/// A Q8_0 mmproj against the f16 export of the same checkpoint.
///
/// `device_tower_matches_the_reference_on_the_real_mmproj` cannot see a
/// loader bug — both of its sides read `load_vision_weights`. Two exports
/// of one checkpoint can: each dequantized matrix must sit inside Q8_0's
/// own error bound of its f16 twin, and the towers built from each must
/// agree on an image. Measured on Clef-Flash's block-5 matrices, a clean
/// load sits at 0.48 steps; swapping two blocks of one row reads 30-38,
/// a transposed matrix ~130.
///
/// Needs both files: `LLMCUDA_MMPROJ_Q8_0` and `LLMCUDA_MMPROJ_F16` (the
/// export commands are in docs/MODEL.md's Clef section). The weight
/// bounds need no device; the tower comparison SKIPS without one.
#[test]
fn q8_0_mmproj_tracks_its_f16_export() {
    let paths = ["LLMCUDA_MMPROJ_Q8_0", "LLMCUDA_MMPROJ_F16"].map(std::env::var_os);
    let [Some(q8_path), Some(f16_path)] = paths else {
        println!("SKIPPED: set LLMCUDA_MMPROJ_Q8_0 and LLMCUDA_MMPROJ_F16");
        return;
    };
    let q8_file = llmcuda_gguf::GgufFile::open(&q8_path).expect("q8_0 mmproj parses");
    let f16_file = llmcuda_gguf::GgufFile::open(&f16_path).expect("f16 mmproj parses");
    let cfg = file_cfg(&q8_file);
    assert_eq!(
        cfg.projection_dim,
        file_cfg(&f16_file).projection_dim,
        "the two files are not exports of one tower"
    );
    let quantized = (0..cfg.num_layers)
        .flat_map(|l| {
            ["attn_qkv", "attn_out", "ffn_up", "ffn_down"].map(|t| format!("v.blk.{l}.{t}.weight"))
        })
        .chain(["mm.0.weight".into(), "mm.2.weight".into()])
        .filter(|n| q8_file.tensor(n).unwrap().ggml_type == llmcuda_gguf::GgmlType::Q8_0)
        .count();
    assert!(quantized > 0, "LLMCUDA_MMPROJ_Q8_0 holds no Q8_0 matrix");

    let q8 = load_vision_weights(&q8_file, &cfg).expect("q8_0 mmproj loads");
    let f16 = load_vision_weights(&f16_file, &cfg).expect("f16 mmproj loads");

    // Q8_0 rounds to the nearest step `d = amax_block / 127`, then stores
    // `d` as f16: |error| <= d/2 + 127 * 2^-11 * d < 0.57 d, and
    // d <= amax_tensor / 127. Bound per tensor at 0.6 * amax / 127.
    let mut worst_ratio = 0.0f32;
    let mut worst_cosine = 1.0f64;
    for ((name, a), (_, b)) in matrices(&q8).into_iter().zip(matrices(&f16)) {
        let amax = b.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let r = compare(a, b);
        let ratio = r.max_abs_error as f32 / (amax / 127.0);
        worst_ratio = worst_ratio.max(ratio);
        worst_cosine = worst_cosine.min(r.cosine_similarity as f64);
        assert!(
            ratio < 0.6,
            "{name}: max_abs {:.3e} is {ratio:.2} Q8_0 steps (amax {amax:.3e})",
            r.max_abs_error
        );
    }
    println!(
        "weights: {quantized} Q8_0 matrices; worst max_abs {worst_ratio:.3} steps, \
         worst cosine {worst_cosine:.7}"
    );

    let Some(ctx) = setup() else { return };
    let image = gradient_image(&cfg);
    let encode_with = |w: &VisionWeights| {
        VisionForward::new(&ctx, ctx.default_stream(), &cfg, w, 256)
            .expect("device tower builds")
            .encode_to_host(&image)
            .expect("device encode")
    };
    let (q8_out, f16_out) = (encode_with(&q8), encode_with(&f16));
    let r = compare(&q8_out, &f16_out);
    let amax = f16_out.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    println!(
        "tower q8_0 vs f16: max_abs {:.3e} (of |out| <= {amax:.2})  cosine {:.6}",
        r.max_abs_error, r.cosine_similarity
    );
    // What Q8_0 costs the embeddings, through 27 blocks; the per-matrix
    // bound above is the sharp layout check. Measured on Clef-Flash's two
    // exports: max_abs 1.17e-1 against |out| <= 18.7, cosine 0.999973.
    // Gates leave ~7x on 1 - cosine and ~4x on max_abs.
    assert!(r.cosine_similarity > 0.9998, "q8_0 tower diverged from f16");
    assert!(r.max_abs_error < 0.5, "q8_0 tower max_abs degraded");
}
