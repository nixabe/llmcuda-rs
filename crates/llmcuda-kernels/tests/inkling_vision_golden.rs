//! Cross-implementation check of the `inkling2v` reference against
//! llama.cpp's own execution of the same mmproj file.
//!
//! `tools/oracle/capture_inkling2v.cpp` drives llama.cpp's libmtmd on the
//! CPU over a fixed set of synthetic images — a checkerboard, a textured
//! gradient, noise; sizes that do and do not divide by 40; one shrunk under
//! a lowered token ceiling — and writes, per image, the input bytes, the
//! preprocessed graph input, intermediate graph tensors and the final
//! embeddings. These tests rerun the reference on the stored bytes and
//! compare:
//!
//! - the token count of a sweep of image sizes, exactly;
//! - the preprocessed image, bit for bit, including the resize path;
//! - every captured graph tensor under [`InklingVisionNumerics::LlamaCpp`],
//!   which rounds the pixels to half precision where llama.cpp's graph
//!   does — tight, so that a structural mistake cannot hide in rounding;
//! - the final embeddings under [`InklingVisionNumerics::Fp32`], which
//!   measures what that one rounding is worth at the output;
//! - the flash-attention capture under
//!   [`InklingVisionNumerics::LlamaCppFlashAttn`].
//!
//! Captured goldens are never committed (AGENTS.md). The tests read
//! `LLMCUDA_INKLING2V_GOLDEN_DIR` and SKIP without it, or without the
//! mmproj (`LLMCUDA_INKLING_MMPROJ`, default the in-tree model directory).
//! A skipped test is not a passing test. To regenerate:
//!
//! ```sh
//! cd tools/oracle
//! LD_LIBRARY_PATH=$HOME/cuda-12.8/lib make capture_inkling2v
//! mkdir -p $LLMCUDA_INKLING2V_GOLDEN_DIR
//! CUDA_VISIBLE_DEVICES="" LD_LIBRARY_PATH=$HOME/cuda-12.8/lib ./capture_inkling2v \
//!   <Inkling-2-Mini-*.gguf> <mmproj-F16.gguf> $LLMCUDA_INKLING2V_GOLDEN_DIR 8
//! ```
//!
//! The language model is opened vocabulary-only; about a minute on 8 cores
//! and 260 MB of output.
//!
//! # Thresholds
//!
//! Measured against llama.cpp ae6a7ad2c, CPU backend, on an SSE4.2-only
//! host (its CUDA backend was not run). The final embeddings span about ±2
//! to ±5.7; intermediate tensors are larger.
//! Cosine is computed in fp32, so `1 − cosine` below 6e-8 reads as zero.
//!
//! | image | tokens | `LlamaCpp`, worst tensor | `LlamaCpp`, output | `Fp32`, output |
//! | --- | --- | --- | --- | --- |
//! | checker 64×48 (328 tensors) | 4 | 4.3e-5 | 1.1e-6 | 1.7e-3 |
//! | gradient 333×207 | 54 | 1.9e-5 | 2.0e-6 | 2.2e-3 |
//! | noise 640×480 | 192 | 1.4e-5 | 3.3e-6 | 8.3e-4 |
//! | gradient 500×380, 20-token ceiling (resized) | 15 | 2.3e-5 | 1.9e-6 | 1.8e-3 |
//! | noise 97×611, 20-token ceiling (resized) | 11 | 4.8e-6 | 1.4e-6 | 4.1e-4 |
//!
//! Figures are max-abs. `1 − cosine` was at most 6e-8 in every `LlamaCpp`
//! comparison and at most 1.8e-6 for the `Fp32` output. Against the
//! flash-attention capture of the gradient, the output is within 2.1e-4
//! under `LlamaCppFlashAttn` and 2.2e-3 under `Fp32`.
//!
//! The bounds below are those figures with a margin of five to ten: fp32
//! summation order differs between this reference and ggml's SIMD, and
//! the margin is for that, not for a different graph.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use llmcuda_gguf::GgufFile;
use llmcuda_kernels::compare::compare;
use llmcuda_kernels::inkling_vision::{
    self, InklingVisionNumerics, InklingVisionWeights, PreprocessedInklingImage,
};
use llmcuda_model::InklingVisionConfig;

const DEFAULT_MMPROJ_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/Inkling-2-Mini-GGUF/mmproj-F16.gguf"
);

/// A bound on one comparison: largest absolute difference and smallest
/// cosine similarity.
#[derive(Clone, Copy)]
struct Bound {
    max_abs: f32,
    min_cosine: f32,
}

/// Any captured tensor, reference under `LlamaCpp` numerics, flash
/// attention off in the capture. Measured worst: 4.3e-5, on a normalized
/// MLP input.
const STRUCTURAL: Bound = Bound {
    max_abs: 3e-4,
    min_cosine: 1.0 - 1e-6,
};

/// Final embeddings under `LlamaCpp` numerics. Measured worst: 3.3e-6.
const STRUCTURAL_OUTPUT: Bound = Bound {
    max_abs: 3e-5,
    min_cosine: 1.0 - 1e-6,
};

/// Final embeddings under `Fp32` numerics, against any capture: the cost
/// of llama.cpp's F16 pixels. Measured worst: 2.2e-3, `1 − cosine` 1.8e-6.
const FP32_OUTPUT: Bound = Bound {
    max_abs: 1e-2,
    min_cosine: 1.0 - 1e-5,
};

/// Final embeddings of the flash-attention capture, reference under
/// `LlamaCppFlashAttn` numerics. Measured: 2.1e-4. Looser than
/// [`STRUCTURAL_OUTPUT`] because the reference only rounds the keys and
/// values; ggml's flash-attention kernel also orders its sums differently.
const FLASH_ATTN_OUTPUT: Bound = Bound {
    max_abs: 2e-3,
    min_cosine: 1.0 - 1e-6,
};

fn golden_dir() -> Option<PathBuf> {
    let Some(dir) = std::env::var_os("LLMCUDA_INKLING2V_GOLDEN_DIR").map(PathBuf::from) else {
        println!(
            "SKIPPED: LLMCUDA_INKLING2V_GOLDEN_DIR not set; see module docs to generate goldens"
        );
        return None;
    };
    if !dir.is_dir() {
        println!("SKIPPED: golden directory {} does not exist", dir.display());
        return None;
    }
    Some(dir)
}

fn weights() -> Option<InklingVisionWeights> {
    let path = std::env::var_os("LLMCUDA_INKLING_MMPROJ")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MMPROJ_PATH));
    if !path.exists() {
        println!(
            "SKIPPED: mmproj not found at {}; set LLMCUDA_INKLING_MMPROJ",
            path.display()
        );
        return None;
    }
    let file = GgufFile::open(&path).expect("mmproj parses");
    let cfg = InklingVisionConfig::inkling2_mini();
    Some(InklingVisionWeights::from_gguf(&file, &cfg).expect("mmproj is an inkling2v tower"))
}

/// One capture: the `LLMCGOLD` container `capture_inkling2v` writes.
struct Golden {
    name: String,
    width: u32,
    height: u32,
    rgb: Vec<u8>,
    max_tokens: u32,
    flash_attn: bool,
    /// Graph tensors by node name, flat in ggml order (`ne[0]` fastest),
    /// which for these is row-major `[rows][channels]`.
    tensors: HashMap<String, Vec<f32>>,
}

fn read_golden(path: &Path) -> Golden {
    let bytes = std::fs::read(path).expect("golden file reads");
    assert_eq!(&bytes[..8], b"LLMCGOLD", "{}: bad magic", path.display());
    let u32_at = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    let u64_at = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
    assert_eq!(u32_at(8), 1, "container version");
    let n_records = u32_at(12);

    let mut at = 16;
    let mut ints: HashMap<String, (Vec<u64>, Vec<i32>)> = HashMap::new();
    let mut tensors = HashMap::new();
    for _ in 0..n_records {
        let name_len = u32_at(at) as usize;
        let name = String::from_utf8(bytes[at + 4..at + 4 + name_len].to_vec()).unwrap();
        at += 4 + name_len;
        let dtype = u32_at(at);
        let ne: Vec<u64> = (0..4).map(|i| u64_at(at + 4 + 8 * i)).collect();
        let n = u64_at(at + 36) as usize;
        at += 44;
        let payload = bytes[at..at + 4 * n].as_chunks::<4>().0;
        at += 4 * n;
        match dtype {
            0 => {
                let values = payload.iter().map(|c| f32::from_le_bytes(*c)).collect();
                tensors.insert(name, values);
            }
            1 => {
                let values = payload.iter().map(|c| i32::from_le_bytes(*c)).collect();
                ints.insert(name, (ne, values));
            }
            other => panic!("{}: record {name} has dtype {other}", path.display()),
        }
    }
    assert_eq!(at, bytes.len(), "{}: trailing bytes", path.display());

    let (ne, rgb) = &ints["image_rgb"];
    Golden {
        name: path.file_stem().unwrap().to_string_lossy().into_owned(),
        width: ne[1] as u32,
        height: ne[2] as u32,
        rgb: rgb.iter().map(|&v| v as u8).collect(),
        max_tokens: ints["image_max_tokens"].1[0] as u32,
        flash_attn: ints["flash_attn"].1[0] != 0,
        tensors,
    }
}

/// Every capture in `dir`, smallest first.
fn goldens(dir: &Path) -> Vec<Golden> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("golden directory lists")
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|e| e == "bin"))
        .collect();
    paths.sort_by_key(|path| std::fs::metadata(path).unwrap().len());
    assert!(!paths.is_empty(), "no captures in {}", dir.display());
    paths.iter().map(|path| read_golden(path)).collect()
}

fn preprocess(cfg: &InklingVisionConfig, g: &Golden) -> PreprocessedInklingImage {
    inkling_vision::preprocess(cfg, &g.rgb, g.width, g.height, g.max_tokens)
        .expect("golden image preprocesses")
}

#[test]
fn token_counts_match_llama_cpp_over_a_sweep_of_sizes() {
    let Some(dir) = golden_dir() else { return };
    let cfg = InklingVisionConfig::inkling2_mini();
    let sweep = std::fs::read_to_string(dir.join("sizes.txt")).expect("sizes.txt reads");
    let mut checked = 0;
    for line in sweep.lines() {
        let f: Vec<u64> = line
            .split_whitespace()
            .map(|t| t.parse().unwrap())
            .collect();
        let (w, h, max, tokens) = (f[0] as u32, f[1] as u32, f[2] as u32, f[5] as usize);
        match inkling_vision::token_count(&cfg, w, h, max) {
            Ok(got) => assert_eq!(got, tokens, "{w}x{h} under {max} tokens"),
            // llama.cpp refuses the image; so must the reference.
            Err(e) => assert_eq!(tokens, 0, "{w}x{h} under {max} tokens: {e}"),
        }
        checked += 1;
    }
    assert!(checked > 0, "sizes.txt is empty");
    println!("{checked} image sizes agree with llama.cpp on the token count");
}

#[test]
fn preprocessing_matches_llama_cpp_bit_for_bit() {
    let Some(dir) = golden_dir() else { return };
    let cfg = InklingVisionConfig::inkling2_mini();
    for g in goldens(&dir) {
        let image = preprocess(&cfg, &g);
        let plan = inkling_vision::plan(&cfg, g.width, g.height, g.max_tokens).unwrap();
        let ours = image.planar(&cfg);
        let theirs = &g.tensors["inp_raw"];
        assert_eq!(ours.len(), theirs.len(), "{}: canvas size", g.name);
        let differing = ours
            .iter()
            .zip(theirs)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!(
            "{}: {}x{} -> canvas {}x{}, resize {:?}: {differing} of {} values differ",
            g.name,
            g.width,
            g.height,
            plan.width,
            plan.height,
            plan.resize,
            ours.len()
        );
        assert_eq!(differing, 0, "{}: preprocessed pixels differ", g.name);
    }
}

/// Compare `ours` with `theirs` and report; returns whether it is in bound.
fn within(label: &str, ours: &[f32], theirs: &[f32], bound: Bound) -> bool {
    assert_eq!(ours.len(), theirs.len(), "{label}: size");
    let r = compare(ours, theirs);
    let scale = theirs.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let ok = r.non_finite_count == 0
        && r.max_abs_error <= bound.max_abs
        && r.cosine_similarity >= bound.min_cosine;
    println!(
        "{label}: max-abs {:.3e} (largest value {scale:.3e}), 1-cos {:.3e}{}",
        r.max_abs_error,
        1.0 - f64::from(r.cosine_similarity),
        if ok { "" } else { "  <-- OUT OF BOUND" }
    );
    ok
}

#[test]
fn every_captured_tensor_matches_under_llama_cpp_numerics() {
    let Some(dir) = golden_dir() else { return };
    let Some(w) = weights() else { return };
    let cfg = InklingVisionConfig::inkling2_mini();
    // Set to print every tensor's figures rather than the worst per image.
    let trace = std::env::var_os("LLMCUDA_INKLING2V_TRACE").is_some();
    let mut failures = Vec::new();
    for g in goldens(&dir).iter().filter(|g| !g.flash_attn) {
        let image = preprocess(&cfg, g);
        // Worst node per image, to keep the report readable.
        let mut worst: Option<(String, f32, f32)> = None;
        let mut compared = 0;
        let out = inkling_vision::encode_tapped(
            &cfg,
            &w,
            &image,
            InklingVisionNumerics::LlamaCpp,
            &mut |name, value| {
                let Some(theirs) = g.tensors.get(name) else {
                    return;
                };
                assert_eq!(value.len(), theirs.len(), "{}: {name} size", g.name);
                let r = compare(value, theirs);
                compared += 1;
                if trace {
                    println!(
                        "{} {name}: max-abs {:.3e}, 1-cos {:.3e}",
                        g.name,
                        r.max_abs_error,
                        1.0 - f64::from(r.cosine_similarity)
                    );
                }
                if r.non_finite_count != 0
                    || r.max_abs_error > STRUCTURAL.max_abs
                    || r.cosine_similarity < STRUCTURAL.min_cosine
                {
                    failures.push(format!(
                        "{} {name}: max-abs {:.3e}, cosine {}",
                        g.name, r.max_abs_error, r.cosine_similarity
                    ));
                }
                if worst.as_ref().is_none_or(|w| r.max_abs_error > w.1) {
                    worst = Some((name.to_string(), r.max_abs_error, r.cosine_similarity));
                }
            },
        );
        // Every node the capture holds must have been produced.
        assert_eq!(
            compared,
            g.tensors.len() - 2,
            "{}: the capture holds nodes the reference did not report",
            g.name
        );
        let (node, max_abs, cosine) = worst.unwrap();
        println!(
            "{}: {compared} tensors; worst {node}: max-abs {max_abs:.3e}, 1-cos {:.3e}",
            g.name,
            1.0 - f64::from(cosine)
        );
        if !within(
            &format!("{} output (LlamaCpp)", g.name),
            &out,
            &g.tensors["output"],
            STRUCTURAL_OUTPUT,
        ) {
            failures.push(format!("{} output", g.name));
        }
    }
    assert!(
        failures.is_empty(),
        "out of bound:\n{}",
        failures.join("\n")
    );
}

#[test]
fn final_embeddings_match_under_the_other_numerics() {
    let Some(dir) = golden_dir() else { return };
    let Some(w) = weights() else { return };
    let cfg = InklingVisionConfig::inkling2_mini();
    let mut failures = Vec::new();
    for g in goldens(&dir) {
        let image = preprocess(&cfg, &g);
        let mut check = |numerics: InklingVisionNumerics, bound: Bound| {
            let started = std::time::Instant::now();
            let out = inkling_vision::encode_tapped(&cfg, &w, &image, numerics, &mut |_, _| {});
            let elapsed = started.elapsed().as_secs_f64();
            assert_eq!(out.len(), image.tokens(&cfg) * cfg.projection_dim as usize);
            let label = format!(
                "{} output ({numerics:?}, {} tokens in {elapsed:.1} s)",
                g.name,
                image.tokens(&cfg)
            );
            if !within(&label, &out, &g.tensors["output"], bound) {
                failures.push(label);
            }
        };
        check(InklingVisionNumerics::Fp32, FP32_OUTPUT);
        if g.flash_attn {
            check(InklingVisionNumerics::LlamaCppFlashAttn, FLASH_ATTN_OUTPUT);
        }
    }
    assert!(
        failures.is_empty(),
        "out of bound:\n{}",
        failures.join("\n")
    );
}
