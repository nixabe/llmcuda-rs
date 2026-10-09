//! The Hugging Face preprocessing path against the implementations it ports:
//! transformers' own `smart_resize` and PyTorch's CPU uint8 antialiased
//! bicubic, both run by `tools/oracle/hf_image.py`.
//!
//! Captured fixtures are never committed (AGENTS.md), so they are read from
//! `LLMCUDA_HF_IMAGE_GOLDEN_DIR` at run time and the test SKIPS without it:
//!
//! ```sh
//! python tools/oracle/hf_image.py $LLMCUDA_HF_IMAGE_GOLDEN_DIR
//! ```
//!
//! Both are exact ports, so both are gated exactly: every size, every byte.

use std::path::PathBuf;

use llmcuda_kernels::vision::{resize_bicubic_aa, smart_resize_hf};
use llmcuda_model::VisionConfig;

fn golden_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("LLMCUDA_HF_IMAGE_GOLDEN_DIR").map(PathBuf::from);
    if dir.is_none() {
        println!("SKIPPED: set LLMCUDA_HF_IMAGE_GOLDEN_DIR (see tools/oracle/hf_image.py)");
    }
    dir
}

#[test]
fn smart_resize_matches_transformers() {
    let Some(dir) = golden_dir() else { return };
    let cfg = VisionConfig::qwen3_6_35b_a3b();
    let text = std::fs::read_to_string(dir.join("smart_resize.txt")).expect("smart_resize.txt");
    let (mut cases, mut errors) = (0, 0);
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        let n = |i: usize| f[i].parse::<u64>().expect("integer field");
        let (w, h, lo, hi) = (n(0) as u32, n(1) as u32, n(2), n(3));
        let got = smart_resize_hf(&cfg, w, h, lo, hi);
        if f[4] == "error" {
            assert!(
                got.is_err(),
                "{line}: expected the aspect-ratio error, got {got:?}"
            );
            errors += 1;
        } else {
            assert_eq!(got, Ok((n(4) as u32, n(5) as u32)), "{line}");
        }
        cases += 1;
    }
    println!("smart_resize: {cases} cases, {errors} refused, all equal");
    assert!(
        cases > 1000 && errors > 0,
        "the fixture is not the generator's"
    );
}

#[test]
fn resize_matches_pytorch_bit_for_bit() {
    let Some(dir) = golden_dir() else { return };
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("fixture directory")
        .map(|e| e.expect("entry").path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("resize-") && n.ends_with(".bin"))
        })
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no resize-*.bin in {}", dir.display());
    let mut values = 0usize;
    for path in &files {
        let bytes = std::fs::read(path).expect("fixture");
        let dim = |i: usize| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
        let (w, h, tw, th) = (dim(0), dim(1), dim(2), dim(3));
        let src_len = (w * h * 3) as usize;
        let (src, want) = bytes[16..].split_at(src_len);
        assert_eq!(want.len(), (tw * th * 3) as usize, "{}", path.display());
        let got = resize_bicubic_aa(src, w, h, tw, th);
        let wrong = got.iter().zip(want).filter(|(a, b)| a != b).count();
        assert_eq!(
            wrong,
            0,
            "{}: {w}x{h} -> {tw}x{th}, {wrong} of {} values differ",
            path.display(),
            want.len()
        );
        values += want.len();
    }
    println!("resize: {} cases, {values} values, all equal", files.len());
}
