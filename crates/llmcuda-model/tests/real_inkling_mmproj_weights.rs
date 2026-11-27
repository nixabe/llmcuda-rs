//! Resolve the `inkling2v` weight schema against the real mmproj file.
//!
//! [`InklingVisionWeightSchema`] derives every expected tensor name and
//! dimension from [`InklingVisionConfig`] alone; this closes the loop
//! against the 610 MB `mmproj-F16.gguf` that ships beside Inkling 2 Mini —
//! every tensor present, every shape as predicted, nothing left over.
//!
//! The path defaults to the in-tree model directory and is overridden by
//! `LLMCUDA_INKLING_MMPROJ`. If the file is absent the tests SKIP with a
//! printed message; per `AGENTS.md` a skipped test is not evidence the
//! schema is right.

use std::path::PathBuf;

use llmcuda_gguf::{GgmlType, GgufFile};
use llmcuda_model::inkling_vision::{
    InklingVisionConfig, InklingVisionRole, InklingVisionWeightSchema,
};
use llmcuda_model::{ModelConfig, VisionConfig};

const DEFAULT_MMPROJ_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/Inkling-2-Mini-GGUF/mmproj-F16.gguf"
);

fn open() -> Option<GgufFile> {
    let path = std::env::var_os("LLMCUDA_INKLING_MMPROJ")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MMPROJ_PATH));
    if !path.exists() {
        println!(
            "SKIPPED: mmproj file not found at {}; set LLMCUDA_INKLING_MMPROJ to override",
            path.display()
        );
        return None;
    }
    Some(GgufFile::open(&path).expect("mmproj file must parse as valid GGUF v3"))
}

/// Only the embedding width of the target is consulted.
fn target() -> ModelConfig {
    ModelConfig {
        hidden_size: 2304,
        ..ModelConfig::qwen3_6_35b_a3b()
    }
}

#[test]
fn metadata_loads_as_the_transcribed_config() {
    let Some(file) = open() else { return };
    let config = InklingVisionConfig::from_gguf(&file, &target()).expect("inkling2v metadata");
    assert_eq!(config, InklingVisionConfig::inkling2_mini());
}

#[test]
fn schema_resolves_with_no_mismatches_and_nothing_left_over() {
    let Some(file) = open() else { return };
    let config = InklingVisionConfig::from_gguf(&file, &target()).unwrap();
    let schema = InklingVisionWeightSchema::new(&config);

    if let Err(errors) = schema.resolve(&file) {
        for e in &errors {
            eprintln!("  {e}");
        }
        panic!("{} tensor(s) did not match the schema", errors.len());
    }
    assert_eq!(schema.unclaimed(&file), Vec::<&str>::new());
    assert_eq!(schema.specs().len(), file.n_tensors());
    println!("resolved {} inkling2v tensors", schema.specs().len());
}

#[test]
fn matrices_are_f16_and_vectors_f32() {
    // The reference loader widens both; anything else in a re-export (a
    // quantized projector, say) has to be noticed rather than misread.
    let Some(file) = open() else { return };
    let schema = InklingVisionWeightSchema::new(&InklingVisionConfig::inkling2_mini());
    for spec in schema.specs() {
        let info = file.tensor(&spec.name).unwrap();
        let expected = if spec.dims.len() >= 2 && spec.role != InklingVisionRole::PositionEmbed {
            GgmlType::F16
        } else {
            GgmlType::F32
        };
        assert_eq!(info.ggml_type, expected, "{}", spec.name);
    }
}

#[test]
fn the_file_is_not_a_qwen_tower_and_says_so() {
    let Some(file) = open() else { return };
    assert!(VisionConfig::from_gguf(&file, &target()).is_err());
    // Absent rather than false: llama.cpp fixes erf-GELU for this projector
    // type and never reads the key.
    assert!(file.get("clip.use_gelu").is_none());
    assert!(file.get("clip.vision.image_max_pixels").is_none());
    assert!(file.get("clip.vision.image_min_pixels").is_none());
}
