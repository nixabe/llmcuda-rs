//! Shared by the `inkling2_loop` tests: locating the model and its capture,
//! and reading the file as the CPU reference's weight source.
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use llmcuda_gguf::{GgmlType, GgufFile};
use llmcuda_kernels::inkling::{Hparams, Weights};
use llmcuda_kernels::quant;

#[path = "../golden.rs"]
pub mod golden;

use golden::Golden;

const DEFAULT_MODEL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_XL.gguf"
);
const DEFAULT_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../.golden/inkling2-mini-q4-golden.bin"
);

/// The model path: `LLMCUDA_INKLING_MODEL`, or the default beside the repo.
pub fn model_path() -> PathBuf {
    std::env::var_os("LLMCUDA_INKLING_MODEL").map_or_else(|| DEFAULT_MODEL.into(), PathBuf::from)
}

/// The model, or `None` with a `SKIPPED` line.
pub fn model() -> Option<GgufFile> {
    let path = model_path();
    if !path.exists() {
        println!("SKIPPED: no Inkling model at {}", path.display());
        return None;
    }
    Some(GgufFile::open(&path).expect("the model parses"))
}

/// The model and the capture taken from it, or `None` with a `SKIPPED` line.
pub fn model_and_golden() -> Option<(GgufFile, Golden)> {
    let file = model()?;
    let path = std::env::var_os("LLMCUDA_INKLING_GOLDEN")
        .map_or_else(|| PathBuf::from(DEFAULT_GOLDEN), PathBuf::from);
    if !path.exists() {
        println!("SKIPPED: no Inkling capture at {}", path.display());
        return None;
    }
    Some((file, Golden::load(path).expect("the capture loads")))
}

/// The reference's hyperparameters, from the file's own metadata.
pub fn hparams(file: &GgufFile) -> Hparams {
    let config = llmcuda_model::ModelConfig::from_gguf(file).expect("a supported architecture");
    Hparams::from_config(&config)
}

type Parts = HashMap<(String, usize), Arc<Vec<f32>>>;

/// The file as fp32 tensors, widened on demand and memoized.
pub struct FileWeights<'a> {
    file: &'a GgufFile,
    whole: Mutex<HashMap<String, Arc<Vec<f32>>>>,
    parts: Mutex<Parts>,
}

impl<'a> FileWeights<'a> {
    /// Read from `file`.
    pub fn new(file: &'a GgufFile) -> Self {
        Self {
            file,
            whole: Mutex::default(),
            parts: Mutex::default(),
        }
    }
}

fn widen(ty: GgmlType, bytes: &[u8]) -> Vec<f32> {
    match ty {
        GgmlType::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        GgmlType::F16 => quant::dequantize_row_f16(bytes),
        GgmlType::Bf16 => quant::dequantize_row_bf16(bytes),
        GgmlType::Q8_0 => quant::dequantize_row_q8_0(bytes).expect("whole q8_0 blocks"),
        GgmlType::Nvfp4 => quant::dequantize_row_nvfp4(bytes).expect("whole nvfp4 blocks"),
        other => panic!("the Inkling reference does not read {}", other.name()),
    }
}

impl Weights for FileWeights<'_> {
    fn tensor(&self, name: &str) -> Option<Arc<Vec<f32>>> {
        if let Some(hit) = self.whole.lock().unwrap().get(name) {
            return Some(hit.clone());
        }
        let info = self.file.tensor(name)?;
        let values = Arc::new(widen(info.ggml_type, self.file.tensor_bytes(name)?));
        self.whole
            .lock()
            .unwrap()
            .insert(name.to_owned(), values.clone());
        Some(values)
    }

    fn part(&self, name: &str, index: usize, len: usize) -> Arc<Vec<f32>> {
        let key = (name.to_owned(), index);
        let mut parts = self.parts.lock().unwrap();
        if let Some(hit) = parts.get(&key) {
            return hit.clone();
        }
        let info = self
            .file
            .tensor(name)
            .unwrap_or_else(|| panic!("missing tensor {name}"));
        let ty = info.ggml_type;
        let block = ty.block_size() as usize;
        assert_eq!(len % block, 0, "{name}: a part must be whole blocks");
        let bytes_per = len / block * ty.type_size() as usize;
        let bytes = self.file.tensor_bytes(name).expect("tensor data");
        let values = Arc::new(widen(
            ty,
            &bytes[index * bytes_per..(index + 1) * bytes_per],
        ));
        // Expert matrices are 10-20 MB each as fp32; keep a bounded number.
        if parts.len() >= 96 {
            parts.clear();
        }
        parts.insert(key, values.clone());
        values
    }
}
