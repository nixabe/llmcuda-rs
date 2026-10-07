//! A prompt piece run as the prefix of a wider pass (`Forward::run_prefix`)
//! must be the pass of its own width, row for row.
//!
//! The prefix run sizes every launch to the piece and hands each block
//! exact-length views of buffers built for more rows, so nothing about the
//! arithmetic should differ from a pass built for the piece: same kernels,
//! same plans, same contraction orders. The gate is therefore bit-identity of
//! the final hidden states (and of the last row's logits, for a model that
//! computes them) — anything looser would hide a view that reads a stale row
//! or a launch still sized to the buffer.
//!
//! Shapes: a cold piece that fills no tile evenly (877 of 1,024), a piece
//! continued by a second one through the same state (300 then 577 — the
//! second attends to the first's keys and folds into its recurrent state),
//! the smallest piece that is still on the tensor cores (`MMA_SPLIT_TOKENS`),
//! and the full width.
//!
//! Below the tile the claim is different and so is the gate: a pass built
//! five rows wide takes the small-batch GEMVs, the wide pass the tensor
//! cores, so the two are the same model at different roundings and are held
//! to a cosine, not to the bit. The runtime never makes that trade — it
//! decomposes such pieces — but the arithmetic is recorded here.
//!
//! Needs a dense model (`qwen35` or `clef`): `LLMCUDA_MODEL`, defaulting to
//! Clef-Flash. SKIPS — reporting that it skipped — without a driver, a
//! supported device, or the file.

use std::path::PathBuf;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use llmcuda_cuda::device::{DeviceInfo, driver_available};
use llmcuda_engine::DeviceWeights;
use llmcuda_engine::forward::{Forward, ForwardError, arena_holds_model_entry};
use llmcuda_gguf::GgufFile;
use llmcuda_model::config::ModelConfig;
use llmcuda_model::weights::WeightSchema;

const DEFAULT_MODEL_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/Clef-Flash-GGUF/Clef-Flash-Q8_0.gguf"
);
/// The wide pass every piece runs as a prefix of.
const WIDTH: usize = 1024;

fn setup() -> Option<(Arc<CudaContext>, GgufFile)> {
    if !driver_available() {
        println!("SKIPPED: no CUDA driver present");
        return None;
    }
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(error) => {
            println!("SKIPPED: could not create a context on device 0: {error}");
            return None;
        }
    };
    let info = DeviceInfo::from_context(0, &ctx).expect("device properties readable");
    if !info.is_supported() {
        println!("SKIPPED: device 0 is below the sm_75 minimum");
        return None;
    }
    let path = std::env::var_os("LLMCUDA_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH));
    if !path.exists() {
        println!(
            "SKIPPED: model file not found at {}; set LLMCUDA_MODEL to override",
            path.display()
        );
        return None;
    }
    Some((ctx, GgufFile::open(path).expect("valid GGUF v3")))
}

fn read(stream: &Arc<CudaStream>, data: &CudaSlice<f32>) -> Vec<f32> {
    let host = stream.clone_dtoh(data).expect("device read-back");
    stream.synchronize().expect("read-back completes");
    host
}

fn prompt(len: usize, vocab: usize) -> Vec<i32> {
    (0..len)
        .map(|i| ((i * 7_919 + 1_234) % vocab) as i32)
        .collect()
}

/// Below the tensor-core tile: the two roundings of one model.
const MIN_COSINE_ACROSS_KERNELS: f64 = 0.9999;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in a.iter().zip(b) {
        dot += f64::from(a) * f64::from(b);
        aa += f64::from(a) * f64::from(a);
        bb += f64::from(b) * f64::from(b);
    }
    dot / (aa.sqrt() * bb.sqrt())
}

/// Bitwise-different elements and the largest difference.
fn compare(a: &[f32], b: &[f32]) -> (usize, f32) {
    a.iter().zip(b).fold((0, 0.0f32), |(n, m), (x, y)| {
        (
            n + usize::from(x.to_bits() != y.to_bits()),
            m.max((x - y).abs()),
        )
    })
}

#[test]
fn a_prefix_run_is_the_pass_of_its_own_width() {
    let Some((ctx, file)) = setup() else {
        return;
    };
    let config = ModelConfig::from_gguf(&file).expect("a supported architecture");
    if config.dense_ffn().is_none() {
        println!("SKIPPED: {} is not a dense model", config.architecture);
        return;
    }
    let stream = ctx.default_stream();
    let schema = WeightSchema::new(&config);
    let directory = schema.resolve(&file).expect("schema resolves");
    let (weights, _) =
        DeviceWeights::load_where_entry(&ctx, &stream, &file, &directory, |role, ty| {
            arena_holds_model_entry(&config, role, ty)
        })
        .expect("weights load");
    let hidden = config.hidden_size as usize;
    let vocab = config.vocab_size as usize;
    let logits = config.decision.is_none();

    let mut wide = Forward::new(
        &ctx,
        &stream,
        &file,
        &directory,
        &weights,
        config.clone(),
        WIDTH,
    )
    .expect("wide pass");

    let tile = llmcuda_cuda::kernels::mma::MMA_SPLIT_TOKENS;
    for pieces in [&[877][..], &[300, 577], &[tile], &[WIDTH], &[5]] {
        let bitwise = pieces.iter().all(|&n| n >= tile);
        let total: usize = pieces.iter().sum();
        let ids = prompt(total, vocab);
        let mut wide_state = wide.new_state(&stream, WIDTH).expect("state");
        let mut exact_state = wide.new_state(&stream, WIDTH).expect("state");
        let mut start = 0;
        for &n in pieces {
            let piece = &ids[start..start + n];
            wide.run_prefix(&stream, &mut wide_state, piece)
                .expect("prefix run");
            let got = read(&stream, wide.final_norm());
            let got_logits = logits.then(|| read(&stream, wide.logits()));

            let mut exact = wide
                .reshape(&ctx, &stream, &file, &directory, &weights, n)
                .expect("exact pass");
            exact
                .run(&stream, &mut exact_state, piece, |_, _| {})
                .expect("exact run");
            let want = read(&stream, exact.final_norm());

            let (got, want) = (&got[..n * hidden], &want[..n * hidden]);
            let (differ, max_abs) = compare(got, want);
            let cos = cosine(got, want);
            println!(
                "pieces {pieces:?}: rows [{start}, {}) of a {WIDTH}-wide pass: \
                 {differ} of {} final-norm values differ, max_abs {max_abs:.3e}, \
                 1-cos {:.3e}",
                start + n,
                n * hidden,
                1.0 - cos
            );
            if bitwise {
                assert_eq!(differ, 0, "final norm, piece at {start} of {pieces:?}");
            } else {
                assert!(cos >= MIN_COSINE_ACROSS_KERNELS, "final norm cosine {cos}");
            }
            if let Some(got_logits) = got_logits {
                let want_logits = read(&stream, exact.logits());
                let (differ, max_abs) = compare(&got_logits, &want_logits);
                println!("    last-row logits: {differ} differ, max_abs {max_abs:.3e}");
                if bitwise {
                    assert_eq!(differ, 0, "logits, piece at {start} of {pieces:?}");
                }
            }
            assert_eq!(wide_state.position(), start + n);
            start += n;
        }
    }
}

#[test]
fn a_prefix_run_refuses_what_it_does_not_cover() {
    let Some((ctx, file)) = setup() else {
        return;
    };
    let config = ModelConfig::from_gguf(&file).expect("a supported architecture");
    if config.dense_ffn().is_none() {
        println!("SKIPPED: {} is not a dense model", config.architecture);
        return;
    }
    let stream = ctx.default_stream();
    let schema = WeightSchema::new(&config);
    let directory = schema.resolve(&file).expect("schema resolves");
    let (weights, _) =
        DeviceWeights::load_where_entry(&ctx, &stream, &file, &directory, |role, ty| {
            arena_holds_model_entry(&config, role, ty)
        })
        .expect("weights load");
    let mut pass =
        Forward::new(&ctx, &stream, &file, &directory, &weights, config, 64).expect("pass");
    let mut state = pass.new_state(&stream, 128).expect("state");
    for len in [0, 65] {
        let ids = vec![1; len];
        assert!(matches!(
            pass.run_prefix(&stream, &mut state, &ids),
            Err(ForwardError::WrongTokenCount { expected: 64, got }) if got == len
        ));
    }
    assert_eq!(state.position(), 0, "a refused run advances nothing");
}
