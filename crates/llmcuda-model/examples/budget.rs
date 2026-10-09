//! Prints the VRAM segmentation table and the per-context bandwidth
//! roofline table for the target model and hardware.
//!
//! `cargo run -p llmcuda-model --example budget`
//!
//! Every number here is either a closed-form derivation from
//! [`llmcuda_model::config::ModelConfig`] or an explicitly labeled estimate —
//! see `crates/llmcuda-model/src/budget.rs`'s module doc for the assumptions
//! and `docs/MODEL.md` for the resource model.

use llmcuda_model::budget::{self, WeightBytesPerParam};
use llmcuda_model::config::ModelConfig;
use tracing::info;

/// 1 GiB in bytes.
const GIB: u64 = 1024 * 1024 * 1024;

/// Historical budget input: ~47.5 GiB usable on a 48 GiB Quadro RTX 8000.
/// Current measured usable capacity is recorded in `docs/DEVELOPMENT.md`.
const USABLE_VRAM_BYTES: u64 = (475 * GIB) / 10;

/// Historical 29.6 GiB weight-budget input. This example does not load or
/// measure the model file; its tensor-directory size is in `docs/MODEL.md`.
const WEIGHTS_BYTES: u64 = (296 * GIB) / 10;

/// Reference serving configuration: `-c 393216 -np 3`.
const REFERENCE_CONTEXT_TOKENS: u64 = 393_216;
const REFERENCE_SLOTS: u32 = 3;

/// f16 KV cache (`-ctk f16 -ctv f16`).
const KV_ELEM_BYTES_F16: u64 = 2;

/// Quadro RTX 8000 memory bandwidth.
const RTX_8000_BANDWIDTH_BYTES_PER_SEC: f64 = 672e9;

fn gib(bytes: u64) -> f64 {
    bytes as f64 / GIB as f64
}

fn main() {
    llmcuda_log::init_from_args();

    let cfg = ModelConfig::qwen3_6_35b_a3b();

    info!("=== VRAM budget per card ({}) ===", cfg.name);
    info!(
        "context={REFERENCE_CONTEXT_TOKENS} slots={REFERENCE_SLOTS} kv=f16 weights={:.1} GiB\n",
        gib(WEIGHTS_BYTES)
    );

    let vram = budget::vram_budget(
        &cfg,
        REFERENCE_CONTEXT_TOKENS,
        REFERENCE_SLOTS,
        WEIGHTS_BYTES,
    );

    info!("{:<32} {:>10}", "Segment", "GiB");
    info!("{:<32} {:>10.2}", "Weights", gib(vram.weights_bytes));
    info!("{:<32} {:>10.2}", "KV pool", gib(vram.kv_pool_bytes));
    info!(
        "{:<32} {:>10.2}",
        "GDN recurrent state (x slots)",
        gib(vram.gdn_state_bytes)
    );
    info!(
        "{:<32} {:>10.2}",
        "Compute buffers (estimate)",
        gib(vram.compute_buffer_bytes)
    );
    info!(
        "{:<32} {:>10.2}",
        "CUDA context overhead (estimate)",
        gib(vram.cuda_context_overhead_bytes)
    );
    info!("{:<32} {:>10.2}", "Total", gib(vram.total_bytes()));
    info!(
        "{:<32} {:>10.2}",
        "Headroom (of usable)",
        vram.headroom_bytes(USABLE_VRAM_BYTES) as f64 / GIB as f64
    );
    info!(
        "\nNote: this budget covers the text path and excludes the vision \
         encoder (mmproj). Serving images requires budgeting that separate \
         allocation as well.\n"
    );

    info!("=== Per-token decode bandwidth roofline ===");
    info!(
        "quantization: expert weights Q6_K, LM head + projections Q8_0 \
         (observed directly in the real GGUF file, tensor by tensor)\n"
    );
    info!(
        "{:<8} {:>12} {:>12} {:>12} {:>14}",
        "Context", "KV GB/tok", "Wt GB/tok", "Total GB/tok", "Roofline tok/s"
    );

    let bpp = WeightBytesPerParam::observed_in_target_file();
    for (label, ctx) in [
        ("4K", 4_096u64),
        ("32K", 32_768),
        ("128K", 131_072),
        ("256K", 262_144),
    ] {
        let bw = budget::decode_bandwidth(
            &cfg,
            ctx,
            KV_ELEM_BYTES_F16,
            bpp,
            RTX_8000_BANDWIDTH_BYTES_PER_SEC,
        );
        info!(
            "{:<8} {:>12.3} {:>12.3} {:>12.3} {:>14.1}",
            label,
            bw.kv_bytes as f64 / 1e9,
            bw.weight_bytes as f64 / 1e9,
            bw.total_bytes as f64 / 1e9,
            bw.roofline_tokens_per_sec
        );
    }

    let weight_bytes = {
        let bw = budget::decode_bandwidth(
            &cfg,
            0,
            KV_ELEM_BYTES_F16,
            bpp,
            RTX_8000_BANDWIDTH_BYTES_PER_SEC,
        );
        bw.weight_bytes
    };
    let crossover = budget::kv_bound_crossover_tokens(&cfg, KV_ELEM_BYTES_F16, weight_bytes);
    info!(
        "\nKV-bound crossover: {crossover} tokens (weight bytes/token exceeded \
         by KV bytes/token past this point)."
    );
    info!(
        "This estimate uses structurally derived projection dimensions and \
         their observed Q8_0 format. See \
         WeightBytesPerParam::observed_in_target_file and docs/MODEL.md."
    );
}
