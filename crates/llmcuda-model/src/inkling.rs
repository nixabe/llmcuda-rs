//! `inkling2_loop` (Inkling 2 Mini): a looped mixture-of-experts transformer.
//!
//! The file holds `block_count` layers of weights; the layers in
//! `[loop_start, loop_end)` run `loop_count` times, each pass after the first
//! steered by messages mixed from the pass before. Every layer application
//! has its own attention cache and short-convolution state, so the model has
//! more cache slots than layers; see [`InklingConfig::slots`].
//!
//! Attention carries no rotary embedding. Positions enter through a learned
//! relative-position bias and, on the full-attention layers, a logarithmic
//! scale on the queries. Most layers attend over a sliding window.
//!
//! Keys and their validation follow llama.cpp's
//! `llama_model_inkling2::load_arch_hparams` (`src/models/inkling2.cpp`) and
//! `llama_model_inkling2_loop::load_arch_hparams`
//! (`src/models/inkling2-loop.cpp`) at ae6a7ad2c.

use llmcuda_gguf::{GgufFile, GgufValue};

use crate::config::{AttentionConfig, FfnConfig, GdnConfig, KvCacheTypes, ModelConfig, MoeConfig};
use crate::gguf::ConfigLoadError;

/// The architecture string.
pub const ARCHITECTURE: &str = "inkling2_loop";

/// The most layers a file may hold: the sliding pattern is a bit mask.
pub const MAX_LAYERS: u32 = 64;

/// What `inkling2_loop` adds to [`ModelConfig`].
///
/// Floats are stored as their bits so the configuration stays `Eq`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InklingConfig {
    /// Rows of the embedding and output tables; the tail past the
    /// vocabulary is padding.
    pub vocab_padded: u32,
    /// Bit `i`: layer `i` attends over a sliding window.
    pub sliding_layers: u64,
    /// The sliding window, and the extent of a sliding layer's bias table.
    pub sliding_window: u32,
    /// Relative-position features per head.
    pub relative_length: u32,
    /// Distances a full-attention layer's bias table covers.
    pub relative_extent: u32,
    /// Taps of the short convolutions.
    pub conv_taps: u32,
    /// Layers per residual block.
    pub res_block: u32,
    /// First looped layer.
    pub loop_start: u32,
    /// One past the last looped layer.
    pub loop_end: u32,
    /// Passes through the looped layers.
    pub loop_count: u32,
    /// Rank of the loop controllers.
    pub gate_rank: u32,
    /// Bits of the scale on the normalized router weights.
    pub expert_weights_scale_bits: u32,
    /// Bits of the SiTU gate saturation.
    pub situ_beta_bits: u32,
    /// Bits of the SiTU linear-branch saturation.
    pub situ_linear_beta_bits: u32,
    /// Bits of the RMSNorm epsilon.
    pub rms_eps_bits: u32,
    /// Bits of the L2 bound on a query or key head.
    pub qk_clip_bits: u32,
    /// Bits of the L2 bound on a head's relative-position features.
    pub relative_clip_bits: u32,
    /// Position below which the logarithmic query scale is one.
    pub log_scale_floor: u32,
    /// Bits of the logarithmic scale's slope.
    pub log_scale_alpha_bits: u32,
    /// Bits of the final logit soft cap; zero disables it.
    pub softcap_bits: u32,
}

impl InklingConfig {
    /// Whether `layer` attends over a sliding window.
    pub const fn is_sliding(&self, layer: u32) -> bool {
        self.sliding_layers >> layer & 1 == 1
    }

    /// Cache slots: one per layer application.
    pub const fn slots(&self, layers: u32) -> u32 {
        layers + (self.loop_count - 1) * (self.loop_end - self.loop_start)
    }

    /// The layer whose weights cache slot `slot` runs on.
    pub const fn slot_layer(&self, layers: u32, slot: u32) -> u32 {
        if slot < layers {
            slot
        } else {
            self.loop_start + (slot - layers) % (self.loop_end - self.loop_start)
        }
    }
}

fn invalid(key: &str, reason: &str) -> ConfigLoadError {
    ConfigLoadError(format!("GGUF `{key}`: {reason}"))
}

pub(crate) fn load(file: &GgufFile) -> Result<ModelConfig, ConfigLoadError> {
    let key = |suffix: &str| format!("{ARCHITECTURE}.{suffix}");
    let int = |suffix: &str| -> Result<u32, ConfigLoadError> {
        let k = key(suffix);
        let v = match file.get(&k) {
            Some(GgufValue::U32(v)) => *v,
            Some(GgufValue::U64(v)) => {
                u32::try_from(*v).map_err(|_| invalid(&k, "integer exceeds u32"))?
            }
            None => return Err(invalid(&k, "required metadata is missing")),
            _ => return Err(invalid(&k, "expected an unsigned integer")),
        };
        if v == 0 {
            return Err(invalid(&k, "must be non-zero"));
        }
        Ok(v)
    };
    let float = |suffix: &str, zero_ok: bool| -> Result<f32, ConfigLoadError> {
        let k = key(suffix);
        let v = file
            .get_f32(&k)
            .ok_or_else(|| invalid(&k, "required float is missing"))?;
        if !v.is_finite() || v < 0.0 || (v == 0.0 && !zero_ok) {
            return Err(invalid(&k, "must be finite and positive"));
        }
        Ok(v)
    };
    // llama.cpp reads the head counts as a scalar or a per-layer array; only
    // uniform layers are supported here.
    let uniform = |suffix: &str, layers: u32| -> Result<u32, ConfigLoadError> {
        let k = key(suffix);
        match file.get(&k) {
            Some(GgufValue::Array(_)) => {
                let values = file
                    .get_i32_array(&k)
                    .ok_or_else(|| invalid(&k, "expected an integer array"))?;
                match values.first() {
                    Some(&first)
                        if first > 0
                            && values.len() == layers as usize
                            && values.iter().all(|v| *v == first) =>
                    {
                        Ok(first as u32)
                    }
                    _ => Err(invalid(&k, "must hold one equal positive value per layer")),
                }
            }
            _ => int(suffix),
        }
    };

    let layers = int("block_count")?;
    if layers > MAX_LAYERS {
        return Err(invalid(&key("block_count"), "more than 64 layers"));
    }
    let head_dim = int("attention.key_length")?;
    if int("attention.value_length")? != head_dim {
        return Err(invalid(
            &key("attention.value_length"),
            "unequal key/value head dimensions are unsupported",
        ));
    }
    let q_heads = uniform("attention.head_count", layers)?;
    let kv_heads = uniform("attention.head_count_kv", layers)?;
    if !q_heads.is_multiple_of(kv_heads) {
        return Err(invalid(
            &key("attention.head_count"),
            "must be a multiple of head_count_kv",
        ));
    }

    let pattern_key = key("attention.sliding_window_pattern");
    let pattern = file
        .get_bool_array(&pattern_key)
        .ok_or_else(|| invalid(&pattern_key, "required bool array is missing"))?;
    if pattern.len() != layers as usize {
        return Err(invalid(&pattern_key, "must hold one entry per layer"));
    }
    let sliding_layers = pattern
        .iter()
        .enumerate()
        .fold(0u64, |m, (i, s)| m | u64::from(*s) << i);

    let tokens = file
        .get_string_array("tokenizer.ggml.tokens")
        .ok_or_else(|| invalid("tokenizer.ggml.tokens", "required string array is missing"))?;
    let vocab = int("vocab_size_unpadded")?;
    let vocab_padded = int("vocab_size")?;
    if tokens.len() != vocab as usize || vocab > vocab_padded {
        return Err(invalid(
            &key("vocab_size_unpadded"),
            "must equal the tokenizer size and not exceed vocab_size",
        ));
    }

    let res_block = int("attn_res.block_size")?;
    let (loop_start, loop_end, loop_count) =
        (int("loop.start")?, int("loop.end")?, int("loop.count")?);
    if file
        .get(&key("loop.varying"))
        .is_some_and(|v| matches!(v, GgufValue::Bool(true)))
    {
        return Err(invalid(
            &key("loop.varying"),
            "a variable loop is unsupported",
        ));
    }
    if loop_count < 2
        || loop_start >= loop_end
        || loop_end > layers
        || !loop_start.is_multiple_of(res_block)
        || !loop_end.is_multiple_of(res_block)
    {
        return Err(invalid(
            &key("loop.start"),
            "the loop must be whole residual blocks inside the model, run at least twice",
        ));
    }
    let conv_taps = int("shortconv.l_cache")?;
    if !(2..=64).contains(&conv_taps) {
        return Err(invalid(&key("shortconv.l_cache"), "must be within 2..=64"));
    }
    let experts = int("expert_count")?;
    let experts_per_token = uniform("expert_used_count", layers)?;
    if experts_per_token > experts {
        return Err(invalid(&key("expert_used_count"), "exceeds expert_count"));
    }

    let softcap = match file.get_f32(&key("final_logit_softcapping")) {
        Some(v) if v.is_finite() && v >= 0.0 => v,
        Some(_) => return Err(invalid(&key("final_logit_softcapping"), "must be finite")),
        None => 0.0,
    };
    let weights_scale = match file.get(&key("expert_weights_scale")) {
        Some(_) => float("expert_weights_scale", false)?,
        None => 1.0,
    };
    let inkling = InklingConfig {
        vocab_padded,
        sliding_layers,
        sliding_window: int("attention.sliding_window")?,
        relative_length: int("attention.relative_length")?,
        relative_extent: int("attention.relative_extent")?,
        conv_taps,
        res_block,
        loop_start,
        loop_end,
        loop_count,
        gate_rank: int("loop.gate_rank")?,
        expert_weights_scale_bits: weights_scale.to_bits(),
        situ_beta_bits: float("activation.situ_beta", false)?.to_bits(),
        situ_linear_beta_bits: float("activation.situ_linear_beta", false)?.to_bits(),
        rms_eps_bits: float("attention.layer_norm_rms_epsilon", false)?.to_bits(),
        qk_clip_bits: float("attention.qk_clip", false)?.to_bits(),
        relative_clip_bits: float("attention.relative_clip", false)?.to_bits(),
        log_scale_floor: int("attention.log_scale_floor")?,
        log_scale_alpha_bits: float("attention.log_scale_alpha", true)?.to_bits(),
        softcap_bits: softcap.to_bits(),
    };
    let native_context = int("context_length")?;
    let config = ModelConfig {
        name: ARCHITECTURE,
        architecture: ARCHITECTURE,
        advertised_params: 0,
        num_layers: layers,
        hidden_size: int("embedding_length")?,
        vocab_size: vocab,
        pattern_period: 1,
        attention_offset: 0,
        // Unused: this architecture has no recurrent layers.
        gdn: GdnConfig {
            value_heads: 1,
            qk_heads: 1,
            head_dim: 128,
            conv_kernel: 4,
            chunk_len: 64,
        },
        attention: AttentionConfig {
            q_heads,
            kv_heads,
            head_dim,
            rope_dim: 0,
        },
        ffn: FfnConfig::Moe(MoeConfig {
            num_experts: experts,
            experts_per_token,
            shared_experts: 0,
            expert_intermediate: int("expert_feed_forward_length")?,
        }),
        native_context,
        yarn_context: native_context,
        has_mtp: false,
        k2: None,
        decision: None,
        inkling: Some(inkling),
        kv_cache: KvCacheTypes::F16,
    };
    crate::verify::check_config(&config).map_err(|e| ConfigLoadError(e.to_string()))?;
    Ok(config)
}
