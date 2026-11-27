//! Inkling 2 Mini vision tower (mmproj, projector type `inkling2v`).
//!
//! Like the Qwen tower in [`crate::vision`], the weights ship as a separate
//! GGUF (`general.architecture = clip`, `general.type = mmproj`) whose path
//! the server takes explicitly. Unlike it, nothing about the image is
//! resized to suit the tower: the image is padded to a multiple of
//! `patch_size * spatial_merge_size` and every 40×40 pixel block becomes one
//! language-model token.
//!
//! The tower is a ViT with a class token: a single bias-free patch conv, a
//! learned 32×32 position grid resized bilinearly to the patch grid, a
//! pre-LayerNorm, 23 pre-LN blocks with 2D rotary attention, LayerScale on
//! both residual branches and an erf-GELU MLP, then a 2×2 patch merge into a
//! bias-free two-layer erf-GELU projector. There is no post-LayerNorm and no
//! fused QKV matrix.
//!
//! Transcribed from llama.cpp at ae6a7ad2c: `tools/mtmd/models/inkling2v.cpp`
//! (`mtmd_graph_inkling2v::build`) and the `PROJECTOR_TYPE_INKLING2V` arms of
//! `tools/mtmd/mtmd-model.cpp` (hyperparameters, tensor loading).

use core::fmt;

use llmcuda_gguf::GgufFile;

use crate::weights::WeightError;

/// The `clip.vision.projector_type` this module implements.
pub const INKLING_PROJECTOR_TYPE: &str = "inkling2v";

/// Structural description of the `inkling2v` tower and projector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InklingVisionConfig {
    /// Transformer blocks in the tower.
    pub num_layers: u32,
    /// Hidden width of the tower.
    pub hidden_size: u32,
    /// Attention heads per block. Bidirectional, no GQA.
    pub num_heads: u32,
    /// MLP intermediate width (erf-GELU, up/down only — no gate).
    pub ffn_size: u32,
    /// Native square input the position grid was trained at, in pixels.
    pub image_size: u32,
    /// Square patch edge in pixels.
    pub patch_size: u32,
    /// Edge of the spatial merge window; `merge^2` adjacent patches are
    /// concatenated before the projector.
    pub spatial_merge: u32,
    /// Output width of the projector — must equal the LLM hidden size.
    pub projection_dim: u32,
    /// LayerNorm epsilon (full LayerNorm with bias, not RMSNorm).
    pub ln_eps: f32,
    /// Rotary base of the 2D RoPE applied to queries and keys.
    ///
    /// Not in the file: llama.cpp sets `hparams.rope_theta = 10000.0f` for
    /// this projector type.
    pub rope_theta: f32,
    /// Per-channel normalization: `(pixel - mean) / std`, applied after
    /// scaling bytes to `[0, 1]`.
    pub image_mean: [f32; 3],
    /// See [`Self::image_mean`].
    pub image_std: [f32; 3],
    /// Most language-model tokens one image may produce before it is
    /// shrunk.
    ///
    /// Not in the file: llama.cpp calls `set_limit_image_tokens(1, 4096)`
    /// for this projector type, and `--image-max-tokens` overrides it.
    pub max_image_tokens: u32,
}

impl InklingVisionConfig {
    /// The tower shipped in `mmproj-F16.gguf` for Inkling 2 Mini.
    ///
    /// Transcribed from that file's `clip.vision.*` metadata; the
    /// `real_inkling_mmproj` test holds it to the file.
    pub const fn inkling2_mini() -> Self {
        Self {
            num_layers: 23,
            hidden_size: 1024,
            num_heads: 16,
            ffn_size: 4096,
            image_size: 640,
            patch_size: 20,
            spatial_merge: 2,
            projection_dim: 2304,
            ln_eps: 1e-5,
            rope_theta: 10_000.0,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            max_image_tokens: 4096,
        }
    }

    /// Read an `inkling2v` tower from its own mmproj metadata.
    ///
    /// Every other projector type is refused, as is a tower whose
    /// projector does not land in `target`'s embedding width. The rotary
    /// base and the token ceiling are projector-family constants (see the
    /// field docs), not metadata.
    pub fn from_gguf(
        file: &GgufFile,
        target: &crate::ModelConfig,
    ) -> Result<Self, crate::ConfigLoadError> {
        use crate::metadata::Metadata;
        let m = Metadata(file);
        m.text("general.architecture", crate::vision::MMPROJ_ARCHITECTURE)?;
        m.text("clip.vision.projector_type", INKLING_PROJECTOR_TYPE)?;
        let c = Self {
            num_layers: m.positive("clip.vision.block_count")?,
            hidden_size: m.positive("clip.vision.embedding_length")?,
            num_heads: m.positive("clip.vision.attention.head_count")?,
            ffn_size: m.positive("clip.vision.feed_forward_length")?,
            image_size: m.positive("clip.vision.image_size")?,
            patch_size: m.positive("clip.vision.patch_size")?,
            spatial_merge: m.positive("clip.vision.spatial_merge_size")?,
            projection_dim: m.positive("clip.vision.projection_dim")?,
            ln_eps: m.float("clip.vision.attention.layer_norm_epsilon")?,
            image_mean: m.rgb("clip.vision.image_mean", false)?,
            image_std: m.rgb("clip.vision.image_std", true)?,
            ..Self::inkling2_mini()
        };
        if c.projection_dim != target.hidden_size {
            return Err(Metadata::error(
                "clip.vision.projection_dim",
                "must match the target hidden width",
            ));
        }
        // Each head splits in two halves, one per image axis, and each half
        // rotates adjacent pairs.
        if !c.hidden_size.is_multiple_of(c.num_heads) || !c.head_dim().is_multiple_of(4) {
            return Err(Metadata::error(
                "clip.vision.attention.head_count",
                "must divide hidden width into heads divisible by four",
            ));
        }
        if !c.image_size.is_multiple_of(c.patch_size) || c.spatial_merge != 2 {
            return Err(Metadata::error(
                "clip.vision.spatial_merge_size",
                "requires a two-by-two merger and an integral native patch grid",
            ));
        }
        Ok(c)
    }

    /// Dimension of each attention head.
    pub const fn head_dim(&self) -> u32 {
        self.hidden_size / self.num_heads
    }

    /// Patches concatenated by one spatial merge window.
    pub const fn merge_factor(&self) -> u32 {
        self.spatial_merge * self.spatial_merge
    }

    /// Width of a merged patch vector entering the projector.
    pub const fn merger_input_dim(&self) -> u32 {
        self.hidden_size * self.merge_factor()
    }

    /// Edge of the learned position grid (`image_size / patch_size`).
    pub const fn pos_grid_edge(&self) -> u32 {
        self.image_size / self.patch_size
    }

    /// Elements of one flattened patch fed to the patch conv:
    /// `channels × patch × patch`.
    pub const fn patch_elems(&self) -> u32 {
        3 * self.patch_size * self.patch_size
    }

    /// Pixels along one edge that become one language-model token, and the
    /// multiple every image edge is padded to: `patch_size * spatial_merge`.
    pub const fn token_edge(&self) -> u32 {
        self.patch_size * self.spatial_merge
    }

    /// Language-model tokens produced by a `grid_h × grid_w` patch grid.
    ///
    /// Both edges must be multiples of [`Self::spatial_merge`];
    /// preprocessing guarantees that by construction.
    pub const fn output_tokens(&self, grid_h: u32, grid_w: u32) -> u32 {
        (grid_h * grid_w) / self.merge_factor()
    }
}

impl Default for InklingVisionConfig {
    fn default() -> Self {
        Self::inkling2_mini()
    }
}

/// What an `inkling2v` tensor is used for.
///
/// Names are the ones llama.cpp's loader asks for (`TN_CLASS_EMBD`,
/// `TN_CLASS_POS_EMBD`, `TN_LLAVA_PROJ` and the generic `v.blk.N.*` set in
/// `tools/mtmd/mtmd-impl.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum InklingVisionRole {
    /// `v.class_embd` — the class token. No `.weight` suffix.
    ClassEmbed,
    /// `v.class_position_embd.weight` — the class token's position.
    ClassPositionEmbed,
    /// `v.patch_embd.weight` — the patch conv. No bias exists.
    PatchEmbed,
    /// `v.position_embd.weight` — learned 32×32 position grid.
    PositionEmbed,
    /// `v.pre_ln.weight` — LayerNorm ahead of the first block.
    PreLnWeight,
    /// `v.pre_ln.bias`.
    PreLnBias,

    /// `v.blk.N.ln1.weight` — LayerNorm before attention.
    Ln1Weight,
    /// `v.blk.N.ln1.bias`.
    Ln1Bias,
    /// `v.blk.N.attn_q.weight`.
    AttnQWeight,
    /// `v.blk.N.attn_q.bias`.
    AttnQBias,
    /// `v.blk.N.attn_k.weight`.
    AttnKWeight,
    /// `v.blk.N.attn_k.bias`.
    AttnKBias,
    /// `v.blk.N.attn_v.weight`.
    AttnVWeight,
    /// `v.blk.N.attn_v.bias`.
    AttnVBias,
    /// `v.blk.N.attn_out.weight`.
    AttnOutWeight,
    /// `v.blk.N.attn_out.bias`.
    AttnOutBias,
    /// `v.blk.N.ls1.weight` — LayerScale on the attention branch.
    Ls1,
    /// `v.blk.N.ln2.weight` — LayerNorm before the MLP.
    Ln2Weight,
    /// `v.blk.N.ln2.bias`.
    Ln2Bias,
    /// `v.blk.N.ffn_up.weight`.
    FfnUpWeight,
    /// `v.blk.N.ffn_up.bias`.
    FfnUpBias,
    /// `v.blk.N.ffn_down.weight`.
    FfnDownWeight,
    /// `v.blk.N.ffn_down.bias`.
    FfnDownBias,
    /// `v.blk.N.ls2.weight` — LayerScale on the MLP branch.
    Ls2,

    /// `mm.1.weight` — first projector matrix over the merged patch
    /// vector. No bias exists.
    Projector1,
    /// `mm.2.weight` — projection into the LLM embedding width. No bias.
    Projector2,
}

impl InklingVisionRole {
    /// The GGUF name suffix, without the `v.blk.N.` prefix.
    ///
    /// Global tensors return their whole name.
    pub const fn suffix(self) -> &'static str {
        use InklingVisionRole::*;
        match self {
            ClassEmbed => "v.class_embd",
            ClassPositionEmbed => "v.class_position_embd.weight",
            PatchEmbed => "v.patch_embd.weight",
            PositionEmbed => "v.position_embd.weight",
            PreLnWeight => "v.pre_ln.weight",
            PreLnBias => "v.pre_ln.bias",
            Ln1Weight => "ln1.weight",
            Ln1Bias => "ln1.bias",
            AttnQWeight => "attn_q.weight",
            AttnQBias => "attn_q.bias",
            AttnKWeight => "attn_k.weight",
            AttnKBias => "attn_k.bias",
            AttnVWeight => "attn_v.weight",
            AttnVBias => "attn_v.bias",
            AttnOutWeight => "attn_out.weight",
            AttnOutBias => "attn_out.bias",
            Ls1 => "ls1.weight",
            Ln2Weight => "ln2.weight",
            Ln2Bias => "ln2.bias",
            FfnUpWeight => "ffn_up.weight",
            FfnUpBias => "ffn_up.bias",
            FfnDownWeight => "ffn_down.weight",
            FfnDownBias => "ffn_down.bias",
            Ls2 => "ls2.weight",
            Projector1 => "mm.1.weight",
            Projector2 => "mm.2.weight",
        }
    }

    /// Whether this role names a global (non-block) tensor.
    pub const fn is_global(self) -> bool {
        use InklingVisionRole::*;
        matches!(
            self,
            ClassEmbed
                | ClassPositionEmbed
                | PatchEmbed
                | PositionEmbed
                | PreLnWeight
                | PreLnBias
                | Projector1
                | Projector2
        )
    }
}

impl fmt::Display for InklingVisionRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.suffix())
    }
}

/// One `inkling2v` tensor with its expected shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InklingVisionTensorSpec {
    /// Full GGUF tensor name.
    pub name: String,
    /// What this tensor is.
    pub role: InklingVisionRole,
    /// Block index, or `None` for global tensors.
    pub layer: Option<u32>,
    /// Expected dimensions in GGUF/ggml order (`dims[0]` fastest-varying).
    pub dims: Vec<u64>,
}

/// Every tensor the `inkling2v` tower needs, derived from an
/// [`InklingVisionConfig`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InklingVisionWeightSchema {
    specs: Vec<InklingVisionTensorSpec>,
}

impl InklingVisionWeightSchema {
    /// Build the schema for `config`.
    pub fn new(config: &InklingVisionConfig) -> Self {
        let h = u64::from(config.hidden_size);
        let ffn = u64::from(config.ffn_size);
        let patch = u64::from(config.patch_size);
        let grid = u64::from(config.pos_grid_edge());
        let merged = u64::from(config.merger_input_dim());
        let proj = u64::from(config.projection_dim);

        let mut specs = Vec::new();
        let mut push = |role: InklingVisionRole, layer: Option<u32>, dims: Vec<u64>| {
            let name = match layer {
                Some(i) => format!("v.blk.{i}.{}", role.suffix()),
                None => role.suffix().to_string(),
            };
            specs.push(InklingVisionTensorSpec {
                name,
                role,
                layer,
                dims,
            });
        };

        use InklingVisionRole::*;
        push(ClassEmbed, None, vec![h]);
        push(ClassPositionEmbed, None, vec![h]);
        push(PatchEmbed, None, vec![patch, patch, 3, h]);
        push(PositionEmbed, None, vec![h, grid * grid]);
        push(PreLnWeight, None, vec![h]);
        push(PreLnBias, None, vec![h]);
        push(Projector1, None, vec![merged, merged]);
        push(Projector2, None, vec![merged, proj]);

        for layer in 0..config.num_layers {
            let l = Some(layer);
            push(Ln1Weight, l, vec![h]);
            push(Ln1Bias, l, vec![h]);
            for (weight, bias) in [
                (AttnQWeight, AttnQBias),
                (AttnKWeight, AttnKBias),
                (AttnVWeight, AttnVBias),
                (AttnOutWeight, AttnOutBias),
            ] {
                push(weight, l, vec![h, h]);
                push(bias, l, vec![h]);
            }
            push(Ls1, l, vec![h]);
            push(Ln2Weight, l, vec![h]);
            push(Ln2Bias, l, vec![h]);
            push(FfnUpWeight, l, vec![h, ffn]);
            push(FfnUpBias, l, vec![ffn]);
            push(FfnDownWeight, l, vec![ffn, h]);
            push(FfnDownBias, l, vec![h]);
            push(Ls2, l, vec![h]);
        }

        Self { specs }
    }

    /// Every tensor in the schema.
    pub fn specs(&self) -> &[InklingVisionTensorSpec] {
        &self.specs
    }

    /// Look up one tensor by role and layer.
    pub fn find(
        &self,
        role: InklingVisionRole,
        layer: Option<u32>,
    ) -> Option<&InklingVisionTensorSpec> {
        self.specs
            .iter()
            .find(|s| s.role == role && s.layer == layer)
    }

    /// Match the schema against an mmproj `file`.
    ///
    /// Checks the declared architecture (`clip`), the projector type
    /// (`inkling2v`), and every tensor's existence and shape. Returns all
    /// mismatches rather than the first, matching
    /// [`crate::weights::WeightSchema::resolve`]. See [`Self::unclaimed`]
    /// for the other direction.
    pub fn resolve(&self, file: &GgufFile) -> Result<(), Vec<WeightError>> {
        let mut errors = Vec::new();

        match file.get_str("general.architecture") {
            Some(crate::vision::MMPROJ_ARCHITECTURE) => {}
            found => errors.push(WeightError::ArchitectureMismatch {
                expected: crate::vision::MMPROJ_ARCHITECTURE.to_string(),
                found: found.unwrap_or("<absent>").to_string(),
            }),
        }
        match file.get_str("clip.vision.projector_type") {
            Some(INKLING_PROJECTOR_TYPE) => {}
            found => errors.push(WeightError::ArchitectureMismatch {
                expected: INKLING_PROJECTOR_TYPE.to_string(),
                found: found.unwrap_or("<absent>").to_string(),
            }),
        }

        for spec in &self.specs {
            let Some(info) = file.tensor(&spec.name) else {
                errors.push(WeightError::Missing {
                    name: spec.name.clone(),
                });
                continue;
            };
            if info.dims != spec.dims {
                errors.push(WeightError::ShapeMismatch {
                    name: spec.name.clone(),
                    expected: spec.dims.clone(),
                    found: info.dims.clone(),
                });
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Names of the tensors in `file` this schema does not claim.
    ///
    /// llama.cpp's graph picks up several optional tensors when a file has
    /// them — a patch bias, a post-LayerNorm, a fused QKV matrix, projector
    /// biases. The published file has none and the reference implements
    /// none, so a loader should refuse a file for which this is not empty
    /// rather than run a silently different model.
    pub fn unclaimed<'a>(&self, file: &'a GgufFile) -> Vec<&'a str> {
        file.tensors()
            .iter()
            .map(|t| t.name.as_str())
            .filter(|name| !self.specs.iter().any(|s| s.name == *name))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> InklingVisionConfig {
        InklingVisionConfig::inkling2_mini()
    }

    #[test]
    fn heads_split_into_two_rotary_halves_of_adjacent_pairs() {
        assert_eq!(cfg().head_dim(), 64);
        assert_eq!(cfg().head_dim() * cfg().num_heads, cfg().hidden_size);
        assert!(cfg().head_dim().is_multiple_of(4));
    }

    #[test]
    fn projector_matches_the_gguf_tensor_shapes() {
        // mm.1 is 4096x4096, mm.2 is 4096x2304 in the file.
        assert_eq!(cfg().merger_input_dim(), 4096);
        assert_eq!(cfg().projection_dim, 2304);
    }

    #[test]
    fn position_grid_is_32_by_32() {
        assert_eq!(cfg().pos_grid_edge(), 32);
        assert_eq!(cfg().patch_elems(), 1200);
    }

    #[test]
    fn one_token_covers_a_40_pixel_square() {
        let c = cfg();
        assert_eq!(c.token_edge(), 40);
        let edge = c.pos_grid_edge();
        assert_eq!(c.output_tokens(edge, edge), 256);
    }

    #[test]
    fn schema_tensor_count_matches_the_published_mmproj() {
        // mmproj-F16.gguf holds 422 tensors: 18 per block across 23 blocks,
        // plus 8 globals.
        let schema = InklingVisionWeightSchema::new(&cfg());
        assert_eq!(schema.specs().len(), 422);
    }

    #[test]
    fn schema_names_are_unique_and_classified_consistently() {
        let schema = InklingVisionWeightSchema::new(&cfg());
        let mut names: Vec<&str> = schema.specs().iter().map(|s| s.name.as_str()).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(names.len(), before, "duplicate tensor name");
        for spec in schema.specs() {
            assert_eq!(spec.role.is_global(), spec.layer.is_none(), "{}", spec.name);
        }
    }

    #[test]
    fn the_class_token_is_the_one_tensor_without_a_weight_suffix() {
        let schema = InklingVisionWeightSchema::new(&cfg());
        let bare: Vec<&str> = schema
            .specs()
            .iter()
            .map(|s| s.name.as_str())
            .filter(|n| !n.ends_with(".weight") && !n.ends_with(".bias"))
            .collect();
        assert_eq!(bare, ["v.class_embd"]);
    }
}
