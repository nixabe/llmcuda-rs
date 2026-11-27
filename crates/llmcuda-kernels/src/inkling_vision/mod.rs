//! CPU reference for the Inkling 2 Mini vision tower (`inkling2v` mmproj).
//!
//! The oracle a device implementation of the tower is compared against,
//! stage by stage: [`preprocess()`] turns an RGB8 image into normalized
//! patches, and [`encode`] turns those into the rows the language model
//! consumes as input embeddings — one per 40×40 pixel block, row-major,
//! with no separator tokens. (The text model takes these rows as they are,
//! skipping the RMS norm it applies to its own token embeddings.)
//!
//! [`token_count`] and [`plan`] answer how many rows an image will become
//! from its size alone, which is what prompt-token accounting needs.
//!
//! The reference is itself validated against llama.cpp's execution of the
//! same mmproj file, tensor by tensor, in `tests/inkling_vision_golden.rs`.
//! llama.cpp's graph has one half-precision step — the normalized pixels
//! are rounded to F16 on their way into the patch convolution — and
//! [`encode`] reproduces it, because the rows llama.cpp hands its language
//! model are the target; [`InklingVisionNumerics`] names the alternatives.
//!
//! Ported from llama.cpp at ae6a7ad2c; each submodule cites its sources.

pub mod preprocess;
pub mod tower;

pub use preprocess::{
    InklingImageError, InklingImagePlan, MAX_RESIZE_EDGE, PAD_RAW_VALUE, PreprocessedInklingImage,
    calc_size_preserved_ratio, normalize, plan, preprocess, resize_bicubic_pillow, token_count,
};
pub use tower::{
    InklingVisionBlockWeights, InklingVisionLoadError, InklingVisionNumerics, InklingVisionTap,
    InklingVisionWeights, attention, block, dot, embeddings, encode, encode_tapped, erf, gelu_erf,
    layer_norm, layer_norm_rows, linear, merge_patches, patch_embed, project,
    resized_position_embedding, rope_2d, rope_position, rope_rows,
};
