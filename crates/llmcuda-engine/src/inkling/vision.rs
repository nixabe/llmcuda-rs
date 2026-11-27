//! The `inkling2v` vision tower on the device: preprocessed images to the
//! rows the language model takes in place of token embeddings.
//!
//! The reference, and the tower's provenance in llama.cpp, is
//! [`llmcuda_kernels::inkling_vision`]. This is that pipeline over
//! [`VisionKernels`], the kernel set of the `qwen3vl_merger` tower, with
//! binary16 activations as there. Three things are arranged at load so
//! those kernels fit a different tower:
//!
//! - **Token order.** The transformer has no mask and takes its positions
//!   explicitly, so the order of its rows is free. Patches go in 2x2-cell
//!   order with the class token *last*: the merge is then four consecutive
//!   rows read as one, and the class token is simply not among them.
//! - **Rotary layout.** llama.cpp rotates adjacent pairs, dims `[0, 32)` of
//!   a head by x and `[32, 64)` by y. `vision_rope_qk` rotates pairs
//!   `(j, j + 32)`, the first sixteen by one position and the rest by the
//!   other. The rows of the query and key projections are permuted to
//!   match (dim `2i` to slot `i`, `2i + 1` to `i + 32`, and the y half
//!   beside them); `q . k` does not see a permutation both share.
//! - **Layer scale.** A block's `ls1` and `ls2` multiply the output of its
//!   output projection and of its feed-forward, per channel; they are
//!   folded into those projections' rows and biases, so the projection can
//!   accumulate straight onto the residual.
//!
//! One instance per worker, every buffer allocated at construction for
//! `max_patches`; encoding allocates nothing on the device.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::f16;

use llmcuda_cuda::kernels::vision::VisionKernels;
use llmcuda_kernels::inkling_vision::{
    InklingVisionWeights, PreprocessedInklingImage, resized_position_embedding,
};
use llmcuda_kernels::vision::cell_order_index;
use llmcuda_model::InklingVisionConfig;

use crate::vision::VisionForwardError;

struct Block {
    ln1_w: CudaSlice<f16>,
    ln1_b: CudaSlice<f16>,
    qkv_w: CudaSlice<f16>,
    qkv_b: CudaSlice<f16>,
    out_w: CudaSlice<f16>,
    out_b: CudaSlice<f16>,
    ln2_w: CudaSlice<f16>,
    ln2_b: CudaSlice<f16>,
    up_w: CudaSlice<f16>,
    up_b: CudaSlice<f16>,
    down_w: CudaSlice<f16>,
    down_b: CudaSlice<f16>,
}

/// The tower resident on one device.
pub struct InklingVisionForward {
    kernels: VisionKernels,
    stream: Arc<CudaStream>,
    cfg: InklingVisionConfig,
    max_patches: usize,
    head_chunk: usize,

    patch_embed: CudaSlice<f16>,
    pre_ln_w: CudaSlice<f16>,
    pre_ln_b: CudaSlice<f16>,
    blocks: Vec<Block>,
    mm1_w: CudaSlice<f16>,
    mm2_w: CudaSlice<f16>,
    /// The learned position grid, resized per image on the host.
    pos_embed_host: Vec<f32>,
    /// The class token's row: its embedding plus its position embedding.
    class_row: Vec<f16>,

    patches_in: CudaSlice<f16>,
    embedded: CudaSlice<f16>,
    x: CudaSlice<f16>,
    normed: CudaSlice<f16>,
    qkv: CudaSlice<f16>,
    attn: CudaSlice<f16>,
    ffn: CudaSlice<f16>,
    scores: CudaSlice<f16>,
    pos_embed_img: CudaSlice<f16>,
    positions: CudaSlice<i32>,
    merged_mid: CudaSlice<f16>,
    out_f16: CudaSlice<f16>,
    out_f32: CudaSlice<f32>,

    stage_f16: Vec<f16>,
    stage_pos: Vec<i32>,
}

fn half(values: impl IntoIterator<Item = f32>) -> Vec<f16> {
    values.into_iter().map(f16::from_f32).collect()
}

/// Where dim `d` of a head goes so that `vision_rope_qk`'s pairs are
/// llama.cpp's: see the module documentation.
fn rope_slot(d: usize, head_dim: usize) -> usize {
    let (half_dim, quarter) = (head_dim / 2, head_dim / 4);
    let (axis, within) = (d / half_dim, d % half_dim);
    axis * quarter + within / 2 + (within % 2) * half_dim
}

impl InklingVisionForward {
    /// Upload `weights` and allocate for images of up to `max_patches`
    /// patches (`max_patches / 4` output rows).
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: Arc<CudaStream>,
        cfg: &InklingVisionConfig,
        weights: &InklingVisionWeights,
        max_patches: usize,
    ) -> Result<Self, VisionForwardError> {
        let hidden = cfg.hidden_size as usize;
        let ffn = cfg.ffn_size as usize;
        let d_head = cfg.head_dim() as usize;
        let heads = cfg.num_heads as usize;
        let patch_len = cfg.patch_elems() as usize;
        let merged = cfg.merger_input_dim() as usize;
        let proj = cfg.projection_dim as usize;
        let merge = cfg.merge_factor() as usize;
        assert!(
            max_patches > 0 && max_patches.is_multiple_of(merge),
            "max_patches must be a positive multiple of the merge factor"
        );
        let check = |what: &'static str, got: usize, want: usize| {
            if got == want {
                Ok(())
            } else {
                Err(VisionForwardError::WrongWeights { what })
            }
        };
        check("patch_embed", weights.patch_embed.len(), patch_len * hidden)?;
        check("blocks", weights.blocks.len(), cfg.num_layers as usize)?;
        check("mm1", weights.mm1_w.len(), merged * merged)?;
        check("mm2", weights.mm2_w.len(), merged * proj)?;
        check("head_dim", d_head % 4, 0)?;
        let edge = cfg.pos_grid_edge() as usize;
        check("pos_embed", weights.pos_embed.len(), edge * edge * hidden)?;

        let kernels = VisionKernels::new(ctx, stream.clone())?;
        let up = |v: Vec<f16>| -> Result<CudaSlice<f16>, VisionForwardError> {
            Ok(stream.clone_htod(&v)?)
        };
        let plain = |v: &[f32]| half(v.iter().copied());
        // Rows of a projection scaled per output channel.
        let scaled = |w: &[f32], scale: &[f32], n_in: usize| {
            half(
                w.chunks_exact(n_in)
                    .zip(scale)
                    .flat_map(|(row, s)| row.iter().map(move |v| v * s)),
            )
        };
        let blocks = weights
            .blocks
            .iter()
            .map(|b| {
                check("attn_q", b.q_w.len(), hidden * hidden)?;
                check("ls1", b.ls1.len(), hidden)?;
                check("ls2", b.ls2.len(), hidden)?;
                // The fused projection: query and key rows in rotary
                // order, value rows as stored.
                let mut qkv_w = vec![0.0f32; 3 * hidden * hidden];
                let mut qkv_b = vec![0.0f32; 3 * hidden];
                for (part, (w, bias)) in [(&b.q_w, &b.q_b), (&b.k_w, &b.k_b), (&b.v_w, &b.v_b)]
                    .into_iter()
                    .enumerate()
                {
                    for row in 0..hidden {
                        let to = if part < 2 {
                            row / d_head * d_head + rope_slot(row % d_head, d_head)
                        } else {
                            row
                        };
                        let at = (part * hidden + to) * hidden;
                        qkv_w[at..at + hidden]
                            .copy_from_slice(&w[row * hidden..(row + 1) * hidden]);
                        qkv_b[part * hidden + to] = bias[row];
                    }
                }
                Ok(Block {
                    ln1_w: up(plain(&b.ln1_w))?,
                    ln1_b: up(plain(&b.ln1_b))?,
                    qkv_w: up(plain(&qkv_w))?,
                    qkv_b: up(plain(&qkv_b))?,
                    out_w: up(scaled(&b.out_w, &b.ls1, hidden))?,
                    out_b: up(half(b.out_b.iter().zip(&b.ls1).map(|(v, s)| v * s)))?,
                    ln2_w: up(plain(&b.ln2_w))?,
                    ln2_b: up(plain(&b.ln2_b))?,
                    up_w: up(plain(&b.up_w))?,
                    up_b: up(plain(&b.up_b))?,
                    down_w: up(scaled(&b.down_w, &b.ls2, ffn))?,
                    down_b: up(half(b.down_b.iter().zip(&b.ls2).map(|(v, s)| v * s)))?,
                })
            })
            .collect::<Result<Vec<_>, VisionForwardError>>()?;

        let rows = max_patches + 1;
        let head_chunk = heads.min(4);
        let max_tokens = max_patches / merge;
        Ok(Self {
            patch_embed: up(plain(&weights.patch_embed))?,
            pre_ln_w: up(plain(&weights.pre_ln_w))?,
            pre_ln_b: up(plain(&weights.pre_ln_b))?,
            blocks,
            mm1_w: up(plain(&weights.mm1_w))?,
            mm2_w: up(plain(&weights.mm2_w))?,
            pos_embed_host: weights.pos_embed.clone(),
            class_row: half(
                weights
                    .class_embed
                    .iter()
                    .zip(&weights.class_pos_embed)
                    .map(|(a, b)| a + b),
            ),
            patches_in: stream.alloc_zeros(max_patches * patch_len)?,
            embedded: stream.alloc_zeros(rows * hidden)?,
            x: stream.alloc_zeros(rows * hidden)?,
            normed: stream.alloc_zeros(rows * hidden)?,
            qkv: stream.alloc_zeros(rows * 3 * hidden)?,
            attn: stream.alloc_zeros(rows * hidden)?,
            ffn: stream.alloc_zeros(rows * ffn)?,
            scores: stream.alloc_zeros(head_chunk * rows * rows)?,
            pos_embed_img: stream.alloc_zeros(max_patches * hidden)?,
            positions: stream.alloc_zeros(2 * rows)?,
            merged_mid: stream.alloc_zeros(max_tokens * merged)?,
            out_f16: stream.alloc_zeros(max_tokens * proj)?,
            out_f32: stream.alloc_zeros(max_tokens * proj)?,
            stage_f16: Vec::with_capacity(max_patches * patch_len.max(hidden)),
            stage_pos: Vec::with_capacity(2 * rows),
            kernels,
            stream,
            cfg: *cfg,
            max_patches,
            head_chunk,
        })
    }

    /// The patch budget this instance was allocated for.
    pub fn max_patches(&self) -> usize {
        self.max_patches
    }

    /// The tower's configuration.
    pub fn config(&self) -> &InklingVisionConfig {
        &self.cfg
    }

    /// Encode one image: its `(grid_w / 2) * (grid_h / 2)` rows of
    /// `projection_dim` floats, in raster order over the merged grid, at
    /// the start of the returned buffer. The stream is synchronized.
    pub fn encode(
        &mut self,
        image: &PreprocessedInklingImage,
    ) -> Result<&CudaSlice<f32>, VisionForwardError> {
        let cfg = self.cfg;
        let hidden = cfg.hidden_size as usize;
        let heads = cfg.num_heads as usize;
        let d_head = cfg.head_dim() as usize;
        let ffn = cfg.ffn_size as usize;
        let patch_len = cfg.patch_elems() as usize;
        let (gw, gh) = (image.grid_w, image.grid_h);
        let patches = (gw * gh) as usize;
        if patches > self.max_patches {
            return Err(VisionForwardError::TooManyPatches {
                patches,
                max: self.max_patches,
            });
        }
        assert_eq!(
            image.patches.len(),
            patches * patch_len,
            "patch buffer size"
        );
        assert!(
            gw.is_multiple_of(2) && gh.is_multiple_of(2),
            "whole 2x2 cells"
        );
        let stream = &self.stream;
        let rows = patches + 1;

        // Per-image inputs in cell order: pixels, the resized position
        // grid, and each row's (x + 1, y + 1); the class token sits last,
        // at (0, 0).
        let cell = |raster: usize| cell_order_index(raster as u32 % gw, raster as u32 / gw, gw);
        self.stage_f16.clear();
        self.stage_f16.resize(patches * patch_len, f16::ZERO);
        for (raster, patch) in image.patches.chunks_exact(patch_len).enumerate() {
            let at = cell(raster) * patch_len;
            for (to, v) in self.stage_f16[at..at + patch_len].iter_mut().zip(patch) {
                *to = f16::from_f32(*v);
            }
        }
        stream.memcpy_htod(
            &self.stage_f16,
            &mut self.patches_in.slice_mut(..patches * patch_len),
        )?;
        let pos = resized_position_embedding(
            &self.pos_embed_host,
            hidden,
            cfg.pos_grid_edge() as usize,
            gw,
            gh,
        );
        self.stage_f16.clear();
        self.stage_f16.resize(patches * hidden, f16::ZERO);
        for (raster, row) in pos.chunks_exact(hidden).enumerate() {
            let at = cell(raster) * hidden;
            for (to, v) in self.stage_f16[at..at + hidden].iter_mut().zip(row) {
                *to = f16::from_f32(*v);
            }
        }
        stream.memcpy_htod(
            &self.stage_f16,
            &mut self.pos_embed_img.slice_mut(..patches * hidden),
        )?;
        self.stage_pos.clear();
        self.stage_pos.resize(2 * rows, 0);
        for raster in 0..patches {
            let at = cell(raster);
            self.stage_pos[2 * at] = (raster as u32 % gw) as i32 + 1;
            self.stage_pos[2 * at + 1] = (raster as u32 / gw) as i32 + 1;
        }
        stream.memcpy_htod(&self.stage_pos, &mut self.positions.slice_mut(..2 * rows))?;

        // Patch embedding (no bias) plus position embedding; the class row.
        self.kernels.proj(
            &self.patches_in,
            &self.patch_embed,
            None,
            &mut self.embedded,
            patches,
            patch_len,
            hidden,
            0.0,
        )?;
        self.kernels.add_assign(
            stream,
            &mut self.embedded,
            &self.pos_embed_img,
            patches * hidden,
        )?;
        stream.memcpy_htod(
            &self.class_row,
            &mut self.embedded.slice_mut(patches * hidden..rows * hidden),
        )?;
        // The residual stream starts from the pre-norm's output.
        self.kernels.layer_norm(
            stream,
            &self.embedded,
            &mut self.x,
            &self.pre_ln_w,
            &self.pre_ln_b,
            rows,
            hidden,
            cfg.ln_eps,
        )?;

        let scale = 1.0 / (d_head as f32).sqrt();
        for blk in &self.blocks {
            self.kernels.layer_norm(
                stream,
                &self.x,
                &mut self.normed,
                &blk.ln1_w,
                &blk.ln1_b,
                rows,
                hidden,
                cfg.ln_eps,
            )?;
            self.kernels.proj(
                &self.normed,
                &blk.qkv_w,
                Some(&blk.qkv_b),
                &mut self.qkv,
                rows,
                hidden,
                3 * hidden,
                0.0,
            )?;
            self.kernels.rope_qk(
                stream,
                &mut self.qkv,
                &self.positions,
                rows,
                hidden,
                d_head,
                cfg.rope_theta,
            )?;
            let mut h0 = 0usize;
            while h0 < heads {
                let chunk = self.head_chunk.min(heads - h0);
                self.kernels.attn_scores(
                    &self.qkv,
                    h0 * d_head,
                    hidden + h0 * d_head,
                    &mut self.scores,
                    rows,
                    3 * hidden,
                    d_head,
                    chunk,
                    scale,
                )?;
                self.kernels
                    .softmax_rows(stream, &mut self.scores, chunk * rows, rows)?;
                self.kernels.attn_output(
                    &self.scores,
                    &self.qkv,
                    2 * hidden + h0 * d_head,
                    &mut self.attn,
                    h0 * d_head,
                    rows,
                    3 * hidden,
                    hidden,
                    d_head,
                    chunk,
                )?;
                h0 += chunk;
            }
            // Output projection, layer scale folded in, onto the residual.
            self.kernels.proj(
                &self.attn,
                &blk.out_w,
                Some(&blk.out_b),
                &mut self.x,
                rows,
                hidden,
                hidden,
                1.0,
            )?;
            self.kernels.layer_norm(
                stream,
                &self.x,
                &mut self.normed,
                &blk.ln2_w,
                &blk.ln2_b,
                rows,
                hidden,
                cfg.ln_eps,
            )?;
            self.kernels.proj(
                &self.normed,
                &blk.up_w,
                Some(&blk.up_b),
                &mut self.ffn,
                rows,
                hidden,
                ffn,
                0.0,
            )?;
            self.kernels.gelu_erf(stream, &mut self.ffn, rows * ffn)?;
            self.kernels.proj(
                &self.ffn,
                &blk.down_w,
                Some(&blk.down_b),
                &mut self.x,
                rows,
                ffn,
                hidden,
                1.0,
            )?;
        }

        // No final norm. Four consecutive rows are one merged row, and the
        // class token, last, is in none of them.
        let merge = cfg.merge_factor() as usize;
        let merged = cfg.merger_input_dim() as usize;
        let proj = cfg.projection_dim as usize;
        let tokens = patches / merge;
        self.kernels.proj(
            &self.x,
            &self.mm1_w,
            None,
            &mut self.merged_mid,
            tokens,
            merged,
            merged,
            0.0,
        )?;
        self.kernels
            .gelu_erf(stream, &mut self.merged_mid, tokens * merged)?;
        self.kernels.proj(
            &self.merged_mid,
            &self.mm2_w,
            None,
            &mut self.out_f16,
            tokens,
            merged,
            proj,
            0.0,
        )?;
        self.kernels
            .to_f32(stream, &self.out_f16, &mut self.out_f32, tokens * proj)?;
        stream.synchronize()?;
        Ok(&self.out_f32)
    }

    /// [`Self::encode`], read back to the host.
    pub fn encode_to_host(
        &mut self,
        image: &PreprocessedInklingImage,
    ) -> Result<Vec<f32>, VisionForwardError> {
        let tokens = (image.grid_w / 2 * (image.grid_h / 2)) as usize;
        let len = tokens * self.cfg.projection_dim as usize;
        let stream = Arc::clone(&self.stream);
        let out = self.encode(image)?;
        Ok(stream.clone_dtoh(&out.slice(..len))?)
    }
}

#[cfg(test)]
mod tests {
    use super::rope_slot;

    #[test]
    fn rotary_slots_pair_adjacent_dims_across_the_half() {
        // x pairs (0, 1), (2, 3) .. fill slots (0, 32), (1, 33) ..; y pairs
        // (32, 33) .. fill (16, 48) ..: a permutation of the head.
        assert_eq!((rope_slot(0, 64), rope_slot(1, 64)), (0, 32));
        assert_eq!((rope_slot(30, 64), rope_slot(31, 64)), (15, 47));
        assert_eq!((rope_slot(32, 64), rope_slot(33, 64)), (16, 48));
        assert_eq!((rope_slot(62, 64), rope_slot(63, 64)), (31, 63));
        let mut seen = [false; 64];
        for d in 0..64 {
            seen[rope_slot(d, 64)] = true;
        }
        assert!(seen.iter().all(|s| *s));
    }
}
