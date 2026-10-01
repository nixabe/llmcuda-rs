//! Attention dispatch chosen from the declared architecture at construction.
use super::attention::{
    AttentionBlockError, AttentionKernelSet, AttentionLayerWeights, AttnScratch,
    GatedAttentionBlock, KvCache, RopeSource,
};
use super::k2::{K2AttentionBlock, K2AttentionKernels, K2AttentionWeights};
use crate::DeviceWeights;
use cudarc::driver::{CudaContext, CudaEvent, CudaSlice, CudaStream};
use llmcuda_gguf::GgufFile;
use llmcuda_model::{Directory, ModelConfig};
use std::sync::Arc;

pub(crate) enum MixerKernelSet {
    Qwen(Arc<AttentionKernelSet>),
    K2(Arc<K2AttentionKernels>),
}
impl MixerKernelSet {
    pub(crate) fn new(
        ctx: &Arc<CudaContext>,
        c: &ModelConfig,
        tokens: usize,
    ) -> Result<Self, AttentionBlockError> {
        if c.k2.is_some() {
            Ok(Self::K2(Arc::new(K2AttentionKernels::new(ctx, c)?)))
        } else {
            Ok(Self::Qwen(Arc::new(AttentionKernelSet::new(
                ctx, c, tokens,
            )?)))
        }
    }
    pub(crate) fn k2_ops(&self) -> Option<Arc<llmcuda_cuda::kernels::k2::K2Kernels>> {
        match self {
            Self::K2(k) => Some(Arc::clone(&k.ops)),
            Self::Qwen(_) => None,
        }
    }
}
pub(crate) enum MixerLayerWeights {
    Qwen(Arc<AttentionLayerWeights>),
    K2(Arc<K2AttentionWeights>),
}
#[allow(clippy::large_enum_variant)]
pub(crate) enum MixerBlock {
    Qwen(GatedAttentionBlock),
    K2(K2AttentionBlock),
}
impl MixerBlock {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        kernels: Arc<MixerKernelSet>,
        stream: &Arc<CudaStream>,
        weights: &DeviceWeights,
        file: &GgufFile,
        d: &Directory<'_>,
        c: &ModelConfig,
        layer: u32,
        tokens: usize,
        eps: f32,
        theta: f32,
    ) -> Result<Self, AttentionBlockError> {
        match &*kernels {
            MixerKernelSet::Qwen(k) => Ok(Self::Qwen(GatedAttentionBlock::new(
                Arc::clone(k),
                stream,
                weights,
                file,
                d,
                c,
                layer,
                tokens,
                eps,
                theta,
            )?)),
            MixerKernelSet::K2(k) => Ok(Self::K2(K2AttentionBlock::new(
                Arc::clone(k),
                Arc::new(K2AttentionWeights::upload(stream, file, d, c, layer)?),
                c,
                tokens,
                eps,
                theta,
            ))),
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_shared(
        kernels: Arc<MixerKernelSet>,
        stream: &Arc<CudaStream>,
        weights: Arc<MixerLayerWeights>,
        c: &ModelConfig,
        layer: u32,
        tokens: usize,
        eps: f32,
        theta: f32,
    ) -> Result<Self, AttentionBlockError> {
        match (&*kernels, &*weights) {
            (MixerKernelSet::Qwen(k), MixerLayerWeights::Qwen(w)) => {
                Ok(Self::Qwen(GatedAttentionBlock::from_shared(
                    Arc::clone(k),
                    stream,
                    Arc::clone(w),
                    c,
                    layer,
                    tokens,
                    eps,
                    theta,
                )?))
            }
            (MixerKernelSet::K2(k), MixerLayerWeights::K2(w)) => Ok(Self::K2(
                K2AttentionBlock::new(Arc::clone(k), Arc::clone(w), c, tokens, eps, theta),
            )),
            _ => Err(AttentionBlockError::NotAnAttentionLayer { layer }),
        }
    }
    pub(crate) fn shared_weights(&self) -> Arc<MixerLayerWeights> {
        Arc::new(match self {
            Self::Qwen(b) => MixerLayerWeights::Qwen(b.shared_weights()),
            Self::K2(b) => MixerLayerWeights::K2(Arc::clone(&b.weights)),
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &mut self,
        stream: &Arc<CudaStream>,
        sc: &mut AttnScratch,
        hidden_state: &CudaSlice<f32>,
        cache: &mut KvCache,
        pos_offset: usize,
        positions: &CudaSlice<i32>,
        rope: RopeSource<'_>,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), AttentionBlockError> {
        match self {
            Self::Qwen(b) => b.forward(
                stream,
                sc,
                hidden_state,
                cache,
                pos_offset,
                positions,
                rope,
                out,
            ),
            Self::K2(b) => b.forward(
                stream,
                sc,
                hidden_state,
                cache,
                pos_offset,
                positions,
                rope,
                out,
            ),
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_batch_prefill(
        &mut self,
        stream: &Arc<CudaStream>,
        sc: &mut AttnScratch,
        hidden_state: &CudaSlice<f32>,
        caches: &mut [&mut KvCache],
        chunk_tokens: usize,
        pos_offsets: &[usize],
        positions: &[&CudaSlice<i32>],
        rope_positions: &[&CudaSlice<i32>],
        out: &mut CudaSlice<f32>,
    ) -> Result<(), AttentionBlockError> {
        match self {
            Self::Qwen(b) => b.forward_batch_prefill(
                stream,
                sc,
                hidden_state,
                caches,
                chunk_tokens,
                pos_offsets,
                positions,
                rope_positions,
                out,
            ),
            Self::K2(b) => b.batch(
                stream,
                sc,
                hidden_state,
                caches,
                chunk_tokens,
                pos_offsets,
                positions,
                rope_positions,
                out,
            ),
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_batch_decode(
        &mut self,
        stream: &Arc<CudaStream>,
        secondary_streams: &[Arc<CudaStream>],
        fork: &CudaEvent,
        joins: &[CudaEvent],
        sc: &mut AttnScratch,
        hidden_state: &CudaSlice<f32>,
        caches: &mut [&mut KvCache],
        pos_offsets: &[usize],
        positions: &[&CudaSlice<i32>],
        rope_positions: &[&CudaSlice<i32>],
        out: &mut CudaSlice<f32>,
    ) -> Result<(), AttentionBlockError> {
        match self {
            Self::Qwen(b) => b.forward_batch_decode(
                stream,
                secondary_streams,
                fork,
                joins,
                sc,
                hidden_state,
                caches,
                pos_offsets,
                positions,
                rope_positions,
                out,
            ),
            Self::K2(b) => b.batch(
                stream,
                sc,
                hidden_state,
                caches,
                1,
                pos_offsets,
                positions,
                rope_positions,
                out,
            ),
        }
    }
    pub(crate) fn disable_tensor_cores(&mut self) {
        match self {
            Self::Qwen(b) => b.disable_tensor_cores(),
            Self::K2(_b) => {}
        }
    }
    pub(crate) fn enable_exact_decode(&mut self) {
        match self {
            Self::Qwen(b) => b.enable_exact_decode(),
            Self::K2(_b) => {}
        }
    }
    pub(crate) fn disable_decode_mma(&mut self) {
        match self {
            Self::Qwen(b) => b.disable_decode_mma(),
            Self::K2(_b) => {
                _b.kernels.attention.disable_decode_mma();
            }
        }
    }
    pub(crate) fn set_decode_mma_wpo(&mut self, wpo: usize) {
        match self {
            Self::Qwen(b) => b.set_decode_mma_wpo(wpo),
            Self::K2(_b) => {
                _b.kernels.attention.set_decode_mma_wpo(wpo);
            }
        }
    }
    pub(crate) fn set_decode_mma_blocks(&mut self, blocks: usize) {
        match self {
            Self::Qwen(b) => b.set_decode_mma_blocks(blocks),
            Self::K2(_b) => {
                _b.kernels.attention.set_decode_mma_blocks(blocks);
            }
        }
    }
    pub(crate) fn owned_k2_bytes(&self) -> usize {
        match self {
            Self::K2(b) => b.weights.bytes(),
            Self::Qwen(_) => 0,
        }
    }
    pub(crate) fn tensor_cores_enabled(&self) -> bool {
        match self {
            Self::Qwen(b) => b.tensor_cores_enabled(),
            Self::K2(_) => false,
        }
    }
}
