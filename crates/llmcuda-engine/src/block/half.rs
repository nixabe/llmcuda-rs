//! The fp16 tensor-core operand for prefill-width projections.
//!
//! The mixer blocks' tensor-core projections stage their activations once and
//! read them from every projection that shares an input (`qkv` and `gate`
//! both contract the normed hidden state). On the integer path that staging
//! is `quantize_rows`; here it is a rounding to fp16, and the projection is
//! [`HgemmKernels::gemm`] over the same split Q8_0 weights — or over a bf16
//! tensor, which has no integer path at all.
//!
//! Whether a model takes this path is decided once, at construction:
//! [`enabled_for`] reads the model's FFN kind and `LLMCUDA_HALF_GEMM`.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use llmcuda_cuda::kernels::hgemm::{HalfWeight, HgemmError, HgemmKernels};

/// Whether the mixer blocks' prefill projections run on the fp16 tensor
/// cores for a model whose feed-forward is `dense`.
///
/// On by default for the dense models (`qwen35`, `clef`), off for
/// `qwen35moe` until it has been measured there: the engine's primary target
/// is not changed by a default it has no numbers for. `LLMCUDA_HALF_GEMM=0`
/// turns it off everywhere and `=1` on everywhere, for A/B.
pub fn enabled_for(dense: bool) -> bool {
    match std::env::var("LLMCUDA_HALF_GEMM").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => dense,
    }
}

/// The compiled GEMM, its staged fp16 activations and its split-K workspace.
pub struct HalfStage {
    kernels: HgemmKernels,
    /// `[tokens][k]` fp16, sized by [`Self::reserve`].
    x: Option<CudaSlice<u16>>,
    /// Split-K partials, sized by [`Self::reserve`] for the widest split any
    /// reserved shape plans.
    workspace: Option<CudaSlice<f32>>,
}

impl HalfStage {
    /// Compile for the context's device. Nothing is allocated until
    /// [`Self::reserve`].
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, HgemmError> {
        Ok(Self {
            kernels: HgemmKernels::new(ctx)?,
            x: None,
            workspace: None,
        })
    }

    /// Grow the buffers to serve `n_rows x k` projections of up to
    /// `max_tokens` tokens. Called for every shape at construction, so the
    /// forward pass never allocates (`AGENTS.md` rule 6).
    pub fn reserve(
        &mut self,
        stream: &Arc<CudaStream>,
        max_tokens: usize,
        n_rows: usize,
        k: usize,
    ) -> Result<(), HgemmError> {
        let x = max_tokens * k;
        if self.x.as_ref().is_none_or(|b| b.len() < x) {
            self.x = Some(stream.alloc_zeros::<u16>(x)?);
        }
        let ws = self.kernels.workspace_elems(max_tokens, n_rows, k);
        if ws > 0 && self.workspace.as_ref().is_none_or(|b| b.len() < ws) {
            self.workspace = Some(stream.alloc_zeros::<f32>(ws)?);
        }
        Ok(())
    }

    /// Round `elements` of `x` to fp16 as the operand of the projections that
    /// follow.
    pub fn stage(
        &mut self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        elements: usize,
    ) -> Result<(), HgemmError> {
        let buf = self.x.as_mut().expect("reserve() runs at construction");
        self.kernels.to_half(stream, x, buf, elements)
    }

    /// `out[t][n] = sum_k staged[t][k] * w[n][k]` for `t < tokens`, `out`
    /// rows `n_rows` apart.
    #[allow(clippy::too_many_arguments)]
    pub fn project(
        &mut self,
        stream: &Arc<CudaStream>,
        w: HalfWeight<'_>,
        out: &mut CudaSlice<f32>,
        k_dim: usize,
        n_rows: usize,
        tokens: usize,
    ) -> Result<(), HgemmError> {
        let x = self.x.as_ref().expect("reserve() runs at construction");
        self.kernels.gemm(
            stream,
            w,
            x,
            out,
            self.workspace.as_mut(),
            tokens,
            n_rows,
            k_dim,
            n_rows,
            false,
        )
    }
}
