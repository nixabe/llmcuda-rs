//! Gated DeltaNet, recurrent (decode) form, on the device.
//!
//! The critical path: 30 of Qwen3.6's 40 layers. Unlike Gated Attention this
//! has no flash-attention codebase to port from, so the reference is
//! `llmcuda_kernels::gdn::recurrent`, which was itself cross-derived from
//! llama.cpp's `ggml_compute_forward_gated_delta_net_one_chunk` and vLLM's
//! `fused_recurrent_gated_delta_rule_fwd_kernel`.
//!
//! ## Shape
//!
//! Per head: state `S` is `head_dim x head_dim` fp32, laid out `S[v][k]` with
//! the key index contiguous. head_dim is 128 and there are 32 value heads, so
//! one layer's state for one sequence is 2 MiB — constant in sequence length,
//! which is the whole point of the hybrid architecture.
//!
//! The 32 value heads share 16 query/key heads by **modulo**:
//! `qk_head = v_head % 16`, so value heads 0 and 16 share a query/key head —
//! *not* 0 and 1. llama.cpp's fused op indexes it as
//! `fastmodulo(h_idx, n_k_heads)` (`ggml/src/ggml-cuda/gated_delta_net.cu:37`),
//! and its non-fused fallback reaches the same mapping through
//! `ggml_repeat_4d` (`src/models/qwen35moe.cpp:466`), which *tiles* rather than
//! stretches. Dividing instead — `v_head / 2`, the other plausible reading —
//! pairs every value head with the wrong query and stays finite and fluent.
//! Against the captured `final_output-N`, `h % 16` reproduces it to
//! `max_abs 7.15e-7`, `h / 2` to `5.59e-1` at cosine `0.975`.
//!
//! ## Why two kernels
//!
//! `q` and `k` are L2-normalized per query/key head. Doing that inside the
//! state kernel would mean every block redundantly re-normalizing a vector
//! shared by 128 blocks, and would add two block-wide reductions to the hot
//! path. Splitting it out costs one extra launch and lets the state kernel
//! touch global memory exactly once in each direction:
//!
//! - read `S[v][k]`, write `S[v][k]` — nothing else.
//!
//! That matters because this kernel is purely bandwidth-bound. At 2 MiB of
//! state per layer per sequence, 30 layers, read and written once per token,
//! the floor is ~120 MiB/token of state traffic.
//!
//! ## One warp per state row
//!
//! A warp owns row `S[v_index][*]`, four consecutive key indices per lane as
//! one `float4`. One *thread* per row is rejected — it needs the row
//! twice, because 128 floats will not stay in one thread's registers — and
//! a whole *block* per row, one thread per element, reads the row once but
//! pays two block-wide reductions (two `__syncthreads`, a shared round
//! trip, a serial cross-warp sum) per row and measured 44% of the
//! streaming roofline. The warp form is
//! the structure between the two: the row stays in four registers per lane,
//! both reductions are barrier-free shuffle butterflies, and every load and
//! store is a 16-byte vector.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, DriverError, LaunchConfig,
    PushKernelArg,
};

/// The epsilon inside the L2 normalization.
///
/// It is **added to the sum of squares, under the root** — see the
/// `gdn_normalize_qk` source. That is flash-linear-attention's `l2norm`,
/// the norm the model was trained with; transformers hardcodes `1e-6` for it
/// (`modeling_qwen3_5_moe.py`, `l2norm(query, dim=-1, eps=1e-6)`) and
/// llama.cpp passes `hparams.f_norm_rms_eps`, which is `1e-6` on every
/// `qwen35`/`qwen35moe` file, to `build_gdn_l2_norm`
/// (`src/models/models.h`).
///
/// Under this form the value matters: it moves every q and k by
/// `eps/(2*sum)`. `llmcuda_kernels::gdn::recurrent::l2_normalize` is called
/// with the same constant.
pub const L2_EPS: f32 = 1e-6;

/// Warps per block of the recurrent step kernel — one warp per state row.
const STEP_WARPS: u32 = 4;

/// Widest batch one `gdn_recurrent_step` launch covers — the kernel carries
/// this many per-sequence state pointer slots.
pub const STEP_MAX_BATCH: usize = 8;

/// Something went wrong compiling or launching a GDN kernel.
#[derive(Debug)]
pub enum GdnError {
    /// NVRTC rejected the source, or the module failed to load.
    Compile(String),
    /// The driver failed.
    Driver(DriverError),
    /// A head dimension the kernel cannot service.
    UnsupportedHeadDim { head_dim: usize },
    /// Value heads are not a whole multiple of query/key heads.
    UnevenHeadGrouping { value_heads: usize, qk_heads: usize },
}

impl std::fmt::Display for GdnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Compile(m) => write!(f, "kernel compilation failed: {m}"),
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::UnsupportedHeadDim { head_dim } => write!(
                f,
                "head_dim {head_dim} must be a positive multiple of 128 and at most 512",
            ),
            Self::UnevenHeadGrouping {
                value_heads,
                qk_heads,
            } => write!(
                f,
                "{value_heads} value heads do not divide evenly among {qk_heads} query/key heads",
            ),
        }
    }
}

impl std::error::Error for GdnError {}

impl From<DriverError> for GdnError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

/// Buffers reused across decode steps.
///
/// Normalized `q` and `k` are scratch, not state: they are overwritten every
/// step. Holding them here rather than allocating per step keeps the decode
/// path free of `cudaMalloc`, which is a synchronising call and would make
/// graph capture impossible.
pub struct GdnScratch {
    q_norm: CudaSlice<f32>,
    k_norm: CudaSlice<f32>,
}

/// Compiled Gated DeltaNet kernels.
pub struct GdnKernels {
    normalize: CudaFunction,
    step: CudaFunction,
    head_dim: usize,
    value_heads: usize,
    qk_heads: usize,
}

impl GdnKernels {
    /// Compile for a specific head geometry.
    ///
    /// The geometry is fixed at construction rather than passed per launch
    /// because it is fixed by the model, and validating it once means the
    /// launch path has nothing left to reject.
    pub fn new(
        ctx: &Arc<CudaContext>,
        head_dim: usize,
        value_heads: usize,
        qk_heads: usize,
    ) -> Result<Self, GdnError> {
        // A warp owns four consecutive key indices per 128-element chunk of
        // a state row, and the kernel spells out four chunks, so the head
        // dimension must be a whole number of 128-element chunks and at most
        // 512.
        if head_dim == 0 || !head_dim.is_multiple_of(128) || head_dim > 512 {
            return Err(GdnError::UnsupportedHeadDim { head_dim });
        }
        if qk_heads == 0 || !value_heads.is_multiple_of(qk_heads) {
            return Err(GdnError::UnevenHeadGrouping {
                value_heads,
                qk_heads,
            });
        }
        let rust = ctx.load_module(cudarc::nvrtc::Ptx::from_src(super::RUST_KERNELS_PTX))?;
        Ok(Self {
            normalize: rust.load_function("gdn_normalize_qk")?,
            step: rust.load_function("gdn_recurrent_step")?,
            head_dim,
            value_heads,
            qk_heads,
        })
    }

    /// How many value heads share each query/key head.
    ///
    /// The *count*, not the mapping: value head `h` reads query/key head
    /// `h % qk_heads`, so the heads sharing a query/key head are `hq`,
    /// `hq + qk_heads`, `hq + 2 * qk_heads`, … and not a contiguous run.
    pub fn heads_per_kv(&self) -> usize {
        self.value_heads / self.qk_heads
    }

    /// The `1/sqrt(head_dim)` scale folded into `q`.
    pub fn output_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    /// Allocate the per-step scratch buffers, sized for the widest batch
    /// [`Self::step_batch_raw`] accepts so a batch of any admissible width never
    /// reallocates.
    pub fn scratch(&self, stream: &Arc<CudaStream>) -> Result<GdnScratch, GdnError> {
        let n = STEP_MAX_BATCH * self.qk_heads * self.head_dim;
        Ok(GdnScratch {
            q_norm: stream.alloc_zeros::<f32>(n)?,
            k_norm: stream.alloc_zeros::<f32>(n)?,
        })
    }

    /// Bytes of recurrent state for one sequence at this geometry.
    pub fn state_bytes(&self) -> usize {
        self.value_heads * self.head_dim * self.head_dim * size_of::<f32>()
    }

    /// Advance one sequence's `state` by one token and write its output.
    ///
    /// [`Self::step_batch_raw`] at a batch of one — the arithmetic is the same
    /// kernel, so a sequence decodes identically here and in a batch.
    ///
    /// `q` and `k` are `[qk_heads][head_dim]` raw — not normalized, not
    /// scaled; this does both. `v` is `[value_heads][head_dim]`, `log_decay`
    /// and `beta` are `[value_heads]`, and `out` is `[value_heads][head_dim]`.
    ///
    /// `state` is `[value_heads][head_dim][head_dim]` and is updated in place.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &self,
        stream: &Arc<CudaStream>,
        scratch: &mut GdnScratch,
        state: &mut CudaSlice<f32>,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        log_decay: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), GdnError> {
        let (ptr, _sync) = state.device_ptr(stream);
        // SAFETY: `ptr` is `state`'s own base address, held mutably for the
        // duration of this call; the batch of one touches nothing else.
        unsafe { self.step_batch_raw(stream, scratch, &[ptr], q, k, v, log_decay, beta, out) }
    }

    /// Advance `n` sequences' states by one token each, in one launch pair.
    ///
    /// Per-sequence launches are slow twice over: 2n launches per layer of
    /// launch floor, and — for the state kernel — one sequence's 4 MiB of
    /// traffic per launch where one launch can stream all of them. Every
    /// activation buffer is the batch scratch
    /// (`[n][width]`, sequence-major); only the state is per-sequence, and it
    /// rides in as `STEP_MAX_BATCH` scalar pointer slots so graph capture
    /// sees exactly the pointer stability the per-sequence launches gave it.
    ///
    /// # Safety
    ///
    /// Each entry of `states` must be the base address of a live
    /// `[value_heads][head_dim][head_dim]` f32 state allocation, exclusively
    /// held by the caller for this call, all distinct.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn step_batch_raw(
        &self,
        stream: &Arc<CudaStream>,
        scratch: &mut GdnScratch,
        states: &[u64],
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        log_decay: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), GdnError> {
        let n = states.len();
        assert!(
            (1..=STEP_MAX_BATCH).contains(&n),
            "step batch of {n} outside 1..={STEP_MAX_BATCH}"
        );
        let head_dim = self.head_dim as i32;
        let qk_heads = self.qk_heads as i32;
        // One float per warp, which is the most `block_reduce_sum` stores.
        let shared = (self.head_dim.div_ceil(32) * size_of::<f32>()) as u32;

        // The normalize kernel indexes `blockIdx.x * head_dim` over
        // contiguous `[n][qk_heads][head_dim]` buffers, so a batch is simply
        // more blocks — same kernel, same per-head arithmetic.
        let norm_cfg = LaunchConfig {
            grid_dim: ((n * self.qk_heads) as u32, 1, 1),
            block_dim: (self.head_dim as u32, 1, 1),
            shared_mem_bytes: shared,
        };
        let eps = L2_EPS;
        let scale = self.output_scale();
        let mut builder = stream.launch_builder(&self.normalize);
        builder
            .arg(q)
            .arg(k)
            .arg(&mut scratch.q_norm)
            .arg(&mut scratch.k_norm)
            .arg(&head_dim)
            .arg(&eps)
            .arg(&scale);
        // SAFETY: one block per (sequence, qk head) pair and one thread per
        // head element; the scratch is allocated to
        // `STEP_MAX_BATCH * qk_heads * head_dim` and `n` is bounded above.
        unsafe { builder.launch(norm_cfg) }?;

        // Unused pointer slots repeat slot 0; `grid.z = n` means no block
        // ever selects them.
        let mut slots = [states[0]; STEP_MAX_BATCH];
        slots[..n].copy_from_slice(states);

        // One warp per state row, `STEP_WARPS` rows per block, one grid.z
        // slice per sequence, no shared memory: both reductions are
        // intra-warp butterflies.
        let step_cfg = LaunchConfig {
            grid_dim: (
                (self.head_dim as u32).div_ceil(STEP_WARPS),
                self.value_heads as u32,
                n as u32,
            ),
            block_dim: (32 * STEP_WARPS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&self.step);
        for slot in &slots {
            builder.arg(slot);
        }
        builder
            .arg(&scratch.q_norm)
            .arg(&scratch.k_norm)
            .arg(v)
            .arg(log_decay)
            .arg(beta)
            .arg(out)
            .arg(&head_dim)
            .arg(&qk_heads);
        // SAFETY: the grid is (head_dim / STEP_WARPS, value_heads, n) with a
        // warp per state row, so the flat state index stays within one
        // sequence's `value_heads * head_dim * head_dim` allocation, whose
        // validity and exclusivity the caller asserts; every batch buffer is
        // indexed by `z` strides the caller allocated it with.
        unsafe { builder.launch(step_cfg) }?;
        Ok(())
    }
}
