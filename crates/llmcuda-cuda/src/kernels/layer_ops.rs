//! RMSNorm, partial rotary embedding, SwiGLU, sigmoid gating, tensor add,
//! softplus, and the GDN short convolution.
//!
//! Every operation a forward pass needs that [`super::gdn`],
//! [`super::gdn_chunked`], [`super::attention`] and [`super::moe`] do not
//! already cover. They live here, as standalone tested kernels, rather than
//! inline in the blocks that use them: an engine-side inline kernel is
//! invisible to the kernel inventory and gets no differential test against the
//! `llmcuda-kernels` oracle. Fusing one *into* a block is still legitimate — the
//! GDN gate kernel computes softplus, the log-decay and beta's sigmoid in one
//! pass — but the standalone op has to exist and be the thing a fused variant
//! is checked against.
//!
//! References: `llmcuda_kernels::norm::rms_norm`,
//! `llmcuda_kernels::rope::apply_rope`, `llmcuda_kernels::norm::swiglu`,
//! `llmcuda_kernels::norm::sigmoid_gate`, `llmcuda_kernels::norm::residual_add`,
//! `llmcuda_kernels::norm::softplus`,
//! `llmcuda_kernels::conv::causal_depthwise_conv1d`.
//!
//! # Geometry is per launch, not per instance
//!
//! [`super::gdn`] fixes its geometry at construction because the model fixes
//! it. These cannot: RMSNorm alone runs at three widths in one pass — hidden
//! 2048 for the input, post-mixer and final norms; head_dim 256 for the
//! attention layers' `attn_q_norm`/`attn_k_norm`; head_dim 128 for the GDN
//! output norm. So the width is a launch argument, validated per launch. The
//! sigmoid gate has the same problem in a sharper form: its gate is
//! elementwise for the attention output gate and **one scalar per token** for
//! the MoE shared expert, so the shape is a [`GateShape`] argument and the
//! gate buffer's length is checked against it on every launch.
//!
//! # What is exact, and what cannot be
//!
//! - **The tensor add, the rotary tail, and the convolution are bit-exact**
//!   and gated with `Tolerance::exact()`. The rotary tail is copied, not
//!   computed — partial rotary is easy to accidentally rotate, truncate or
//!   reorder past its boundary, and a tolerance would pass all three. The
//!   convolution rounds each product before adding it (the Rust PTX is built
//!   with multiply-add fusion off) because an FMA would round once where the
//!   reference rounds twice, and a tolerance can hide a reversed tap order on
//!   smooth input where exact equality cannot.
//! - **RMSNorm, SwiGLU, the sigmoid gate and softplus cannot be.** RMSNorm
//!   reduces 2048 squares in a shuffle tree where the reference sums
//!   sequentially. The other three are elementwise, so their entire
//!   disagreement is `expf` against `f32::exp` — about an ulp. All four are
//!   gated on a measured tolerance sized for that, not on a matmul-era one.
//!
//! # One language
//!
//! Every kernel here is Rust, compiled ahead of time
//! ([`RUST_KERNELS_PTX`](super::RUST_KERNELS_PTX)); this module no longer
//! invokes NVRTC. The source is [`RUST_KERNELS_SRC`](super::RUST_KERNELS_SRC).

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};

/// Widest convolution the launchers will service.
///
/// The fused Gated DeltaNet decode kernels stage `conv_kernel - 1` window
/// values in a register array of this size (`GDN_MAX_CONV_KERNEL`), and a
/// cache these launchers advance has to be one those kernels can read. The
/// Rust convolution kernels here need no bound of their own.
///
/// Qwen3.6 needs 4 (`qwen35moe.ssm.conv_kernel`). 8 is headroom, not a
/// surveyed requirement: llama.cpp reads `ssm_d_conv` from the file rather
/// than hardcoding it per architecture. Anything wider is rejected at the
/// launch path.
pub const MAX_CONV_KERNEL: usize = 8;

/// Largest block the RMSNorm and rotary kernels will launch.
///
/// One thread per head element for the rotary kernel, and the upper bound on
/// the strided reduction for RMSNorm.
const MAX_BLOCK: usize = 1024;

/// The NVRTC translation unit for the layer ops that are still CUDA C++:
/// Something went wrong compiling or launching a layer-op kernel.
#[derive(Debug)]
pub enum LayerOpsError {
    /// NVRTC rejected the source, or the module failed to load.
    Compile(String),
    /// The driver failed.
    Driver(DriverError),
    /// A row width the kernel cannot service.
    UnsupportedWidth { width: usize },
    /// A head geometry the rotary kernel cannot service.
    UnsupportedRotaryShape { head_dim: usize, rope_dim: usize },
    /// A convolution width the kernel cannot service.
    UnsupportedConvKernel { conv_kernel: usize },
    /// A buffer is not the length the declared geometry requires.
    ShapeMismatch {
        what: &'static str,
        expected: usize,
        got: usize,
    },
}

impl std::fmt::Display for LayerOpsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Compile(m) => write!(f, "kernel compilation failed: {m}"),
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::UnsupportedWidth { width } => {
                write!(f, "row width {width} must be non-zero")
            }
            Self::UnsupportedRotaryShape { head_dim, rope_dim } => write!(
                f,
                "rope_dim {rope_dim} must be even and at most head_dim {head_dim}, which must \
                 itself be non-zero and at most 1024 (it is the block width)",
            ),
            Self::UnsupportedConvKernel { conv_kernel } => write!(
                f,
                "conv_kernel {conv_kernel} must be in 1..={MAX_CONV_KERNEL}",
            ),
            Self::ShapeMismatch {
                what,
                expected,
                got,
            } => write!(f, "{what} must hold {expected} floats, holds {got}"),
        }
    }
}

impl std::error::Error for LayerOpsError {}

impl From<DriverError> for LayerOpsError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

/// The shape of the gate tensor [`LayerOpsKernels::sigmoid_gate`] is given.
///
/// Qwen3.6 has two sigmoid gates and they are not the same op. This is an
/// argument rather than an assumption because guessing wrong is not a crash:
/// a broadcast gate read elementwise walks `rows * width` floats off the end
/// of a `rows`-float buffer, and an elementwise gate read per-row applies
/// every token's first gate value to that whole token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateShape {
    /// One gate value per element: `gate` is `[rows][width]`, like `x`.
    ///
    /// The Gated Attention output gate — `attn_gate` in
    /// `src/models/qwen35moe.cpp`, `q_heads * head_dim = 4096` values per
    /// token.
    Elementwise,
    /// One gate value per row, broadcast across that row: `gate` is `[rows]`.
    ///
    /// The MoE shared expert's gate. `ffn_gate_inp_shexp` is a single
    /// `[hidden]` row, so its projection collapses to one scalar per token,
    /// which `build_layer_ffn` multiplies the whole shared-expert output by.
    PerRow,
}

impl GateShape {
    /// How many gate values a `[rows][width]` activation needs in this shape.
    ///
    /// Also the length of the `sig` output, which is the gate's shape rather
    /// than `x`'s.
    const fn gate_len(self, rows: usize, width: usize) -> usize {
        match self {
            Self::Elementwise => rows * width,
            Self::PerRow => rows,
        }
    }
}

/// The compiled layer-op kernels.
pub struct LayerOpsKernels {
    rms_norm: CudaFunction,
    rms_norm_swiglu: CudaFunction,
    rms_norm_swiglu_q8: CudaFunction,
    rope: CudaFunction,
    swiglu: CudaFunction,
    sigmoid_gate: CudaFunction,
    add: CudaFunction,
    softplus: CudaFunction,
    conv1d: CudaFunction,
    conv1d_state: CudaFunction,
    conv1d_step: CudaFunction,
    conv1d_step_batch: CudaFunction,
}

impl LayerOpsKernels {
    /// Load the layer ops: the Rust ones from
    /// [`RUST_KERNELS_PTX`](super::RUST_KERNELS_PTX), which lists them, and
    /// the rest compiled from CUDA C++ by NVRTC.
    ///
    /// All loading happens before hot-path launches or graph capture.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, LayerOpsError> {
        let rust = ctx.load_module(cudarc::nvrtc::Ptx::from_src(super::RUST_KERNELS_PTX))?;
        Ok(Self {
            rms_norm: rust.load_function("rms_norm_rows")?,
            rms_norm_swiglu: rust.load_function("rms_norm_swiglu_rows")?,
            rms_norm_swiglu_q8: rust.load_function("rms_norm_swiglu_q8_rows")?,
            rope: rust.load_function("rope_partial")?,
            swiglu: rust.load_function("swiglu_mul")?,
            sigmoid_gate: rust.load_function("sigmoid_gate_mul")?,
            add: rust.load_function("tensor_add")?,
            softplus: rust.load_function("softplus_elementwise")?,
            conv1d: rust.load_function("conv1d_causal_depthwise")?,
            conv1d_state: rust.load_function("conv1d_update_state")?,
            conv1d_step: rust.load_function("conv1d_step")?,
            conv1d_step_batch: rust.load_function("conv1d_step_batch")?,
        })
    }

    /// RMSNorm over `[rows][width]`, then `out = silu(gate) * normed`.
    ///
    /// One launch for what `rms_norm` and `swiglu` did in two. The normed
    /// intermediate is still written; see the kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_swiglu(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        normed: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        width: usize,
        eps: f32,
    ) -> Result<(), LayerOpsError> {
        if width == 0 {
            return Err(LayerOpsError::UnsupportedWidth { width });
        }
        let n = rows * width;
        check_len("rms_norm x", n, x.len())?;
        check_len("rms_norm weight", width, weight.len())?;
        check_len("swiglu gate", n, gate.len())?;
        check_len("rms_norm out", n, normed.len())?;
        check_len("swiglu out", n, out.len())?;
        if rows == 0 {
            return Ok(());
        }

        let block = block_for_width(width);
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (block as u32, 1, 1),
            shared_mem_bytes: (block.div_ceil(32) * size_of::<f32>()) as u32,
        };
        let width_i32 = width as i32;
        let mut builder = stream.launch_builder(&self.rms_norm_swiglu);
        builder
            .arg(x)
            .arg(weight)
            .arg(gate)
            .arg(&mut *normed)
            .arg(&mut *out)
            .arg(&width_i32)
            .arg(&eps);
        // SAFETY: as `rms_norm` below, with `gate` and `out` checked to the
        // same `rows * width` and indexed by the same expression.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// [`Self::rms_norm_swiglu`] followed by
    /// [`super::mma::MmaKernels::quantize_rows`], in one launch that writes
    /// neither float intermediate.
    ///
    /// `q` receives `rows * width` int8 and `scales` one fp32 per 32 of them,
    /// both laid out as the quantizer lays out `[rows * width / k][k]` for
    /// any `k` that is a multiple of `width`: bit-identical to the two-launch
    /// form. Both may be longer than this call fills.
    ///
    /// Returns `Ok(false)` without launching when `width` is not a whole
    /// number of 32-element blocks that fit one thread block, which is the
    /// shape the fusion needs; the caller then runs the two launches.
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_swiglu_quantize(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scales: &mut CudaSlice<f32>,
        rows: usize,
        width: usize,
        eps: f32,
    ) -> Result<bool, LayerOpsError> {
        if width == 0 || !width.is_multiple_of(32) || width > MAX_BLOCK {
            return Ok(false);
        }
        let n = rows * width;
        check_len("rms_norm x", n, x.len())?;
        check_len("rms_norm weight", width, weight.len())?;
        check_len("swiglu gate", n, gate.len())?;
        if q.len() < n {
            return Err(LayerOpsError::ShapeMismatch {
                what: "swiglu quants",
                expected: n,
                got: q.len(),
            });
        }
        if scales.len() < n / 32 {
            return Err(LayerOpsError::ShapeMismatch {
                what: "swiglu scales",
                expected: n / 32,
                got: scales.len(),
            });
        }
        if rows == 0 {
            return Ok(true);
        }

        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (width as u32, 1, 1),
            shared_mem_bytes: ((width / 32) * size_of::<f32>()) as u32,
        };
        let width_i32 = width as i32;
        let mut builder = stream.launch_builder(&self.rms_norm_swiglu_q8);
        builder
            .arg(x)
            .arg(weight)
            .arg(gate)
            .arg(&mut *q)
            .arg(&mut *scales)
            .arg(&width_i32)
            .arg(&eps);
        // SAFETY: one block per row and one thread per element, so every
        // thread reads `x` and `gate` and writes `q` at `row * width + j`
        // inside `rows * width`, and lane 0 of each warp writes `scales` at
        // that index over 32; all five lengths were checked above. Shared
        // memory covers one float per warp, all the reduction stores.
        unsafe { builder.launch(cfg) }?;
        Ok(true)
    }

    /// RMSNorm over `rows` independent rows of `width` elements each.
    ///
    /// `x` and `out` are `[rows][width]`; `weight` is one `width`-long vector
    /// shared by every row. Both hidden-size norms (`rows = tokens`,
    /// `width = 2048`) and per-head norms (`rows = tokens * heads`,
    /// `width = head_dim`) are this one call.
    ///
    /// `out` may alias `x`: every thread reads `x[base + j]` and writes
    /// `out[base + j]` for the same `j`, after the reduction's final
    /// `__syncthreads()`, so no thread can read an element another has
    /// already overwritten.
    ///
    /// Eight plain arguments rather than a config struct, for the same reason
    /// [`super::gdn_chunked::GdnChunkedKernels::prefill`] takes ten: every one
    /// is a distinct per-call tensor or shape, not related configuration that
    /// would be set once and reused.
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        width: usize,
        eps: f32,
    ) -> Result<(), LayerOpsError> {
        if width == 0 {
            return Err(LayerOpsError::UnsupportedWidth { width });
        }
        check_len("rms_norm x", rows * width, x.len())?;
        check_len("rms_norm weight", width, weight.len())?;
        check_len("rms_norm out", rows * width, out.len())?;
        if rows == 0 {
            return Ok(());
        }

        let block = block_for_width(width);
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (block as u32, 1, 1),
            // One float per warp, which is the most `block_reduce_sum` stores.
            shared_mem_bytes: (block.div_ceil(32) * size_of::<f32>()) as u32,
        };
        let width_i32 = width as i32;
        let mut builder = stream.launch_builder(&self.rms_norm);
        builder
            .arg(x)
            .arg(weight)
            .arg(out)
            .arg(&width_i32)
            .arg(&eps);
        // SAFETY: one block per row over buffers of `rows * width` floats,
        // with the in-row loop bounded by `width`, so `blockIdx.x * width + j`
        // is in range for every thread. `weight` is indexed by `j < width` and
        // holds `width` floats. Shared memory covers one float per warp, which
        // is all the reduction writes.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// Partial rotary embedding over `[seq_len][heads][head_dim]`.
    ///
    /// `positions` holds one absolute position per token. Dimensions
    /// `[rope_dim, head_dim)` are copied through bit-exactly.
    ///
    /// Unlike [`Self::rms_norm`], **`out` must not alias `x`**: element `j`
    /// of a rotated pair is written from elements `i` and `i + half`, so an
    /// in-place kernel would race. Enforced by the borrow checker taking `x`
    /// shared and `out` exclusive.
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        out: &mut CudaSlice<f32>,
        seq_len: usize,
        heads: usize,
        head_dim: usize,
        rope_dim: usize,
        theta_base: f32,
    ) -> Result<(), LayerOpsError> {
        if head_dim == 0
            || head_dim > MAX_BLOCK
            || !rope_dim.is_multiple_of(2)
            || rope_dim > head_dim
        {
            return Err(LayerOpsError::UnsupportedRotaryShape { head_dim, rope_dim });
        }
        let n = seq_len * heads * head_dim;
        check_len("rope x", n, x.len())?;
        check_len("rope out", n, out.len())?;
        check_len("rope positions", seq_len, positions.len())?;
        if seq_len == 0 || heads == 0 {
            return Ok(());
        }

        let cfg = LaunchConfig {
            grid_dim: (heads as u32, seq_len as u32, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let heads_i32 = heads as i32;
        let head_dim_i32 = head_dim as i32;
        let rope_dim_i32 = rope_dim as i32;
        let mut builder = stream.launch_builder(&self.rope);
        builder
            .arg(x)
            .arg(positions)
            .arg(out)
            .arg(&heads_i32)
            .arg(&head_dim_i32)
            .arg(&rope_dim_i32)
            .arg(&theta_base);
        // SAFETY: the grid is (heads, seq_len) and the block is head_dim
        // threads, so `(t * heads + h) * head_dim + j` covers exactly the
        // `seq_len * heads * head_dim` floats both buffers hold. The paired
        // reads at `i` and `i + half` stay below `rope_dim <= head_dim`.
        // `positions` is indexed by `t < seq_len` and holds `seq_len` entries.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// `out = silu(gate) * up` over `n` elements.
    ///
    /// The standalone form. [`super::moe`]'s grouped GEMM fuses the same
    /// activation into its epilogue; this exists for the paths that do not go
    /// through it and as the thing the fused version is checked against.
    pub fn swiglu(
        &self,
        stream: &Arc<CudaStream>,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), LayerOpsError> {
        check_len("swiglu gate", n, gate.len())?;
        check_len("swiglu up", n, up.len())?;
        check_len("swiglu out", n, out.len())?;
        if n == 0 {
            return Ok(());
        }

        let n_i64 = n as i64;
        let mut builder = stream.launch_builder(&self.swiglu);
        builder.arg(gate).arg(up).arg(out).arg(&n_i64);
        // SAFETY: the grid-stride loop is bounded by `n`, which is the length
        // checked above for all three buffers.
        unsafe { builder.launch(elementwise_cfg(n)) }?;
        Ok(())
    }

    /// `out = sigmoid(gate) * x` over a `[rows][width]` activation, writing
    /// the sigmoid itself to `sig`.
    ///
    /// `shape` says whether `gate` is elementwise or one scalar per row; see
    /// [`GateShape`]. `gate` and `sig` are both that shape's length —
    /// `rows * width` or `rows` — and `x` and `out` are always `rows * width`.
    /// Every one of those lengths is checked here rather than assumed, which
    /// is the point of taking the shape as an argument at all.
    ///
    /// `sig` exists so a divergence localizes to the nonlinearity or to the
    /// product rather than to "the gate", and so the caller keeps llama.cpp's
    /// `attn_gate_sigmoid` waypoint to compare against. In [`GateShape::PerRow`]
    /// it costs `rows` floats.
    ///
    /// `out` may alias `x`: each thread reads and writes the same index and
    /// reads `gate`, which is a different buffer. `sig` must not alias either.
    #[allow(clippy::too_many_arguments)]
    pub fn sigmoid_gate(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        sig: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        width: usize,
        shape: GateShape,
    ) -> Result<(), LayerOpsError> {
        if width == 0 {
            return Err(LayerOpsError::UnsupportedWidth { width });
        }
        let n = rows * width;
        let gate_len = shape.gate_len(rows, width);
        check_len("sigmoid_gate x", n, x.len())?;
        check_len("sigmoid_gate gate", gate_len, gate.len())?;
        check_len("sigmoid_gate sigmoid", gate_len, sig.len())?;
        check_len("sigmoid_gate out", n, out.len())?;
        if rows == 0 {
            return Ok(());
        }

        let n_i64 = n as i64;
        let width_i32 = width as i32;
        let broadcast_i32 = i32::from(shape == GateShape::PerRow);
        // Elementwise gating is the Rust kernel; the per-row broadcast stays
        // CUDA C++, where the Rust port loses at prefill sizes.
        let kernel = match shape {
            GateShape::Elementwise | GateShape::PerRow => &self.sigmoid_gate,
        };
        let mut builder = stream.launch_builder(kernel);
        builder
            .arg(x)
            .arg(gate)
            .arg(sig)
            .arg(out)
            .arg(&n_i64)
            .arg(&width_i32)
            .arg(&broadcast_i32);
        // SAFETY: the grid-stride loop is bounded by `n = rows * width`, the
        // checked length of `x` and `out`. The gate index is `i` when not
        // broadcasting and `i / width < rows` when broadcasting, which are
        // exactly the two lengths `gate` and `sig` were checked against just
        // above; `width != 0` so the division is defined.
        unsafe { builder.launch(elementwise_cfg(n)) }?;
        Ok(())
    }

    /// `out = a + b` over `n` elements — the residual add.
    ///
    /// Bit-identical to `llmcuda_kernels::norm::residual_add`, not merely
    /// close: one rounding per element, in the operand order given.
    ///
    /// `out` may alias either input; each thread touches one index of each.
    pub fn add(
        &self,
        stream: &Arc<CudaStream>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), LayerOpsError> {
        check_len("add a", n, a.len())?;
        check_len("add b", n, b.len())?;
        check_len("add out", n, out.len())?;
        if n == 0 {
            return Ok(());
        }

        let n_i64 = n as i64;
        let mut builder = stream.launch_builder(&self.add);
        builder.arg(a).arg(b).arg(out).arg(&n_i64);
        // SAFETY: the grid-stride loop is bounded by `n`, which is the length
        // checked above for all three buffers.
        unsafe { builder.launch(elementwise_cfg(n)) }?;
        Ok(())
    }

    /// `out = log(1 + exp(x))` over `n` elements, with ggml's passthrough
    /// above `x = 20`.
    ///
    /// The GDN alpha gate's nonlinearity. The passthrough is not optional:
    /// without it `expf` overflows above `x ≈ 88.7` and the result is `inf`
    /// where the answer is `x`. See `llmcuda_kernels::norm::softplus`.
    ///
    /// `out` may alias `x`.
    pub fn softplus(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), LayerOpsError> {
        check_len("softplus x", n, x.len())?;
        check_len("softplus out", n, out.len())?;
        if n == 0 {
            return Ok(());
        }

        let n_i64 = n as i64;
        let mut builder = stream.launch_builder(&self.softplus);
        builder.arg(x).arg(out).arg(&n_i64);
        // SAFETY: the grid-stride loop is bounded by `n`, which is the length
        // checked above for both buffers.
        unsafe { builder.launch(elementwise_cfg(n)) }?;
        Ok(())
    }

    /// One decode-step causal convolution over `n` sequences in one launch.
    ///
    /// `x` and `out` are `[n][channels]` batch scratch; `states` are the
    /// per-sequence conv cache base addresses (`[channels][conv_kernel-1]`
    /// f32 each), updated in place exactly as [`Self::conv1d`] at one token
    /// updates its single cache.
    ///
    /// # Safety
    ///
    /// Each entry of `states` must be the base address of a live
    /// `[channels][conv_kernel - 1]` f32 allocation, exclusively held by the
    /// caller for this call, all distinct.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn conv1d_step_batch_raw(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        states: &[u64],
        out: &mut CudaSlice<f32>,
        channels: usize,
        conv_kernel: usize,
    ) -> Result<(), LayerOpsError> {
        const MAX_BATCH: usize = 8;
        let n = states.len();
        assert!(
            (1..=MAX_BATCH).contains(&n),
            "conv step batch of {n} outside 1..={MAX_BATCH}"
        );
        if conv_kernel == 0 || conv_kernel > MAX_CONV_KERNEL {
            return Err(LayerOpsError::UnsupportedConvKernel { conv_kernel });
        }
        if channels == 0 {
            return Err(LayerOpsError::UnsupportedWidth { width: channels });
        }
        check_len("conv1d batch x", n * channels, x.len())?;
        check_len("conv1d batch weight", channels * conv_kernel, weight.len())?;
        check_len("conv1d batch out", n * channels, out.len())?;

        const BLOCK: usize = 256;
        let channels_i32 = channels as i32;
        let conv_kernel_i32 = conv_kernel as i32;
        let mut slots = [states[0]; MAX_BATCH];
        slots[..n].copy_from_slice(states);
        let cfg = LaunchConfig {
            grid_dim: (channels.div_ceil(BLOCK) as u32, n as u32, 1),
            block_dim: (BLOCK as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&self.conv1d_step_batch);
        builder.arg(x).arg(weight);
        for slot in &slots {
            builder.arg(slot);
        }
        builder
            .arg(&mut *out)
            .arg(&channels_i32)
            .arg(&conv_kernel_i32);
        // SAFETY: threads past `channels` return before touching memory;
        // `grid.y = n` bounds the pointer slots to the caller-asserted live
        // ones, each indexed within `[ch * carry, ch * carry + carry)`; `x`
        // and `out` were length-checked to `n * channels` above.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// The GDN short causal depthwise convolution, advancing `state`.
    ///
    /// `x` and `out` are `[seq_len][channels]` with channels contiguous;
    /// `weight` is `[channels][conv_kernel]`, the `ssm_conv1d.weight` layout
    /// verbatim; `state` is `[channels][conv_kernel - 1]` oldest-first and is
    /// updated in place to the last `conv_kernel - 1` inputs.
    ///
    /// Two launches, not one: the state update must observe the *old* state,
    /// which the convolution also reads, so they cannot be fused without
    /// either a second state buffer or a grid-wide barrier.
    ///
    /// This is the pure convolution, matching `ggml_ssm_conv`'s boundary.
    /// llama.cpp applies SiLU to the result as a separate op before slicing
    /// q/k/v apart (`src/models/qwen35moe.cpp`, `conv_output_silu`); that is
    /// the caller's job here too.
    #[allow(clippy::too_many_arguments)]
    pub fn conv1d(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        state: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        seq_len: usize,
        channels: usize,
        conv_kernel: usize,
    ) -> Result<(), LayerOpsError> {
        if conv_kernel == 0 || conv_kernel > MAX_CONV_KERNEL {
            return Err(LayerOpsError::UnsupportedConvKernel { conv_kernel });
        }
        if channels == 0 {
            return Err(LayerOpsError::UnsupportedWidth { width: channels });
        }
        check_len("conv1d x", seq_len * channels, x.len())?;
        check_len("conv1d weight", channels * conv_kernel, weight.len())?;
        check_len("conv1d state", channels * (conv_kernel - 1), state.len())?;
        check_len("conv1d out", seq_len * channels, out.len())?;
        if seq_len == 0 {
            return Ok(());
        }

        const BLOCK: usize = 256;
        let channels_i32 = channels as i32;
        let conv_kernel_i32 = conv_kernel as i32;

        // One token needs no batch-wide view of the window, so the output and
        // the cache advance are one launch. See `conv1d_step`.
        if seq_len == 1 {
            let cfg = LaunchConfig {
                grid_dim: (channels.div_ceil(BLOCK) as u32, 1, 1),
                block_dim: (BLOCK as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = stream.launch_builder(&self.conv1d_step);
            builder
                .arg(x)
                .arg(weight)
                .arg(&mut *state)
                .arg(&mut *out)
                .arg(&channels_i32)
                .arg(&conv_kernel_i32);
            // SAFETY: threads past `channels` return before touching memory;
            // `state` is indexed in `[ch * carry, ch * carry + carry)` and
            // `weight` in `[ch * conv_kernel, ...)`, both length-checked
            // above.
            unsafe { builder.launch(cfg) }?;
            return Ok(());
        }

        let cfg = LaunchConfig {
            grid_dim: (channels.div_ceil(BLOCK) as u32, seq_len as u32, 1),
            block_dim: (BLOCK as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&self.conv1d);
        builder
            .arg(x)
            .arg(weight)
            .arg(&*state)
            .arg(&mut *out)
            .arg(&channels_i32)
            .arg(&conv_kernel_i32);
        // SAFETY: threads past `channels` return before touching memory. The
        // window position `t - (conv_kernel - 1) + i` is either in
        // `[0, seq_len)` and indexes `x`, or in `[-(conv_kernel-1), 0)` and
        // indexes `state` at `ch * carry + (u + carry)` which is in
        // `[0, carry)`. Both buffers were length-checked above.
        unsafe { builder.launch(cfg) }?;
        self.launch_conv1d_state(stream, x, state, seq_len, channels, conv_kernel)
    }

    /// Advance a conv cache past `seq_len` tokens of `x` without convolving.
    ///
    /// The second half of [`Self::conv1d`], for a caller whose convolution
    /// ran elsewhere (the GDN prefill prelude reads the old cache and leaves
    /// it old). Same kernel, same lengths, so the cache lands on the same
    /// floats either way.
    pub fn conv1d_advance_state(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        state: &mut CudaSlice<f32>,
        seq_len: usize,
        channels: usize,
        conv_kernel: usize,
    ) -> Result<(), LayerOpsError> {
        if conv_kernel == 0 || conv_kernel > MAX_CONV_KERNEL {
            return Err(LayerOpsError::UnsupportedConvKernel { conv_kernel });
        }
        if channels == 0 {
            return Err(LayerOpsError::UnsupportedWidth { width: channels });
        }
        check_len("conv1d x", seq_len * channels, x.len())?;
        check_len("conv1d state", channels * (conv_kernel - 1), state.len())?;
        if seq_len == 0 {
            return Ok(());
        }
        self.launch_conv1d_state(stream, x, state, seq_len, channels, conv_kernel)
    }

    fn launch_conv1d_state(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        state: &mut CudaSlice<f32>,
        seq_len: usize,
        channels: usize,
        conv_kernel: usize,
    ) -> Result<(), LayerOpsError> {
        const BLOCK: usize = 256;
        let channels_i32 = channels as i32;
        let conv_kernel_i32 = conv_kernel as i32;
        let seq_len_i32 = seq_len as i32;
        let state_cfg = LaunchConfig {
            grid_dim: (channels.div_ceil(BLOCK).min(1024) as u32, 1, 1),
            block_dim: (BLOCK as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&self.conv1d_state);
        builder
            .arg(x)
            .arg(&mut *state)
            .arg(&seq_len_i32)
            .arg(&channels_i32)
            .arg(&conv_kernel_i32);
        // SAFETY: same index bounds as above; `seq_len >= 1` and the lengths
        // were checked at the top of both callers, which are the only paths
        // that launch it.
        unsafe { builder.launch(state_cfg) }?;
        Ok(())
    }
}

/// Block width for a row of `width` elements: a whole number of warps, capped
/// at [`MAX_BLOCK`], with wider rows covered by the kernel's strided loop.
///
/// A whole number of warps is required, not preferred: `block_reduce_sum`
/// shuffles with a full `0xffffffff` mask, so every lane of every warp in the
/// block must reach it. Threads past `width` contribute a zero partial.
fn block_for_width(width: usize) -> usize {
    width.next_multiple_of(32).clamp(32, MAX_BLOCK)
}

/// Launch geometry for the four grid-stride elementwise kernels.
///
/// A grid capped at 1024 blocks rather than one derived from `n`: grid-stride
/// keeps the launch shape independent of a host-side length, so the launch can
/// be captured in a CUDA graph and replayed at a different `n` (`AGENTS.md`
/// rule 5). The cap is what makes the shape *fixed*; without it the grid would
/// track `n` and every new length would need a new capture.
fn elementwise_cfg(n: usize) -> LaunchConfig {
    const BLOCK: usize = 256;
    LaunchConfig {
        grid_dim: (n.div_ceil(BLOCK).min(1024) as u32, 1, 1),
        block_dim: (BLOCK as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Rejects a buffer whose length disagrees with the declared geometry.
fn check_len(what: &'static str, expected: usize, got: usize) -> Result<(), LayerOpsError> {
    if got == expected {
        Ok(())
    } else {
        Err(LayerOpsError::ShapeMismatch {
            what,
            expected,
            got,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_gate_length_a_shape_demands_is_the_one_the_model_actually_has() {
        // Qwen3.6's two gates, spelled as literals because this crate does not
        // depend on `llmcuda-model`: attention q_heads 16 * head_dim 256 = 4096
        // values per token, and the MoE shared expert exactly one.
        const TOKENS: usize = 64;
        assert_eq!(
            GateShape::Elementwise.gate_len(TOKENS, 4096),
            TOKENS * 4096,
            "the attention output gate is elementwise over the q projection",
        );
        assert_eq!(
            GateShape::PerRow.gate_len(TOKENS, 2048),
            TOKENS,
            "the shared expert's gate is one scalar per token",
        );
        // A degenerate row count must not smuggle in a non-empty gate.
        assert_eq!(GateShape::Elementwise.gate_len(0, 4096), 0);
        assert_eq!(GateShape::PerRow.gate_len(0, 2048), 0);
    }

    #[test]
    fn the_elementwise_grid_is_capped_and_does_not_track_n() {
        // Grid-stride over a fixed grid: AGENTS.md rule 5 wants the launch
        // shape independent of a host-side length so it survives graph
        // capture and replay at a different n.
        for n in [1usize, 255, 256, 1 << 20, 1 << 28] {
            let cfg = elementwise_cfg(n);
            assert_eq!(cfg.block_dim, (256, 1, 1), "at n={n}");
            assert!((1..=1024).contains(&cfg.grid_dim.0), "at n={n}");
            assert_eq!(cfg.shared_mem_bytes, 0);
        }
        // Small launches still cover every element without the stride loop,
        // and large ones saturate the cap.
        assert_eq!(elementwise_cfg(1).grid_dim.0, 1);
        assert_eq!(elementwise_cfg(1 << 28).grid_dim.0, 1024);
    }

    #[test]
    fn the_block_width_is_always_a_whole_number_of_warps() {
        // block_reduce_sum shuffles with a full mask, so a partial warp would
        // hang or read garbage. Every width the model uses, plus the awkward
        // ones.
        for width in [1usize, 31, 32, 33, 128, 256, 2048, 4096, 8192] {
            let block = block_for_width(width);
            assert_eq!(block % 32, 0, "width {width} gave block {block}");
            assert!((32..=MAX_BLOCK).contains(&block), "width {width}");
        }
        // The three widths RMSNorm actually runs at in one forward pass.
        assert_eq!(block_for_width(2048), 1024);
        assert_eq!(block_for_width(256), 256);
        assert_eq!(block_for_width(128), 128);
    }

    #[test]
    fn a_length_that_matches_the_geometry_is_accepted_and_one_that_does_not_is_not() {
        assert!(check_len("x", 4, 4).is_ok());
        assert!(check_len("x", 4, 5).is_err());
    }
}
