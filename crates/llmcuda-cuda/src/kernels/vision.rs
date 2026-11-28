//! Device ops for the qwen3vl vision tower: LayerNorm with bias, GELU,
//! vision M-RoPE, row softmax, f16 utility copies, and cuBLASLt f16 GEMM.
//!
//! Reference: `llmcuda_kernels::vision` — the tower structure, layouts and the
//! cited llama.cpp provenance live there; this module only reproduces those
//! ops on the device. Activations are f16 (matching llama.cpp's CUDA clip
//! path); every kernel computes in f32 and rounds once on store.
//!
//! # Why cuBLASLt, in a workspace that hand-writes its matmuls
//!
//! The tower is dense f16 GEMM over up to a few thousand rows — exactly the
//! shape cuBLASLt covers well on Turing — and it runs once per image at
//! admission, never inside a captured graph and never on the decode path
//! this project's kernels are specialized for. `MmaKernels` is int8 and
//! `LmHeadKernels` is Q6_K/Q8_0; neither speaks f16 weights, and writing a
//! third GEMM family for an off-hot-path pass would buy correctness risk
//! with no benchmark upside. llama.cpp makes the same call: its prefill
//! matmuls go through cuBLAS.
//!
//! # Geometry is per launch
//!
//! Image grids vary per request, so like [`super::layer_ops`] every launch
//! takes its shape as arguments and validates buffer lengths against them.
//! The vision pass is never captured (it runs before prefill on the same
//! stream), so host-sized launches are sound here; AGENTS.md rule 5 binds
//! captured paths only.

use std::sync::Arc;

use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};
use half::f16;

/// Errors from the vision device ops.
#[derive(Debug)]
pub enum VisionError {
    /// A kernel failed to load from the module.
    Build(String),
    /// cuBLASLt refused the matmul (no heuristic, bad descriptor).
    Blas(cudarc::cublaslt::result::CublasError),
    /// The driver failed.
    Driver(DriverError),
    /// A buffer is not the length the declared geometry requires.
    WrongLength {
        name: &'static str,
        expected: usize,
        got: usize,
    },
}

impl std::fmt::Display for VisionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Build(msg) => write!(f, "vision kernel build failed: {msg}"),
            Self::Blas(e) => write!(f, "cublasLt: {e:?}"),
            Self::Driver(e) => write!(f, "driver: {e}"),
            Self::WrongLength {
                name,
                expected,
                got,
            } => write!(f, "{name}: expected {expected} elements, got {got}"),
        }
    }
}

impl std::error::Error for VisionError {}

impl From<DriverError> for VisionError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

impl From<cudarc::cublaslt::result::CublasError> for VisionError {
    fn from(e: cudarc::cublaslt::result::CublasError) -> Self {
        Self::Blas(e)
    }
}

fn check_len(name: &'static str, expected: usize, got: usize) -> Result<(), VisionError> {
    if expected == got {
        Ok(())
    } else {
        Err(VisionError::WrongLength {
            name,
            expected,
            got,
        })
    }
}

/// Pre-sized buffers may exceed the launch's geometry; equality is only
/// demanded of weights.
fn check_min(name: &'static str, expected: usize, got: usize) -> Result<(), VisionError> {
    if got >= expected {
        Ok(())
    } else {
        Err(VisionError::WrongLength {
            name,
            expected,
            got,
        })
    }
}

const BLOCK: u32 = 256;

fn grid_1d(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(BLOCK as usize) as u32, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// The compiled vision kernel set plus a cuBLASLt handle, bound to one
/// stream.
pub struct VisionKernels {
    layer_norm: CudaFunction,
    gelu: CudaFunction,
    gelu_erf: CudaFunction,
    rope_qk: CudaFunction,
    softmax_rows: CudaFunction,
    add_f16: CudaFunction,
    half_to_float: CudaFunction,
    blas: CudaBlasLT,
}

impl VisionKernels {
    /// Load the kernels on `ctx`, binding GEMMs to `stream`.
    pub fn new(ctx: &Arc<CudaContext>, stream: Arc<CudaStream>) -> Result<Self, VisionError> {
        let rust = ctx
            .load_module(cudarc::nvrtc::Ptx::from_src(super::RUST_KERNELS_PTX))
            .map_err(VisionError::Driver)?;
        let r = |name: &str| {
            rust.load_function(name)
                .map_err(|e| VisionError::Build(format!("{name}: {e}")))
        };
        Ok(Self {
            layer_norm: r("vision_layer_norm")?,
            gelu: r("vision_gelu")?,
            gelu_erf: r("vision_gelu_erf")?,
            rope_qk: r("vision_rope_qk")?,
            softmax_rows: r("vision_softmax_rows")?,
            add_f16: r("vision_add_f16")?,
            half_to_float: r("vision_half_to_float")?,
            blas: CudaBlasLT::new(stream)?,
        })
    }

    /// LayerNorm with weight and bias over each of `rows` rows of `width`.
    ///
    /// `out` must not alias `x` (the residual stream keeps the un-normed
    /// value); enforced by the exclusive borrow.
    #[allow(clippy::too_many_arguments)]
    pub fn layer_norm(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f16>,
        out: &mut CudaSlice<f16>,
        gamma: &CudaSlice<f16>,
        beta: &CudaSlice<f16>,
        rows: usize,
        width: usize,
        eps: f32,
    ) -> Result<(), VisionError> {
        check_min("layer_norm x", rows * width, x.len())?;
        check_min("layer_norm out", rows * width, out.len())?;
        check_len("layer_norm gamma", width, gamma.len())?;
        check_len("layer_norm beta", width, beta.len())?;
        if rows == 0 {
            return Ok(());
        }
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let width_i32 = width as i32;
        let mut b = stream.launch_builder(&self.layer_norm);
        b.arg(x)
            .arg(out)
            .arg(gamma)
            .arg(beta)
            .arg(&width_i32)
            .arg(&eps);
        // SAFETY: one block per row; every in-row index is bounded by
        // `width`, and both buffers hold `rows * width` halves.
        unsafe { b.launch(cfg) }?;
        Ok(())
    }

    /// In-place tanh-approximation GELU over the first `n` elements.
    pub fn gelu(
        &self,
        stream: &Arc<CudaStream>,
        x: &mut CudaSlice<f16>,
        n: usize,
    ) -> Result<(), VisionError> {
        if x.len() < n {
            return Err(VisionError::WrongLength {
                name: "gelu x",
                expected: n,
                got: x.len(),
            });
        }
        if n == 0 {
            return Ok(());
        }
        let n_ll = n as i64;
        let mut b = stream.launch_builder(&self.gelu);
        b.arg(x).arg(&n_ll);
        // SAFETY: the guard `i >= n` bounds every access; `x` holds at
        // least `n` halves.
        unsafe { b.launch(grid_1d(n)) }?;
        Ok(())
    }

    /// In-place exact GELU, `x / 2 * (1 + erf(x / sqrt 2))`, over the first
    /// `n` elements: the `inkling2v` tower's activation, where the tanh
    /// form is `qwen3vl_merger`'s.
    pub fn gelu_erf(
        &self,
        stream: &Arc<CudaStream>,
        x: &mut CudaSlice<f16>,
        n: usize,
    ) -> Result<(), VisionError> {
        if x.len() < n {
            return Err(VisionError::WrongLength {
                name: "gelu_erf x",
                expected: n,
                got: x.len(),
            });
        }
        if n == 0 {
            return Ok(());
        }
        let n_ll = n as i64;
        let mut b = stream.launch_builder(&self.gelu_erf);
        b.arg(x).arg(&n_ll);
        // SAFETY: the guard `i >= n` bounds every access; `x` holds at
        // least `n` halves.
        unsafe { b.launch(grid_1d(n)) }?;
        Ok(())
    }

    /// Vision M-RoPE over the q and k slices of the fused qkv buffer.
    ///
    /// `positions` holds `(row, col)` i32 pairs per token, in the same
    /// token order as the rows of `qkv`.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_qk(
        &self,
        stream: &Arc<CudaStream>,
        qkv: &mut CudaSlice<f16>,
        positions: &CudaSlice<i32>,
        tokens: usize,
        hidden: usize,
        head_dim: usize,
        theta_base: f32,
    ) -> Result<(), VisionError> {
        check_min("rope qkv", tokens * 3 * hidden, qkv.len())?;
        check_min("rope positions", tokens * 2, positions.len())?;
        assert!(
            head_dim.is_multiple_of(4) && hidden.is_multiple_of(head_dim),
            "vision rope needs head_dim % 4 == 0 and hidden % head_dim == 0"
        );
        if tokens == 0 {
            return Ok(());
        }
        let heads = hidden / head_dim;
        let cfg = LaunchConfig {
            grid_dim: (tokens as u32, 2, heads as u32),
            block_dim: ((head_dim / 2) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let hidden_i32 = hidden as i32;
        let head_dim_i32 = head_dim as i32;
        let mut b = stream.launch_builder(&self.rope_qk);
        b.arg(qkv)
            .arg(positions)
            .arg(&hidden_i32)
            .arg(&head_dim_i32)
            .arg(&theta_base);
        // SAFETY: grid is (tokens, 2, heads), block head_dim/2 threads; the
        // furthest access is token*3*hidden + slice*hidden + head*head_dim
        // + head_dim - 1, inside the checked qkv length. positions is
        // indexed by 2*token + 1 < 2*tokens.
        unsafe { b.launch(cfg) }?;
        Ok(())
    }

    /// Row-wise softmax in place over `rows` rows of `row_len`.
    pub fn softmax_rows(
        &self,
        stream: &Arc<CudaStream>,
        scores: &mut CudaSlice<f16>,
        rows: usize,
        row_len: usize,
    ) -> Result<(), VisionError> {
        if scores.len() < rows * row_len {
            return Err(VisionError::WrongLength {
                name: "softmax scores",
                expected: rows * row_len,
                got: scores.len(),
            });
        }
        if rows == 0 {
            return Ok(());
        }
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let row_len_i32 = row_len as i32;
        let mut b = stream.launch_builder(&self.softmax_rows);
        b.arg(scores).arg(&row_len_i32);
        // SAFETY: one block per row, in-row indices bounded by row_len, and
        // the buffer holds at least rows * row_len halves (checked above).
        unsafe { b.launch(cfg) }?;
        Ok(())
    }

    /// `a[..n] += b[..n]` elementwise.
    pub fn add_assign(
        &self,
        stream: &Arc<CudaStream>,
        a: &mut CudaSlice<f16>,
        b: &CudaSlice<f16>,
        n: usize,
    ) -> Result<(), VisionError> {
        if a.len() < n || b.len() < n {
            return Err(VisionError::WrongLength {
                name: "add_assign",
                expected: n,
                got: a.len().min(b.len()),
            });
        }
        if n == 0 {
            return Ok(());
        }
        let n_ll = n as i64;
        let mut builder = stream.launch_builder(&self.add_f16);
        builder.arg(a).arg(b).arg(&n_ll);
        // SAFETY: guarded by `i >= n`; both buffers checked to hold n.
        unsafe { builder.launch(grid_1d(n)) }?;
        Ok(())
    }

    /// Widen `src[..n]` into `dst[..n]`.
    pub fn to_f32(
        &self,
        stream: &Arc<CudaStream>,
        src: &CudaSlice<f16>,
        dst: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), VisionError> {
        if src.len() < n || dst.len() < n {
            return Err(VisionError::WrongLength {
                name: "to_f32",
                expected: n,
                got: src.len().min(dst.len()),
            });
        }
        if n == 0 {
            return Ok(());
        }
        let n_ll = n as i64;
        let mut b = stream.launch_builder(&self.half_to_float);
        b.arg(src).arg(dst).arg(&n_ll);
        // SAFETY: guarded by `i >= n`; both buffers checked to hold n.
        unsafe { b.launch(grid_1d(n)) }?;
        Ok(())
    }

    /// `y[tokens × out] = x[tokens × in] · wᵀ + bias`, all row-major f16,
    /// f32 accumulation.
    ///
    /// `w` is a GGUF-layout projection: `out` rows of length `in`. The
    /// row-major views map onto cuBLASLt's column-major world as
    /// `C(out×n) = Aᵀ(out×in) · B(in×n)` with A = w and B = x, both at
    /// their natural leading dimensions, so no copies or transposes are
    /// materialized. `x_ld`/`y_ld` allow reading from and writing into
    /// slices of wider row buffers (the fused qkv rows).
    #[allow(clippy::too_many_arguments)]
    pub fn proj(
        &self,
        x: &CudaSlice<f16>,
        w: &CudaSlice<f16>,
        bias: Option<&CudaSlice<f16>>,
        y: &mut CudaSlice<f16>,
        tokens: usize,
        in_dim: usize,
        out_dim: usize,
        beta: f32,
    ) -> Result<(), VisionError> {
        check_len("proj w", in_dim * out_dim, w.len())?;
        if x.len() < tokens * in_dim || y.len() < tokens * out_dim {
            return Err(VisionError::WrongLength {
                name: "proj x/y",
                expected: tokens * in_dim.max(out_dim),
                got: x.len().min(y.len()),
            });
        }
        if let Some(b) = bias {
            check_len("proj bias", out_dim, b.len())?;
        }
        if tokens == 0 {
            return Ok(());
        }
        let cfg = MatmulConfig {
            transa: true,
            transb: false,
            transc: false,
            m: out_dim as u64,
            n: tokens as u64,
            k: in_dim as u64,
            alpha: 1.0,
            lda: in_dim as i64,
            ldb: in_dim as i64,
            beta,
            ldc: out_dim as i64,
            stride_a: None,
            stride_b: None,
            stride_c: None,
            stride_bias: None,
            batch_size: None,
        };
        // SAFETY: dimensions and leading strides are validated against the
        // buffer lengths above; bias is out_dim long, matching m.
        unsafe { self.blas.matmul(cfg, w, x, y, bias, None) }?;
        Ok(())
    }

    /// Batched per-head attention scores:
    /// `scores[h][q][k] = scale · Q[q] · K[k]` over `heads` consecutive
    /// heads, where token `t`'s slice for the first head starts at
    /// `t * row_len + q_offset` (resp. `k_offset`) and later heads advance
    /// by `head_dim` — the fused-qkv layout, with the q/k slice base and
    /// any head-chunk offset folded into the offsets.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_scores(
        &self,
        buf: &CudaSlice<f16>,
        q_offset: usize,
        k_offset: usize,
        scores: &mut CudaSlice<f16>,
        tokens: usize,
        row_len: usize,
        head_dim: usize,
        heads: usize,
        scale: f32,
    ) -> Result<(), VisionError> {
        if scores.len() < heads * tokens * tokens {
            return Err(VisionError::WrongLength {
                name: "attn scores",
                expected: heads * tokens * tokens,
                got: scores.len(),
            });
        }
        if tokens == 0 || heads == 0 {
            return Ok(());
        }
        let cfg = MatmulConfig {
            // C(k_tokens × q_tokens) per head = Kᵀ-of-memory transposed ·
            // Q-memory: both operands are the (head_dim × tokens)
            // column-major views their row-major memory naturally is.
            transa: true,
            transb: false,
            transc: false,
            m: tokens as u64,
            n: tokens as u64,
            k: head_dim as u64,
            alpha: scale,
            lda: row_len as i64,
            ldb: row_len as i64,
            beta: 0.0,
            ldc: tokens as i64,
            stride_a: Some(head_dim as i64),
            stride_b: Some(head_dim as i64),
            stride_c: Some((tokens * tokens) as i64),
            stride_bias: None,
            batch_size: Some(heads as i32),
        };
        let q_view = slice_from(buf, q_offset);
        let k_view = slice_from(buf, k_offset);
        // SAFETY: per-head strides walk `heads` slices of `head_dim` within
        // rows of `row_len`; the furthest element read is
        // (tokens-1)*row_len + offset + heads*head_dim - 1, within the
        // caller's buffer; scores length is checked above.
        unsafe { self.blas.matmul(cfg, &k_view, &q_view, scores, None, None) }?;
        Ok(())
    }

    /// Batched per-head attention output:
    /// `out[q] = Σ_k probs[h][q][k] · V[k]`, mirroring
    /// [`Self::attn_scores`]'s offset convention: token `t`'s first-head
    /// value slice starts at `t * v_row_len + v_offset`, the output's at
    /// `t * out_row_len + out_offset`.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_output(
        &self,
        probs: &CudaSlice<f16>,
        v: &CudaSlice<f16>,
        v_offset: usize,
        out: &mut CudaSlice<f16>,
        out_offset: usize,
        tokens: usize,
        v_row_len: usize,
        out_row_len: usize,
        head_dim: usize,
        heads: usize,
    ) -> Result<(), VisionError> {
        if probs.len() < heads * tokens * tokens {
            return Err(VisionError::WrongLength {
                name: "attn probs",
                expected: heads * tokens * tokens,
                got: probs.len(),
            });
        }
        if tokens == 0 || heads == 0 {
            return Ok(());
        }
        let cfg = MatmulConfig {
            // C(head_dim × q_tokens) per head = V-memory (head_dim × k) ·
            // probsᵀ-memory (k × q).
            transa: false,
            transb: false,
            transc: false,
            m: head_dim as u64,
            n: tokens as u64,
            k: tokens as u64,
            alpha: 1.0,
            lda: v_row_len as i64,
            ldb: tokens as i64,
            beta: 0.0,
            ldc: out_row_len as i64,
            stride_a: Some(head_dim as i64),
            stride_b: Some((tokens * tokens) as i64),
            stride_c: Some(head_dim as i64),
            stride_bias: None,
            batch_size: Some(heads as i32),
        };
        let v_view = slice_from(v, v_offset);
        let probs_view = slice_from(probs, 0);
        let mut out_view = slice_from_mut(out, out_offset);
        // SAFETY: same windowing argument as attn_scores; the output's
        // furthest write is (tokens-1)*out_row_len + out_offset +
        // heads*head_dim - 1, within the caller's buffer.
        unsafe {
            self.blas
                .matmul(cfg, &v_view, &probs_view, &mut out_view, None, None)
        }?;
        Ok(())
    }
}

/// A shared sub-view of a device buffer starting at `offset` elements.
fn slice_from(s: &CudaSlice<f16>, offset: usize) -> cudarc::driver::CudaView<'_, f16> {
    s.slice(offset..)
}

/// An exclusive sub-view of a device buffer starting at `offset` elements.
fn slice_from_mut(s: &mut CudaSlice<f16>, offset: usize) -> cudarc::driver::CudaViewMut<'_, f16> {
    s.slice_mut(offset..)
}
