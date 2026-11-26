//! Q8_0 and Q6_K dequantization on the device.
//!
//! These unpack the two quantized formats present in the target model file
//! into fp32. The eventual hot path fuses this into a GEMM prologue rather
//! than materialising a dequantized tensor — 29.6 GiB of weights would become
//! 132 GiB in fp32 and fit on no card — but a standalone kernel is what makes
//! the bit-unpacking testable in isolation, and it is the fallback for the
//! handful of small tensors where fusion is not worth it.
//!
//! ## Ported from
//!
//! The bit layouts are `block_q8_0` and `block_q6_K` in
//! `ggml/src/ggml-common.h`, and the unpacking follows
//! `dequantize_row_q8_0` / `dequantize_row_q6_K` in `ggml/src/ggml-quants.c`
//! — the same source as the scalar reference in `llmcuda_kernels::quant`, so
//! that the differential test compares two transcriptions of one algorithm
//! rather than two guesses.
//!
//! ## Why the arithmetic is written to be bit-identical
//!
//! Both formats compute their result with multiplications only — `q * d` for
//! Q8_0, `(d * scale) * q` for Q6_K — and no additions. There is therefore no
//! opportunity for the compiler to contract anything into an FMA, and with
//! the same operand order the device result is bit-identical to the scalar
//! reference. That makes the differential gate exact equality rather than a
//! tolerance, which is a far stronger statement: a tolerance can hide a
//! systematically wrong scale, and exact equality cannot.
//!
//! Keeping it that way means **not** reassociating: writing `d * (scale * q)`
//! for Q6_K would still be correct mathematically and would break bit
//! equality, because `scale * q` can round differently than `d * scale`.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};

/// Elements per Q8_0 block.
pub const QK8_0: usize = 32;
/// Elements per k-quant superblock.
pub const QK_K: usize = 256;
/// Serialized bytes per Q8_0 block.
pub const BLOCK_Q8_0_BYTES: usize = 34;
/// Serialized bytes per Q6_K superblock.
pub const BLOCK_Q6_K_BYTES: usize = 210;

/// Compiled dequantization kernels, ready to launch.
pub struct Dequantizer {
    q8_0: cudarc::driver::CudaFunction,
    q6_k: cudarc::driver::CudaFunction,
}

/// Something went wrong compiling or launching a dequantization kernel.
#[derive(Debug)]
pub enum DequantError {
    /// NVRTC rejected the source, or the module failed to load.
    Compile(String),
    /// The driver failed.
    Driver(DriverError),
    /// The input byte count is not a whole number of blocks.
    ///
    /// Rejected rather than truncated: a partial trailing block would read
    /// past the tensor and dequantize whatever follows it in the arena.
    Ragged {
        format: &'static str,
        bytes: usize,
        block_bytes: usize,
    },
}

impl std::fmt::Display for DequantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Compile(m) => write!(f, "kernel compilation failed: {m}"),
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::Ragged {
                format,
                bytes,
                block_bytes,
            } => write!(
                f,
                "{bytes} bytes is not a whole number of {format} blocks of {block_bytes} B",
            ),
        }
    }
}

impl std::error::Error for DequantError {}

impl From<DriverError> for DequantError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

const THREADS: u32 = 256;

impl Dequantizer {
    /// Compile the kernels for `ctx`.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, DequantError> {
        let rust = ctx.load_module(cudarc::nvrtc::Ptx::from_src(super::RUST_KERNELS_PTX))?;
        Ok(Self {
            q8_0: rust.load_function("dequantize_q8_0")?,
            q6_k: rust.load_function("dequantize_q6_k")?,
        })
    }

    /// Dequantize a Q8_0 tensor into a freshly allocated fp32 buffer.
    pub fn q8_0(
        &self,
        stream: &Arc<CudaStream>,
        src: &CudaSlice<u8>,
    ) -> Result<CudaSlice<f32>, DequantError> {
        let bytes = src.len();
        if !bytes.is_multiple_of(BLOCK_Q8_0_BYTES) {
            return Err(DequantError::Ragged {
                format: "q8_0",
                bytes,
                block_bytes: BLOCK_Q8_0_BYTES,
            });
        }
        let n = bytes / BLOCK_Q8_0_BYTES * QK8_0;
        let mut dst = stream.alloc_zeros::<f32>(n)?;
        let n64 = n as i64;

        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(THREADS as usize) as u32, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&self.q8_0);
        builder.arg(src).arg(&mut dst).arg(&n64);
        // SAFETY: the kernel bounds-checks against `n_elements`, the output
        // is allocated to exactly `n` floats, and `src` is `n/32 * 34` bytes
        // — the block count the kernel derives from `i / 32`.
        unsafe { builder.launch(cfg) }?;
        Ok(dst)
    }

    /// Dequantize a Q6_K tensor into a freshly allocated fp32 buffer.
    pub fn q6_k(
        &self,
        stream: &Arc<CudaStream>,
        src: &CudaSlice<u8>,
    ) -> Result<CudaSlice<f32>, DequantError> {
        let bytes = src.len();
        if !bytes.is_multiple_of(BLOCK_Q6_K_BYTES) {
            return Err(DequantError::Ragged {
                format: "q6_K",
                bytes,
                block_bytes: BLOCK_Q6_K_BYTES,
            });
        }
        let superblocks = bytes / BLOCK_Q6_K_BYTES;
        let n = superblocks * QK_K;
        let mut dst = stream.alloc_zeros::<f32>(n)?;
        let sb64 = superblocks as i64;

        // 64 threads per superblock, each writing four outputs.
        let threads_needed = superblocks * 64;
        let cfg = LaunchConfig {
            grid_dim: (threads_needed.div_ceil(THREADS as usize) as u32, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&self.q6_k);
        builder.arg(src).arg(&mut dst).arg(&sb64);
        // SAFETY: the kernel bounds-checks against `n_superblocks * 64`, and
        // the output holds exactly `n_superblocks * 256` floats, which is
        // what four writes per thread cover.
        unsafe { builder.launch(cfg) }?;
        Ok(dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_sizes_match_the_ggml_layout() {
        // Duplicated from `llmcuda_kernels::quant` on purpose: this crate must
        // not depend on that one (it is the oracle, not a library), so the
        // constants are asserted independently against ggml-common.h.
        assert_eq!(BLOCK_Q8_0_BYTES, 2 + QK8_0);
        assert_eq!(BLOCK_Q6_K_BYTES, QK_K / 2 + QK_K / 4 + QK_K / 16 + 2);
    }

    #[test]
    fn the_q6_k_source_offsets_partition_the_superblock() {
        // The kernel indexes `ql` at +0, `qh` at +128, `scales` at +192 and
        // the delta at +208. Those must tile 210 bytes exactly; an off-by-one
        // would read a neighbouring field and still produce finite output.
        assert_eq!(QK_K / 2, 128);
        assert_eq!(128 + QK_K / 4, 192);
        assert_eq!(192 + QK_K / 16, 208);
        assert_eq!(208 + 2, BLOCK_Q6_K_BYTES);
    }

    #[test]
    fn the_source_multiplies_in_the_reference_order() {
        // Bit-identical agreement with the scalar reference depends on the
        // operand order `d * scale * q`. This guards the comment above from
        // being quietly invalidated by a "simplification".
        assert!(
            crate::kernels::rust_kernel_source("dequantize_q6_k")
                .contains("*y.add(l) = d * *sc.add(is) as f32 * (raw1 - 32) as f32;"),
            "Q6_K reassociated away from (d * scale) * q",
        );
        assert!(
            crate::kernels::rust_kernel_source("dequantize_q8_0").contains("q as f32 * d"),
            "Q8_0 reassociated away from q * d",
        );
    }

    #[test]
    fn quants_are_read_as_signed() {
        // Reading int8 codes as unsigned is the single most likely
        // transcription error here, and it produces plausible magnitudes.
        // Q8_0 is Rust.
        assert!(
            crate::kernels::rust_kernel_source("dequantize_q8_0")
                .contains("*base.add(2 + (i % 32) as usize) as i8")
        );
        assert!(
            crate::kernels::rust_kernel_source("dequantize_q6_k")
                .contains("let sc = base.add(192 + half * 8) as *const i8;")
        );
    }
}
