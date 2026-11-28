//! K2 rotary coefficients shared by the heads and layers of a fixed pass.
//! The double-precision frequency and angle evaluation matches `attention`
//! exactly; only the resulting fp32 sine/cosine values are cached.
use super::{
    attention::{KvFormat, KvFormats},
    moe::MoeError,
};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use std::sync::Arc;
/// Fixed coefficients for one physical token shape, refreshed on the device
/// at layer zero, including on each graph replay.
pub struct K2Rope {
    prepare: CudaFunction,
    apply: CudaFunction,
    append: CudaFunction,
    /// The cache formats `append` was chosen for.
    kv: KvFormats,
    coefficients: CudaSlice<f32>,
    tokens: usize,
    dim: usize,
    rope_dim: usize,
}
impl K2Rope {
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        tokens: usize,
        dim: usize,
        rope_dim: usize,
    ) -> Result<Self, MoeError> {
        Self::with_kv_formats(ctx, stream, tokens, dim, rope_dim, KvFormats::F16)
    }

    /// [`Self::new`] whose batched decode append writes `kv`'s formats. A
    /// q8_0 half launches `k2_rotary_append_kv8`, which stays bit-identical
    /// to `attn_kv_append_kv8`.
    pub fn with_kv_formats(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        tokens: usize,
        dim: usize,
        rope_dim: usize,
        kv: KvFormats,
    ) -> Result<Self, MoeError> {
        if tokens == 0
            || dim == 0
            || dim > 1024
            || !rope_dim.is_multiple_of(2)
            || rope_dim == 0
            || rope_dim > dim
            || rope_dim > 256
        {
            return Err(MoeError::WrongElementCount {
                which: "K2 rotary geometry",
                expected: dim,
                found: rope_dim,
            });
        }
        if kv.is_quantized() && !dim.is_multiple_of(32) {
            return Err(MoeError::WrongElementCount {
                which: "K2 q8_0 cache head width",
                expected: dim.next_multiple_of(32),
                found: dim,
            });
        }
        let rust = ctx.load_module(cudarc::nvrtc::Ptx::from_src(super::RUST_KERNELS_PTX))?;
        Ok(Self {
            prepare: rust.load_function("k2_rotary_prepare")?,
            apply: rust.load_function("k2_rotary_apply")?,
            // The fp16 cache and the quantized one are separate entry
            // points; the quantized one takes the formats as arguments.
            append: rust.load_function(if kv.is_quantized() {
                "k2_rotary_append_kv8"
            } else {
                "k2_rotary_append"
            })?,
            kv,
            coefficients: stream.alloc_zeros(tokens * rope_dim)?,
            tokens,
            dim,
            rope_dim,
        })
    }
    /// Prepare one contiguous batch of at most eight sequence chunks.
    ///
    /// # Safety
    /// Each position slot must point to a live device i32 on this stream.
    pub unsafe fn prepare_raw(
        &mut self,
        stream: &Arc<CudaStream>,
        positions: &[u64],
        chunk: usize,
        offset: usize,
        theta: f32,
    ) -> Result<(), MoeError> {
        if !theta.is_finite()
            || theta <= 0.0
            || positions.is_empty()
            || positions.len() > 8
            || chunk == 0
            || offset + positions.len() * chunk > self.tokens
        {
            return Err(MoeError::WrongElementCount {
                which: "K2 rotary positions",
                expected: self.tokens,
                found: offset + positions.len() * chunk,
            });
        }
        let mut p = [positions[0]; 8];
        p[..positions.len()].copy_from_slice(positions);
        let (chunk_i, rd, off) = (chunk as i32, self.rope_dim as i32, offset as i32);
        let mut launch = stream.launch_builder(&self.prepare);
        launch
            .arg(&mut self.coefficients)
            .arg(&chunk_i)
            .arg(&rd)
            .arg(&theta)
            .arg(&off);
        for slot in &p {
            launch.arg(slot);
        }
        // SAFETY: caller owns live position slots; checked output token range.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: ((chunk as u32).div_ceil(8), positions.len() as u32, 1),
                block_dim: ((self.rope_dim / 2) as u32, 8, 1),
                shared_mem_bytes: 0,
            })?;
        }
        Ok(())
    }
    pub fn apply(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        heads: usize,
        offset: usize,
    ) -> Result<(), MoeError> {
        if input.is_empty() || heads == 0 || !input.len().is_multiple_of(heads * self.dim) {
            return Err(MoeError::WrongElementCount {
                which: "K2 rotary heads",
                expected: self.dim,
                found: heads,
            });
        }
        let tokens = input.len() / (heads * self.dim);
        check("K2 rotary output", input.len(), out.len())?;
        if offset + tokens > self.tokens {
            return Err(MoeError::WrongElementCount {
                which: "K2 rotary range",
                expected: self.tokens,
                found: offset + tokens,
            });
        }
        let (h, d, rd, off, t) = (
            heads as i32,
            self.dim as i32,
            self.rope_dim as i32,
            offset as i32,
            tokens as i32,
        );
        // SAFETY: input/output and coefficient ranges have been checked.
        unsafe {
            stream
                .launch_builder(&self.apply)
                .arg(input)
                .arg(out)
                .arg(&self.coefficients)
                .arg(&h)
                .arg(&d)
                .arg(&rd)
                .arg(&off)
                .arg(&t)
                .launch(LaunchConfig {
                    grid_dim: ((tokens as u32).div_ceil(8), heads as u32, 1),
                    block_dim: (self.dim as u32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }
    /// Rotate and append one decode row for every sequence.
    ///
    /// # Safety
    /// Position slots must address live i32 values. Cache slots must be
    /// distinct live caches, in the formats this rope was built for,
    /// covering those positions and these heads.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn append_raw(
        &self,
        stream: &Arc<CudaStream>,
        q: &CudaSlice<f32>,
        qr: &mut CudaSlice<f32>,
        k: &CudaSlice<f32>,
        kr: &mut CudaSlice<f32>,
        v: &CudaSlice<f32>,
        positions: &[u64],
        keys: &[u64],
        values: &[u64],
        qh: usize,
        kh: usize,
    ) -> Result<(), MoeError> {
        let n = positions.len();
        if n == 0 || n > 8 || n != self.tokens || qh == 0 || kh == 0 {
            return Err(MoeError::WrongElementCount {
                which: "K2 rotary decode batch",
                expected: self.tokens,
                found: n,
            });
        }
        check("K2 key slots", n, keys.len())?;
        check("K2 value slots", n, values.len())?;
        check("K2 query", n * qh * self.dim, q.len())?;
        check("K2 query output", q.len(), qr.len())?;
        check("K2 key", n * kh * self.dim, k.len())?;
        check("K2 key output", k.len(), kr.len())?;
        check("K2 value", k.len(), v.len())?;
        let fill = |s: &[u64]| {
            let mut a = [s[0]; 8];
            a[..n].copy_from_slice(s);
            a
        };
        let (ap, kc, vc) = (fill(positions), fill(keys), fill(values));
        let (qh_i, kh_i, d, rd) = (qh as i32, kh as i32, self.dim as i32, self.rope_dim as i32);
        let mut launch = stream.launch_builder(&self.append);
        launch
            .arg(q)
            .arg(qr)
            .arg(k)
            .arg(kr)
            .arg(v)
            .arg(&self.coefficients);
        for slot in ap.iter().chain(&kc).chain(&vc) {
            launch.arg(slot);
        }
        launch.arg(&qh_i).arg(&kh_i).arg(&d).arg(&rd);
        let (kq8, vq8) = (
            i32::from(self.kv.k == KvFormat::Q8_0),
            i32::from(self.kv.v == KvFormat::Q8_0),
        );
        if self.kv.is_quantized() {
            launch.arg(&kq8).arg(&vq8);
        }
        // SAFETY: checked input shapes; caller guarantees live independent caches.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: ((qh + kh) as u32, n as u32, 1),
                block_dim: (self.dim as u32, 1, 1),
                shared_mem_bytes: 0,
            })?;
        }
        Ok(())
    }
}
fn check(which: &'static str, expected: usize, found: usize) -> Result<(), MoeError> {
    if expected == found {
        Ok(())
    } else {
        Err(MoeError::WrongElementCount {
            which,
            expected,
            found,
        })
    }
}
