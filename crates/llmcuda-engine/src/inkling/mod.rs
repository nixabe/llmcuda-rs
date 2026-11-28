//! The `inkling2_loop` forward pass (Inkling 2 Mini) on one device.
//!
//! [`InklingModel`] holds the weights, [`InklingState`] one sequence's
//! attention caches and short-convolution carries, and [`InklingPass`] the
//! scratch for a fixed number of rows. A pass runs *segments*: runs of
//! consecutive positions, each of one sequence. A prefill chunk is one
//! segment of many rows; a batched decode step is one row from each of
//! several sequences. Everything a row does with weights runs over all rows
//! at once; only what reads or writes a sequence's own state (the
//! convolutions, the cache append, attention) runs per segment.
//!
//! The arithmetic follows `llmcuda_kernels::inkling`, which is held to a
//! llama.cpp capture; `tests/inkling_forward.rs` holds this module to both.
//! Contractions run on the fp16 tensor cores ([`HgemmKernels`]): activations
//! are rounded to binary16 and weights to the binary16 of their stored value
//! (exact for NVFP4 and for bf16 in range), with fp32 accumulation. The
//! router's logits alone stay fp32 end to end, as llama.cpp keeps them.
//!
//! This is the first, plain version: every layer application keeps a
//! full-length cache, and attention is a scalar scan. Neither is the shape
//! the model should be served in.

use std::mem::ManuallyDrop;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DriverError};
use llmcuda_cuda::kernels::hgemm::{
    HalfWeight, HgemmError, HgemmKernels, bf16_half_exponent, nv4_tile_major, nv4_tiles_fit,
};
use llmcuda_cuda::kernels::inkling::{InkDispatch, InklingError, InklingKernels};
use llmcuda_cuda::kernels::k2::K2Kernels;
use llmcuda_cuda::kernels::layer_ops::{GateShape, LayerOpsError, LayerOpsKernels};
use llmcuda_cuda::kernels::mma::{MmaError, MmaKernels};
use llmcuda_cuda::kernels::moe::MoeError;
use llmcuda_gguf::{GgmlType, GgufFile};
use llmcuda_model::{InklingConfig, ModelConfig};

use crate::viewslice::subslice;

mod decode;
pub(crate) mod serve;
mod snapshot;
pub mod vision;

pub use decode::InklingDecode;

/// Why loading or running the model failed.
#[derive(Debug)]
pub struct InklingRunError(pub String);

impl core::fmt::Display for InklingRunError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

impl core::error::Error for InklingRunError {}

macro_rules! run_error_from {
    ($($ty:ty),*) => {$(
        impl From<$ty> for InklingRunError {
            fn from(e: $ty) -> Self {
                Self(e.to_string())
            }
        }
    )*};
}
run_error_from!(
    DriverError,
    InklingError,
    HgemmError,
    LayerOpsError,
    MmaError,
    MoeError
);

fn bad(what: impl Into<String>) -> InklingRunError {
    InklingRunError(what.into())
}

type Res<T = ()> = Result<T, InklingRunError>;

/// A matrix `[rows][k]` in a format the fp16 GEMM contracts.
enum Dense {
    Q8 {
        q: CudaSlice<i8>,
        scales: CudaSlice<u16>,
    },
    Bf16 {
        bytes: CudaSlice<u8>,
        exponent: i32,
    },
}

struct Matrix {
    w: Dense,
    rows: usize,
    k: usize,
}

impl Matrix {
    fn half(&self) -> HalfWeight<'_> {
        match &self.w {
            Dense::Q8 { q, scales } => HalfWeight::Q8_0 { q, scales },
            Dense::Bf16 { bytes, exponent } => HalfWeight::Bf16 {
                bytes,
                exponent: *exponent,
            },
        }
    }
}

/// One layer's experts for one projection: `[experts][rows][k]`.
enum ExpertStack {
    /// Tile-major (`nv4_tile_major`), partners `partner` rows apart.
    Nvfp4 {
        bytes: CudaSlice<u8>,
        scales: Option<CudaSlice<f32>>,
        partner: usize,
    },
    Dense(Dense),
}

struct Experts {
    w: ExpertStack,
    rows: usize,
    k: usize,
}

struct Controller {
    down: Matrix,
    gain: Matrix,
    forget: Matrix,
}

struct LoopLayer {
    msg_query: CudaSlice<f32>,
    /// Q, K, V, FFN.
    ctrl: [Controller; 4],
}

struct Layer {
    sliding: bool,
    q: Matrix,
    k: Matrix,
    v: Matrix,
    o: Matrix,
    gate: Matrix,
    rel: Matrix,
    sink: Matrix,
    k_conv_proj: Option<Matrix>,
    v_conv_proj: Option<Matrix>,
    /// One norm on a sliding layer; Q, K and V input norms on a full one.
    norms: Vec<CudaSlice<f32>>,
    q_clip: CudaSlice<f32>,
    rel_clip: CudaSlice<f32>,
    rel_bias: CudaSlice<f32>,
    k_conv: CudaSlice<f32>,
    v_conv: CudaSlice<f32>,
    attn_conv: CudaSlice<f32>,
    ffn_conv: CudaSlice<f32>,
    attn_query: Option<CudaSlice<f32>>,
    ffn_query: CudaSlice<f32>,
    ffn_norm: CudaSlice<f32>,
    router: CudaSlice<f32>,
    route_bias: CudaSlice<f32>,
    gate_up: Experts,
    down: Experts,
    looped: Option<LoopLayer>,
    /// The same projections, packed for the decode GEMM.
    decode: decode::DecodeLayer,
}

/// The resident model.
pub struct InklingModel {
    config: ModelConfig,
    k: InklingConfig,
    kernels: InklingKernels,
    hg: HgemmKernels,
    ops: LayerOpsKernels,
    route: K2Kernels,
    dg: llmcuda_cuda::kernels::inkling::DecodeGemm,
    dec: llmcuda_cuda::kernels::inkling::InklingDecodeKernels,
    layers: Vec<Layer>,
    tok_embd: CudaSlice<u8>,
    embd_bf16: bool,
    tok_norm: CudaSlice<f32>,
    out_norm: CudaSlice<f32>,
    out_query: CudaSlice<f32>,
    head: Matrix,
    weight_bytes: usize,
}

struct Loader<'a> {
    stream: &'a Arc<CudaStream>,
    file: &'a GgufFile,
    mma: MmaKernels,
    bytes: usize,
    /// Host staging for a repacked expert stack, reused across tensors.
    repack: Vec<u8>,
}

impl<'a> Loader<'a> {
    fn info(&self, name: &str) -> Res<(GgmlType, &'a [u8], usize)> {
        let file: &'a GgufFile = self.file;
        let info = file
            .tensor(name)
            .ok_or_else(|| bad(format!("missing tensor {name}")))?;
        let bytes = file
            .tensor_bytes(name)
            .ok_or_else(|| bad(format!("tensor {name} has no data")))?;
        let elements = info.dims.iter().product::<u64>() as usize;
        Ok((info.ggml_type, bytes, elements))
    }

    /// A small tensor widened to fp32, on the host.
    fn host_floats(&self, name: &str, elements: usize) -> Res<Vec<f32>> {
        let (ty, bytes, n) = self.info(name)?;
        if n != elements {
            return Err(bad(format!("{name}: {n} elements, expected {elements}")));
        }
        let values: Vec<f32> = match ty {
            GgmlType::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
            GgmlType::Bf16 => llmcuda_kernels::quant::dequantize_row_bf16(bytes),
            GgmlType::F16 => llmcuda_kernels::quant::dequantize_row_f16(bytes),
            other => {
                return Err(bad(format!(
                    "{name}: {} is not a float format",
                    other.name()
                )));
            }
        };
        Ok(values)
    }

    /// A small tensor as fp32 on the device.
    fn floats(&mut self, name: &str, elements: usize) -> Res<CudaSlice<f32>> {
        let values = self.host_floats(name, elements)?;
        self.bytes += values.len() * 4;
        Ok(self.stream.clone_htod(&values)?)
    }

    fn optional_floats(&mut self, name: &str, elements: usize) -> Res<Option<CudaSlice<f32>>> {
        if self.file.tensor(name).is_none() {
            return Ok(None);
        }
        self.floats(name, elements).map(Some)
    }

    fn dense_from(
        &mut self,
        name: &str,
        ty: GgmlType,
        bytes: &[u8],
        elements: usize,
    ) -> Res<Dense> {
        match ty {
            GgmlType::Q8_0 => {
                let src = self.stream.clone_htod(&bytes[..elements / 32 * 34])?;
                let mut q = self.stream.alloc_zeros::<i8>(elements)?;
                let mut scales = self.stream.alloc_zeros::<u16>(elements / 32)?;
                self.mma
                    .repack_q8_0_half(self.stream, &src, &mut q, &mut scales, elements)?;
                self.stream.synchronize()?;
                self.bytes += elements + elements / 16;
                Ok(Dense::Q8 { q, scales })
            }
            GgmlType::Bf16 => {
                let bytes = &bytes[..elements * 2];
                self.bytes += bytes.len();
                Ok(Dense::Bf16 {
                    exponent: bf16_half_exponent(bytes),
                    bytes: self.stream.clone_htod(bytes)?,
                })
            }
            // Stored fp32, contracted as bf16: round to nearest even. The
            // shipped fp32 controllers are upcasts of bf16, so this is exact
            // for them.
            GgmlType::F32 => {
                let mut out = Vec::with_capacity(elements * 2);
                for c in &bytes.as_chunks::<4>().0[..elements] {
                    let b = u32::from_le_bytes(*c);
                    let rounded = (b.wrapping_add(0x7FFF + ((b >> 16) & 1)) >> 16) as u16;
                    out.extend_from_slice(&rounded.to_le_bytes());
                }
                self.bytes += out.len();
                Ok(Dense::Bf16 {
                    exponent: bf16_half_exponent(&out),
                    bytes: self.stream.clone_htod(&out)?,
                })
            }
            other => Err(bad(format!(
                "{name}: {} has no tensor-core path here",
                other.name()
            ))),
        }
    }

    /// A projection `[rows][k]`; `rows` may be a prefix of the stored rows.
    fn matrix(&mut self, name: &str, rows: usize, k: usize) -> Res<Matrix> {
        let (ty, bytes, n) = self.info(name)?;
        if n < rows * k || !n.is_multiple_of(k) {
            return Err(bad(format!("{name}: {n} elements, expected {rows} x {k}")));
        }
        Ok(Matrix {
            w: self.dense_from(name, ty, bytes, rows * k)?,
            rows,
            k,
        })
    }

    /// One layer's experts for one projection. NVFP4 is repacked for the
    /// decode GEMM, whose lanes pair row `g` with row `g + partner`; the
    /// prefill GEMM reads the same bytes through the same map.
    fn experts(
        &mut self,
        name: &str,
        experts: usize,
        rows: usize,
        k: usize,
        partner: usize,
    ) -> Res<Experts> {
        let (ty, bytes, n) = self.info(name)?;
        if n != experts * rows * k {
            return Err(bad(format!(
                "{name}: {n} elements, expected {experts} x {rows} x {k}"
            )));
        }
        let w = if ty == GgmlType::Nvfp4 {
            self.bytes += bytes.len();
            if !k.is_multiple_of(64) || !nv4_tiles_fit(rows, partner) {
                return Err(bad(format!(
                    "{name}: {rows} x {k} is not whole NVFP4 tiles"
                )));
            }
            let per = rows * k / 64 * 36;
            if bytes.len() != experts * per {
                return Err(bad(format!("{name}: {} bytes of NVFP4", bytes.len())));
            }
            self.repack.clear();
            self.repack.resize(bytes.len(), 0);
            let threads = std::thread::available_parallelism().map_or(4, |n| n.get().min(16));
            let share = experts.div_ceil(threads) * per;
            std::thread::scope(|scope| {
                for (src, dst) in bytes.chunks(share).zip(self.repack.chunks_mut(share)) {
                    scope.spawn(move || {
                        for (src, dst) in src.chunks_exact(per).zip(dst.chunks_exact_mut(per)) {
                            nv4_tile_major(src, dst, rows, k, partner);
                        }
                    });
                }
            });
            ExpertStack::Nvfp4 {
                bytes: self.stream.clone_htod(&self.repack)?,
                scales: self.optional_floats(&name.replace(".weight", ".scale"), experts)?,
                partner,
            }
        } else {
            ExpertStack::Dense(self.dense_from(name, ty, bytes, n)?)
        };
        Ok(Experts { w, rows, k })
    }
}

impl InklingModel {
    /// Upload the model in `file`.
    pub fn load(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>, file: &GgufFile) -> Res<Self> {
        let config = ModelConfig::from_gguf(file).map_err(|e| bad(e.to_string()))?;
        let k = config
            .inkling
            .ok_or_else(|| bad("not an inkling2_loop file"))?;
        let moe = config.moe().ok_or_else(|| bad("no routed experts"))?;
        let n = config.hidden_size as usize;
        let (heads, kv_heads, dim) = (
            config.attention.q_heads as usize,
            config.attention.kv_heads as usize,
            config.attention.head_dim as usize,
        );
        let (n_rel, taps) = (k.relative_length as usize, k.conv_taps as usize);
        let (experts, ff) = (moe.num_experts as usize, moe.expert_intermediate as usize);
        let rank = k.gate_rank as usize;
        let mut l = Loader {
            stream,
            file,
            mma: MmaKernels::new(ctx)?,
            bytes: 0,
            repack: Vec::new(),
        };

        let mut layers = Vec::with_capacity(config.num_layers as usize);
        for i in 0..config.num_layers {
            let name = |s: &str| format!("blk.{i}.{s}");
            let sliding = k.is_sliding(i);
            let extent = if sliding {
                k.sliding_window
            } else {
                k.relative_extent
            } as usize;
            let norms = if sliding {
                vec![l.floats(&name("attn_norm.weight"), n)?]
            } else {
                vec![
                    l.floats(&name("attn_q_inp_norm.weight"), n)?,
                    l.floats(&name("attn_k_inp_norm.weight"), n)?,
                    l.floats(&name("attn_v_inp_norm.weight"), n)?,
                ]
            };
            let looped = if (k.loop_start..k.loop_end).contains(&i) {
                let mut ctrl = Vec::with_capacity(4);
                for which in ["q", "k", "v", "ffn"] {
                    let part = |p: &str| name(&format!("loop_{which}_ctrl_{p}.weight"));
                    ctrl.push(Controller {
                        down: l.matrix(&part("down"), rank, 2 * n)?,
                        gain: l.matrix(&part("gain"), n, rank)?,
                        forget: l.matrix(&part("forget"), n, rank)?,
                    });
                }
                Some(LoopLayer {
                    msg_query: l.floats(&name("loop_msg_query.weight"), 4 * n)?,
                    ctrl: ctrl.try_into().ok().expect("four controllers"),
                })
            } else {
                None
            };
            layers.push(Layer {
                sliding,
                q: l.matrix(&name("attn_q.weight"), heads * dim, n)?,
                k: l.matrix(&name("attn_k.weight"), kv_heads * dim, n)?,
                v: l.matrix(&name("attn_v.weight"), kv_heads * dim, n)?,
                o: l.matrix(&name("attn_output.weight"), n, heads * dim)?,
                gate: l.matrix(&name("attn_gate.weight"), heads * dim, n)?,
                rel: l.matrix(&name("attn_rel.weight"), heads * n_rel, n)?,
                sink: l.matrix(&name("attn_sink_proj.weight"), heads, n)?,
                k_conv_proj: if sliding {
                    None
                } else {
                    Some(l.matrix(&name("attn_k_conv_proj.weight"), kv_heads * taps, n)?)
                },
                v_conv_proj: if sliding {
                    None
                } else {
                    Some(l.matrix(&name("attn_v_conv_proj.weight"), kv_heads * taps, n)?)
                },
                norms,
                q_clip: l.floats(&name("attn_q_clip.weight"), heads * dim)?,
                rel_clip: l.floats(&name("attn_rel_clip.weight"), heads * n_rel)?,
                rel_bias: l.floats(&name("attn_rel_bias.weight"), extent * n_rel)?,
                k_conv: l.floats(&name("attn_k_conv.weight"), kv_heads * dim * taps)?,
                v_conv: l.floats(&name("attn_v_conv.weight"), kv_heads * dim * taps)?,
                attn_conv: l.floats(&name("attn_post_conv.weight"), n * taps)?,
                ffn_conv: l.floats(&name("ffn_post_conv.weight"), n * taps)?,
                attn_query: if i == 0 {
                    None
                } else {
                    Some(l.floats(
                        &name("attn_res_query.weight"),
                        if sliding { n } else { 3 * n },
                    )?)
                },
                ffn_query: l.floats(&name("ffn_res_query.weight"), n)?,
                ffn_norm: l.floats(&name("ffn_norm.weight"), n)?,
                router: l.floats(&name("ffn_gate_inp.weight"), experts * n)?,
                route_bias: match l.optional_floats(&name("exp_probs_b.bias"), experts)? {
                    Some(b) => b,
                    None => stream.alloc_zeros::<f32>(experts)?,
                },
                gate_up: l.experts(&name("ffn_gate_up_exps.weight"), experts, 2 * ff, n, ff)?,
                down: l.experts(&name("ffn_down_exps.weight"), experts, n, ff, 8)?,
                looped,
                decode: decode::DecodeLayer::load(&mut l, i, &config)?,
            });
        }

        let (ty, bytes, _) = l.info("token_embd.weight")?;
        let embd_bf16 = match ty {
            GgmlType::Q8_0 => false,
            GgmlType::Bf16 => true,
            other => {
                return Err(bad(format!(
                    "token_embd.weight: {} unsupported",
                    other.name()
                )));
            }
        };
        l.bytes += bytes.len();
        let tok_embd = stream.clone_htod(bytes)?;
        let model = Self {
            tok_embd,
            embd_bf16,
            tok_norm: l.floats("token_embd_norm.weight", n)?,
            out_norm: l.floats("output_norm.weight", n)?,
            out_query: l.floats("output_res_query.weight", n)?,
            // Only the real vocabulary's rows: the padded tail is never a
            // token.
            head: l.matrix("output.weight", config.vocab_size as usize, n)?,
            weight_bytes: l.bytes,
            layers,
            kernels: InklingKernels::new(ctx)?,
            hg: HgemmKernels::new(ctx)?,
            ops: LayerOpsKernels::new(ctx)?,
            route: K2Kernels::new(ctx)?,
            dg: llmcuda_cuda::kernels::inkling::DecodeGemm::new(ctx)?,
            dec: llmcuda_cuda::kernels::inkling::InklingDecodeKernels::new(ctx)?,
            config,
            k,
        };
        stream.synchronize()?;
        Ok(model)
    }

    /// The configuration the file declared.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Run prefill on the plain reference kernels instead of the
    /// tensor-core attention and the tiled fp32 projection. For A/B
    /// measurement and tests.
    pub fn use_reference_kernels(&self, on: bool) {
        self.kernels.use_reference_kernels(on);
    }

    /// Choose the decode attention kernel. For A/B measurement and tests.
    pub fn use_decode_attention(&self, which: llmcuda_cuda::kernels::inkling::DecodeAttention) {
        self.dec.use_decode_attention(which);
    }

    /// Bytes of weights resident on the device.
    pub fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }

    fn slots(&self) -> usize {
        self.k.slots(self.config.num_layers) as usize
    }

    fn kv_width(&self) -> usize {
        (self.config.attention.kv_heads * self.config.attention.head_dim) as usize
    }
}

/// One layer application's share of a sequence's state.
struct SlotState {
    /// Rows of `k` and `v` when they are a ring: position `j` at row
    /// `j % ring`. `None` when they hold every position.
    ring: Option<usize>,
    k: CudaSlice<u16>,
    v: CudaSlice<u16>,
    k_carry: CudaSlice<f32>,
    v_carry: CudaSlice<f32>,
    attn_carry: CudaSlice<f32>,
    ffn_carry: CudaSlice<f32>,
}

/// One sequence: its caches, convolution carries and position.
pub struct InklingState {
    slots: Vec<SlotState>,
    position: usize,
    capacity: usize,
    /// The widest pass the ring caches leave room for.
    max_pass_rows: usize,
    d_position: CudaSlice<i32>,
}

impl InklingState {
    /// An empty sequence that can grow to `capacity` positions, every
    /// layer application keeping keys and values for all of them.
    pub fn new(stream: &Arc<CudaStream>, model: &InklingModel, capacity: usize) -> Res<Self> {
        Self::build(stream, model, capacity, None)
    }

    /// An empty sequence that can grow to `capacity` positions in passes of
    /// at most `max_pass_rows` rows, with the sliding layers' caches bounded.
    ///
    /// A sliding layer reads `window` positions back from each row, and a
    /// pass appends its rows before it attends, so `window + max_pass_rows`
    /// rows are all it can reach. Its caches are rings of the next power of
    /// two (position `j` at row `j % ring`), instead of `capacity` rows —
    /// which at this model's 40 sliding applications of 48 is most of a
    /// long sequence's memory. Results are the same bits.
    ///
    /// The ring's size depends on `max_pass_rows` alone, not on `capacity`:
    /// sequences of any two lengths can share a decode step.
    pub fn bounded(
        stream: &Arc<CudaStream>,
        model: &InklingModel,
        capacity: usize,
        max_pass_rows: usize,
    ) -> Res<Self> {
        if max_pass_rows == 0 {
            return Err(bad("a pass has at least one row"));
        }
        Self::build(stream, model, capacity, Some(max_pass_rows))
    }

    fn build(
        stream: &Arc<CudaStream>,
        model: &InklingModel,
        capacity: usize,
        max_pass_rows: Option<usize>,
    ) -> Res<Self> {
        let n = model.config.hidden_size as usize;
        let kv = model.kv_width();
        let carry = model.k.conv_taps as usize - 1;
        let layers = model.config.num_layers;
        let mut slots = Vec::with_capacity(model.slots());
        for slot in 0..model.slots() {
            let sliding = model.k.is_sliding(model.k.slot_layer(layers, slot as u32));
            // A ring whatever the capacity, even where `capacity` rows would
            // be fewer: a decode step takes one row mask for all its rows,
            // so every sequence that may share a step holds the same ring.
            let ring = max_pass_rows
                .filter(|_| sliding)
                .map(|pass| (model.k.sliding_window as usize + pass).next_power_of_two());
            let rows = ring.unwrap_or(capacity);
            slots.push(SlotState {
                ring,
                k: stream.alloc_zeros::<u16>(rows * kv)?,
                v: stream.alloc_zeros::<u16>(rows * kv)?,
                k_carry: stream.alloc_zeros::<f32>(kv * carry)?,
                v_carry: stream.alloc_zeros::<f32>(kv * carry)?,
                attn_carry: stream.alloc_zeros::<f32>(n * carry)?,
                ffn_carry: stream.alloc_zeros::<f32>(n * carry)?,
            });
        }
        Ok(Self {
            slots,
            position: 0,
            capacity,
            max_pass_rows: max_pass_rows.unwrap_or(usize::MAX),
            d_position: stream.alloc_zeros::<i32>(1)?,
        })
    }

    /// Bytes of keys and values this sequence holds on the device.
    pub fn cache_bytes(&self) -> usize {
        self.slots.iter().map(|s| (s.k.len() + s.v.len()) * 2).sum()
    }

    /// Positions consumed so far.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Forget everything: position zero, carries zero. The caches need no
    /// clearing; nothing past the position is ever read.
    pub fn reset(&mut self, stream: &Arc<CudaStream>) -> Res {
        for s in &mut self.slots {
            for c in [
                &mut s.k_carry,
                &mut s.v_carry,
                &mut s.attn_carry,
                &mut s.ffn_carry,
            ] {
                stream.memset_zeros(c)?;
            }
        }
        self.position = 0;
        Ok(())
    }
}

/// Device time per stage, from CUDA events recorded at the stage
/// boundaries and read back once the pass has finished.
///
/// Events rather than a synchronized host clock: synchronizing at every
/// boundary measured ~115 us a stage of its own, which at a decode step's
/// ~440 stages was more than twice the step.
struct Profile {
    events: Vec<cudarc::driver::CudaEvent>,
    /// The stage each recorded event closes; `None` opens a pass.
    closes: Vec<Option<&'static str>>,
    stages: Vec<(&'static str, f64)>,
}

impl Profile {
    fn start() -> Self {
        Self {
            events: Vec::new(),
            closes: Vec::new(),
            stages: Vec::new(),
        }
    }

    fn record(&mut self, stream: &Arc<CudaStream>, closes: Option<&'static str>) -> Res {
        let at = self.closes.len();
        if at == self.events.len() {
            self.events.push(
                stream
                    .context()
                    .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?,
            );
        }
        self.events[at].record(stream)?;
        self.closes.push(closes);
        Ok(())
    }

    /// Open a pass: what ran since the last mark was not a stage of it.
    fn restart(&mut self, stream: &Arc<CudaStream>) -> Res {
        self.resolve()?;
        self.record(stream, None)
    }

    /// Close the stage that just ran.
    fn mark(&mut self, stream: &Arc<CudaStream>, stage: &'static str) -> Res {
        self.record(stream, Some(stage))
    }

    /// Wait for the recorded events and add their intervals to the stages.
    fn resolve(&mut self) -> Res {
        if let Some(last) = self.closes.len().checked_sub(1) {
            self.events[last].synchronize()?;
        }
        for at in 1..self.closes.len() {
            let Some(stage) = self.closes[at] else {
                continue;
            };
            let dt = f64::from(self.events[at - 1].elapsed_ms(&self.events[at])?) * 1e-3;
            match self.stages.iter_mut().find(|(name, _)| *name == stage) {
                Some((_, total)) => *total += dt,
                None => self.stages.push((stage, dt)),
            }
        }
        self.closes.clear();
        Ok(())
    }
}

/// Scratch for passes of exactly `rows` rows.
pub struct InklingPass {
    model: Arc<InklingModel>,
    rows: usize,
    profile: Option<Profile>,
    d_tokens: CudaSlice<i32>,
    h: CudaSlice<f32>,
    input: CudaSlice<f32>,
    prelude: CudaSlice<f32>,
    /// `[states][rows][n]`: the residual blocks, then the current state.
    blocks: CudaSlice<f32>,
    bank: CudaSlice<f32>,
    mixed: CudaSlice<f32>,
    messages: CudaSlice<f32>,
    controlled: [CudaSlice<f32>; 3],
    normed: [CudaSlice<f32>; 3],
    normed_h: [CudaSlice<u16>; 3],
    pair_h: CudaSlice<u16>,
    ctrl_gate: CudaSlice<f32>,
    ctrl_gate_h: CudaSlice<u16>,
    gain: CudaSlice<f32>,
    forget: CudaSlice<f32>,
    q: CudaSlice<f32>,
    k_raw: CudaSlice<f32>,
    v_raw: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    r: CudaSlice<f32>,
    sink: CudaSlice<f32>,
    k_taps: CudaSlice<f32>,
    v_taps: CudaSlice<f32>,
    /// The reference path's bias table, allocated the first time that
    /// path runs: `rows * heads * (extent + 1)` floats nothing else needs.
    table: Option<CudaSlice<f32>>,
    attended: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    gate_sig: CudaSlice<f32>,
    gated: CudaSlice<f32>,
    gated_h: CudaSlice<u16>,
    o_raw: CudaSlice<f32>,
    attn_out: CudaSlice<f32>,
    ffn_inp: CudaSlice<f32>,
    x_mix: CudaSlice<f32>,
    x_ctrl: CudaSlice<f32>,
    x: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    ids: CudaSlice<i32>,
    route_weights: CudaSlice<f32>,
    /// Expert ids of every routing since [`Self::log_routes`], for tests.
    route_log: Option<Vec<i32>>,
    dispatch: InkDispatch,
    xs: CudaSlice<u16>,
    gate_up: CudaSlice<f32>,
    hidden_h: CudaSlice<u16>,
    y: CudaSlice<f32>,
    moe: CudaSlice<f32>,
    ffn_out: CudaSlice<f32>,
    final_norm: CudaSlice<f32>,
    final_h: CudaSlice<u16>,
    head_logits: CudaSlice<f32>,
}

/// A run of consecutive positions of one sequence within a pass.
pub struct Segment<'a> {
    /// The sequence.
    pub state: &'a mut InklingState,
    /// Rows of the pass it takes.
    pub rows: usize,
}

/// A row range of a `[rows][width]` buffer.
///
/// # Safety
/// As [`subslice`]: the range is inside `base` and the view does not outlive
/// it.
unsafe fn rows_of<T>(
    stream: &Arc<CudaStream>,
    base: &CudaSlice<T>,
    row0: usize,
    rows: usize,
    width: usize,
) -> ManuallyDrop<CudaSlice<T>> {
    unsafe { subslice(stream, base, row0 * width, rows * width) }
}

/// Rows of a pass whose inputs are given: `rows` rows of `from`, starting
/// at its row `first`, in place of the pass's rows from `at`.
#[derive(Clone, Copy)]
pub struct GivenRows<'a> {
    /// The first row of the pass these replace.
    pub at: usize,
    /// `[..][hidden]` fp32 rows.
    pub from: &'a CudaSlice<f32>,
    /// The first row of `from` to take.
    pub first: usize,
    /// How many.
    pub rows: usize,
}

impl InklingPass {
    /// Allocate for passes of exactly `rows` rows.
    pub fn new(model: &Arc<InklingModel>, stream: &Arc<CudaStream>, rows: usize) -> Res<Self> {
        Self::with_logit_rows(model, stream, rows, rows)
    }

    /// As [`Self::new`], with room for the logits of only `logit_rows` rows
    /// at a time: a row of logits is the vocabulary wide, and a server reads
    /// one a pass.
    pub fn with_logit_rows(
        model: &Arc<InklingModel>,
        stream: &Arc<CudaStream>,
        rows: usize,
        logit_rows: usize,
    ) -> Res<Self> {
        let c = &model.config;
        // Several kernels take a block a row on a grid axis that ends at
        // 65535.
        if rows == 0 || rows > 65_535 {
            return Err(bad(format!("a pass takes 1..=65535 rows, not {rows}")));
        }
        let k = &model.k;
        let n = c.hidden_size as usize;
        let (heads, dim) = (c.attention.q_heads as usize, c.attention.head_dim as usize);
        let kv = model.kv_width();
        let moe = c.moe().expect("routed experts");
        let (experts, topk, ff) = (
            moe.num_experts as usize,
            moe.experts_per_token as usize,
            moe.expert_intermediate as usize,
        );
        let block_states = (c.num_layers / k.res_block) as usize + 1;
        let bank_states = ((k.loop_end - k.loop_start) / k.res_block) as usize;
        let dispatch = InkDispatch::new(stream, rows, topk, experts)?;
        let slots = dispatch.slots();
        let f = |len: usize| stream.alloc_zeros::<f32>(len.max(1));
        let hf = |len: usize| stream.alloc_zeros::<u16>(len.max(1));
        let taps = model.config.attention.kv_heads as usize * k.conv_taps as usize;
        Ok(Self {
            model: model.clone(),
            rows,
            profile: None,
            d_tokens: stream.alloc_zeros::<i32>(rows)?,
            h: f(rows * n)?,
            input: f(rows * n)?,
            prelude: f(rows * n)?,
            blocks: f(block_states * rows * n)?,
            bank: f(bank_states * rows * n)?,
            mixed: f(3 * rows * n)?,
            messages: f(4 * rows * n)?,
            controlled: [f(rows * n)?, f(rows * n)?, f(rows * n)?],
            normed: [f(rows * n)?, f(rows * n)?, f(rows * n)?],
            normed_h: [hf(rows * n)?, hf(rows * n)?, hf(rows * n)?],
            pair_h: hf(rows * 2 * n)?,
            ctrl_gate: f(rows * k.gate_rank as usize)?,
            ctrl_gate_h: hf(rows * k.gate_rank as usize)?,
            gain: f(rows * n)?,
            forget: f(rows * n)?,
            q: f(rows * heads * dim)?,
            k_raw: f(rows * kv)?,
            v_raw: f(rows * kv)?,
            k: f(rows * kv)?,
            v: f(rows * kv)?,
            r: f(rows * heads * k.relative_length as usize)?,
            sink: f(rows * heads)?,
            k_taps: f(rows * taps)?,
            v_taps: f(rows * taps)?,
            table: None,
            attended: f(rows * heads * dim)?,
            gate: f(rows * heads * dim)?,
            gate_sig: f(rows * heads * dim)?,
            gated: f(rows * heads * dim)?,
            gated_h: hf(rows * heads * dim)?,
            o_raw: f(rows * n)?,
            attn_out: f(rows * n)?,
            ffn_inp: f(rows * n)?,
            x_mix: f(rows * n)?,
            x_ctrl: f(rows * n)?,
            x: f(rows * n)?,
            logits: f(rows * experts)?,
            ids: stream.alloc_zeros::<i32>(rows * topk)?,
            route_weights: f(rows * topk)?,
            route_log: None,
            xs: hf(slots * n)?,
            gate_up: f(slots * 2 * ff)?,
            hidden_h: hf(slots * ff)?,
            y: f(slots * n)?,
            dispatch,
            moe: f(rows * n)?,
            ffn_out: f(rows * n)?,
            final_norm: f(rows * n)?,
            final_h: hf(rows * n)?,
            head_logits: f(logit_rows.min(rows) * c.vocab_size as usize)?,
        })
    }

    /// Rows a pass runs.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Record the experts of every routing from here on: `[rows][top_k]`
    /// per layer application, in execution order. A diagnostic; it reads
    /// the device back at every layer.
    pub fn log_routes(&mut self) {
        self.route_log = Some(Vec::new());
    }

    /// What [`Self::log_routes`] has recorded.
    pub fn routes(&self) -> &[i32] {
        self.route_log.as_deref().unwrap_or(&[])
    }

    /// Start timing stages: every stage boundary synchronizes the stream
    /// from here on, so a profiled pass is slower than a plain one.
    pub fn enable_profiling(&mut self) {
        self.profile = Some(Profile::start());
    }

    /// Seconds per stage since profiling was enabled, in first-seen order.
    pub fn stage_seconds(&mut self) -> Res<&[(&'static str, f64)]> {
        match self.profile.as_mut() {
            Some(p) => {
                p.resolve()?;
                Ok(&p.stages)
            }
            None => Ok(&[]),
        }
    }

    /// Close the stage that just ran.
    fn mark(&mut self, stream: &Arc<CudaStream>, stage: &'static str) -> Res {
        match self.profile.as_mut() {
            Some(p) => p.mark(stream, stage),
            None => Ok(()),
        }
    }

    /// The final normalized hidden states of the last pass, `[rows][n]`.
    pub fn hidden(&self) -> &CudaSlice<f32> {
        &self.final_norm
    }

    /// The logits [`Self::logits`] last computed, `[rows][vocab]`.
    pub fn head_logits(&self) -> &CudaSlice<f32> {
        &self.head_logits
    }

    /// `out = m . x` for every row, through the fp16 tensor cores.
    fn gemm(
        model: &InklingModel,
        stream: &Arc<CudaStream>,
        m: &Matrix,
        x: &CudaSlice<u16>,
        out: &mut CudaSlice<f32>,
        rows: usize,
    ) -> Res {
        model.hg.gemm(
            stream,
            m.half(),
            x,
            out,
            None,
            rows,
            m.rows,
            m.k,
            m.rows,
            false,
        )?;
        Ok(())
    }

    fn gemm_experts(
        model: &InklingModel,
        stream: &Arc<CudaStream>,
        e: &Experts,
        d: &InkDispatch,
        x: &CudaSlice<u16>,
        out: &mut CudaSlice<f32>,
        experts: usize,
    ) -> Res {
        let (w, scales) = match &e.w {
            ExpertStack::Nvfp4 {
                bytes,
                scales,
                partner,
            } => (
                HalfWeight::Nvfp4 {
                    bytes,
                    scale: 1.0,
                    partner: *partner,
                },
                scales.as_ref(),
            ),
            ExpertStack::Dense(Dense::Q8 { q, scales }) => (HalfWeight::Q8_0 { q, scales }, None),
            ExpertStack::Dense(Dense::Bf16 { bytes, exponent }) => (
                HalfWeight::Bf16 {
                    bytes,
                    exponent: *exponent,
                },
                None,
            ),
        };
        model.hg.gemm_grouped(
            stream,
            w,
            experts,
            d.owners(),
            scales,
            x,
            out,
            e.rows,
            e.k,
            false,
        )?;
        Ok(())
    }

    /// `out = (1 - tanh(forget)) * cur + tanh(gain) * msg`, both gates read
    /// from the normalized pair.
    fn control(
        &mut self,
        stream: &Arc<CudaStream>,
        c: &Controller,
        cur: &CudaSlice<f32>,
        msg: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> Res {
        let model = self.model.clone();
        let n = model.config.hidden_size as usize;
        let rows = self.rows;
        let rank = model.k.gate_rank as usize;
        model
            .kernels
            .ctrl_pair(stream, cur, msg, &mut self.pair_h, n, rows)?;
        Self::gemm(
            &model,
            stream,
            &c.down,
            &self.pair_h,
            &mut self.ctrl_gate,
            rows,
        )?;
        model
            .hg
            .to_half(stream, &self.ctrl_gate, &mut self.ctrl_gate_h, rows * rank)?;
        Self::gemm(
            &model,
            stream,
            &c.gain,
            &self.ctrl_gate_h,
            &mut self.gain,
            rows,
        )?;
        Self::gemm(
            &model,
            stream,
            &c.forget,
            &self.ctrl_gate_h,
            &mut self.forget,
            rows,
        )?;
        model
            .kernels
            .ctrl_apply(stream, cur, msg, &self.gain, &self.forget, out, rows * n)?;
        Ok(())
    }

    /// One layer application. `self.h` holds the input and takes the output;
    /// `blocks_live` is how many residual blocks the stack holds.
    #[allow(clippy::too_many_lines)]
    fn layer(
        &mut self,
        stream: &Arc<CudaStream>,
        segments: &mut [Segment<'_>],
        slot: usize,
        layer_index: usize,
        blocks_live: &mut usize,
        with_messages: bool,
    ) -> Res {
        let model = self.model.clone();
        let layer = &model.layers[layer_index];
        let c = &model.config;
        let kc = &model.k;
        let n = c.hidden_size as usize;
        let rows = self.rows;
        let plane = rows * n;
        let (heads, kv_heads, dim) = (
            c.attention.q_heads as usize,
            c.attention.kv_heads as usize,
            c.attention.head_dim as usize,
        );
        let kv = kv_heads * dim;
        let n_rel = kc.relative_length as usize;
        let taps = kc.conv_taps as usize;
        let eps = f32::from_bits(kc.rms_eps_bits);
        let log = (!layer.sliding).then(|| {
            (
                f32::from_bits(kc.log_scale_alpha_bits),
                kc.log_scale_floor as f32,
            )
        });
        let (window, extent) = if layer.sliding {
            (kc.sliding_window as usize, kc.sliding_window as usize)
        } else {
            (0, kc.relative_extent as usize)
        };
        let looped = if with_messages {
            Some(
                layer
                    .looped
                    .as_ref()
                    .ok_or_else(|| bad("messages outside the loop"))?,
            )
        } else {
            None
        };

        // The layer input joins the stack as the current state; if this
        // layer opens a block it simply stays there.
        stream.memcpy_dtod(&self.h, &mut self.input)?;
        {
            let mut top = unsafe { subslice(stream, &self.blocks, *blocks_live * plane, plane) };
            stream.memcpy_dtod(&self.h, &mut *top)?;
        }

        // Attention inputs: how many distinct ones, and where they come from.
        let n_mixed = match &layer.attn_query {
            Some(query) => {
                let q = if layer.sliding { 1 } else { 3 };
                model.kernels.mix(
                    stream,
                    &self.blocks,
                    query,
                    &mut self.mixed,
                    n,
                    *blocks_live + 1,
                    q,
                    rows,
                )?;
                q
            }
            None => 0,
        };
        let distinct = if looped.is_some() { 3 } else { n_mixed.max(1) };
        for j in 0..distinct {
            let source = match n_mixed {
                0 => unsafe { subslice(stream, &self.input, 0, plane) },
                m => unsafe { subslice(stream, &self.mixed, j.min(m - 1) * plane, plane) },
            };
            let norm = &layer.norms[j.min(layer.norms.len() - 1)];
            if let Some(lp) = looped {
                let msg = unsafe { subslice(stream, &self.messages, j * plane, plane) };
                let mut out = std::mem::replace(&mut self.controlled[j], stream.alloc_zeros(1)?);
                let r = self.control(stream, &lp.ctrl[j], &source, &msg, &mut out);
                self.controlled[j] = out;
                r?;
                model.ops.rms_norm(
                    stream,
                    &self.controlled[j],
                    norm,
                    &mut self.normed[j],
                    rows,
                    n,
                    eps,
                )?;
            } else {
                model
                    .ops
                    .rms_norm(stream, &source, norm, &mut self.normed[j], rows, n, eps)?;
            }
            model
                .hg
                .to_half(stream, &self.normed[j], &mut self.normed_h[j], plane)?;
        }
        self.mark(stream, "attn inputs (mix, control, norm)")?;
        let (iq, ik, iv) = (0, 1.min(distinct - 1), 2.min(distinct - 1));

        Self::gemm(
            &model,
            stream,
            &layer.q,
            &self.normed_h[iq],
            &mut self.q,
            rows,
        )?;
        Self::gemm(
            &model,
            stream,
            &layer.rel,
            &self.normed_h[iq],
            &mut self.r,
            rows,
        )?;
        Self::gemm(
            &model,
            stream,
            &layer.sink,
            &self.normed_h[iq],
            &mut self.sink,
            rows,
        )?;
        Self::gemm(
            &model,
            stream,
            &layer.gate,
            &self.normed_h[iq],
            &mut self.gate,
            rows,
        )?;
        Self::gemm(
            &model,
            stream,
            &layer.k,
            &self.normed_h[ik],
            &mut self.k_raw,
            rows,
        )?;
        Self::gemm(
            &model,
            stream,
            &layer.v,
            &self.normed_h[iv],
            &mut self.v_raw,
            rows,
        )?;
        if let (Some(kp), Some(vp)) = (&layer.k_conv_proj, &layer.v_conv_proj) {
            Self::gemm(
                &model,
                stream,
                kp,
                &self.normed_h[ik],
                &mut self.k_taps,
                rows,
            )?;
            Self::gemm(
                &model,
                stream,
                vp,
                &self.normed_h[iv],
                &mut self.v_taps,
                rows,
            )?;
        }

        self.mark(stream, "attn projections")?;
        // Key and value convolutions, each sequence over its own carry.
        let dynamic = layer.k_conv_proj.is_some();
        let mut row0 = 0;
        for seg in segments.iter_mut() {
            let st = &mut seg.state.slots[slot];
            let r = seg.rows;
            for (raw, weight, tap_buf, carry, out) in [
                (
                    &self.k_raw,
                    &layer.k_conv,
                    &self.k_taps,
                    &mut st.k_carry,
                    &mut self.k,
                ),
                (
                    &self.v_raw,
                    &layer.v_conv,
                    &self.v_taps,
                    &mut st.v_carry,
                    &mut self.v,
                ),
            ] {
                let x = unsafe { rows_of(stream, raw, row0, r, kv) };
                let mut o = unsafe { rows_of(stream, out, row0, r, kv) };
                let t = unsafe { rows_of(stream, tap_buf, row0, r, kv_heads * taps) };
                model.kernels.conv(
                    stream,
                    &x,
                    weight,
                    dynamic.then_some((&*t, kv_heads)),
                    carry,
                    &mut o,
                    kv,
                    taps,
                    r,
                )?;
            }
            row0 += r;
        }

        // Clips and the value norm; the position scale needs each segment's
        // own base position.
        model
            .kernels
            .rms_groups(stream, &mut self.v, dim, eps, rows * kv_heads)?;
        let mut row0 = 0;
        for seg in segments.iter_mut() {
            let r = seg.rows;
            let pos = &seg.state.d_position;
            let mut qv = unsafe { rows_of(stream, &self.q, row0, r, heads * dim) };
            model.kernels.clip(
                stream,
                &mut qv,
                Some(&layer.q_clip),
                pos,
                dim,
                heads,
                f32::from_bits(kc.qk_clip_bits),
                log,
                r,
            )?;
            let mut kview = unsafe { rows_of(stream, &self.k, row0, r, kv) };
            model.kernels.clip(
                stream,
                &mut kview,
                None,
                pos,
                dim,
                kv_heads,
                f32::from_bits(kc.qk_clip_bits),
                None,
                r,
            )?;
            let mut rv = unsafe { rows_of(stream, &self.r, row0, r, heads * n_rel) };
            model.kernels.clip(
                stream,
                &mut rv,
                Some(&layer.rel_clip),
                pos,
                n_rel,
                heads,
                f32::from_bits(kc.relative_clip_bits),
                log,
                r,
            )?;
            row0 += r;
        }
        // The reference path looks the bias up in a table of every
        // distance for every row; the fused kernel computes what it reads.
        let fused = model.kernels.fuses_the_bias(dim, n_rel);
        if !fused {
            if self.table.is_none() {
                // Sized for the widest layer: sliding and full layers differ.
                let widest = kc.sliding_window.max(kc.relative_extent) as usize;
                self.table = Some(stream.alloc_zeros::<f32>(self.rows * heads * (widest + 1))?);
            }
            model.kernels.rel_table(
                stream,
                &layer.rel_bias,
                &self.r,
                &self.sink,
                self.table.as_mut().expect("just allocated"),
                n_rel,
                extent,
                heads,
                rows,
            )?;
        }

        self.mark(stream, "conv, clips, bias table")?;
        // Append to each sequence's cache, then attend over it.
        let mut row0 = 0;
        for seg in segments.iter_mut() {
            let r = seg.rows;
            let at = seg.state.position;
            let st = &mut seg.state.slots[slot];
            // A ring takes the rows in at most two runs: up to its end, then
            // from its start.
            let first = st.ring.map_or(r, |ring| r.min(ring - at % ring));
            for (src0, dst0, count) in [
                (row0, st.ring.map_or(at, |ring| at % ring), first),
                (row0 + first, 0, r - first),
            ] {
                if count == 0 {
                    continue;
                }
                let kview = unsafe { rows_of(stream, &self.k, src0, count, kv) };
                let mut dst = unsafe { rows_of(stream, &st.k, dst0, count, kv) };
                model.hg.to_half(stream, &kview, &mut dst, count * kv)?;
                let vview = unsafe { rows_of(stream, &self.v, src0, count, kv) };
                let mut dst = unsafe { rows_of(stream, &st.v, dst0, count, kv) };
                model.hg.to_half(stream, &vview, &mut dst, count * kv)?;
            }
            let qv = unsafe { rows_of(stream, &self.q, row0, r, heads * dim) };
            let mut out = unsafe { rows_of(stream, &self.attended, row0, r, heads * dim) };
            if fused {
                let rv = unsafe { rows_of(stream, &self.r, row0, r, heads * n_rel) };
                let sv = unsafe { rows_of(stream, &self.sink, row0, r, heads) };
                model.kernels.attn_fused(
                    stream,
                    &qv,
                    &st.k,
                    &st.v,
                    &rv,
                    &sv,
                    &layer.rel_bias,
                    &mut out,
                    &seg.state.d_position,
                    window,
                    extent,
                    heads,
                    kv_heads,
                    r,
                    st.ring,
                )?;
            } else {
                let table = self.table.as_ref().expect("filled above");
                let tv = unsafe { rows_of(stream, table, row0, r, heads * (extent + 1)) };
                model.kernels.attn(
                    stream,
                    &qv,
                    &st.k,
                    &st.v,
                    &tv,
                    &mut out,
                    &seg.state.d_position,
                    window,
                    extent,
                    heads,
                    kv_heads,
                    dim,
                    r,
                    st.ring,
                )?;
            }
            row0 += r;
        }

        self.mark(stream, "attention")?;
        model.ops.sigmoid_gate(
            stream,
            &self.attended,
            &self.gate,
            &mut self.gate_sig,
            &mut self.gated,
            rows,
            heads * dim,
            GateShape::Elementwise,
        )?;
        model
            .hg
            .to_half(stream, &self.gated, &mut self.gated_h, rows * heads * dim)?;
        Self::gemm(
            &model,
            stream,
            &layer.o,
            &self.gated_h,
            &mut self.o_raw,
            rows,
        )?;
        let mut row0 = 0;
        for seg in segments.iter_mut() {
            let r = seg.rows;
            let x = unsafe { rows_of(stream, &self.o_raw, row0, r, n) };
            let mut o = unsafe { rows_of(stream, &self.attn_out, row0, r, n) };
            model.kernels.conv(
                stream,
                &x,
                &layer.attn_conv,
                None,
                &mut seg.state.slots[slot].attn_carry,
                &mut o,
                n,
                taps,
                r,
            )?;
            row0 += r;
        }

        // The first layer of a block starts a new residual stream; the one
        // before it joins the mixed blocks (it is already on the stack).
        if (layer_index as u32).is_multiple_of(kc.res_block) {
            *blocks_live += 1;
            stream.memcpy_dtod(&self.attn_out, &mut self.ffn_inp)?;
        } else {
            model.ops.add(
                stream,
                &self.input,
                &self.attn_out,
                &mut self.ffn_inp,
                plane,
            )?;
        }

        self.mark(stream, "gate, output projection, conv")?;
        // Feed-forward: mix, control, norm, route, experts, convolve.
        {
            let mut top = unsafe { subslice(stream, &self.blocks, *blocks_live * plane, plane) };
            stream.memcpy_dtod(&self.ffn_inp, &mut *top)?;
        }
        model.kernels.mix(
            stream,
            &self.blocks,
            &layer.ffn_query,
            &mut self.x_mix,
            n,
            *blocks_live + 1,
            1,
            rows,
        )?;
        if let Some(lp) = looped {
            let msg = unsafe { subslice(stream, &self.messages, 3 * plane, plane) };
            let mix = std::mem::replace(&mut self.x_mix, stream.alloc_zeros(1)?);
            let mut out = std::mem::replace(&mut self.x_ctrl, stream.alloc_zeros(1)?);
            let r = self.control(stream, &lp.ctrl[3], &mix, &msg, &mut out);
            self.x_mix = mix;
            self.x_ctrl = out;
            r?;
            model.ops.rms_norm(
                stream,
                &self.x_ctrl,
                &layer.ffn_norm,
                &mut self.x,
                rows,
                n,
                eps,
            )?;
        } else {
            model.ops.rms_norm(
                stream,
                &self.x_mix,
                &layer.ffn_norm,
                &mut self.x,
                rows,
                n,
                eps,
            )?;
        }
        let moe = c.moe().expect("routed experts");
        let (experts, topk, ff) = (
            moe.num_experts as usize,
            moe.experts_per_token as usize,
            moe.expert_intermediate as usize,
        );
        self.mark(stream, "ffn input (mix, control, norm)")?;
        model.kernels.linear(
            stream,
            &layer.router,
            &self.x,
            &mut self.logits,
            experts,
            n,
            n,
            rows,
        )?;
        model.route.route(
            stream,
            &self.logits,
            &layer.route_bias,
            &mut self.ids,
            &mut self.route_weights,
            topk,
            f32::from_bits(kc.expert_weights_scale_bits),
        )?;
        self.mark(stream, "router logits, route")?;
        if let Some(log) = self.route_log.as_mut() {
            let ids = stream.clone_dtoh(&self.ids)?;
            log.extend_from_slice(&ids[..rows * topk]);
        }
        model
            .kernels
            .dispatch(stream, &self.ids, &mut self.dispatch)?;
        model
            .kernels
            .gather_half(stream, &self.x, &mut self.dispatch, &mut self.xs, n)?;
        self.mark(stream, "dispatch, gather")?;
        Self::gemm_experts(
            &model,
            stream,
            &layer.gate_up,
            &self.dispatch,
            &self.xs,
            &mut self.gate_up,
            experts,
        )?;
        self.mark(stream, "expert gate/up")?;
        model.kernels.situ_half(
            stream,
            &self.gate_up,
            &self.dispatch,
            &mut self.hidden_h,
            ff,
            f32::from_bits(kc.situ_beta_bits),
            f32::from_bits(kc.situ_linear_beta_bits),
        )?;
        Self::gemm_experts(
            &model,
            stream,
            &layer.down,
            &self.dispatch,
            &self.hidden_h,
            &mut self.y,
            experts,
        )?;
        self.mark(stream, "situ, expert down")?;
        model.kernels.moe_combine(
            stream,
            &self.y,
            &self.route_weights,
            &self.dispatch,
            &mut self.moe,
            n,
        )?;
        let mut row0 = 0;
        for seg in segments.iter_mut() {
            let r = seg.rows;
            let x = unsafe { rows_of(stream, &self.moe, row0, r, n) };
            let mut o = unsafe { rows_of(stream, &self.ffn_out, row0, r, n) };
            model.kernels.conv(
                stream,
                &x,
                &layer.ffn_conv,
                None,
                &mut seg.state.slots[slot].ffn_carry,
                &mut o,
                n,
                taps,
                r,
            )?;
            row0 += r;
        }
        model
            .ops
            .add(stream, &self.ffn_inp, &self.ffn_out, &mut self.h, plane)?;
        self.mark(stream, "combine, conv, residual")?;
        Ok(())
    }

    /// Run `tokens` (one id a row, segments in order) and leave the final
    /// normalized hidden states in [`Self::hidden`]. Every segment's
    /// position advances by its rows.
    ///
    /// `tap`, when given, receives each layer application's output by cache
    /// slot, read back from the device.
    pub fn run(
        &mut self,
        stream: &Arc<CudaStream>,
        segments: &mut [Segment<'_>],
        tokens: &[i32],
        tap: Option<&mut dyn FnMut(usize, Vec<f32>)>,
    ) -> Res {
        self.run_with_rows(stream, segments, tokens, &[], tap)
    }

    /// [`Self::run`], with some rows' inputs given rather than looked up:
    /// an image's rows from the vision tower stand where its placeholder
    /// tokens are. They enter as the model's input directly, past the token
    /// embedding's norm, as llama.cpp feeds an embedding batch
    /// (`src/models/inkling2.cpp`: `tok_norm` only `if (ubatch.token)`).
    pub fn run_with_rows(
        &mut self,
        stream: &Arc<CudaStream>,
        segments: &mut [Segment<'_>],
        tokens: &[i32],
        given: &[GivenRows<'_>],
        mut tap: Option<&mut dyn FnMut(usize, Vec<f32>)>,
    ) -> Res {
        let model = self.model.clone();
        let c = &model.config;
        let kc = &model.k;
        let n = c.hidden_size as usize;
        let rows = self.rows;
        let plane = rows * n;
        if tokens.len() != rows || segments.iter().map(|s| s.rows).sum::<usize>() != rows {
            return Err(bad(format!("the pass runs exactly {rows} rows")));
        }
        for seg in segments.iter_mut() {
            if seg.state.position + seg.rows > seg.state.capacity {
                return Err(bad("a sequence would outgrow its cache"));
            }
            if seg.rows > seg.state.max_pass_rows {
                return Err(bad(format!(
                    "a pass of {} rows is wider than the {} this sequence's ring caches were sized for",
                    seg.rows, seg.state.max_pass_rows
                )));
            }
            stream.memcpy_htod(&[seg.state.position as i32], &mut seg.state.d_position)?;
        }
        if tokens.iter().any(|t| *t < 0 || *t as u32 >= c.vocab_size) {
            return Err(bad("a token id is outside the vocabulary"));
        }
        if let Some(p) = self.profile.as_mut() {
            p.restart(stream)?;
        }
        stream.memcpy_htod(tokens, &mut self.d_tokens)?;
        model.kernels.embed(
            stream,
            &model.tok_embd,
            &self.d_tokens,
            &mut self.input,
            n,
            rows,
            model.embd_bf16,
        )?;
        let eps = f32::from_bits(kc.rms_eps_bits);
        model.ops.rms_norm(
            stream,
            &self.input,
            &model.tok_norm,
            &mut self.h,
            rows,
            n,
            eps,
        )?;
        for g in given {
            if g.at + g.rows > rows || (g.first + g.rows) * n > g.from.len() {
                return Err(bad("given rows fall outside the pass or their source"));
            }
            let src = g.from.slice(g.first * n..(g.first + g.rows) * n);
            let mut dst = self.h.slice_mut(g.at * n..(g.at + g.rows) * n);
            stream.memcpy_dtod(&src, &mut dst)?;
        }

        let layers = c.num_layers as usize;
        let (start, end) = (kc.loop_start as usize, kc.loop_end as usize);
        self.mark(stream, "embed")?;
        let mut live = 0usize;
        let mut emit = |pass: &Self, slot: usize| -> Res {
            if let Some(t) = tap.as_mut() {
                t(slot, stream.clone_dtoh(&pass.h)?);
            }
            Ok(())
        };
        for layer in 0..start {
            self.layer(stream, segments, layer, layer, &mut live, false)?;
            emit(self, layer)?;
        }
        // Every pass starts from the prelude output.
        stream.memcpy_dtod(&self.h, &mut self.prelude)?;
        let prelude_live = live;
        for pass in 0..kc.loop_count as usize {
            if pass > 0 {
                // The bank of the previous pass: its blocks after the first
                // core block, then its output.
                let banked = live - prelude_live - 1;
                {
                    let src = unsafe {
                        subslice(
                            stream,
                            &self.blocks,
                            (prelude_live + 1) * plane,
                            banked * plane,
                        )
                    };
                    let mut dst = unsafe { subslice(stream, &self.bank, 0, banked * plane) };
                    stream.memcpy_dtod(&*src, &mut *dst)?;
                    let mut out = unsafe { subslice(stream, &self.bank, banked * plane, plane) };
                    stream.memcpy_dtod(&self.h, &mut *out)?;
                }
                stream.memcpy_dtod(&self.prelude, &mut self.h)?;
                live = prelude_live;
                for layer in start..end {
                    let slot = layers + (pass - 1) * (end - start) + (layer - start);
                    let lp = model.layers[layer]
                        .looped
                        .as_ref()
                        .ok_or_else(|| bad("a looped layer has no controllers"))?;
                    model.kernels.mix(
                        stream,
                        &self.bank,
                        &lp.msg_query,
                        &mut self.messages,
                        n,
                        banked + 1,
                        4,
                        rows,
                    )?;
                    self.layer(stream, segments, slot, layer, &mut live, true)?;
                    emit(self, slot)?;
                }
            } else {
                for layer in start..end {
                    self.layer(stream, segments, layer, layer, &mut live, false)?;
                    emit(self, layer)?;
                }
            }
        }
        for layer in end..layers {
            self.layer(stream, segments, layer, layer, &mut live, false)?;
            emit(self, layer)?;
        }

        {
            let mut top = unsafe { subslice(stream, &self.blocks, live * plane, plane) };
            stream.memcpy_dtod(&self.h, &mut *top)?;
        }
        model.kernels.mix(
            stream,
            &self.blocks,
            &model.out_query,
            &mut self.x_mix,
            n,
            live + 1,
            1,
            rows,
        )?;
        model.ops.rms_norm(
            stream,
            &self.x_mix,
            &model.out_norm,
            &mut self.final_norm,
            rows,
            n,
            eps,
        )?;
        for seg in segments.iter_mut() {
            seg.state.position += seg.rows;
        }
        self.mark(stream, "loop messages, output mix and norm")?;
        Ok(())
    }

    /// Logits of rows `[row0, row0 + count)` of the last pass over the real
    /// vocabulary, soft-capped, into the head of [`Self::head_logits`].
    pub fn logits(&mut self, stream: &Arc<CudaStream>, row0: usize, count: usize) -> Res {
        let model = self.model.clone();
        let n = model.config.hidden_size as usize;
        let vocab = model.config.vocab_size as usize;
        if row0 + count > self.rows {
            return Err(bad("logit rows outside the pass"));
        }
        if count * vocab > self.head_logits.len() {
            return Err(bad("more logit rows than this pass has room for"));
        }
        if let Some(p) = self.profile.as_mut() {
            p.restart(stream)?;
        }
        let src = unsafe { rows_of(stream, &self.final_norm, row0, count, n) };
        model
            .hg
            .to_half(stream, &src, &mut self.final_h, count * n)?;
        Self::gemm(
            &model,
            stream,
            &model.head,
            &self.final_h,
            &mut self.head_logits,
            count,
        )?;
        let cap = f32::from_bits(model.k.softcap_bits);
        if cap != 0.0 {
            model
                .kernels
                .softcap(stream, &mut self.head_logits, count * vocab, cap)?;
        }
        self.mark(stream, "head")?;
        Ok(())
    }
}
