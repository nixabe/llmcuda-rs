//! Clef's joint schema head on the device.
//!
//! Reads the backbone's final RMS-normed hidden state at every prompt
//! position and returns one score per answer option. The scalar oracle is
//! [`llmcuda_kernels::decision::joint_schema_head`]; the kernels are
//! [`llmcuda_cuda::kernels::decision`], and the order they run in here is
//! llama.cpp's `llama_model_clef::graph::build_head` (`src/models/clef.cpp`).
//!
//! [`DecisionHead::load`] builds kernels, dequantizes the head's weights to
//! f32 on the host, uploads them once and preallocates a workspace sized by
//! [`DecisionLimits`]; [`DecisionHead::forward`] queues the head on a stream
//! and copies back one f32 per option, allocating nothing (AGENTS.md rule 6).
//!
//! # Shape of a pass
//!
//! Prompt-length work is the memory projection (`hidden -> width` for every
//! position) and, per head layer, one K/V projection of that memory: twelve
//! `width -> width` GEMMs over every position, against ~34 MFLOP per token in
//! total. Everything else is sized by the question and option counts. The
//! hidden-state LayerNorm is never materialized — its row statistics are
//! folded into the GEMM and span means that read it — so the largest
//! prompt-sized buffers are the memory (`[positions][width]`) and one layer's
//! K/V (`[positions][2 * width]`).

use std::cell::Cell;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr};
use llmcuda_cuda::kernels::decision::{
    Attention as AttnLaunch, DecisionKernelError, DecisionKernels, Gemm, RowFormat, RowNorm,
    Scratch,
};
use llmcuda_cuda::kernels::lm_head::HeadTensor;
use llmcuda_gguf::{GgmlType, GgufFile};
use llmcuda_kernels::decision::{
    Attention, HeadLayer, HeadWeights, LayerNorm, Linear, OptionSpan, QuestionSpan,
};
use llmcuda_kernels::quant;
use llmcuda_model::ModelConfig;
use tracing::debug;

/// Workspace bounds, fixed at load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecisionLimits {
    /// Longest prompt the head will read.
    pub max_positions: usize,
    /// Most questions in one request.
    pub max_questions: usize,
    /// Most options across all questions of one request.
    pub max_options: usize,
}

/// Loading or running the head failed.
#[derive(Debug)]
pub enum DecisionHeadError {
    /// A tensor is missing, misshapen or in an unsupported format.
    Weights(String),
    /// A request exceeds the workspace bounds or is malformed.
    Request(String),
    /// Kernel compilation failed.
    Compile(String),
    /// The CUDA driver reported an error.
    Driver(cudarc::driver::DriverError),
}

impl core::fmt::Display for DecisionHeadError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Weights(e) => write!(f, "decision head weights: {e}"),
            Self::Request(e) => write!(f, "decision request: {e}"),
            Self::Compile(e) => write!(f, "decision head kernels: {e}"),
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
        }
    }
}

impl core::error::Error for DecisionHeadError {}

impl From<cudarc::driver::DriverError> for DecisionHeadError {
    fn from(value: cudarc::driver::DriverError) -> Self {
        Self::Driver(value)
    }
}

impl From<DecisionKernelError> for DecisionHeadError {
    fn from(value: DecisionKernelError) -> Self {
        match value {
            DecisionKernelError::Compile(e) => Self::Compile(e),
            DecisionKernelError::Driver(e) => Self::Driver(e),
            // A geometry the launchers refuse is one `forward` failed to
            // reject first, so it is reported against the request.
            DecisionKernelError::Shape(e) => Self::Request(e),
        }
    }
}

fn weights_err(msg: impl Into<String>) -> DecisionHeadError {
    DecisionHeadError::Weights(msg.into())
}

fn request_err(msg: impl Into<String>) -> DecisionHeadError {
    DecisionHeadError::Request(msg.into())
}

/// The head's device allocations, counted as they are made.
struct Alloc<'a> {
    stream: &'a Arc<CudaStream>,
    bytes: Cell<usize>,
}

impl Alloc<'_> {
    fn add(&self, bytes: usize) {
        self.bytes.set(self.bytes.get() + bytes);
    }
}

/// One resident f32 buffer and its device address.
struct Buf {
    slice: CudaSlice<f32>,
    ptr: u64,
}

impl Buf {
    fn upload(al: &Alloc<'_>, data: &[f32]) -> Result<Self, DecisionHeadError> {
        let slice = al.stream.clone_htod(data)?;
        al.add(4 * slice.len());
        let ptr = slice.device_ptr(al.stream).0;
        Ok(Self { slice, ptr })
    }

    fn zeros(al: &Alloc<'_>, len: usize) -> Result<Self, DecisionHeadError> {
        let slice = al.stream.alloc_zeros::<f32>(len.max(1))?;
        al.add(4 * slice.len());
        let ptr = slice.device_ptr(al.stream).0;
        Ok(Self { slice, ptr })
    }
}

struct DevNorm {
    w: Buf,
    b: Buf,
}

impl DevNorm {
    fn upload(al: &Alloc<'_>, n: &LayerNorm) -> Result<Self, DecisionHeadError> {
        Ok(Self {
            w: Buf::upload(al, &n.weight)?,
            b: Buf::upload(al, &n.bias)?,
        })
    }

    fn row(&self, stats: u64) -> RowNorm {
        RowNorm {
            stats,
            weight: self.w.ptr,
            bias: self.b.ptr,
        }
    }
}

/// `[out][inp]` row-major, optional bias.
struct DevLinear {
    w: Buf,
    b: Option<Buf>,
    out: usize,
    inp: usize,
}

impl DevLinear {
    /// Several linears over the same input, stacked along the output.
    fn stacked(al: &Alloc<'_>, parts: &[&Linear]) -> Result<Self, DecisionHeadError> {
        let inp = parts[0].inp;
        let out: usize = parts.iter().map(|l| l.out).sum();
        let mut w = Vec::with_capacity(out * inp);
        let mut b = Vec::with_capacity(out);
        let biased = parts.iter().any(|l| l.bias.is_some());
        for l in parts {
            debug_assert_eq!(l.inp, inp);
            w.extend_from_slice(&l.weight);
            match &l.bias {
                Some(v) => b.extend_from_slice(v),
                None => b.resize(b.len() + l.out, 0.0),
            }
        }
        Ok(Self {
            w: Buf::upload(al, &w)?,
            b: if biased {
                Some(Buf::upload(al, &b)?)
            } else {
                None
            },
            out,
            inp,
        })
    }

    fn single(al: &Alloc<'_>, l: &Linear) -> Result<Self, DecisionHeadError> {
        Self::stacked(al, &[l])
    }

    /// Bias-free linears over concatenated inputs: row `n` is `[a[n] | b[n]]`.
    fn side_by_side(al: &Alloc<'_>, a: &Linear, b: &Linear) -> Result<Self, DecisionHeadError> {
        debug_assert_eq!(a.out, b.out);
        let inp = a.inp + b.inp;
        let mut w = Vec::with_capacity(a.out * inp);
        for n in 0..a.out {
            w.extend_from_slice(&a.weight[n * a.inp..(n + 1) * a.inp]);
            w.extend_from_slice(&b.weight[n * b.inp..(n + 1) * b.inp]);
        }
        Ok(Self {
            w: Buf::upload(al, &w)?,
            b: None,
            out: a.out,
            inp,
        })
    }
}

struct DevSelfAttn {
    norm: DevNorm,
    /// q, k, v stacked: `[3 * width][width]`.
    qkv: DevLinear,
    o: DevLinear,
}

struct DevLayer {
    self_attn: Option<DevSelfAttn>,
    cross_norm: DevNorm,
    cross_norm_kv: Option<DevNorm>,
    cross_q: DevLinear,
    /// k, v stacked: `[2 * width][width]`.
    cross_kv: DevLinear,
    cross_o: DevLinear,
    ffn_norm: DevNorm,
    ffn_up: DevLinear,
    ffn_down: DevLinear,
}

/// Every resident tensor of the head.
struct DevWeights {
    hidden_norm: DevNorm,
    proj_memory: DevLinear,
    /// `proj_question` then `proj_option_question`: `[2 * width][hidden]`.
    proj_questions: DevLinear,
    /// `[proj_option_context | proj_option_lexical]`: `[width][2 * hidden]`.
    proj_options: DevLinear,
    proj_global: DevLinear,
    type_embd: Buf,
    layers: Vec<DevLayer>,
    option_summary_norm: DevNorm,
    field_norm: DevNorm,
    option_norm: DevNorm,
    scorer: DevLinear,
    scorer_out: Buf,
    scorer_out_bias: f32,
    scales: [f32; 3],
}

/// Preallocated activations, sized by [`DecisionLimits`].
struct Workspace {
    /// i32 request metadata: question spans, option spans, the last-position
    /// span and the prompt's token ids (see the `META_*` offsets).
    meta: CudaSlice<i32>,
    meta_ptr: u64,
    /// `[positions]` (mean, rstd) of the hidden states.
    hstats: Buf,
    /// `[positions][width]`.
    memory: Buf,
    /// `[positions]` (mean, rstd) of the memory rows.
    mstats: Buf,
    /// `[positions][2 * width]`: one layer's keys then values.
    kv: Buf,
    /// `[questions][hidden]` span means of the normed hidden state.
    qmean: Buf,
    /// `[options][2 * hidden]`: context span mean, then lexical mean.
    optin: Buf,
    /// `[hidden]`: the normed last position.
    global: Buf,
    /// `[options][width]`: the routed option vectors.
    opts: Buf,
    /// `[max(options, questions)]` (mean, rstd).
    rstats: Buf,
    /// Attention queries (and the self-attention's keys and values).
    qbuf: Buf,
    /// `[max(options, questions)][width]` attention output.
    attn: Buf,
    /// Feed-forward hidden, later the scorer's features.
    up: Buf,
    /// `[questions][2 * width]`: `proj_question` then `proj_option_question`.
    qproj: Buf,
    /// `[width]`.
    gproj: Buf,
    /// `[questions][width]`.
    fields: Buf,
    /// `[questions][width]`, after `field_norm`.
    fieldn: Buf,
    /// `[options][width]`, after `option_norm`.
    optn: Buf,
    /// `[options][width]`, the scorer's GELU output.
    scorer_h: Buf,
    /// `[options]` summary softmax weights.
    weights: Buf,
    /// `[options][2]` (prior, cosine).
    aux: Buf,
    /// `[options]`.
    scores: CudaSlice<f32>,
    scores_ptr: u64,
    split: Buf,
    split_capacity: usize,
    attn_part: Buf,
    attn_capacity: usize,
}

/// Elements of split-K partials kept for small-M projections.
const SPLIT_SCRATCH: usize = 1 << 20;
/// Query-split rows of attention partials kept for long-key attention.
const ATTN_SPLIT_ROWS: usize = 4096;

/// The head, resident on one device.
pub struct DecisionHead {
    limits: DecisionLimits,
    kernels: DecisionKernels,
    hidden: usize,
    width: usize,
    ff: usize,
    heads: usize,
    routing_layers: usize,
    eps: f32,
    weights: DevWeights,
    ws: Workspace,
    /// Host staging for [`Workspace::meta`], `meta_len` long.
    meta_host: Vec<i32>,
    /// Host staging for the scores, `max_options` long.
    scores_host: Vec<f32>,
    /// Bytes of resident weights.
    weight_bytes: usize,
    /// Bytes of every device allocation, weights included.
    device_bytes: usize,
}

impl DecisionHead {
    /// Build kernels, upload weights from `file`, and preallocate the
    /// workspace.
    pub fn load(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        file: &GgufFile,
        config: &ModelConfig,
        limits: DecisionLimits,
    ) -> Result<Self, DecisionHeadError> {
        let host = Self::host_weights(file, config)?;
        Self::from_weights(ctx, stream, &host, limits)
    }

    /// Build the head from weights already in the scalar reference's layout.
    /// [`Self::load`] is this after [`Self::host_weights`].
    pub fn from_weights(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        w: &HeadWeights,
        limits: DecisionLimits,
    ) -> Result<Self, DecisionHeadError> {
        check_geometry(w)?;
        if limits.max_positions == 0 || limits.max_questions == 0 || limits.max_options == 0 {
            return Err(request_err(format!("degenerate limits {limits:?}")));
        }
        if i32::try_from(limits.max_positions).is_err() {
            return Err(request_err("max_positions exceeds i32".to_string()));
        }
        let kernels = DecisionKernels::new(ctx)?;
        let (hidden, width) = (w.hidden, w.width);
        let ff = w.layers.first().map_or(width, |l| l.ffn_up.out);
        let head_dim = width / w.heads;
        let al = Alloc {
            stream,
            bytes: Cell::new(0),
        };

        let mut layers = Vec::with_capacity(w.layers.len());
        for l in &w.layers {
            let self_attn = match &l.self_attn {
                Some((norm, a)) => Some(DevSelfAttn {
                    norm: DevNorm::upload(&al, norm)?,
                    qkv: DevLinear::stacked(&al, &[&a.q, &a.k, &a.v])?,
                    o: DevLinear::single(&al, &a.o)?,
                }),
                None => None,
            };
            layers.push(DevLayer {
                self_attn,
                cross_norm: DevNorm::upload(&al, &l.cross_norm)?,
                cross_norm_kv: l
                    .cross_norm_kv
                    .as_ref()
                    .map(|n| DevNorm::upload(&al, n))
                    .transpose()?,
                cross_q: DevLinear::single(&al, &l.cross_attn.q)?,
                cross_kv: DevLinear::stacked(&al, &[&l.cross_attn.k, &l.cross_attn.v])?,
                cross_o: DevLinear::single(&al, &l.cross_attn.o)?,
                ffn_norm: DevNorm::upload(&al, &l.ffn_norm)?,
                ffn_up: DevLinear::single(&al, &l.ffn_up)?,
                ffn_down: DevLinear::single(&al, &l.ffn_down)?,
            });
        }
        let weights = DevWeights {
            hidden_norm: DevNorm::upload(&al, &w.hidden_norm)?,
            proj_memory: DevLinear::single(&al, &w.proj_memory)?,
            proj_questions: DevLinear::stacked(&al, &[&w.proj_question, &w.proj_option_question])?,
            proj_options: DevLinear::side_by_side(
                &al,
                &w.proj_option_context,
                &w.proj_option_lexical,
            )?,
            proj_global: DevLinear::single(&al, &w.proj_global)?,
            type_embd: Buf::upload(&al, &w.type_embd)?,
            layers,
            option_summary_norm: DevNorm::upload(&al, &w.option_summary_norm)?,
            field_norm: DevNorm::upload(&al, &w.field_norm)?,
            option_norm: DevNorm::upload(&al, &w.option_norm)?,
            scorer: DevLinear::single(&al, &w.scorer)?,
            scorer_out: Buf::upload(&al, &w.scorer_out.weight)?,
            scorer_out_bias: w.scorer_out.bias.as_ref().map_or(0.0, |b| b[0]),
            scales: w.scales,
        };
        let weight_bytes = al.bytes.get();

        let DecisionLimits {
            max_positions: p,
            max_questions: nq,
            max_options: no,
        } = limits;
        let rows = nq.max(no);
        let meta_len = meta_len(limits);
        let meta = stream.alloc_zeros::<i32>(meta_len)?;
        al.add(4 * meta_len);
        let meta_ptr = meta.device_ptr(stream).0;
        let scores = stream.alloc_zeros::<f32>(no)?;
        al.add(4 * no);
        let scores_ptr = scores.device_ptr(stream).0;
        let attn_capacity =
            DecisionKernels::attention_scratch(ATTN_SPLIT_ROWS, w.heads, head_dim, 1);
        let ws = Workspace {
            meta,
            meta_ptr,
            hstats: Buf::zeros(&al, 2 * p)?,
            memory: Buf::zeros(&al, p * width)?,
            mstats: Buf::zeros(&al, 2 * p)?,
            kv: Buf::zeros(&al, p * 2 * width)?,
            qmean: Buf::zeros(&al, nq * hidden)?,
            optin: Buf::zeros(&al, no * 2 * hidden)?,
            global: Buf::zeros(&al, hidden)?,
            opts: Buf::zeros(&al, no * width)?,
            rstats: Buf::zeros(&al, 2 * rows)?,
            qbuf: Buf::zeros(&al, (no * width).max(nq * 3 * width))?,
            attn: Buf::zeros(&al, rows * width)?,
            up: Buf::zeros(&al, rows * ff.max(4 * width))?,
            qproj: Buf::zeros(&al, nq * 2 * width)?,
            gproj: Buf::zeros(&al, width)?,
            fields: Buf::zeros(&al, nq * width)?,
            fieldn: Buf::zeros(&al, nq * width)?,
            optn: Buf::zeros(&al, no * width)?,
            scorer_h: Buf::zeros(&al, no * width)?,
            weights: Buf::zeros(&al, no)?,
            aux: Buf::zeros(&al, 2 * no)?,
            scores,
            scores_ptr,
            split: Buf::zeros(&al, SPLIT_SCRATCH)?,
            split_capacity: SPLIT_SCRATCH,
            attn_part: Buf::zeros(&al, attn_capacity)?,
            attn_capacity,
        };
        let device_bytes = al.bytes.get();
        let mib = |b: usize| b as f64 / (1 << 20) as f64;
        debug!(
            max_positions = p,
            max_questions = nq,
            max_options = no,
            weights_mib = mib(weight_bytes),
            workspace_mib = mib(device_bytes - weight_bytes),
            "decision head resident"
        );
        Ok(Self {
            limits,
            kernels,
            hidden,
            width,
            ff,
            heads: w.heads,
            routing_layers: w.routing_layers,
            eps: w.eps,
            weights,
            ws,
            meta_host: vec![0; meta_len],
            scores_host: vec![0.0; no],
            weight_bytes,
            device_bytes,
        })
    }

    /// The head's weights dequantized to f32 on the host, in the layout the
    /// scalar reference takes.
    pub fn host_weights(
        file: &GgufFile,
        config: &ModelConfig,
    ) -> Result<HeadWeights, DecisionHeadError> {
        let decision = config
            .decision
            .ok_or_else(|| weights_err("the model declares no decision head"))?;
        let hidden = config.hidden_size as usize;
        let heads = decision.heads as usize;
        let routing = decision.routing_layers as usize;
        let joint = decision.joint_layers as usize;
        let eps = decision.layer_norm_eps();
        let width = file
            .tensor("decision.proj_memory.weight")
            .and_then(|t| t.dims.get(1).copied())
            .ok_or_else(|| weights_err("missing tensor decision.proj_memory.weight"))?
            as usize;
        let ff = if routing + joint > 0 {
            file.tensor("dec.blk.0.ffn_up.weight")
                .and_then(|t| t.dims.get(1).copied())
                .ok_or_else(|| weights_err("missing tensor dec.blk.0.ffn_up.weight"))?
                as usize
        } else {
            width
        };

        let norm = |name: &str, n: usize| -> Result<LayerNorm, DecisionHeadError> {
            Ok(LayerNorm {
                weight: tensor_f32(file, &format!("{name}.weight"), &[n])?,
                bias: tensor_f32(file, &format!("{name}.bias"), &[n])?,
            })
        };
        let linear = |name: &str, inp: usize, out: usize, bias: bool| {
            let weight = tensor_f32(file, &format!("{name}.weight"), &[inp, out])?;
            let bias = if bias {
                Some(tensor_f32(file, &format!("{name}.bias"), &[out])?)
            } else {
                None
            };
            Ok::<_, DecisionHeadError>(Linear {
                weight,
                bias,
                out,
                inp,
            })
        };
        let attention = |prefix: &str| -> Result<Attention, DecisionHeadError> {
            Ok(Attention {
                q: linear(&format!("{prefix}_q"), width, width, true)?,
                k: linear(&format!("{prefix}_k"), width, width, true)?,
                v: linear(&format!("{prefix}_v"), width, width, true)?,
                o: linear(&format!("{prefix}_o"), width, width, true)?,
            })
        };

        let mut layers = Vec::with_capacity(routing + joint);
        for il in 0..routing + joint {
            let p = format!("dec.blk.{il}");
            let is_routing = il < routing;
            layers.push(HeadLayer {
                self_attn: if is_routing {
                    None
                } else {
                    Some((
                        norm(&format!("{p}.attn_norm"), width)?,
                        attention(&format!("{p}.attn"))?,
                    ))
                },
                cross_norm: norm(&format!("{p}.cross_attn_norm"), width)?,
                cross_norm_kv: if is_routing {
                    Some(norm(&format!("{p}.cross_attn_norm_kv"), width)?)
                } else {
                    None
                },
                cross_attn: attention(&format!("{p}.cross_attn"))?,
                ffn_norm: norm(&format!("{p}.ffn_norm"), width)?,
                ffn_up: linear(&format!("{p}.ffn_up"), width, ff, true)?,
                ffn_down: linear(&format!("{p}.ffn_down"), ff, width, true)?,
            });
        }

        let scales = tensor_f32(file, "decision.scales", &[3])?;
        let w = HeadWeights {
            hidden,
            width,
            heads,
            eps,
            hidden_norm: norm("decision.hidden_norm", hidden)?,
            proj_memory: linear("decision.proj_memory", hidden, width, false)?,
            proj_question: linear("decision.proj_question", hidden, width, false)?,
            proj_option_question: linear("decision.proj_option_question", hidden, width, false)?,
            proj_global: linear("decision.proj_global", hidden, width, false)?,
            proj_option_context: linear("decision.proj_option_context", hidden, width, false)?,
            proj_option_lexical: linear("decision.proj_option_lexical", hidden, width, false)?,
            type_embd: tensor_f32(file, "token_types.weight", &[width, 3])?,
            layers,
            routing_layers: routing,
            option_summary_norm: norm("decision.option_summary_norm", width)?,
            field_norm: norm("decision.field_norm", width)?,
            option_norm: norm("decision.option_norm", width)?,
            scorer: linear("decision.scorer", 4 * width, width, true)?,
            scorer_out: linear("decision.scorer_out", width, 1, true)?,
            scales: [scales[0], scales[1], scales[2]],
        };
        check_geometry(&w)?;
        Ok(w)
    }

    /// The bounds this head was built for.
    pub fn limits(&self) -> DecisionLimits {
        self.limits
    }

    /// Device bytes this head holds: `(weights, weights + workspace)`. Fixed
    /// at load; the f32 weights do not depend on the limits.
    pub fn device_bytes(&self) -> (usize, usize) {
        (self.weight_bytes, self.device_bytes)
    }

    /// Score every option. `hidden` is `[positions][hidden]` f32; `tokens`
    /// are the prompt's ids (the lexical vectors gather their output rows
    /// from `lm_head`). `scores` is cleared and filled with one value per
    /// option, in `options` order.
    ///
    /// Queues the whole head on `stream` and synchronizes it before
    /// returning; allocates nothing on the device or, given a `scores` with
    /// capacity for every option, on the host.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        stream: &Arc<CudaStream>,
        hidden: &CudaSlice<f32>,
        positions: usize,
        tokens: &[i32],
        questions: &[QuestionSpan],
        options: &[OptionSpan],
        lm_head: HeadTensor<'_>,
        scores: &mut Vec<f32>,
    ) -> Result<(), DecisionHeadError> {
        scores.clear();
        let fmt = self.validate(hidden, positions, tokens, questions, options, &lm_head)?;
        self.stage(positions, tokens, questions, options);
        let used = meta_tokens_at(self.limits) + positions;
        {
            let mut dst = self.ws.meta.slice_mut(0..used);
            stream.memcpy_htod(&self.meta_host[..used], &mut dst)?;
        }
        // The guards record their read events when dropped, after the head
        // is queued, so a later writer on another stream orders after it.
        let (hidden_ptr, hidden_read) = hidden.device_ptr(stream);
        let (table_ptr, table_read) = lm_head.bytes.device_ptr(stream);
        self.run(
            stream,
            hidden_ptr,
            table_ptr,
            fmt,
            positions,
            questions.len(),
            options.len(),
        )?;
        drop((hidden_read, table_read));
        let no = options.len();
        stream.memcpy_dtoh(&self.ws.scores.slice(0..no), &mut self.scores_host[..no])?;
        stream.synchronize()?;
        scores.extend_from_slice(&self.scores_host[..no]);
        Ok(())
    }

    /// Copy the memory projection (`[positions][width]`) of the last
    /// [`Self::forward`] back to the host. Diagnostic: allocates, and is not
    /// for the serving path.
    pub fn read_memory(
        &self,
        stream: &Arc<CudaStream>,
        positions: usize,
    ) -> Result<Vec<f32>, DecisionHeadError> {
        if positions > self.limits.max_positions {
            return Err(request_err(format!(
                "{positions} positions, limit {}",
                self.limits.max_positions
            )));
        }
        let n = positions * self.width;
        let mut out = vec![0.0; n];
        let view = self.ws.memory.slice.slice(0..n);
        stream.memcpy_dtoh(&view, &mut out)?;
        stream.synchronize()?;
        Ok(out)
    }

    fn validate(
        &self,
        hidden: &CudaSlice<f32>,
        positions: usize,
        tokens: &[i32],
        questions: &[QuestionSpan],
        options: &[OptionSpan],
        lm_head: &HeadTensor<'_>,
    ) -> Result<RowFormat, DecisionHeadError> {
        let l = self.limits;
        if positions == 0 || positions != tokens.len() || positions > l.max_positions {
            return Err(request_err(format!(
                "{positions} positions for {} tokens, limit {}",
                tokens.len(),
                l.max_positions
            )));
        }
        if hidden.len() < positions * self.hidden {
            return Err(request_err(format!(
                "hidden state holds {} floats, {positions} positions need {}",
                hidden.len(),
                positions * self.hidden
            )));
        }
        if questions.is_empty() || questions.len() > l.max_questions {
            return Err(request_err(format!(
                "{} questions, limit 1..={}",
                questions.len(),
                l.max_questions
            )));
        }
        if options.is_empty() || options.len() > l.max_options {
            return Err(request_err(format!(
                "{} options, limit 1..={}",
                options.len(),
                l.max_options
            )));
        }
        let fmt = RowFormat::from_head(lm_head.format);
        let row_bytes = fmt.row_bytes(self.hidden).ok_or_else(|| {
            request_err(format!(
                "a {}-wide output row is not whole {fmt:?} blocks",
                self.hidden
            ))
        })?;
        let vocab = lm_head.bytes.len() / row_bytes;
        let in_range = |start: usize, end: usize| start < end && end <= positions;
        for (i, q) in questions.iter().enumerate() {
            if !in_range(q.start, q.end) {
                return Err(request_err(format!(
                    "question {i} spans [{}, {}) of a {positions}-token prompt",
                    q.start, q.end
                )));
            }
        }
        for (i, o) in options.iter().enumerate() {
            if o.question >= questions.len() || !in_range(o.start, o.end) {
                return Err(request_err(format!(
                    "option {i} (question {}) spans [{}, {}) of a {positions}-token prompt",
                    o.question, o.start, o.end
                )));
            }
            if let Some(&t) = tokens[o.start..o.end]
                .iter()
                .find(|&&t| t < 0 || t as usize >= vocab)
            {
                return Err(request_err(format!(
                    "option {i} holds token {t}, outside the {vocab}-row output embedding"
                )));
            }
        }
        // Every question needs an option: the reference has no defined
        // output for an empty softmax. Checked without a set: questions are
        // few and this is O(questions * options) integer compares at most.
        for qi in 0..questions.len() {
            if !options.iter().any(|o| o.question == qi) {
                return Err(request_err(format!("question {qi} has no options")));
            }
        }
        Ok(fmt)
    }

    /// Pack spans and tokens into the host staging buffer.
    fn stage(
        &mut self,
        positions: usize,
        tokens: &[i32],
        questions: &[QuestionSpan],
        options: &[OptionSpan],
    ) {
        let l = self.limits;
        let m = &mut self.meta_host;
        for (i, q) in questions.iter().enumerate() {
            let at = META_QUESTIONS + 3 * i;
            m[at] = q.start as i32;
            m[at + 1] = q.end as i32;
            m[at + 2] = q.kind as i32;
        }
        let o_at = meta_options_at(l);
        for (i, o) in options.iter().enumerate() {
            let at = o_at + 3 * i;
            m[at] = o.start as i32;
            m[at + 1] = o.end as i32;
            m[at + 2] = o.question as i32;
        }
        let g_at = meta_global_at(l);
        m[g_at] = positions as i32 - 1;
        m[g_at + 1] = positions as i32;
        let t_at = meta_tokens_at(l);
        m[t_at..t_at + positions].copy_from_slice(tokens);
    }

    fn split(&self) -> Scratch {
        Scratch {
            ptr: self.ws.split.ptr,
            capacity: self.ws.split_capacity,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        stream: &Arc<CudaStream>,
        x: u64,
        ldx: usize,
        w: &DevLinear,
        y: u64,
        ldy: usize,
        m: usize,
        norm: Option<RowNorm>,
        gelu: bool,
        accumulate: bool,
    ) -> Result<(), DecisionHeadError> {
        let g = Gemm {
            x,
            ldx,
            w: w.w.ptr,
            ldw: w.inp,
            bias: w.b.as_ref().map_or(0, |b| b.ptr),
            y,
            ldy,
            m,
            n: w.out,
            k: w.inp,
            norm,
            gelu,
            accumulate,
        };
        // SAFETY: every pointer is a workspace or weight buffer of this head
        // (or the caller's hidden state, validated against `positions`), each
        // sized at load for the limits `forward` validated against.
        unsafe { self.kernels.gemm(stream, &g, self.split()) }?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn attend(
        &self,
        stream: &Arc<CudaStream>,
        q: u64,
        ldq: usize,
        k: u64,
        v: u64,
        ldkv: usize,
        nq: usize,
        nk: usize,
    ) -> Result<(), DecisionHeadError> {
        let a = AttnLaunch {
            q,
            ldq,
            k,
            ldk: ldkv,
            v,
            ldv: ldkv,
            o: self.ws.attn.ptr,
            ldo: self.width,
            nq,
            nk,
            heads: self.heads,
            head_dim: self.width / self.heads,
        };
        let scratch = Scratch {
            ptr: self.ws.attn_part.ptr,
            capacity: self.ws.attn_capacity,
        };
        // SAFETY: as in `gemm`.
        unsafe { self.kernels.attention(stream, &a, scratch) }?;
        Ok(())
    }

    /// LayerNorm statistics of `rows` rows of `x` into `stats`.
    fn stats(
        &self,
        stream: &Arc<CudaStream>,
        x: u64,
        rows: usize,
        cols: usize,
        stats: u64,
    ) -> Result<(), DecisionHeadError> {
        // SAFETY: as in `gemm`; no normed output is written.
        unsafe {
            self.kernels
                .layer_norm(stream, x, cols, rows, cols, self.eps, 0, 0, stats, 0, 0)
        }?;
        Ok(())
    }

    /// The feed-forward residual `x += down(gelu(up(LN(x))))` over `rows`.
    fn ffn(
        &self,
        stream: &Arc<CudaStream>,
        layer: &DevLayer,
        x: u64,
        rows: usize,
    ) -> Result<(), DecisionHeadError> {
        let (ws, width) = (&self.ws, self.width);
        self.stats(stream, x, rows, width, ws.rstats.ptr)?;
        let norm = Some(layer.ffn_norm.row(ws.rstats.ptr));
        self.gemm(
            stream,
            x,
            width,
            &layer.ffn_up,
            ws.up.ptr,
            self.ff,
            rows,
            norm,
            true,
            false,
        )?;
        self.gemm(
            stream,
            ws.up.ptr,
            self.ff,
            &layer.ffn_down,
            x,
            width,
            rows,
            None,
            false,
            true,
        )
    }

    /// Queue the whole head. Metadata is already on the device.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        stream: &Arc<CudaStream>,
        hidden_ptr: u64,
        table_ptr: u64,
        fmt: RowFormat,
        p: usize,
        nq: usize,
        no: usize,
    ) -> Result<(), DecisionHeadError> {
        let (ws, w) = (&self.ws, &self.weights);
        let (h, width, eps) = (self.hidden, self.width, self.eps);
        let l = self.limits;
        let qspans = ws.meta_ptr;
        let ospans = ws.meta_ptr + 4 * meta_options_at(l) as u64;
        let gspan = ws.meta_ptr + 4 * meta_global_at(l) as u64;
        let tokens = ws.meta_ptr + 4 * meta_tokens_at(l) as u64;
        let k = &self.kernels;

        // The hidden-state LayerNorm, as row statistics only.
        self.stats(stream, hidden_ptr, p, h, ws.hstats.ptr)?;
        let hn = w.hidden_norm.row(ws.hstats.ptr);
        // memory = proj_memory(LN(hidden)), every position.
        self.gemm(
            stream,
            hidden_ptr,
            h,
            &w.proj_memory,
            ws.memory.ptr,
            width,
            p,
            Some(hn),
            false,
            false,
        )?;
        // Span means of LN(hidden), the normed last position, and the
        // options' output-embedding means.
        // SAFETY: spans were validated against `p`, token ids against the
        // table's row count, and every output buffer is sized by the limits.
        unsafe {
            k.span_mean(stream, hidden_ptr, h, h, hn, qspans, 3, nq, ws.qmean.ptr, h)?;
            k.span_mean(
                stream,
                hidden_ptr,
                h,
                h,
                hn,
                ospans,
                3,
                no,
                ws.optin.ptr,
                2 * h,
            )?;
            k.span_mean(stream, hidden_ptr, h, h, hn, gspan, 2, 1, ws.global.ptr, h)?;
            k.lexical(
                stream,
                table_ptr,
                fmt,
                h,
                tokens,
                ospans,
                3,
                no,
                ws.optin.ptr + 4 * h as u64,
                2 * h,
            )?;
        }
        // One query per option: context and lexical projections in one GEMM,
        // plus its question's projection.
        self.gemm(
            stream,
            ws.qmean.ptr,
            h,
            &w.proj_questions,
            ws.qproj.ptr,
            2 * width,
            nq,
            None,
            false,
            false,
        )?;
        self.gemm(
            stream,
            ws.optin.ptr,
            2 * h,
            &w.proj_options,
            ws.opts.ptr,
            width,
            no,
            None,
            false,
            false,
        )?;
        // SAFETY: owners were validated `< nq`.
        unsafe {
            k.add_owner_rows(
                stream,
                ws.opts.ptr,
                width,
                ws.qproj.ptr + 4 * width as u64,
                2 * width,
                ospans,
                3,
                no,
                width,
            )?;
        }
        self.gemm(
            stream,
            ws.global.ptr,
            h,
            &w.proj_global,
            ws.gproj.ptr,
            width,
            1,
            None,
            false,
            false,
        )?;

        let kv_v = ws.kv.ptr + 4 * width as u64;
        // The options read the prompt.
        if self.routing_layers > 0 {
            self.stats(stream, ws.memory.ptr, p, width, ws.mstats.ptr)?;
        }
        for layer in &w.layers[..self.routing_layers] {
            let kv_norm = layer
                .cross_norm_kv
                .as_ref()
                .ok_or_else(|| weights_err("routing layer without a memory norm"))?;
            self.gemm(
                stream,
                ws.memory.ptr,
                width,
                &layer.cross_kv,
                ws.kv.ptr,
                2 * width,
                p,
                Some(kv_norm.row(ws.mstats.ptr)),
                false,
                false,
            )?;
            self.stats(stream, ws.opts.ptr, no, width, ws.rstats.ptr)?;
            self.gemm(
                stream,
                ws.opts.ptr,
                width,
                &layer.cross_q,
                ws.qbuf.ptr,
                width,
                no,
                Some(layer.cross_norm.row(ws.rstats.ptr)),
                false,
                false,
            )?;
            self.attend(
                stream,
                ws.qbuf.ptr,
                width,
                ws.kv.ptr,
                kv_v,
                2 * width,
                no,
                p,
            )?;
            self.gemm(
                stream,
                ws.attn.ptr,
                width,
                &layer.cross_o,
                ws.opts.ptr,
                width,
                no,
                None,
                false,
                true,
            )?;
            self.ffn(stream, layer, ws.opts.ptr, no)?;
        }

        // One vector per question: its text, a summary of its options, the
        // end of the prompt and its type.
        // SAFETY: owners `< nq`, types `< 3`, buffers sized by the limits.
        unsafe {
            k.question_fields(
                stream,
                ws.opts.ptr,
                width,
                no,
                ospans,
                3,
                ws.qproj.ptr,
                2 * width,
                ws.gproj.ptr,
                w.type_embd.ptr,
                qspans,
                3,
                nq,
                (w.option_summary_norm.w.ptr, w.option_summary_norm.b.ptr),
                eps,
                ws.weights.ptr,
                ws.fields.ptr,
                width,
                width,
            )?;
        }

        // The questions read each other and the prompt.
        for layer in &w.layers[self.routing_layers..] {
            let sa = layer
                .self_attn
                .as_ref()
                .ok_or_else(|| weights_err("joint layer without self-attention"))?;
            self.stats(stream, ws.fields.ptr, nq, width, ws.rstats.ptr)?;
            self.gemm(
                stream,
                ws.fields.ptr,
                width,
                &sa.qkv,
                ws.qbuf.ptr,
                3 * width,
                nq,
                Some(sa.norm.row(ws.rstats.ptr)),
                false,
                false,
            )?;
            self.attend(
                stream,
                ws.qbuf.ptr,
                3 * width,
                ws.qbuf.ptr + 4 * width as u64,
                ws.qbuf.ptr + 8 * width as u64,
                3 * width,
                nq,
                nq,
            )?;
            self.gemm(
                stream,
                ws.attn.ptr,
                width,
                &sa.o,
                ws.fields.ptr,
                width,
                nq,
                None,
                false,
                true,
            )?;
            self.stats(stream, ws.fields.ptr, nq, width, ws.rstats.ptr)?;
            self.gemm(
                stream,
                ws.fields.ptr,
                width,
                &layer.cross_q,
                ws.qbuf.ptr,
                width,
                nq,
                Some(layer.cross_norm.row(ws.rstats.ptr)),
                false,
                false,
            )?;
            self.gemm(
                stream,
                ws.memory.ptr,
                width,
                &layer.cross_kv,
                ws.kv.ptr,
                2 * width,
                p,
                None,
                false,
                false,
            )?;
            self.attend(
                stream,
                ws.qbuf.ptr,
                width,
                ws.kv.ptr,
                kv_v,
                2 * width,
                nq,
                p,
            )?;
            self.gemm(
                stream,
                ws.attn.ptr,
                width,
                &layer.cross_o,
                ws.fields.ptr,
                width,
                nq,
                None,
                false,
                true,
            )?;
            self.ffn(stream, layer, ws.fields.ptr, nq)?;
        }

        // Final norms, the scorer's features, prior and cosine, the scorer.
        // SAFETY: as above.
        unsafe {
            k.layer_norm(
                stream,
                ws.fields.ptr,
                width,
                nq,
                width,
                eps,
                w.field_norm.w.ptr,
                w.field_norm.b.ptr,
                0,
                ws.fieldn.ptr,
                width,
            )?;
            k.layer_norm(
                stream,
                ws.opts.ptr,
                width,
                no,
                width,
                eps,
                w.option_norm.w.ptr,
                w.option_norm.b.ptr,
                0,
                ws.optn.ptr,
                width,
            )?;
            k.features(
                stream,
                ws.fieldn.ptr,
                width,
                ws.optn.ptr,
                width,
                ospans,
                3,
                no,
                ws.qmean.ptr,
                h,
                ws.global.ptr,
                ws.optin.ptr + 4 * h as u64,
                2 * h,
                h,
                width,
                ws.up.ptr,
                4 * width,
                ws.aux.ptr,
            )?;
        }
        self.gemm(
            stream,
            ws.up.ptr,
            4 * width,
            &w.scorer,
            ws.scorer_h.ptr,
            width,
            no,
            None,
            true,
            false,
        )?;
        // SAFETY: as above.
        unsafe {
            k.final_scores(
                stream,
                ws.scorer_h.ptr,
                width,
                no,
                width,
                w.scorer_out.ptr,
                w.scorer_out_bias,
                ws.aux.ptr,
                w.scales,
                ws.scores_ptr,
            )?;
        }
        Ok(())
    }
}

/// i32 offset of the question spans (`[max_questions][start, end, type]`).
const META_QUESTIONS: usize = 0;

/// i32 offset of the option spans (`[max_options][start, end, question]`).
const fn meta_options_at(l: DecisionLimits) -> usize {
    META_QUESTIONS + 3 * l.max_questions
}

/// i32 offset of the last-position span (`[start, end]`).
const fn meta_global_at(l: DecisionLimits) -> usize {
    meta_options_at(l) + 3 * l.max_options
}

/// i32 offset of the token ids (`[max_positions]`).
const fn meta_tokens_at(l: DecisionLimits) -> usize {
    meta_global_at(l) + 2
}

const fn meta_len(l: DecisionLimits) -> usize {
    meta_tokens_at(l) + l.max_positions
}

/// Geometry the kernels support.
fn check_geometry(w: &HeadWeights) -> Result<(), DecisionHeadError> {
    let (h, width) = (w.hidden, w.width);
    if w.heads == 0 || !width.is_multiple_of(w.heads) {
        return Err(weights_err(format!(
            "width {width} is not divisible into {} heads",
            w.heads
        )));
    }
    let dh = width / w.heads;
    if !matches!(dh, 32 | 64 | 128) {
        return Err(weights_err(format!(
            "head dim {dh} (the attention kernel serves 32, 64 and 128)"
        )));
    }
    if !h.is_multiple_of(4) || !width.is_multiple_of(4) {
        return Err(weights_err(format!(
            "hidden {h} and width {width} must be multiples of 4"
        )));
    }
    if w.routing_layers > w.layers.len() {
        return Err(weights_err("more routing layers than layers"));
    }
    let ff = w.layers.first().map_or(width, |l| l.ffn_up.out);
    if !ff.is_multiple_of(4) {
        return Err(weights_err(format!(
            "feed-forward width {ff} must be a multiple of 4"
        )));
    }
    for (i, l) in w.layers.iter().enumerate() {
        let routing = i < w.routing_layers;
        if routing != l.cross_norm_kv.is_some() || routing == l.self_attn.is_some() {
            return Err(weights_err(format!(
                "layer {i} has the wrong shape for a {} layer",
                if routing { "routing" } else { "joint" }
            )));
        }
        if l.ffn_up.out != ff || l.ffn_down.inp != ff {
            return Err(weights_err(format!(
                "layer {i} feed-forward width differs from layer 0's {ff}"
            )));
        }
    }
    if w.scorer.inp != 4 * width || w.scorer.out != width || w.scorer_out.inp != width {
        return Err(weights_err("scorer shapes do not match the width"));
    }
    if w.type_embd.len() != 3 * width {
        return Err(weights_err("type embedding is not [3][width]"));
    }
    Ok(())
}

/// A tensor dequantized to f32, its GGUF dims checked against `dims`
/// (innermost first; trailing unit dims ignored).
fn tensor_f32(file: &GgufFile, name: &str, dims: &[usize]) -> Result<Vec<f32>, DecisionHeadError> {
    let info = file
        .tensor(name)
        .ok_or_else(|| weights_err(format!("missing tensor {name}")))?;
    let trim = |d: &[usize]| {
        let mut v = d.to_vec();
        while v.len() > 1 && v.last() == Some(&1) {
            v.pop();
        }
        v
    };
    let got: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
    if trim(&got) != trim(dims) {
        return Err(weights_err(format!(
            "{name} has dims {got:?}, expected {dims:?}"
        )));
    }
    let bytes = file
        .tensor_bytes(name)
        .ok_or_else(|| weights_err(format!("missing tensor {name}")))?;
    let block = |r: Result<Vec<f32>, quant::BlockSizeError>| {
        r.map_err(|e| weights_err(format!("{name}: {e}")))
    };
    let values = match info.ggml_type {
        GgmlType::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        GgmlType::F16 => quant::dequantize_row_f16(bytes),
        GgmlType::Bf16 => quant::dequantize_row_bf16(bytes),
        GgmlType::Q8_0 => block(quant::dequantize_row_q8_0(bytes))?,
        GgmlType::Q4_0 => block(quant::dequantize_row_q4_0(bytes))?,
        GgmlType::Q4K => block(quant::dequantize_row_q4_k(bytes))?,
        GgmlType::Q5K => block(quant::dequantize_row_q5_k(bytes))?,
        GgmlType::Q6K => block(quant::dequantize_row_q6_k(bytes))?,
    };
    let want: usize = dims.iter().product();
    if values.len() != want {
        return Err(weights_err(format!(
            "{name} dequantized to {} values, expected {want}",
            values.len()
        )));
    }
    Ok(values)
}
