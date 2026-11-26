//! The decode shape of the `inkling2_loop` forward: a handful of rows, each
//! the next position of its own sequence.
//!
//! [`super::InklingPass`] runs this shape correctly and slowly — its
//! launches are sized for hundreds of rows, and at one row a layer is some
//! sixty of them. Here a layer is ten: the attention inputs, one or two
//! projection packs, keys and values into the cache, attention, the output
//! projection, everything up to the routing, the two expert projections,
//! and the tail. The contractions are the decode GEMM
//! ([`llmcuda_cuda::kernels::inkling::DecodeGemm`]): the arithmetic class of
//! the prefill path — binary16 activations, the binary16 of each stored
//! weight, fp32 accumulation — in a different summation order, so the two
//! paths agree to rounding and not to the bit.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use llmcuda_cuda::kernels::hgemm::{HalfWeight, bf16_half_exponent};
use llmcuda_cuda::kernels::inkling::{
    ControlWeights, DEC_ROW_POINTERS, DG_BLOCK_ROWS, DG_MAX_ROWS, ProjOffsets,
};
use llmcuda_gguf::GgmlType;
use llmcuda_model::ModelConfig;

use super::{Dense, ExpertStack, InklingModel, InklingState, Loader, Res, bad};
use crate::viewslice::subslice;

/// The split-Q8_0 projections of one layer, stacked: `[rows][n]`.
struct PackQ8 {
    q: CudaSlice<i8>,
    scales: CudaSlice<u16>,
    rows: usize,
    b1: usize,
    b2: usize,
}

/// The bf16 projections of one layer, stacked.
struct PackBf16 {
    bytes: CudaSlice<u8>,
    exponent: i32,
    rows: usize,
    b1: usize,
    b2: usize,
}

/// A loop controller's weights as fp32: down, gain, forget.
type ControlF32 = [CudaSlice<f32>; 3];

/// One layer's attention projections packed for the decode GEMM, and the
/// small tensors the fused kernels read in one piece.
///
/// A pack stacks the projections that share a format, grouped by the input
/// they read (the query's, the key's, the value's), each group padded to
/// whole GEMM blocks so one launch contracts all of them. Both packs write
/// one output row, the Q8_0 pack's rows first.
pub(super) struct DecodeLayer {
    q8: Option<PackQ8>,
    bf16: Option<PackBf16>,
    offsets: ProjOffsets,
    /// `[1 or 3][n]`.
    norms: CudaSlice<f32>,
    /// The Q, K and V controllers, each tensor the three concatenated.
    ctrl_qkv: Option<ControlF32>,
    ctrl_ffn: Option<ControlF32>,
}

impl DecodeLayer {
    pub(super) fn load(l: &mut Loader<'_>, layer: u32, config: &ModelConfig) -> Res<Self> {
        let k = config.inkling.expect("an inkling2_loop configuration");
        let n = config.hidden_size as usize;
        let (heads, kv_heads, dim) = (
            config.attention.q_heads as usize,
            config.attention.kv_heads as usize,
            config.attention.head_dim as usize,
        );
        let sliding = k.is_sliding(layer);
        let taps = kv_heads * k.conv_taps as usize;
        let name = |s: &str| format!("blk.{layer}.{s}");
        // (tensor, rows, input group); the field order of `ProjOffsets`.
        let mut wanted = vec![
            ("attn_q.weight", heads * dim, 0),
            ("attn_rel.weight", heads * k.relative_length as usize, 0),
            ("attn_sink_proj.weight", heads, 0),
            ("attn_gate.weight", heads * dim, 0),
            ("attn_k.weight", kv_heads * dim, 1),
            ("attn_v.weight", kv_heads * dim, 2),
        ];
        if !sliding {
            wanted.push(("attn_k_conv_proj.weight", taps, 1));
            wanted.push(("attn_v_conv_proj.weight", taps, 2));
        }

        let (mut q8_q, mut q8_s, mut bf) = (Vec::<i8>::new(), Vec::<u16>::new(), Vec::<u8>::new());
        let mut q8_bounds = [0usize; 3];
        let mut bf_bounds = [0usize; 3];
        // (pack is bf16, row within the pack) per wanted tensor.
        let mut placed = vec![(false, 0usize); wanted.len()];
        for group in 0..3 {
            for (index, (tensor, rows, g)) in wanted.iter().enumerate() {
                if *g != group {
                    continue;
                }
                let (ty, bytes, elements) = l.info(&name(tensor))?;
                if elements != rows * n {
                    return Err(bad(format!("{}: unexpected shape", name(tensor))));
                }
                match ty {
                    GgmlType::Q8_0 => {
                        placed[index] = (false, q8_q.len() / n);
                        for block in bytes.as_chunks::<34>().0 {
                            q8_s.push(u16::from_le_bytes([block[0], block[1]]));
                            q8_q.extend(block[2..].iter().map(|b| *b as i8));
                        }
                    }
                    GgmlType::Bf16 => {
                        placed[index] = (true, bf.len() / (2 * n));
                        bf.extend_from_slice(bytes);
                    }
                    GgmlType::F32 => {
                        placed[index] = (true, bf.len() / (2 * n));
                        for c in bytes.as_chunks::<4>().0 {
                            let b = u32::from_le_bytes(*c);
                            let rounded = (b.wrapping_add(0x7FFF + ((b >> 16) & 1)) >> 16) as u16;
                            bf.extend_from_slice(&rounded.to_le_bytes());
                        }
                    }
                    other => {
                        return Err(bad(format!(
                            "{}: {} has no decode projection",
                            name(tensor),
                            other.name()
                        )));
                    }
                }
            }
            // Whole GEMM blocks per input group, so a block reads one plane.
            let q8_rows = (q8_q.len() / n).next_multiple_of(DG_BLOCK_ROWS);
            q8_q.resize(q8_rows * n, 0);
            q8_s.resize(q8_rows * n / 32, 0);
            q8_bounds[group] = q8_rows;
            let bf_rows = (bf.len() / (2 * n)).next_multiple_of(DG_BLOCK_ROWS);
            bf.resize(bf_rows * n * 2, 0);
            bf_bounds[group] = bf_rows;
        }
        let (q8_rows, bf_rows) = (q8_bounds[2], bf_bounds[2]);
        let at = |index: usize| {
            let (is_bf, row) = placed[index];
            (if is_bf { q8_rows + row } else { row }) as i32
        };
        let offsets = ProjOffsets {
            ldo: (q8_rows + bf_rows) as i32,
            q: at(0),
            rel: at(1),
            sink: at(2),
            gate: at(3),
            k: at(4),
            v: at(5),
            k_taps: if sliding { -1 } else { at(6) },
            v_taps: if sliding { -1 } else { at(7) },
        };
        l.bytes += q8_q.len() + q8_s.len() * 2 + bf.len();
        let q8 = if q8_rows > 0 {
            Some(PackQ8 {
                q: l.stream.clone_htod(&q8_q)?,
                scales: l.stream.clone_htod(&q8_s)?,
                rows: q8_rows,
                b1: q8_bounds[0],
                b2: q8_bounds[1],
            })
        } else {
            None
        };
        let bf16 = if bf_rows > 0 {
            Some(PackBf16 {
                exponent: bf16_half_exponent(&bf),
                bytes: l.stream.clone_htod(&bf)?,
                rows: bf_rows,
                b1: bf_bounds[0],
                b2: bf_bounds[1],
            })
        } else {
            None
        };

        let norm_names: &[&str] = if sliding {
            &["attn_norm.weight"]
        } else {
            &[
                "attn_q_inp_norm.weight",
                "attn_k_inp_norm.weight",
                "attn_v_inp_norm.weight",
            ]
        };
        let mut norms = Vec::with_capacity(norm_names.len() * n);
        for tensor in norm_names {
            norms.extend(l.host_floats(&name(tensor), n)?);
        }

        let rank = k.gate_rank as usize;
        let mut control = |which: &[&str]| -> Res<ControlF32> {
            let mut parts = [Vec::new(), Vec::new(), Vec::new()];
            for w in which {
                for (part, (suffix, len)) in parts.iter_mut().zip([
                    ("down", rank * 2 * n),
                    ("gain", n * rank),
                    ("forget", n * rank),
                ]) {
                    part.extend(
                        l.host_floats(&name(&format!("loop_{w}_ctrl_{suffix}.weight")), len)?,
                    );
                }
            }
            l.bytes += parts.iter().map(|p| p.len() * 4).sum::<usize>();
            Ok([
                l.stream.clone_htod(&parts[0])?,
                l.stream.clone_htod(&parts[1])?,
                l.stream.clone_htod(&parts[2])?,
            ])
        };
        let looped = (k.loop_start..k.loop_end).contains(&layer);
        let ctrl_qkv = looped.then(|| control(&["q", "k", "v"])).transpose()?;
        let ctrl_ffn = looped.then(|| control(&["ffn"])).transpose()?;
        l.bytes += norms.len() * 4;
        Ok(Self {
            q8,
            bf16,
            offsets,
            norms: l.stream.clone_htod(&norms)?,
            ctrl_qkv,
            ctrl_ffn,
        })
    }
}

fn control(c: &ControlF32, rank: usize) -> ControlWeights<'_> {
    ControlWeights {
        down: &c[0],
        gain: &c[1],
        forget: &c[2],
        rank,
    }
}

fn stack_of(e: &super::Experts) -> (HalfWeight<'_>, Option<&CudaSlice<f32>>) {
    match &e.w {
        ExpertStack::Nvfp4 { bytes, scales } => {
            (HalfWeight::Nvfp4 { bytes, scale: 1.0 }, scales.as_ref())
        }
        ExpertStack::Dense(dense) => (dense_half(dense), None),
    }
}

fn dense_half(d: &Dense) -> HalfWeight<'_> {
    match d {
        Dense::Q8 { q, scales } => HalfWeight::Q8_0 { q, scales },
        Dense::Bf16 { bytes, exponent } => HalfWeight::Bf16 {
            bytes,
            exponent: *exponent,
        },
    }
}

/// Scratch for decode steps of exactly `rows` sequences.
pub struct InklingDecode {
    model: Arc<InklingModel>,
    rows: usize,
    profile: Option<super::Profile>,
    route_log: Option<Vec<i32>>,
    d_tokens: CudaSlice<i32>,
    d_pos: CudaSlice<i32>,
    /// `[slots][rows][6]` device addresses, rebuilt each step.
    ptrs: CudaSlice<u64>,
    host_ptrs: Vec<u64>,
    host_pos: Vec<i32>,
    embedded: CudaSlice<f32>,
    h: CudaSlice<f32>,
    prelude: CudaSlice<f32>,
    blocks: CudaSlice<f32>,
    bank: CudaSlice<f32>,
    messages: CudaSlice<f32>,
    work: CudaSlice<f32>,
    inputs_h: CudaSlice<u16>,
    proj: CudaSlice<f32>,
    gated_h: CudaSlice<u16>,
    o_raw: CudaSlice<f32>,
    ffn_inp: CudaSlice<f32>,
    x_work: CudaSlice<f32>,
    inv: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    route_logits: CudaSlice<f32>,
    x_h: CudaSlice<u16>,
    ids: CudaSlice<i32>,
    route_weights: CudaSlice<f32>,
    hidden_h: CudaSlice<u16>,
    partial: CudaSlice<f32>,
    x_mix: CudaSlice<f32>,
    final_norm: CudaSlice<f32>,
    final_h: CudaSlice<u16>,
    head_logits: CudaSlice<f32>,
}

impl InklingDecode {
    /// Allocate for steps of exactly `rows` sequences.
    pub fn new(model: &Arc<InklingModel>, stream: &Arc<CudaStream>, rows: usize) -> Res<Self> {
        if rows == 0 || rows > DG_MAX_ROWS {
            return Err(bad(format!("a decode step takes 1..={DG_MAX_ROWS} rows")));
        }
        let c = &model.config;
        let k = &model.k;
        let n = c.hidden_size as usize;
        let (heads, dim) = (c.attention.q_heads as usize, c.attention.head_dim as usize);
        let moe = c.moe().expect("routed experts");
        let (topk, ff) = (
            moe.experts_per_token as usize,
            moe.expert_intermediate as usize,
        );
        let block_states = (c.num_layers / k.res_block) as usize + 1;
        let bank_states = ((k.loop_end - k.loop_start) / k.res_block) as usize;
        let ldo = model
            .layers
            .iter()
            .map(|l| l.decode.offsets.ldo as usize)
            .max()
            .unwrap_or(0);
        let slots = model.slots();
        let f = |len: usize| stream.alloc_zeros::<f32>(len.max(1));
        let hf = |len: usize| stream.alloc_zeros::<u16>(len.max(1));
        Ok(Self {
            model: model.clone(),
            rows,
            profile: None,
            route_log: None,
            d_tokens: stream.alloc_zeros::<i32>(rows)?,
            d_pos: stream.alloc_zeros::<i32>(rows)?,
            ptrs: stream.alloc_zeros::<u64>(slots * rows * DEC_ROW_POINTERS)?,
            host_ptrs: vec![0; slots * rows * DEC_ROW_POINTERS],
            host_pos: vec![0; rows],
            embedded: f(rows * n)?,
            h: f(rows * n)?,
            prelude: f(rows * n)?,
            blocks: f(block_states * rows * n)?,
            bank: f(bank_states * rows * n)?,
            messages: f(4 * rows * n)?,
            work: f(3 * rows * n)?,
            inputs_h: hf(3 * rows * n)?,
            proj: f(rows * ldo)?,
            gated_h: hf(rows * heads * dim)?,
            o_raw: f(rows * n)?,
            ffn_inp: f(rows * n)?,
            x_work: f(rows * n)?,
            inv: f(rows * 6)?,
            gate: f(rows * 3 * k.gate_rank as usize)?,
            route_logits: f(rows * moe.num_experts as usize)?,
            x_h: hf(rows * n)?,
            ids: stream.alloc_zeros::<i32>(rows * topk)?,
            route_weights: f(rows * topk)?,
            hidden_h: hf(rows * topk * ff)?,
            partial: f(rows * topk * n)?,
            x_mix: f(rows * n)?,
            final_norm: f(rows * n)?,
            final_h: hf(rows * n)?,
            head_logits: f(rows * c.vocab_size as usize)?,
        })
    }

    /// Sequences a step advances.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Record the experts of every routing from here on; see
    /// [`super::InklingPass::log_routes`].
    pub fn log_routes(&mut self) {
        self.route_log = Some(Vec::new());
    }

    /// What [`Self::log_routes`] has recorded.
    pub fn routes(&self) -> &[i32] {
        self.route_log.as_deref().unwrap_or(&[])
    }

    /// Start timing stages; see [`super::InklingPass::enable_profiling`].
    pub fn enable_profiling(&mut self) {
        self.profile = Some(super::Profile::start());
    }

    /// Seconds per stage since profiling was enabled, in first-seen order.
    pub fn stage_seconds(&self) -> &[(&'static str, f64)] {
        self.profile.as_ref().map_or(&[], |p| &p.stages)
    }

    fn mark(&mut self, stream: &Arc<CudaStream>, stage: &'static str) -> Res {
        match self.profile.as_mut() {
            Some(p) => p.mark(stream, stage),
            None => Ok(()),
        }
    }

    /// The final normalized hidden states of the last step, `[rows][n]`.
    pub fn hidden(&self) -> &CudaSlice<f32> {
        &self.final_norm
    }

    /// The logits [`Self::logits`] last computed, `[rows][vocab]`.
    pub fn head_logits(&self) -> &CudaSlice<f32> {
        &self.head_logits
    }

    #[allow(clippy::too_many_lines)]
    fn layer(
        &mut self,
        stream: &Arc<CudaStream>,
        slot: usize,
        layer_index: usize,
        live: &mut usize,
        with_messages: bool,
    ) -> Res {
        let model = self.model.clone();
        let layer = &model.layers[layer_index];
        let d = &layer.decode;
        let c = &model.config;
        let kc = &model.k;
        let n = c.hidden_size as usize;
        let rows = self.rows;
        let (heads, kv_heads, dim) = (
            c.attention.q_heads as usize,
            c.attention.kv_heads as usize,
            c.attention.head_dim as usize,
        );
        let taps = kc.conv_taps as usize;
        let rank = kc.gate_rank as usize;
        let eps = f32::from_bits(kc.rms_eps_bits);
        let moe = c.moe().expect("routed experts");
        let (experts, topk, ff) = (
            moe.num_experts as usize,
            moe.experts_per_token as usize,
            moe.expert_intermediate as usize,
        );
        let ptrs = unsafe {
            subslice(
                stream,
                &self.ptrs,
                slot * rows * DEC_ROW_POINTERS,
                rows * DEC_ROW_POINTERS,
            )
        };
        let (ctrl_qkv, ctrl_ffn) = if with_messages {
            let missing = || bad("messages outside the loop");
            (
                Some(d.ctrl_qkv.as_ref().ok_or_else(missing)?),
                Some(d.ctrl_ffn.as_ref().ok_or_else(missing)?),
            )
        } else {
            (None, None)
        };

        let distinct = model.dec.inputs(
            stream,
            &self.h,
            &self.blocks,
            layer
                .attn_query
                .as_ref()
                .map(|q| (q, if layer.sliding { 1 } else { 3 })),
            ctrl_qkv.map(|_| &self.messages),
            &d.norms,
            &mut self.work,
            &mut self.inputs_h,
            &mut self.inv,
            n,
            *live,
            eps,
            rows,
        )?;
        if let Some(w) = ctrl_qkv {
            model.dec.control(
                stream,
                &mut self.work,
                &self.messages,
                control(w, rank),
                &d.norms,
                &mut self.inputs_h,
                &self.inv,
                &mut self.gate,
                n,
                3,
                eps,
                false,
                rows,
            )?;
        }
        self.mark(stream, "attn inputs")?;
        // One input: every projection reads plane 0.
        let ldo = d.offsets.ldo as usize;
        if let Some(p) = &d.q8 {
            let (b1, b2) = if distinct == 1 {
                (p.rows, p.rows)
            } else {
                (p.b1, p.b2)
            };
            model.dg.dense(
                stream,
                HalfWeight::Q8_0 {
                    q: &p.q,
                    scales: &p.scales,
                },
                &self.inputs_h,
                &mut self.proj,
                p.rows,
                n,
                rows,
                ldo,
                rows,
                b1,
                b2,
            )?;
        }
        if let Some(p) = &d.bf16 {
            let (b1, b2) = if distinct == 1 {
                (p.rows, p.rows)
            } else {
                (p.b1, p.b2)
            };
            let base = d.q8.as_ref().map_or(0, |q| q.rows);
            let mut out = unsafe { subslice(stream, &self.proj, base, rows * ldo - base) };
            model.dg.dense(
                stream,
                HalfWeight::Bf16 {
                    bytes: &p.bytes,
                    exponent: p.exponent,
                },
                &self.inputs_h,
                &mut out,
                p.rows,
                n,
                rows,
                ldo,
                rows,
                b1,
                b2,
            )?;
        }

        self.mark(stream, "projection packs")?;
        model.dec.kv(
            stream,
            &self.proj,
            &ptrs,
            &self.d_pos,
            &layer.k_conv,
            &layer.v_conv,
            d.offsets,
            kv_heads,
            dim,
            taps,
            f32::from_bits(kc.qk_clip_bits),
            eps,
            rows,
        )?;
        self.mark(stream, "keys and values")?;
        let (window, extent) = if layer.sliding {
            (kc.sliding_window as usize, kc.sliding_window as usize)
        } else {
            (0, kc.relative_extent as usize)
        };
        model.dec.attn(
            stream,
            &self.proj,
            &ptrs,
            &self.d_pos,
            &layer.q_clip,
            &layer.rel_clip,
            &layer.rel_bias,
            &mut self.gated_h,
            d.offsets,
            heads,
            kv_heads,
            dim,
            kc.relative_length as usize,
            window,
            extent,
            f32::from_bits(kc.qk_clip_bits),
            f32::from_bits(kc.relative_clip_bits),
            (!layer.sliding).then(|| {
                (
                    f32::from_bits(kc.log_scale_alpha_bits),
                    kc.log_scale_floor as f32,
                )
            }),
            rows,
        )?;
        self.mark(stream, "attention")?;
        let whole = layer.o.rows.next_multiple_of(DG_BLOCK_ROWS);
        model.dg.dense(
            stream,
            dense_half(&layer.o.w),
            &self.gated_h,
            &mut self.o_raw,
            layer.o.rows,
            layer.o.k,
            rows,
            n,
            rows,
            whole,
            whole,
        )?;

        self.mark(stream, "output projection")?;
        let opens = (layer_index as u32).is_multiple_of(kc.res_block);
        let gate_up_nv = matches!(layer.gate_up.w, ExpertStack::Nvfp4 { .. });
        let down_nv = matches!(layer.down.w, ExpertStack::Nvfp4 { .. });
        let msg =
            ctrl_ffn.map(|_| unsafe { subslice(stream, &self.messages, 3 * rows * n, rows * n) });
        model.dec.ffn_in(
            stream,
            &self.o_raw,
            &self.h,
            &layer.attn_conv,
            &ptrs,
            &mut self.blocks,
            &layer.ffn_query,
            msg.as_deref(),
            &layer.ffn_norm,
            &mut self.ffn_inp,
            &mut self.x_work,
            &mut self.x_h,
            &mut self.inv,
            n,
            *live,
            opens,
            taps,
            gate_up_nv,
            eps,
            rows,
        )?;
        if let (Some(w), Some(msg)) = (ctrl_ffn, msg.as_deref()) {
            model.dec.control(
                stream,
                &mut self.x_work,
                msg,
                control(w, rank),
                &layer.ffn_norm,
                &mut self.x_h,
                &self.inv,
                &mut self.gate,
                n,
                1,
                eps,
                gate_up_nv,
                rows,
            )?;
        }
        model.dec.route(
            stream,
            &self.x_work,
            &layer.router,
            &layer.route_bias,
            &mut self.route_logits,
            &mut self.ids,
            &mut self.route_weights,
            n,
            experts,
            topk,
            f32::from_bits(kc.expert_weights_scale_bits),
            rows,
        )?;
        *live += usize::from(opens);

        self.mark(stream, "ffn input and routing")?;
        if let Some(log) = self.route_log.as_mut() {
            log.extend(stream.clone_dtoh(&self.ids)?);
        }
        let pairs = rows * topk;
        let (w, escale) = stack_of(&layer.gate_up);
        model.dg.gate_up(
            stream,
            w,
            experts,
            &self.ids,
            escale,
            &self.x_h,
            &mut self.hidden_h,
            2 * ff,
            n,
            pairs,
            topk,
            down_nv,
            f32::from_bits(kc.situ_beta_bits),
            f32::from_bits(kc.situ_linear_beta_bits),
        )?;
        self.mark(stream, "expert gate/up")?;
        let (w, escale) = stack_of(&layer.down);
        model.dg.down(
            stream,
            w,
            experts,
            &self.ids,
            escale,
            &self.hidden_h,
            &self.route_weights,
            &mut self.partial,
            n,
            ff,
            pairs,
        )?;
        self.mark(stream, "expert down")?;
        model.dec.ffn_out(
            stream,
            &self.partial,
            &self.ffn_inp,
            &layer.ffn_conv,
            &ptrs,
            &mut self.h,
            n,
            topk,
            taps,
            rows,
        )?;
        self.mark(stream, "layer tail")?;
        Ok(())
    }

    /// Advance every sequence by one token and leave the final normalized
    /// hidden states in [`Self::hidden`].
    pub fn step(
        &mut self,
        stream: &Arc<CudaStream>,
        states: &mut [&mut InklingState],
        tokens: &[i32],
    ) -> Res {
        let model = self.model.clone();
        let c = &model.config;
        let kc = &model.k;
        let n = c.hidden_size as usize;
        let rows = self.rows;
        let plane = rows * n;
        if tokens.len() != rows || states.len() != rows {
            return Err(bad(format!("the step advances exactly {rows} sequences")));
        }
        if tokens.iter().any(|t| *t < 0 || *t as u32 >= c.vocab_size) {
            return Err(bad("a token id is outside the vocabulary"));
        }
        let slots = model.slots();
        for (row, state) in states.iter().enumerate() {
            if state.position + 1 > state.capacity {
                return Err(bad("a sequence would outgrow its cache"));
            }
            self.host_pos[row] = state.position as i32;
            for (slot, s) in state.slots.iter().enumerate() {
                let at = (slot * rows + row) * DEC_ROW_POINTERS;
                self.host_ptrs[at] = s.k.device_ptr(stream).0;
                self.host_ptrs[at + 1] = s.v.device_ptr(stream).0;
                self.host_ptrs[at + 2] = s.k_carry.device_ptr(stream).0;
                self.host_ptrs[at + 3] = s.v_carry.device_ptr(stream).0;
                self.host_ptrs[at + 4] = s.attn_carry.device_ptr(stream).0;
                self.host_ptrs[at + 5] = s.ffn_carry.device_ptr(stream).0;
            }
        }
        debug_assert_eq!(self.host_ptrs.len(), slots * rows * DEC_ROW_POINTERS);
        if let Some(p) = self.profile.as_mut() {
            p.restart(stream)?;
        }
        stream.memcpy_htod(&self.host_ptrs, &mut self.ptrs)?;
        stream.memcpy_htod(&self.host_pos, &mut self.d_pos)?;
        stream.memcpy_htod(tokens, &mut self.d_tokens)?;
        model.kernels.embed(
            stream,
            &model.tok_embd,
            &self.d_tokens,
            &mut self.embedded,
            n,
            rows,
            model.embd_bf16,
        )?;
        let eps = f32::from_bits(kc.rms_eps_bits);
        model.ops.rms_norm(
            stream,
            &self.embedded,
            &model.tok_norm,
            &mut self.h,
            rows,
            n,
            eps,
        )?;

        self.mark(stream, "embed")?;
        let layers = c.num_layers as usize;
        let (start, end) = (kc.loop_start as usize, kc.loop_end as usize);
        let mut live = 0usize;
        for layer in 0..start {
            self.layer(stream, layer, layer, &mut live, false)?;
        }
        stream.memcpy_dtod(&self.h, &mut self.prelude)?;
        let prelude_live = live;
        for layer in start..end {
            self.layer(stream, layer, layer, &mut live, false)?;
        }
        for pass in 1..kc.loop_count as usize {
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
                self.layer(stream, slot, layer, &mut live, true)?;
            }
        }
        for layer in end..layers {
            self.layer(stream, layer, layer, &mut live, false)?;
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
        self.mark(stream, "loop messages, output mix and norm")?;
        for state in states.iter_mut() {
            state.position += 1;
        }
        Ok(())
    }

    /// Logits of every row of the last step over the real vocabulary,
    /// soft-capped, into [`Self::head_logits`].
    pub fn logits(&mut self, stream: &Arc<CudaStream>) -> Res {
        let model = self.model.clone();
        let n = model.config.hidden_size as usize;
        let vocab = model.config.vocab_size as usize;
        let rows = self.rows;
        model
            .hg
            .to_half(stream, &self.final_norm, &mut self.final_h, rows * n)?;
        let whole = vocab.next_multiple_of(DG_BLOCK_ROWS);
        model.dg.dense(
            stream,
            dense_half(&model.head.w),
            &self.final_h,
            &mut self.head_logits,
            vocab,
            n,
            rows,
            vocab,
            rows,
            whole,
            whole,
        )?;
        let cap = f32::from_bits(model.k.softcap_bits);
        if cap != 0.0 {
            model
                .kernels
                .softcap(stream, &mut self.head_logits, rows * vocab, cap)?;
        }
        self.mark(stream, "head")?;
        Ok(())
    }
}
