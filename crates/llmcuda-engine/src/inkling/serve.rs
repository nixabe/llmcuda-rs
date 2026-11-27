//! The serving runtime for `inkling2_loop`: what a worker's device thread
//! runs in place of [`crate::runtime::DeviceRuntime`] for this architecture.
//!
//! It answers the same commands — admit, remove, execute a scheduler step —
//! and returns the same step record, so the scheduler, the pager and the
//! engine above it are unchanged. It does less: no prefix snapshots (a step
//! never retains anything, and a resume is refused), no images, no decision
//! requests, no speculation. Every token is chosen on the host from a row of
//! logits, greedy or sampled, grammar applied first.
//!
//! Nothing is allocated per step. Prefill passes are prebuilt at every power
//! of two up to the chunk, so a grant of any length is a handful of exact
//! passes — and any cut of a prompt into passes is the same bits
//! (`inkling_forward.rs`). Decode steps are prebuilt at every width.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};
use llmcuda_gguf::GgufFile;
use llmcuda_grammar::ToolConstraint;
use llmcuda_model::{InklingConfig, ModelConfig};
use llmcuda_sched::request::{BatchDescription, NewRequest, RequestId};
use smallvec::SmallVec;

use super::{InklingDecode, InklingModel, InklingPass, InklingRunError, InklingState, Segment};
use crate::runtime::{RuntimeConfig, RuntimeError, RuntimeStep};
use crate::sampling::{Sampler, SamplingParams};
use crate::state::{SnapshotArena, SnapshotArenaSize, SnapshotSlots};
use crate::viewslice::subslice;

impl From<InklingRunError> for RuntimeError {
    fn from(error: InklingRunError) -> Self {
        Self::Inkling(error.0)
    }
}

struct Sequence {
    state: InklingState,
    prompt: Vec<i32>,
    prefilled: usize,
    /// The token the next decode step consumes: the last one emitted.
    next_token: Option<i32>,
    emitted: u32,
    max_output: u32,
    sampler: Option<Sampler>,
    constraint: Option<Box<ToolConstraint>>,
}

impl Sequence {
    /// Pick a token from a row of logits, the grammar applied first.
    fn choose(&mut self, logits: &mut [f32], scratch: &mut Vec<(f32, u32)>) -> i32 {
        if let Some(constraint) = self.constraint.as_mut() {
            constraint.mask(logits);
        }
        let token = match self.sampler.as_mut() {
            Some(sampler) => sampler.sample(logits, scratch),
            None => crate::runtime::argmax(logits),
        };
        if let Some(constraint) = self.constraint.as_mut() {
            constraint.observe(token);
        }
        token
    }
}

/// One card's `inkling2_loop` model, prebuilt passes and live sequences.
pub(crate) struct InklingRuntime {
    stream: Arc<CudaStream>,
    model: Arc<InklingModel>,
    /// Prefill passes, widest first: every power of two up to the chunk.
    passes: Vec<(usize, InklingPass)>,
    /// Decode steps by width, `decode[w - 1]`.
    decode: Vec<InklingDecode>,
    sequences: HashMap<RequestId, Sequence>,
    /// The sequences of the decode step in flight, lifted out of the map so
    /// their states can be borrowed together.
    stepping: Vec<(RequestId, Sequence)>,
    step_tokens: Vec<i32>,
    host_logits: Vec<f32>,
    scratch: Vec<(f32, u32)>,
    eos: SmallVec<[i32; 2]>,
    vocab: u32,
    max_pass_rows: usize,
    slots: SnapshotSlots,
}

impl InklingRuntime {
    pub(crate) fn load(
        device_ordinal: usize,
        model_path: &Path,
        config: ModelConfig,
        runtime: RuntimeConfig,
    ) -> Result<Self, RuntimeError> {
        let refuse =
            |what: &str| RuntimeError::Inkling(format!("inkling2_loop does not serve {what}"));
        if runtime.prefill_chunk == 0 {
            return Err(RuntimeError::ZeroPrefillChunk);
        }
        if runtime.max_batch == 0 {
            return Err(RuntimeError::ZeroBatchWidth);
        }
        if runtime.max_batch > super::decode::MAX_DECODE_ROWS {
            return Err(RuntimeError::BatchTooWide {
                requested: runtime.max_batch,
                maximum: super::decode::MAX_DECODE_ROWS,
            });
        }
        if runtime.spec.is_some() || runtime.mtp_drafts > 0 || runtime.dflash.is_some() {
            return Err(refuse("speculative decoding"));
        }
        if runtime.mmproj.is_some() {
            return Err(refuse("images"));
        }
        // The sliding layers' rings are sized for passes this wide; the
        // pager counted them at that size.
        let max_pass_rows = runtime
            .prefill_chunk
            .min(InklingConfig::SERVING_PASS_ROWS as usize);

        let ctx = CudaContext::new(device_ordinal)?;
        let stream = ctx.default_stream();
        let file = GgufFile::open(model_path)?;
        let model = Arc::new(InklingModel::load(&ctx, &stream, &file)?);
        let vocab = config.vocab_size;

        let mut passes = Vec::new();
        let mut width = max_pass_rows.next_power_of_two();
        if width > max_pass_rows {
            width /= 2;
        }
        loop {
            passes.push((
                width,
                InklingPass::with_logit_rows(&model, &stream, width, 1)?,
            ));
            if width == 1 {
                break;
            }
            width /= 2;
        }
        let mut decode = Vec::with_capacity(runtime.max_batch);
        for width in 1..=runtime.max_batch {
            decode.push(InklingDecode::new(&model, &stream, width)?);
        }

        let mut eos: SmallVec<[i32; 2]> = SmallVec::new();
        if runtime.stop_on_eos {
            for key in ["tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"] {
                if let Some(token) = file.get_u32(key)
                    && !eos.contains(&(token as i32))
                {
                    eos.push(token as i32);
                }
            }
        }
        // An arena with nothing in it: the engine asks how many snapshots a
        // worker could take, and the answer here is always none.
        let slots = SnapshotArena::new(&ctx, &config, 1, SnapshotArenaSize::NONE)?.slots();
        Ok(Self {
            stream,
            model,
            passes,
            decode,
            sequences: HashMap::with_capacity(runtime.max_batch * 4),
            stepping: Vec::with_capacity(runtime.max_batch),
            step_tokens: Vec::with_capacity(runtime.max_batch),
            host_logits: vec![0.0; vocab as usize],
            scratch: Vec::with_capacity(vocab as usize),
            eos,
            vocab,
            max_pass_rows,
            slots,
        })
    }

    pub(crate) fn slots(&self) -> SnapshotSlots {
        self.slots.clone()
    }

    pub(crate) fn admit(
        &mut self,
        req: NewRequest,
        prompt: Vec<i32>,
        has_images: bool,
        sampling: SamplingParams,
        constraint: Option<Box<ToolConstraint>>,
    ) -> Result<(), RuntimeError> {
        if has_images {
            return Err(RuntimeError::VisionNotEnabled);
        }
        if self.sequences.contains_key(&req.id) {
            return Err(RuntimeError::DuplicateRequest(req.id));
        }
        if prompt.len() != req.prompt_tokens as usize {
            return Err(RuntimeError::PromptLength {
                declared: req.prompt_tokens,
                actual: prompt.len(),
            });
        }
        if let Some(&token) = prompt
            .iter()
            .find(|&&token| token < 0 || token as u32 >= self.vocab)
        {
            return Err(RuntimeError::TokenOutOfRange {
                token,
                vocab: self.vocab,
            });
        }
        // The whole sequence's cache, reserved for its whole life.
        let state = InklingState::bounded(
            &self.stream,
            &self.model,
            req.full_seq_len() as usize,
            self.max_pass_rows,
        )?;
        self.sequences.insert(
            req.id,
            Sequence {
                state,
                prompt,
                prefilled: 0,
                next_token: None,
                emitted: 0,
                max_output: req.max_output_tokens,
                sampler: (!sampling.is_greedy()).then(|| Sampler::new(sampling)),
                constraint,
            },
        );
        Ok(())
    }

    pub(crate) fn remove(&mut self, id: RequestId) -> bool {
        self.sequences.remove(&id).is_some()
    }

    /// Copy one row of logits to the host and let the sequence choose.
    fn choose_from(
        stream: &Arc<CudaStream>,
        logits: &cudarc::driver::CudaSlice<f32>,
        row: usize,
        host: &mut [f32],
        scratch: &mut Vec<(f32, u32)>,
        seq: &mut Sequence,
    ) -> Result<i32, RuntimeError> {
        let vocab = host.len();
        // SAFETY: the row lies inside `logits`, which outlives the view.
        let view = unsafe { subslice(stream, logits, row * vocab, vocab) };
        stream.memcpy_dtoh(&*view, host)?;
        Ok(seq.choose(host, scratch))
    }

    /// Record a chosen token: an end-of-sequence token ends the request and
    /// is not emitted; anything else is, and feeds the next step.
    fn emit(
        eos: &[i32],
        id: RequestId,
        token: i32,
        seq: &mut Sequence,
        generated: &mut SmallVec<[(RequestId, i32); 16]>,
        stopped: &mut SmallVec<[RequestId; 3]>,
    ) {
        if eos.contains(&token) {
            seq.emitted = seq.max_output;
            seq.next_token = None;
            stopped.push(id);
        } else {
            generated.push((id, token));
            seq.emitted += 1;
            seq.next_token = Some(token);
        }
    }

    /// One decode step over `self.stepping`, and a token chosen for each.
    fn step_decodes(
        &mut self,
        generated: &mut SmallVec<[(RequestId, i32); 16]>,
        stopped: &mut SmallVec<[RequestId; 3]>,
    ) -> Result<(), RuntimeError> {
        let width = self.stepping.len();
        if width == 0 {
            return Ok(());
        }
        if width > self.decode.len() {
            return Err(RuntimeError::BatchTooWide {
                requested: width,
                maximum: self.decode.len(),
            });
        }
        let decode = &mut self.decode[width - 1];
        {
            let mut states: SmallVec<[&mut InklingState; 8]> = self
                .stepping
                .iter_mut()
                .map(|(_, s)| &mut s.state)
                .collect();
            decode.step(&self.stream, &mut states, &self.step_tokens)?;
        }
        decode.logits(&self.stream)?;
        for (row, (id, seq)) in self.stepping.iter_mut().enumerate() {
            let token = Self::choose_from(
                &self.stream,
                decode.head_logits(),
                row,
                &mut self.host_logits,
                &mut self.scratch,
                seq,
            )?;
            Self::emit(&self.eos, *id, token, seq, generated, stopped);
        }
        Ok(())
    }

    pub(crate) fn execute(
        &mut self,
        scheduled: BatchDescription,
    ) -> Result<RuntimeStep, RuntimeError> {
        let mut generated = SmallVec::new();
        let mut stopped = SmallVec::new();

        // Decodes first, all of them in one step.
        self.stepping.clear();
        self.step_tokens.clear();
        for item in &scheduled.decodes {
            let seq = self
                .sequences
                .remove(&item.id)
                .ok_or(RuntimeError::UnknownRequest(item.id))?;
            match seq.next_token {
                Some(token) if seq.emitted < seq.max_output => {
                    self.step_tokens.push(token);
                    self.stepping.push((item.id, seq));
                }
                // Nothing to feed: the request is finished and this step's
                // sweep will collect it.
                _ => {
                    self.sequences.insert(item.id, seq);
                }
            }
        }
        let outcome = self.step_decodes(&mut generated, &mut stopped);
        // Back into the map whatever happened, so a failed step leaves
        // every sequence removable.
        for (id, seq) in self.stepping.drain(..) {
            self.sequences.insert(id, seq);
        }
        outcome?;

        for item in &scheduled.prefills {
            let seq = self
                .sequences
                .get_mut(&item.id)
                .ok_or(RuntimeError::UnknownRequest(item.id))?;
            let mut left = (item.tokens as usize).min(seq.prompt.len() - seq.prefilled);
            while left > 0 {
                let (rows, pass) = self
                    .passes
                    .iter_mut()
                    .find(|(rows, _)| *rows <= left)
                    .expect("the narrowest pass is one row");
                let rows = *rows;
                pass.run(
                    &self.stream,
                    &mut [Segment {
                        state: &mut seq.state,
                        rows,
                    }],
                    &seq.prompt[seq.prefilled..seq.prefilled + rows],
                    None,
                )?;
                seq.prefilled += rows;
                left -= rows;
                // The prompt's last row yields the first output token.
                if seq.prefilled == seq.prompt.len() && seq.max_output > 0 {
                    pass.logits(&self.stream, rows - 1, 1)?;
                    let token = Self::choose_from(
                        &self.stream,
                        pass.head_logits(),
                        0,
                        &mut self.host_logits,
                        &mut self.scratch,
                        seq,
                    )?;
                    Self::emit(&self.eos, item.id, token, seq, &mut generated, &mut stopped);
                }
            }
        }

        let completed: SmallVec<[RequestId; 3]> = self
            .sequences
            .iter()
            .filter_map(|(&id, seq)| {
                (seq.prefilled == seq.prompt.len() && seq.emitted >= seq.max_output).then_some(id)
            })
            .collect();
        for id in &completed {
            self.sequences.remove(id);
        }
        Ok(RuntimeStep {
            scheduled,
            generated,
            completed,
            stopped,
            retained: SmallVec::new(),
            decisions: SmallVec::new(),
        })
    }
}
