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
use crate::state::{SequenceSnapshot, SnapshotArena, SnapshotSlots, StateError};
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
    /// The newest snapshot of this sequence's prefix, taken here or resumed
    /// from, and the parent of the next one.
    last_snapshot: Option<Arc<SequenceSnapshot>>,
    last_snapshot_position: usize,
    /// Whether this sequence took `last_snapshot` itself.
    last_snapshot_own: bool,
    /// Set when the arena ran out: this sequence retains nothing more.
    retention_disabled: bool,
    /// Resumed from a snapshot of its whole prompt: the first output token
    /// is the snapshot's, emitted on the first grant.
    first_from_snapshot: bool,
}

impl Sequence {
    /// Whether a snapshot is due at the position the state has reached.
    fn at_boundary(&self, interval: usize) -> bool {
        let position = self.state.position();
        interval > 0
            && !self.retention_disabled
            && position > self.last_snapshot_position
            && position.is_multiple_of(interval)
    }

    /// Whether what this sequence picks next is the model's argmax and
    /// nothing else, so that another request may inherit it.
    fn picks_the_argmax(&self) -> bool {
        self.sampler.is_none() && self.constraint.is_none()
    }

    /// Take a snapshot at the current position, recording `next` as what
    /// follows it when `record`. An exhausted arena ends this sequence's
    /// retention and is not an error. As `DeviceRuntime`'s `Retention`: the
    /// snapshot before gives up its fixed-size state unless it is the
    /// prompt's last boundary, which a later request is likeliest to want.
    #[allow(clippy::too_many_arguments)]
    fn retain(
        &mut self,
        stream: &Arc<CudaStream>,
        model: &InklingModel,
        arena: &SnapshotArena,
        interval: usize,
        next: i32,
        record: bool,
    ) -> Result<Option<Arc<SequenceSnapshot>>, RuntimeError> {
        let snapshot = self
            .state
            .snapshot(stream, model, arena, self.last_snapshot.clone());
        let mut snapshot = match snapshot {
            Ok(snapshot) => snapshot,
            Err(StateError::SnapshotArenaExhausted) => {
                self.retention_disabled = true;
                return Ok(None);
            }
            Err(error) => return Err(error.into()),
        };
        if record {
            snapshot.set_next_token(next);
        }
        let snapshot = Arc::new(snapshot);
        if self.last_snapshot_own
            && let Some(previous) = self.last_snapshot.as_ref()
        {
            let prompt_tail = self.prompt.len() / interval * interval;
            if previous.position() != prompt_tail {
                previous.yield_state();
            }
        }
        self.last_snapshot_position = snapshot.position();
        self.last_snapshot = Some(Arc::clone(&snapshot));
        self.last_snapshot_own = true;
        Ok(Some(snapshot))
    }

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

/// Snapshots a step took, by request.
type Retained = SmallVec<[(RequestId, Arc<SequenceSnapshot>); 8]>;

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
    arena: SnapshotArena,
    /// Positions between snapshots; zero when the arena holds none.
    retention_interval: usize,
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
        let arena = SnapshotArena::new(
            &ctx,
            &config,
            runtime.retention_interval.max(1),
            runtime.snapshot_slots,
        )?;
        // A prefill piece ends at every boundary, so a boundary has to be a
        // pass this runtime can run.
        let retention_interval = if arena.capacity() == 0 || arena.state_capacity() == 0 {
            0
        } else {
            runtime.retention_interval
        };
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
            arena,
            retention_interval,
        })
    }

    pub(crate) fn slots(&self) -> SnapshotSlots {
        self.arena.slots()
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
                last_snapshot: None,
                last_snapshot_position: 0,
                last_snapshot_own: false,
                retention_disabled: false,
                first_from_snapshot: false,
            },
        );
        Ok(())
    }

    /// Admit a request whose prompt begins with the prefix `snapshot`
    /// holds: the state is restored and prefill continues from there.
    pub(crate) fn admit_restored(
        &mut self,
        req: NewRequest,
        prompt: Vec<i32>,
        has_images: bool,
        snapshot: Arc<SequenceSnapshot>,
        sampling: SamplingParams,
        constraint: Option<Box<ToolConstraint>>,
    ) -> Result<(), RuntimeError> {
        let prefix = snapshot.position();
        if prefix > prompt.len() {
            return Err(RuntimeError::SnapshotPrefix {
                snapshot: prefix,
                prompt: prompt.len(),
            });
        }
        self.admit(req, prompt, has_images, sampling, constraint)?;
        let seq = self
            .sequences
            .get_mut(&req.id)
            .expect("admit inserted the request");
        if let Err(error) = seq.state.restore(&self.stream, &self.model, &snapshot) {
            self.sequences.remove(&req.id);
            return Err(error.into());
        }
        seq.prefilled = prefix;
        seq.last_snapshot_position = prefix;
        if prefix == seq.prompt.len() {
            seq.next_token = snapshot.next_token();
            seq.first_from_snapshot = true;
        }
        seq.last_snapshot = Some(snapshot);
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
        retained: &mut Retained,
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
            // The token just chosen is what follows the prefix the state
            // now holds.
            if seq.at_boundary(self.retention_interval) {
                let record = seq.sampler.is_none();
                if let Some(snapshot) = seq.retain(
                    &self.stream,
                    &self.model,
                    &self.arena,
                    self.retention_interval,
                    token,
                    record,
                )? {
                    retained.push((*id, snapshot));
                }
            }
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
        let mut retained: Retained = SmallVec::new();

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
        let outcome = self.step_decodes(&mut generated, &mut stopped, &mut retained);
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
            // A whole-prompt resume has nothing to prefill: its first token
            // is the one the snapshot recorded.
            if seq.first_from_snapshot {
                seq.first_from_snapshot = false;
                if seq.max_output > 0 {
                    let token = seq.next_token.ok_or(RuntimeError::SnapshotPrefix {
                        snapshot: seq.prefilled,
                        prompt: seq.prompt.len(),
                    })?;
                    if let Some(constraint) = seq.constraint.as_mut() {
                        constraint.observe(token);
                    }
                    Self::emit(&self.eos, item.id, token, seq, &mut generated, &mut stopped);
                }
            }
            while left > 0 {
                // A piece ends at a retention boundary, never across one.
                let reach = match self.retention_interval {
                    0 => left,
                    _ if seq.retention_disabled => left,
                    interval => left.min(interval - seq.prefilled % interval),
                };
                let (rows, pass) = self
                    .passes
                    .iter_mut()
                    .find(|(rows, _)| *rows <= reach)
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
                // The prompt's last row yields the first output token; a
                // boundary inside the prompt yields the model's own
                // prediction, recorded so that a request resuming there
                // with nothing left to prefill knows what follows.
                let first = seq.prefilled == seq.prompt.len() && seq.max_output > 0;
                let boundary = seq.at_boundary(self.retention_interval);
                if !first && !boundary {
                    continue;
                }
                pass.logits(&self.stream, rows - 1, 1)?;
                let (token, record) = if first {
                    let record = seq.picks_the_argmax();
                    let token = Self::choose_from(
                        &self.stream,
                        pass.head_logits(),
                        0,
                        &mut self.host_logits,
                        &mut self.scratch,
                        seq,
                    )?;
                    (token, record)
                } else {
                    let vocab = self.host_logits.len();
                    // SAFETY: the first row of the pass's logits.
                    let view = unsafe { subslice(&self.stream, pass.head_logits(), 0, vocab) };
                    self.stream.memcpy_dtoh(&*view, &mut self.host_logits)?;
                    (crate::runtime::argmax(&self.host_logits), true)
                };
                if boundary
                    && let Some(snapshot) = seq.retain(
                        &self.stream,
                        &self.model,
                        &self.arena,
                        self.retention_interval,
                        token,
                        record,
                    )?
                {
                    retained.push((item.id, snapshot));
                }
                if first {
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
            retained,
            decisions: SmallVec::new(),
        })
    }
}
