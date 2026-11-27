//! The serving runtime for `inkling2_loop`: what a worker's device thread
//! runs in place of [`crate::runtime::DeviceRuntime`] for this architecture.
//!
//! It answers the same commands — admit, remove, execute a scheduler step —
//! and returns the same step record, so the scheduler, the pager and the
//! engine above it are unchanged: prefix snapshots are retained and resumed
//! as there, images are encoded at admission and their rows stand in the
//! prompt, and the n-gram drafters are verified in the decode step itself.
//! It has no decision requests and no trained drafter. Every token is chosen
//! on the host from a row of logits, greedy or sampled, grammar applied
//! first.
//!
//! Nothing is allocated per step. Prefill passes are prebuilt at every power
//! of two up to the chunk, so a grant of any length is a handful of exact
//! passes — and any cut of a prompt into passes is the same bits
//! (`inkling_forward.rs`). Decode steps are prebuilt at every width.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use llmcuda_gguf::GgufFile;
use llmcuda_grammar::ToolConstraint;
use llmcuda_kernels::inkling_vision::{InklingVisionWeights, PreprocessedInklingImage};
use llmcuda_model::{InklingConfig, InklingVisionConfig, ModelConfig};
use llmcuda_sched::request::{BatchDescription, NewRequest, RequestId};
use smallvec::SmallVec;

use super::vision::InklingVisionForward;
use super::{
    GivenRows, InklingDecode, InklingModel, InklingPass, InklingRunError, InklingState, Segment,
};
use crate::image::{SequenceImage, validate_placements};
use crate::runtime::{RuntimeConfig, RuntimeError, RuntimeStep};
use crate::sampling::{Sampler, SamplingParams};
use crate::state::{SequenceSnapshot, SnapshotArena, SnapshotSlots, StateError};
use crate::viewslice::subslice;
use llmcuda_sched::spec::{SelfSpecFactory, SelfSpeculator};

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
    /// Proposes tokens from what this sequence has seen, when the server
    /// speculates.
    spec: Option<SelfSpeculator>,
    /// The tokens drafted for the step in flight.
    draft: Vec<i32>,
    /// The prompt's image spans, in order: first position, rows, and the
    /// first row of `image_rows` that holds them.
    images: SmallVec<[(usize, usize, usize); 2]>,
    /// Every image's rows from the vision tower, encoded at admission.
    image_rows: Option<CudaSlice<f32>>,
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
    /// Present when the server speculates with an n-gram drafter.
    spec: Option<SelfSpecFactory>,
    /// The fewest tokens worth drafting.
    draft_n_min: usize,
    /// Rows of each sequence in a step of windows, and how many it keeps.
    window_lens: Vec<usize>,
    window_keep: Vec<usize>,
    /// The tokens each sequence chose in a step of windows, end to end.
    window_chosen: Vec<i32>,
    /// Present when the server was given a vision projector.
    vision: Option<InklingVisionForward>,
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
        // The n-gram drafters need nothing of the model. The trained ones
        // are Qwen's: an MTP head in the file, a DFlash drafter beside it.
        if runtime.mtp_drafts > 0 || runtime.dflash.is_some() {
            return Err(refuse("a trained drafter; the n-gram drafters work"));
        }
        let spec = runtime.spec.map(SelfSpecFactory::new);
        let drafts = spec.as_ref().map_or(0, SelfSpecFactory::max_draft_tokens);
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
        // Without drafts a step is a row a sequence. With them it is up to
        // as many rows as the step holds, whoever they belong to.
        let widest = if drafts > 0 {
            super::decode::MAX_DECODE_ROWS
        } else {
            runtime.max_batch
        };
        let mut decode = Vec::with_capacity(widest);
        for width in 1..=widest {
            decode.push(if drafts > 0 {
                InklingDecode::with_windows(&model, &stream, width)?
            } else {
                InklingDecode::new(&model, &stream, width)?
            });
        }

        let vision = match runtime.mmproj.as_deref() {
            Some(path) => {
                let failed = |e: String| RuntimeError::Vision(e);
                let projector = GgufFile::open(path).map_err(|e| failed(e.to_string()))?;
                let tower = InklingVisionConfig::from_gguf(&projector, &config)
                    .map_err(|e| failed(e.to_string()))?;
                let weights = InklingVisionWeights::from_gguf(&projector, &tower)
                    .map_err(|e| failed(e.to_string()))?;
                let patches = runtime
                    .max_image_patches
                    .next_multiple_of(tower.merge_factor() as usize)
                    .max(tower.merge_factor() as usize);
                Some(
                    InklingVisionForward::new(&ctx, stream.clone(), &tower, &weights, patches)
                        .map_err(|e| failed(e.to_string()))?,
                )
            }
            None => None,
        };

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
            spec,
            draft_n_min: runtime.draft_n_min,
            window_lens: Vec::with_capacity(widest),
            window_keep: Vec::with_capacity(widest),
            window_chosen: Vec::with_capacity(widest),
            vision,
        })
    }

    pub(crate) fn slots(&self) -> SnapshotSlots {
        self.arena.slots()
    }

    pub(crate) fn admit(
        &mut self,
        req: NewRequest,
        prompt: Vec<i32>,
        images: Vec<SequenceImage>,
        sampling: SamplingParams,
        constraint: Option<Box<ToolConstraint>>,
    ) -> Result<(), RuntimeError> {
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
        let (spans, image_rows) = self.encode_images(&images, prompt.len())?;
        // The whole sequence's cache, reserved for its whole life, and room
        // past its end for the drafts of its last step.
        let drafts = self
            .spec
            .as_ref()
            .map_or(0, SelfSpecFactory::max_draft_tokens);
        let state = InklingState::bounded(
            &self.stream,
            &self.model,
            req.full_seq_len() as usize + drafts,
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
                spec: self.spec.as_ref().map(SelfSpecFactory::new_speculator),
                draft: Vec::with_capacity(drafts),
                images: spans,
                image_rows,
            },
        );
        Ok(())
    }

    /// Encode a request's images into one device buffer, rows in placement
    /// order, and say where each span sits. The one device allocation a
    /// request makes, at admission.
    #[allow(clippy::type_complexity)]
    fn encode_images(
        &mut self,
        images: &[SequenceImage],
        prompt_len: usize,
    ) -> Result<(SmallVec<[(usize, usize, usize); 2]>, Option<CudaSlice<f32>>), RuntimeError> {
        let mut spans = SmallVec::new();
        if images.is_empty() {
            return Ok((spans, None));
        }
        let vision = self.vision.as_mut().ok_or(RuntimeError::VisionNotEnabled)?;
        let placements: SmallVec<[_; 2]> = images.iter().map(|i| i.placement).collect();
        validate_placements(&placements, prompt_len).map_err(RuntimeError::ImagePlacement)?;
        let merge = vision.config().spatial_merge;
        let hidden = vision.config().projection_dim as usize;
        let total: usize = images.iter().map(|i| i.placement.tokens()).sum();
        let rows = self.stream.alloc_zeros::<f32>(total * hidden)?;
        let mut row = 0usize;
        for image in images {
            let (mh, mw) = (image.image.grid_h / merge, image.image.grid_w / merge);
            if (image.placement.grid_h, image.placement.grid_w) != (mh, mw) {
                return Err(RuntimeError::ImagePlacement(format!(
                    "placement grid {}x{} does not match the preprocessed {mh}x{mw}",
                    image.placement.grid_h, image.placement.grid_w
                )));
            }
            // The server preprocessed for this tower: raster patches.
            let pre = PreprocessedInklingImage {
                patches: image.image.patches.clone(),
                grid_w: image.image.grid_w,
                grid_h: image.image.grid_h,
            };
            let out = vision
                .encode(&pre)
                .map_err(|e| RuntimeError::Vision(e.to_string()))?;
            let tokens = image.placement.tokens();
            // SAFETY: both views lie inside their buffers.
            let src = unsafe { subslice(&self.stream, out, 0, tokens * hidden) };
            let mut dst = unsafe { subslice(&self.stream, &rows, row * hidden, tokens * hidden) };
            self.stream.memcpy_dtod(&*src, &mut *dst)?;
            spans.push((image.placement.start, tokens, row));
            row += tokens;
        }
        self.stream.synchronize()?;
        Ok((spans, Some(rows)))
    }

    /// Admit a request whose prompt begins with the prefix `snapshot`
    /// holds: the state is restored and prefill continues from there.
    pub(crate) fn admit_restored(
        &mut self,
        req: NewRequest,
        prompt: Vec<i32>,
        images: Vec<SequenceImage>,
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
        self.admit(req, prompt, images, sampling, constraint)?;
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
        if let Some(spec) = seq.spec.as_mut() {
            spec.observe_all(&seq.prompt[..prefix]);
        }
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
            if let Some(spec) = seq.spec.as_mut() {
                spec.observe(token);
            }
        }
    }

    /// Draft for every sequence of the step, and whether any drafted.
    ///
    /// A draft is cut to what the sequence may still emit, to the rows left
    /// before its next retention boundary (a window may land on one, never
    /// cross it), and dropped when shorter than `draft_n_min`. Then the
    /// longest drafts give up tokens until the step's rows fit.
    fn draft(&mut self) -> bool {
        let widest = self.decode.len();
        let mut rows = self.stepping.len();
        for (_, seq) in &mut self.stepping {
            seq.draft.clear();
            let Some(spec) = seq.spec.as_mut() else {
                continue;
            };
            spec.propose_into(&mut seq.draft);
            let mut room = (seq.max_output - seq.emitted).saturating_sub(1) as usize;
            if self.retention_interval > 0 && !seq.retention_disabled {
                let position = seq.state.position();
                let to_boundary = self.retention_interval - position % self.retention_interval;
                room = room.min(to_boundary - 1);
            }
            seq.draft.truncate(room);
            if seq.draft.len() < self.draft_n_min {
                seq.draft.clear();
            }
            rows += seq.draft.len();
        }
        while rows > widest {
            let longest = self
                .stepping
                .iter_mut()
                .map(|(_, seq)| &mut seq.draft)
                .max_by_key(|draft| draft.len())
                .expect("a step has sequences");
            longest.pop();
            rows -= 1;
        }
        rows > self.stepping.len()
    }

    /// One step over `self.stepping` with at least one draft: every
    /// sequence's next token and its drafts as a window, and from each
    /// window the tokens the model itself chooses, for as long as they are
    /// the drafted ones.
    fn step_windows(
        &mut self,
        generated: &mut SmallVec<[(RequestId, i32); 16]>,
        stopped: &mut SmallVec<[RequestId; 3]>,
        retained: &mut Retained,
    ) -> Result<(), RuntimeError> {
        self.window_lens.clear();
        self.step_tokens.clear();
        for (_, seq) in &self.stepping {
            self.window_lens.push(1 + seq.draft.len());
            self.step_tokens
                .push(seq.next_token.expect("a stepping sequence has a token"));
            self.step_tokens.extend(&seq.draft);
        }
        let rows = self.step_tokens.len();
        let decode = &mut self.decode[rows - 1];
        {
            let mut states: SmallVec<[&mut InklingState; 8]> = self
                .stepping
                .iter_mut()
                .map(|(_, s)| &mut s.state)
                .collect();
            decode.step_windows(
                &self.stream,
                &mut states,
                &self.window_lens,
                &self.step_tokens,
            )?;
        }
        decode.logits(&self.stream)?;

        // Row j of a window is the model after the next token and j
        // drafts. Its choice is emitted; the row after it stands only if
        // that choice is the draft it was computed from.
        self.window_keep.clear();
        self.window_chosen.clear();
        let mut row = 0;
        for (_, seq) in &mut self.stepping {
            let len = 1 + seq.draft.len();
            let mut kept = 0;
            for j in 0..len {
                let token = Self::choose_from(
                    &self.stream,
                    decode.head_logits(),
                    row + j,
                    &mut self.host_logits,
                    &mut self.scratch,
                    seq,
                )?;
                self.window_chosen.push(token);
                kept += 1;
                if j + 1 == len || token != seq.draft[j] || self.eos.contains(&token) {
                    break;
                }
            }
            if let Some(spec) = seq.spec.as_mut()
                && !seq.draft.is_empty()
            {
                spec.accept(kept - 1);
            }
            self.window_keep.push(kept);
            row += len;
        }
        {
            let mut states: SmallVec<[&mut InklingState; 8]> = self
                .stepping
                .iter_mut()
                .map(|(_, s)| &mut s.state)
                .collect();
            decode.commit(&self.stream, &mut states, &self.window_keep)?;
        }
        let mut at = 0;
        for ((id, seq), kept) in self.stepping.iter_mut().zip(&self.window_keep) {
            let chosen = &self.window_chosen[at..at + kept];
            at += kept;
            let last = *chosen.last().expect("a window keeps a row");
            if seq.at_boundary(self.retention_interval) {
                let record = seq.sampler.is_none();
                if let Some(snapshot) = seq.retain(
                    &self.stream,
                    &self.model,
                    &self.arena,
                    self.retention_interval,
                    last,
                    record,
                )? {
                    retained.push((*id, snapshot));
                }
            }
            for token in chosen {
                Self::emit(&self.eos, *id, *token, seq, generated, stopped);
            }
        }
        Ok(())
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
        if self.draft() {
            return self.step_windows(generated, stopped, retained);
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
                {
                    // The image rows this piece covers, in place of their
                    // placeholder tokens.
                    let mut given: SmallVec<[GivenRows<'_>; 2]> = SmallVec::new();
                    if let Some(image_rows) = seq.image_rows.as_ref() {
                        let (lo, hi) = (seq.prefilled, seq.prefilled + rows);
                        for &(start, tokens, first) in &seq.images {
                            let (from, to) = (start.max(lo), (start + tokens).min(hi));
                            if from < to {
                                given.push(GivenRows {
                                    at: from - lo,
                                    from: image_rows,
                                    first: first + (from - start),
                                    rows: to - from,
                                });
                            }
                        }
                    }
                    pass.run_with_rows(
                        &self.stream,
                        &mut [Segment {
                            state: &mut seq.state,
                            rows,
                        }],
                        &seq.prompt[seq.prefilled..seq.prefilled + rows],
                        &given,
                        None,
                    )?;
                }
                if let Some(spec) = seq.spec.as_mut() {
                    spec.observe_all(&seq.prompt[seq.prefilled..seq.prefilled + rows]);
                }
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
