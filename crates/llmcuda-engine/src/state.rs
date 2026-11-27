//! Everything about one sequence that outlives a single forward pass.
//!
//! A forward pass is built for a fixed number of positions — see the module
//! docs on [`crate::block::attention`] for why — so prefill and decode are two
//! differently-shaped [`Forward`](crate::forward::Forward) objects. What makes
//! the second a *continuation* of the first rather than a new sequence is that
//! they share one of these.
//!
//! # What is carried, and what is not
//!
//! | Carried | Where | Size |
//! | --- | --- | --- |
//! | 30 Gated DeltaNet recurrent matrices and convolution windows | [`GdnState`] | fixed, ~4 MiB total |
//! | 10 attention key/value caches | [`KvCache`] | 40 KiB per position |
//! | the write position | [`SequenceState::position`] | — |
//!
//! Activations are *not* carried. They are scratch inside the pass and are
//! overwritten every call; nothing downstream of the residual stream survives
//! a step, which is what makes it safe for two `Forward` objects of different
//! shapes to alternate over one state.
//!
//! # The asymmetry between the two halves
//!
//! Gated DeltaNet state is **constant in sequence length**: the delta rule
//! folds each token into a `[value_heads][head_dim][head_dim]` matrix, so
//! position 100,000 costs exactly what position 1 costs. That is the entire
//! architectural point of the hybrid, and it is why this model can hold a
//! very long context in a cache that a forty-layer dense model would need
//! four times over.
//!
//! Attention state is **linear** in sequence length, and only ten of forty
//! layers have it. See [`KvCache`] for the arithmetic.
//!
//! # Why a reset does not clear the key/value cache
//!
//! [`SequenceState::reset`] zeroes the Gated DeltaNet states and rewinds the
//! position, but leaves the key/value caches alone. That is not an oversight
//! and not a shortcut: the attention kernel reads keys `[0, position]` only,
//! so every slot at or above a rewound position is unreachable until it has
//! been overwritten by the pass that advances back past it. Zeroing 1.25 GiB
//! to hide data that cannot be read would make a reset cost more than the
//! prefill that follows it.
//!
//! The Gated DeltaNet halves are different, and *must* be cleared: they are
//! read unconditionally on the very first token, with no position to bound
//! them. Leaving them would resume the previous prompt's recurrent matrix and
//! its last three convolution taps, which is finite, plausible, and a
//! different sequence.

use std::mem::ManuallyDrop;
use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, PinnedHostSlice};

use llmcuda_model::config::ModelConfig;

use crate::block::attention::{AttentionBlockError, HostKvPrefix, KvCache};
use crate::block::gdn::{GdnBlock, GdnBlockError, GdnState};

/// Something went wrong allocating or resetting sequence state.
#[derive(Debug)]
pub enum StateError {
    /// A Gated DeltaNet state could not be allocated or cleared.
    Gdn(GdnBlockError),
    /// A key/value cache could not be allocated.
    Attention(AttentionBlockError),
    /// The driver rejected a clear.
    Driver(cudarc::driver::DriverError),
    /// A host snapshot belongs to a different model geometry or is too long.
    SnapshotShape,
    /// Every preallocated host snapshot slot is still retained.
    SnapshotArenaExhausted,
    /// The snapshot kept its attention prefix but gave up its recurrent
    /// state, so nothing can resume from it.
    SnapshotStateReleased,
}

impl std::fmt::Display for StateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gdn(e) => write!(f, "Gated DeltaNet state: {e}"),
            Self::Attention(e) => write!(f, "key/value cache: {e}"),
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::SnapshotShape => write!(f, "prefix snapshot does not match sequence geometry"),
            Self::SnapshotArenaExhausted => write!(f, "pinned snapshot arena is exhausted"),
            Self::SnapshotStateReleased => {
                write!(f, "prefix snapshot no longer holds its recurrent state")
            }
        }
    }
}

impl std::error::Error for StateError {}

impl From<GdnBlockError> for StateError {
    fn from(e: GdnBlockError) -> Self {
        Self::Gdn(e)
    }
}

impl From<AttentionBlockError> for StateError {
    fn from(e: AttentionBlockError) -> Self {
        Self::Attention(e)
    }
}

impl From<cudarc::driver::DriverError> for StateError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        Self::Driver(e)
    }
}

/// One sequence's carried state: 30 recurrent states, 10 key/value caches, and
/// the position they are all filled to.
pub struct SequenceState {
    gdn: Vec<GdnState>,
    kv: Vec<KvCache>,
    position: usize,
    /// `position`, on the device, as one `i32`.
    ///
    /// The rotary embedding, the causal bound and the cache append all need
    /// the position, and all three used to take it as a host argument. That
    /// is exactly what stops a decode step from being captured once as a CUDA
    /// graph and replayed: a recorded launch keeps the argument it was
    /// recorded with, so the replay would rotate by, and append at, the
    /// position of the step that was captured — forever. Holding it here and
    /// pushing four bytes before each step makes every step the same launch
    /// sequence. It is also `AGENTS.md` rule 5.
    ///
    /// The host copy stays because the bounds checks are still the host's job
    /// (`position + tokens <= max_seq`); [`Self::publish_position`] is what
    /// keeps the two in step, and it is called on the forward path rather
    /// than by [`Self::advance`] so that a failed pass cannot leave the device
    /// claiming a position no cache was written for.
    ///
    /// Two elements since M-RoPE: `[0]` is the cache-slot / causal position,
    /// `[1]` the rotary base (`position + rope_delta`). For a text-only
    /// sequence `rope_delta` is 0 and the two are equal, which is what keeps
    /// that path byte-identical; after an image the rotary base lags the
    /// slot position because an image advances rope "time" by its longest
    /// grid edge, not its token count (see `llmcuda_kernels::mrope`). One
    /// buffer, one publish — the same four... eight bytes per step.
    d_position: CudaSlice<i32>,
    /// Aliased one-element views of `d_position[0]` / `d_position[1]`,
    /// created once at construction. Kernel wrappers demand exactly-one-
    /// element position buffers, and a captured decode graph binds whichever
    /// pointer it saw at capture — persistent views keep those pointers
    /// stable for the state's whole life (same technique as
    /// `weights::ResidentTensor`).
    slot_position_view: ManuallyDrop<CudaSlice<i32>>,
    rope_position_view: ManuallyDrop<CudaSlice<i32>>,
    /// Rotary-base offset relative to `position`. 0 for text-only; set by
    /// the runtime while prefilling image-bearing sequences and constant
    /// from the end of the prompt onward.
    rope_delta: i32,
    max_seq: usize,
}

pub(crate) struct HostGdnState {
    pub(crate) conv: PinnedHostSlice<f32>,
    pub(crate) recurrent: PinnedHostSlice<f32>,
}

/// One retention interval of attention key/value rows, every layer.
struct SnapshotBuffers {
    attention: Vec<HostKvPrefix>,
}

/// One complete recurrent state, every Gated DeltaNet layer.
///
/// Held apart from [`SnapshotBuffers`] because the two are needed at
/// different densities. Restoring a prefix reads the attention rows of
/// *every* snapshot behind it and the recurrent state of the deepest one
/// only, so a state is worth keeping where a request can resume and nowhere
/// else — while the attention rows have to stay wherever a later resume
/// point depends on them.
struct StateBuffers {
    gdn: Vec<HostGdnState>,
}

struct LeasePool<T> {
    free: Mutex<Vec<T>>,
    capacity: usize,
}

/// How much room a worker's pinned snapshot arena has left.
///
/// The arena is the scarce resource behind prefix sharing: a live sequence
/// that cannot check out a slot stops retaining for the rest of its life, so
/// anything holding snapshots for later has to be able to see how close it is
/// to causing that.
#[derive(Clone)]
pub struct SnapshotSlots {
    intervals: Arc<LeasePool<SnapshotBuffers>>,
    states: Arc<StatePool>,
}

impl SnapshotSlots {
    /// Snapshots a live sequence could take right now: the scarcer of the
    /// two pools, counting a recurrent state another snapshot would give up
    /// on demand as free.
    pub fn available(&self) -> usize {
        self.intervals.available().min(self.states.obtainable())
    }

    /// Attention intervals the arena holds in total.
    pub fn capacity(&self) -> usize {
        self.intervals.capacity
    }

    /// Recurrent states the arena holds in total.
    pub fn state_capacity(&self) -> usize {
        self.states.pool.capacity
    }
}

impl core::fmt::Debug for SnapshotSlots {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SnapshotSlots")
            .field("available", &self.available())
            .field("capacity", &self.capacity())
            .field("state_capacity", &self.state_capacity())
            .finish()
    }
}

/// Whether a snapshot's recurrent state may be given to another snapshot.
///
/// A state starts out held: the snapshot is its sequence's newest, and the
/// place that sequence — or a request continuing it — resumes from. Once the
/// sequence retains a later one, the earlier state is only read by a request
/// that *diverges* inside the following interval, and the runtime marks it
/// [`Self::yield_state`] unless that is likely (the end of a prompt, where
/// the next turn of a conversation forks). A yielded state is still restored
/// from while it lasts; it is simply the first thing to go when the pool is
/// empty. Being resumed from is evidence of a fork point, and holds it again.
struct StateClaim<S> {
    held: Mutex<Held<S>>,
    yielded: std::sync::atomic::AtomicBool,
    /// Set by a resume, and never cleared: it outranks a later yield.
    resumed: std::sync::atomic::AtomicBool,
    order: std::sync::atomic::AtomicU64,
}

impl<S> StateClaim<S> {
    fn present(&self) -> bool {
        self.held
            .lock()
            .expect("snapshot state poisoned")
            .state
            .is_some()
    }

    /// Give the state up, or mark it to go once nothing is waiting on it.
    fn release(&self) {
        let mut held = self.held.lock().expect("snapshot state poisoned");
        if held.waiting == 0 {
            held.state.take();
        } else {
            held.released = true;
        }
    }

    /// One waiting request has restored, or has gone away without doing so.
    fn settle(&self) {
        let mut held = self.held.lock().expect("snapshot state poisoned");
        held.waiting = held.waiting.saturating_sub(1);
        if held.waiting == 0 && held.released {
            held.state.take();
        }
    }

    fn is_yielded(&self) -> bool {
        self.yielded.load(std::sync::atomic::Ordering::Acquire)
            && !self.resumed.load(std::sync::atomic::Ordering::Acquire)
    }
}

/// A claim's state, and who is still waiting to read it.
struct Held<S> {
    state: Option<S>,
    /// Requests that were told they can resume here and have not yet.
    /// Admission and restore are different moments — a request can wait in
    /// its worker's queue between them — and the state must not be released
    /// under one that is waiting.
    waiting: u32,
    /// Released while a request was waiting: goes when the last one is done.
    released: bool,
}

/// The recurrent states of one worker's arena, and which of them may be
/// reassigned.
///
/// Generic over the state so the policy is testable without a device, as
/// `SharedSnapshots` is.
struct ClaimPool<S> {
    pool: Arc<LeasePool<S>>,
    /// Every claim that may still hold a state. Weak, so a dropped snapshot
    /// returns its lease by itself; entries without a state are swept on
    /// the next take.
    claims: Mutex<Vec<std::sync::Weak<StateClaim<Lease<S>>>>>,
    clock: std::sync::atomic::AtomicU64,
}

type StatePool = ClaimPool<StateBuffers>;

impl<S> ClaimPool<S> {
    fn new(states: Vec<S>) -> Arc<Self> {
        Arc::new(Self {
            claims: Mutex::new(Vec::with_capacity(states.len() + 1)),
            pool: LeasePool::new(states),
            clock: std::sync::atomic::AtomicU64::new(0),
        })
    }

    fn tick(&self) -> u64 {
        self.clock.fetch_add(1, std::sync::atomic::Ordering::AcqRel)
    }

    /// States a new snapshot could get: free ones, plus yielded ones.
    fn obtainable(&self) -> usize {
        let yielded = self
            .claims
            .lock()
            .expect("snapshot arena poisoned")
            .iter()
            .filter_map(std::sync::Weak::upgrade)
            .filter(|claim| claim.is_yielded() && claim.present())
            .count();
        self.pool.available() + yielded
    }

    /// A state for a new snapshot: a free one, or else the least recently
    /// used yielded one. `None` when every state is held.
    fn take(self: &Arc<Self>) -> Option<Arc<StateClaim<Lease<S>>>> {
        let lease = self.pool.checkout().or_else(|| self.reassign())?;
        let claim = Arc::new(StateClaim {
            held: Mutex::new(Held {
                state: Some(lease),
                waiting: 0,
                released: false,
            }),
            yielded: std::sync::atomic::AtomicBool::new(false),
            resumed: std::sync::atomic::AtomicBool::new(false),
            order: std::sync::atomic::AtomicU64::new(self.tick()),
        });
        let mut claims = self.claims.lock().expect("snapshot arena poisoned");
        // Only a claim that still holds a state can ever be reassigned, and
        // there are never more of those than states, so the list stays
        // within the capacity it was built with.
        claims.retain(|claim| claim.upgrade().is_some_and(|claim| claim.present()));
        claims.push(Arc::downgrade(&claim));
        drop(claims);
        Some(claim)
    }

    fn reassign(&self) -> Option<Lease<S>> {
        let claims = self.claims.lock().expect("snapshot arena poisoned");
        loop {
            let oldest = claims
                .iter()
                .filter_map(std::sync::Weak::upgrade)
                .filter(|claim| claim.is_yielded() && claim.present())
                .min_by_key(|claim| claim.order.load(std::sync::atomic::Ordering::Acquire))?;
            let mut held = oldest.held.lock().expect("snapshot state poisoned");
            // Re-read under the lock: a resume may have held it since. One
            // that has no longer passes the filter above, so this ends.
            if oldest.is_yielded()
                && held.waiting == 0
                && let Some(lease) = held.state.take()
            {
                return Some(lease);
            }
        }
    }

    /// Hold a state for a request that is about to resume from it, until
    /// that request settles. `false` when it has already gone, and the
    /// caller must look further back.
    fn hold(&self, claim: &StateClaim<Lease<S>>) -> bool {
        let mut held = claim.held.lock().expect("snapshot state poisoned");
        if held.state.is_none() || held.released {
            return false;
        }
        held.waiting += 1;
        claim
            .resumed
            .store(true, std::sync::atomic::Ordering::Release);
        claim
            .order
            .store(self.tick(), std::sync::atomic::Ordering::Release);
        true
    }
}

impl<T> LeasePool<T> {
    fn new(free: Vec<T>) -> Arc<Self> {
        let capacity = free.len();
        Arc::new(Self {
            free: Mutex::new(free),
            capacity,
        })
    }

    /// Slots not currently leased out.
    fn available(&self) -> usize {
        self.free.lock().expect("snapshot arena poisoned").len()
    }

    fn checkout(self: &Arc<Self>) -> Option<Lease<T>> {
        self.free
            .lock()
            .expect("snapshot arena poisoned")
            .pop()
            .map(|value| Lease {
                value: Some(value),
                pool: Arc::clone(self),
            })
    }
}

struct Lease<T> {
    value: Option<T>,
    pool: Arc<LeasePool<T>>,
}

impl<T> Drop for Lease<T> {
    fn drop(&mut self) {
        self.pool
            .free
            .lock()
            .expect("snapshot arena poisoned")
            .push(self.value.take().expect("live lease owns its value"));
    }
}

impl<T> std::ops::Deref for Lease<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value.as_ref().expect("live lease owns its value")
    }
}

impl<T> std::ops::DerefMut for Lease<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.value.as_mut().expect("live lease owns its value")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SnapshotLayout {
    attention_layers: usize,
    /// Positions one snapshot can carry: the retention interval.
    positions: usize,
    /// `u16` words one position occupies in the key and value halves, in
    /// the model's cache formats (q8_0 rows are 17/32 of binary16's).
    k_row: usize,
    v_row: usize,
    gdn_layers: usize,
    conv_elements: usize,
    recurrent_elements: usize,
}

impl SnapshotLayout {
    fn new(config: &ModelConfig, retention_interval: usize) -> Self {
        // `inkling2_loop` has no recurrent layers, and two kinds of cache:
        // a full-attention slot's keys and values grow with the sequence
        // and are the per-interval half; everything of fixed size is the
        // per-state half, one entry a layer application - its four
        // convolution carries and, on a sliding slot, the window of keys
        // and of values its next rows can still reach (binary16, counted
        // here in the f32 words the pinned buffer is made of).
        if let Some(k) = config.inkling.as_ref() {
            let layers = config.num_layers;
            let kv = config.attention.kv_heads as usize * config.attention.head_dim as usize;
            let carry = k.conv_taps as usize - 1;
            return Self {
                attention_layers: k.full_slots(layers) as usize,
                positions: retention_interval,
                k_row: kv,
                v_row: kv,
                gdn_layers: k.slots(layers) as usize,
                conv_elements: carry * 2 * (kv + config.hidden_size as usize),
                recurrent_elements: k.sliding_window as usize * kv,
            };
        }
        let attention_layers = (0..config.num_layers)
            .filter(|&layer| {
                config.layer_kind(layer) == llmcuda_model::config::LayerKind::GatedAttention
            })
            .count();
        let gdn_layers = config.num_layers as usize - attention_layers;
        let kv_dim = config.attention.kv_heads as usize * config.attention.head_dim as usize;
        let formats = crate::block::attention::kv_formats(config);
        let conv_dim = (2 * config.gdn.qk_heads as usize + config.gdn.value_heads as usize)
            * config.gdn.head_dim as usize;
        Self {
            attention_layers,
            positions: retention_interval,
            k_row: formats.k.row_words(kv_dim),
            v_row: formats.v.row_words(kv_dim),
            gdn_layers,
            conv_elements: conv_dim * (config.gdn.conv_kernel as usize - 1),
            recurrent_elements: config.gdn.value_heads as usize
                * config.gdn.head_dim as usize
                * config.gdn.head_dim as usize,
        }
    }

    fn attention_bytes(self) -> usize {
        self.attention_layers * self.positions * (self.k_row + self.v_row) * size_of::<u16>()
    }

    fn gdn_bytes(self) -> usize {
        self.gdn_layers * (self.conv_elements + self.recurrent_elements) * size_of::<f32>()
    }
}

/// Pinned host bytes of one retained attention interval.
///
/// Pure arithmetic over the model configuration, so a caller can turn a
/// memory budget into a slot count — and report what that budget bought —
/// during preflight, before any device is touched.
pub fn snapshot_bytes_per_interval(config: &ModelConfig, retention_interval: usize) -> usize {
    SnapshotLayout::new(config, retention_interval).attention_bytes()
}

/// Pinned host bytes of one retained recurrent state. Independent of the
/// retention interval: the state is the same size at any depth.
pub fn snapshot_bytes_per_state(config: &ModelConfig) -> usize {
    SnapshotLayout::new(config, 1).gdn_bytes()
}

/// How many of each a worker's snapshot arena preallocates.
///
/// Two counts because the two halves of a snapshot are needed at different
/// densities (see [`SnapshotArenaSize::for_budget`]), and at their natural
/// sizes a recurrent state is 1.6 retention intervals of attention rows on
/// the target model — so the split decides what a budget covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SnapshotArenaSize {
    /// Attention intervals: one per retention interval of restorable context.
    pub intervals: usize,
    /// Recurrent states: one per place a request can resume.
    pub states: usize,
}

/// Recurrent states kept per serving sequence. A sequence holds at most
/// three at once that it will not give up — its newest, the end of its
/// prompt, and the point where it left a prefix another request had cached —
/// and the rest of this is the earlier turns and neighbouring conversations
/// a slot serves in rotation.
pub const SNAPSHOT_STATES_PER_SEQUENCE: usize = 8;

impl SnapshotArenaSize {
    /// No retention at all.
    pub const NONE: Self = Self {
        intervals: 0,
        states: 0,
    };

    /// As many intervals as a state for each: the dense arena, where no
    /// snapshot ever gives its state up.
    pub fn dense(snapshots: usize) -> Self {
        Self {
            intervals: snapshots,
            states: snapshots,
        }
    }

    /// The arena that covers `intervals` retention intervals for `sequences`
    /// concurrent sequences.
    pub fn covering(intervals: usize, sequences: usize) -> Self {
        Self {
            intervals,
            states: intervals.min(sequences * SNAPSHOT_STATES_PER_SEQUENCE),
        }
    }

    /// The widest coverage `bytes` buys for `sequences` concurrent sequences.
    ///
    /// States are bought first, up to [`SNAPSHOT_STATES_PER_SEQUENCE`] each,
    /// and never more of them than intervals; what is left is intervals.
    pub fn for_budget(
        bytes: u64,
        sequences: usize,
        bytes_per_interval: usize,
        bytes_per_state: usize,
    ) -> Self {
        let (interval, state) = (bytes_per_interval.max(1) as u64, bytes_per_state as u64);
        let wanted = (sequences * SNAPSHOT_STATES_PER_SEQUENCE) as u64;
        // A state is no use without an interval to hang it on, so a budget
        // too small for `wanted` states holds only as many as it can pair.
        let states = wanted.min(bytes / (interval + state));
        let intervals = (bytes - states * state) / interval;
        Self {
            intervals: intervals as usize,
            states: states as usize,
        }
    }

    /// This size, with neither count above `limit`'s.
    pub fn at_most(self, limit: Self) -> Self {
        Self {
            intervals: self.intervals.min(limit.intervals),
            states: self.states.min(limit.states),
        }
    }

    /// Pinned bytes this size occupies.
    pub fn bytes(self, bytes_per_interval: usize, bytes_per_state: usize) -> u64 {
        self.intervals as u64 * bytes_per_interval as u64
            + self.states as u64 * bytes_per_state as u64
    }
}

/// Fixed-capacity page-locked storage for retained sequence snapshots.
pub(crate) struct SnapshotArena {
    slots: Arc<LeasePool<SnapshotBuffers>>,
    states: Arc<StatePool>,
    layout: SnapshotLayout,
}

impl SnapshotArena {
    pub(crate) fn new(
        ctx: &Arc<CudaContext>,
        config: &ModelConfig,
        retention_interval: usize,
        size: SnapshotArenaSize,
    ) -> Result<Self, StateError> {
        let layout = SnapshotLayout::new(config, retention_interval);
        let mut buffers = Vec::with_capacity(size.intervals);
        for _ in 0..size.intervals {
            let mut attention = Vec::with_capacity(layout.attention_layers);
            for _ in 0..layout.attention_layers {
                // SAFETY: capture initializes the used prefix before publication.
                let k = unsafe { ctx.alloc_pinned(layout.positions * layout.k_row)? };
                let v = unsafe { ctx.alloc_pinned(layout.positions * layout.v_row)? };
                attention.push(HostKvPrefix { k, v });
            }
            buffers.push(SnapshotBuffers { attention });
        }
        let mut states = Vec::with_capacity(size.states);
        for _ in 0..size.states {
            let mut gdn = Vec::with_capacity(layout.gdn_layers);
            for _ in 0..layout.gdn_layers {
                // SAFETY: capture initializes both buffers before publication.
                let conv = unsafe { ctx.alloc_pinned(layout.conv_elements)? };
                let recurrent = unsafe { ctx.alloc_pinned(layout.recurrent_elements)? };
                gdn.push(HostGdnState { conv, recurrent });
            }
            states.push(StateBuffers { gdn });
        }
        Ok(Self {
            slots: LeasePool::new(buffers),
            states: ClaimPool::new(states),
            layout,
        })
    }

    /// Attention intervals this arena holds.
    pub(crate) fn capacity(&self) -> usize {
        self.slots.capacity
    }

    /// Recurrent states this arena holds.
    pub(crate) fn state_capacity(&self) -> usize {
        self.states.pool.capacity
    }

    /// Pinned bytes of one attention interval.
    pub(crate) fn bytes_per_interval(&self) -> usize {
        self.layout.attention_bytes()
    }

    /// Pinned bytes of one recurrent state.
    pub(crate) fn bytes_per_state(&self) -> usize {
        self.layout.gdn_bytes()
    }

    /// A snapshot at `position` whose buffers `fill` writes: the interval
    /// `[start, position)` of every growing cache (`start` is the parent's
    /// position, or zero) and the fixed-size state. For a sequence type
    /// other than [`SequenceState`], which captures itself.
    ///
    /// `fill` must have its copies complete before it returns.
    pub(crate) fn capture(
        &self,
        position: usize,
        parent: Option<Arc<SequenceSnapshot>>,
        fill: impl FnOnce(usize, &mut [HostKvPrefix], &mut [HostGdnState]) -> Result<(), StateError>,
    ) -> Result<SequenceSnapshot, StateError> {
        let start = parent.as_ref().map_or(0, |snapshot| snapshot.position);
        if start > position || position - start > self.layout.positions {
            return Err(StateError::SnapshotShape);
        }
        let mut buffers = self
            .slots
            .checkout()
            .ok_or(StateError::SnapshotArenaExhausted)?;
        let claim = self
            .states
            .take()
            .ok_or(StateError::SnapshotArenaExhausted)?;
        {
            // Nothing else can reach this claim yet; see `snapshot`.
            let mut held = claim.held.lock().expect("snapshot state poisoned");
            let host = held.state.as_mut().expect("a fresh claim holds its state");
            fill(start, &mut buffers.attention, &mut host.gdn)?;
        }
        Ok(SequenceSnapshot {
            position,
            start,
            row_words: self.layout.k_row + self.layout.v_row,
            buffers,
            state: claim,
            states: Arc::clone(&self.states),
            gdn_bytes: self.layout.gdn_bytes(),
            next_token: None,
            parent,
        })
    }

    /// A view of this arena's free-slot count, shareable with the engine
    /// thread.
    ///
    /// Handed out rather than answered over the runtime's command channel so
    /// the engine can ask without waiting on the GPU thread, which is the
    /// thread it would be asking on behalf of.
    pub(crate) fn slots(&self) -> SnapshotSlots {
        SnapshotSlots {
            intervals: Arc::clone(&self.slots),
            states: Arc::clone(&self.states),
        }
    }
}

/// What [`SequenceSnapshot::read`] calls for each interval: its first
/// position, its end, and its rows for every growing cache.
pub(crate) type IntervalVisitor<'a> =
    dyn FnMut(usize, usize, &[HostKvPrefix]) -> Result<(), StateError> + 'a;

/// A complete resumable prefix, stored in page-locked host memory.
///
/// Attention and GDN retain their natural geometries: attention contains only
/// `position` prefix positions, while each GDN layer contains its fixed-size
/// convolution and recurrent state. No group is padded to the other's size.
pub struct SequenceSnapshot {
    position: usize,
    start: usize,
    /// `u16` words per position across both halves of one layer.
    row_words: usize,
    buffers: Lease<SnapshotBuffers>,
    /// The recurrent state at `position`, until it is given up.
    state: Arc<StateClaim<Lease<StateBuffers>>>,
    states: Arc<StatePool>,
    gdn_bytes: usize,
    next_token: Option<i32>,
    parent: Option<Arc<SequenceSnapshot>>,
}

impl SequenceSnapshot {
    pub fn position(&self) -> usize {
        self.position
    }

    /// Read this snapshot back: `interval(start, end, rows)` for each
    /// interval of its chain, oldest first, then `state` on the fixed-size
    /// half. The state's lock is held throughout, so `state` must have its
    /// copies complete before it returns - the pinned buffers may be handed
    /// to another snapshot the moment the lock drops.
    pub(crate) fn read(
        &self,
        interval: &mut IntervalVisitor<'_>,
        state: impl FnOnce(&[HostGdnState]) -> Result<(), StateError>,
    ) -> Result<(), StateError> {
        let held = self.state.held.lock().expect("snapshot state poisoned");
        let host = held
            .state
            .as_ref()
            .ok_or(StateError::SnapshotStateReleased)?;
        self.visit_intervals(interval)?;
        state(&host.gdn)
    }

    fn visit_intervals(&self, interval: &mut IntervalVisitor<'_>) -> Result<(), StateError> {
        if let Some(parent) = self.parent.as_deref() {
            parent.visit_intervals(interval)?;
        }
        interval(self.start, self.position, &self.buffers.attention)
    }

    /// Target-model prediction produced after computing this exact prefix.
    pub fn next_token(&self) -> Option<i32> {
        self.next_token
    }

    pub(crate) fn set_next_token(&mut self, token: i32) {
        self.next_token = Some(token);
    }

    pub fn attention_bytes(&self) -> usize {
        let own = self.buffers.attention.len()
            * (self.position - self.start)
            * self.row_words
            * size_of::<u16>();
        own + self
            .parent
            .as_ref()
            .map_or(0, |parent| parent.attention_bytes())
    }

    /// Pinned bytes of recurrent state this snapshot holds: one state, or
    /// nothing once it has given the state up.
    pub fn gdn_bytes(&self) -> usize {
        if self.has_state() { self.gdn_bytes } else { 0 }
    }

    /// Whether a request can still resume here.
    pub fn has_state(&self) -> bool {
        self.state.present()
    }

    /// Hold this snapshot's state as a fork point without waiting on it:
    /// the arena will not reassign it, and nothing is owed.
    pub fn keep_state(&self) {
        if self.states.hold(&self.state) {
            self.state.settle();
        }
    }

    /// Let this snapshot's recurrent state be reassigned once the arena has
    /// no free one. Its attention rows stay: later snapshots restore through
    /// them.
    pub(crate) fn yield_state(&self) {
        self.state
            .yielded
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// Give the recurrent state back now. For a snapshot nothing will resume
    /// from again — one dropped from the shared cache — whose attention rows
    /// a descendant may still keep alive.
    pub fn release_state(&self) {
        self.state.release();
    }
}

/// A snapshot whose recurrent state is held for one request to resume from.
///
/// Created where the request is told it can resume, dropped once it has —
/// or once it never will, because it was cancelled or refused first.
pub struct ResumeHold(Arc<SequenceSnapshot>);

impl ResumeHold {
    /// Hold `snapshot`'s recurrent state for one request, so it can be
    /// neither reassigned nor released before that request restores it.
    /// `None` when the state has already gone, and the caller must look
    /// further back.
    pub fn new(snapshot: Arc<SequenceSnapshot>) -> Option<Self> {
        snapshot
            .states
            .hold(&snapshot.state)
            .then_some(Self(snapshot))
    }

    pub fn snapshot(&self) -> &Arc<SequenceSnapshot> {
        &self.0
    }
}

impl Drop for ResumeHold {
    fn drop(&mut self) {
        self.0.state.settle();
    }
}

impl SequenceState {
    /// Allocate state for one sequence of up to `max_seq` positions.
    ///
    /// `gdn_layers` and `attention_layers` are counted from `config` rather
    /// than passed, so a state can never be built with the wrong number of
    /// either for the model it will be run against.
    pub fn new(
        stream: &Arc<CudaStream>,
        gdn: &GdnBlock,
        config: &ModelConfig,
        max_seq: usize,
    ) -> Result<Self, StateError> {
        let (mut gdn_states, mut kv) = (Vec::new(), Vec::new());
        for layer in 0..config.num_layers {
            match config.layer_kind(layer) {
                llmcuda_model::config::LayerKind::GatedDeltaNet => {
                    gdn_states.push(gdn.state(stream)?);
                }
                llmcuda_model::config::LayerKind::GatedAttention => {
                    kv.push(KvCache::new(stream, config, max_seq)?);
                }
            }
        }
        let d_position = stream.alloc_zeros::<i32>(2)?;
        // SAFETY: both views are inside the two-element buffer just
        // allocated, and live exactly as long as the struct that owns it.
        let slot_position_view = unsafe { crate::viewslice::subslice(stream, &d_position, 0, 1) };
        let rope_position_view = unsafe { crate::viewslice::subslice(stream, &d_position, 1, 1) };
        Ok(Self {
            gdn: gdn_states,
            kv,
            position: 0,
            d_position,
            slot_position_view,
            rope_position_view,
            rope_delta: 0,
            max_seq,
        })
    }

    /// How many positions have been written.
    ///
    /// This is where the next pass appends, and it is the `pos_offset` the
    /// rotary embedding rotates by — so a state at position 0 and a state at
    /// position 5,000 produce genuinely different attention for the same
    /// token, which is the point.
    pub fn position(&self) -> usize {
        self.position
    }

    /// The longest sequence this state was allocated for.
    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    /// Device bytes held by everything in here.
    pub fn bytes(&self) -> u64 {
        let kv: u64 = self.kv.iter().map(KvCache::bytes).sum();
        let gdn: u64 = self
            .gdn
            .iter()
            .map(|s| ((s.conv.len() + s.recurrent.len()) * size_of::<f32>()) as u64)
            .sum();
        kv + gdn
    }

    /// Return to a cold start: zero the recurrent state and rewind to 0.
    ///
    /// See the module docs on why the key/value caches are left as they are.
    /// The zeroing is a `cuMemsetD8Async` per state, not a reallocation.
    pub fn reset(&mut self, stream: &Arc<CudaStream>) -> Result<(), StateError> {
        for state in &mut self.gdn {
            stream.memset_zeros(&mut state.conv)?;
            stream.memset_zeros(&mut state.recurrent)?;
        }
        self.position = 0;
        self.rope_delta = 0;
        stream.memset_zeros(&mut self.d_position)?;
        Ok(())
    }

    /// Copy the filled prefix and recurrent state into page-locked host RAM.
    pub(crate) fn snapshot(
        &self,
        stream: &Arc<CudaStream>,
        arena: &SnapshotArena,
        parent: Option<Arc<SequenceSnapshot>>,
    ) -> Result<SequenceSnapshot, StateError> {
        let start = parent.as_ref().map_or(0, |snapshot| snapshot.position);
        if start > self.position {
            return Err(StateError::SnapshotShape);
        }
        if self.position - start > arena.layout.positions
            || self
                .kv
                .first()
                .is_some_and(|cache| cache.row_words() != (arena.layout.k_row, arena.layout.v_row))
        {
            return Err(StateError::SnapshotShape);
        }
        let mut buffers = arena
            .slots
            .checkout()
            .ok_or(StateError::SnapshotArenaExhausted)?;
        let claim = arena
            .states
            .take()
            .ok_or(StateError::SnapshotArenaExhausted)?;
        if buffers.attention.len() != self.kv.len() {
            return Err(StateError::SnapshotShape);
        }
        for (cache, prefix) in self.kv.iter().zip(&mut buffers.attention) {
            cache.snapshot_range_into(stream, start, self.position, prefix)?;
        }
        {
            // Nothing else can reach this claim yet: it was created above and
            // has not been published, and only a yielded claim is reassigned.
            let mut held = claim.held.lock().expect("snapshot state poisoned");
            let host = held.state.as_mut().expect("a fresh claim holds its state");
            if host.gdn.len() != self.gdn.len() {
                return Err(StateError::SnapshotShape);
            }
            for (state, host) in self.gdn.iter().zip(&mut host.gdn) {
                stream.memcpy_dtoh(&state.conv, &mut host.conv)?;
                stream.memcpy_dtoh(&state.recurrent, &mut host.recurrent)?;
            }
            stream.synchronize()?;
        }
        Ok(SequenceSnapshot {
            position: self.position,
            start,
            row_words: arena.layout.k_row + arena.layout.v_row,
            buffers,
            state: claim,
            states: Arc::clone(&arena.states),
            gdn_bytes: arena.layout.gdn_bytes(),
            next_token: None,
            parent,
        })
    }

    /// Restore a snapshot created on this or another CUDA context.
    pub fn restore(
        &mut self,
        stream: &Arc<CudaStream>,
        snapshot: &SequenceSnapshot,
    ) -> Result<(), StateError> {
        if snapshot.position > self.max_seq || snapshot.buffers.attention.len() != self.kv.len() {
            return Err(StateError::SnapshotShape);
        }
        // The lock is held across the copies and the synchronize that
        // completes them, so the pinned buffers cannot be handed to another
        // snapshot while the driver is still reading them.
        let held = snapshot.state.held.lock().expect("snapshot state poisoned");
        let host = held
            .state
            .as_ref()
            .ok_or(StateError::SnapshotStateReleased)?;
        if host.gdn.len() != self.gdn.len() {
            return Err(StateError::SnapshotShape);
        }
        self.restore_attention_chain(stream, snapshot)?;
        for (state, host) in self.gdn.iter_mut().zip(&host.gdn) {
            if state.conv.len() != host.conv.len() || state.recurrent.len() != host.recurrent.len()
            {
                return Err(StateError::SnapshotShape);
            }
            stream.memcpy_htod(&host.conv, &mut state.conv)?;
            stream.memcpy_htod(&host.recurrent, &mut state.recurrent)?;
        }
        self.position = snapshot.position;
        self.publish_position(stream)?;
        stream.synchronize()?;
        Ok(())
    }

    fn restore_attention_chain(
        &mut self,
        stream: &Arc<CudaStream>,
        snapshot: &SequenceSnapshot,
    ) -> Result<(), StateError> {
        if let Some(parent) = snapshot.parent.as_deref() {
            self.restore_attention_chain(stream, parent)?;
        }
        for (cache, prefix) in self.kv.iter_mut().zip(&snapshot.buffers.attention) {
            cache.restore_prefix(stream, prefix, snapshot.start, snapshot.position)?;
        }
        Ok(())
    }

    /// Number of Gated DeltaNet layers this state carries.
    pub fn gdn_layers(&self) -> usize {
        self.gdn.len()
    }

    /// Number of attention layers this state carries.
    pub fn attention_layers(&self) -> usize {
        self.kv.len()
    }

    /// The `slot`-th Gated DeltaNet state, counting only GDN layers.
    pub(crate) fn gdn_mut(&mut self, slot: usize) -> &mut GdnState {
        &mut self.gdn[slot]
    }

    /// The `slot`-th key/value cache and the device position together.
    ///
    /// Two accessors would be two borrows of `self`, one of them mutable, in
    /// one expression. They are disjoint fields, so one call that splits them
    /// is the borrow checker's answer rather than a workaround.
    pub(crate) fn kv_and_position_mut(&mut self, slot: usize) -> (&mut KvCache, &CudaSlice<i32>) {
        (&mut self.kv[slot], &self.slot_position_view)
    }

    /// As [`Self::kv_and_position_mut`], plus the rotary-base scalar.
    ///
    /// The slot scalar drives the cache append and the causal bound; the
    /// rotary scalar drives RoPE. They differ only for image-bearing
    /// sequences (`rope_delta != 0`).
    pub(crate) fn kv_and_positions_mut(
        &mut self,
        slot: usize,
    ) -> (&mut KvCache, &CudaSlice<i32>, &CudaSlice<i32>) {
        (
            &mut self.kv[slot],
            &self.slot_position_view,
            &self.rope_position_view,
        )
    }

    /// The rotary-base offset relative to [`Self::position`].
    pub fn rope_delta(&self) -> i32 {
        self.rope_delta
    }

    /// Set the rotary-base offset. Takes effect at the next
    /// [`Self::publish_position`].
    pub(crate) fn set_rope_delta(&mut self, delta: i32) {
        self.rope_delta = delta;
    }

    /// Copy the host position — slot and rotary base — to the device.
    ///
    /// Eight bytes, once per pass, before anything reads them.
    pub(crate) fn publish_position(&mut self, stream: &Arc<CudaStream>) -> Result<(), StateError> {
        let pos = self.position as i32;
        stream.memcpy_htod(&[pos, pos + self.rope_delta], &mut self.d_position)?;
        Ok(())
    }

    /// Record that `tokens` more positions have been written.
    ///
    /// Called by the forward pass after every block has appended, never by a
    /// caller: advancing early would make a mid-pass failure leave the state
    /// claiming positions that no cache actually holds.
    pub(crate) fn advance(&mut self, tokens: usize) {
        self.position += tokens;
    }
}

#[cfg(test)]
mod snapshot_tests {
    use super::*;

    #[test]
    fn default_retention_slot_keeps_natural_groups_separate() {
        let layout = SnapshotLayout::new(&ModelConfig::qwen3_6_35b_a3b(), 2048);
        assert_eq!(layout.attention_bytes(), 40 * 1024 * 1024);
        assert_eq!(layout.gdn_bytes(), 65_863_680);
        assert_eq!(layout.attention_bytes() + layout.gdn_bytes(), 107_806_720);
    }

    #[test]
    fn q8_kv_snapshots_shrink_with_the_cache() {
        use llmcuda_model::{KvCacheType, KvCacheTypes};
        let f16 = ModelConfig::k2_horizon_36b_a4b();
        // 48 layers x 2048 positions x 2 halves x 2 KiB.
        assert_eq!(snapshot_bytes_per_interval(&f16, 2048), 384 << 20);
        assert_eq!(snapshot_bytes_per_state(&f16), 0);
        let q8 = ModelConfig {
            kv_cache: KvCacheTypes {
                k: KvCacheType::Q8_0,
                v: KvCacheType::Q8_0,
            },
            ..f16
        };
        // The same rows at 1088 bytes per half, 17/32 of binary16.
        assert_eq!(snapshot_bytes_per_interval(&q8, 2048), 48 * 2048 * 2 * 1088);
    }

    #[test]
    fn a_budget_buys_states_for_its_sequences_and_intervals_with_the_rest() {
        let config = ModelConfig::qwen3_6_35b_a3b();
        let interval = snapshot_bytes_per_interval(&config, 2048);
        let state = snapshot_bytes_per_state(&config);
        assert_eq!((interval, state), (40 << 20, 65_863_680));

        // 8 GiB for three sequences: 24 states, and 167 intervals where the
        // same bytes held 79 when every interval carried a state of its own.
        let size = SnapshotArenaSize::for_budget(8 << 30, 3, interval, state);
        assert_eq!(size.states, 3 * SNAPSHOT_STATES_PER_SEQUENCE);
        assert_eq!(size.intervals, 167);
        assert!(size.bytes(interval, state) <= 8 << 30);
        assert_eq!((8u64 << 30) / (interval + state) as u64, 79);

        // Too little for every sequence's states: as many as the budget
        // holds with an interval each, never more states than intervals.
        let small = SnapshotArenaSize::for_budget(1 << 30, 3, interval, state);
        assert_eq!(
            small,
            SnapshotArenaSize {
                intervals: 11,
                states: 9
            }
        );

        // Full coverage of a context stops at the states its sequences use.
        assert_eq!(
            SnapshotArenaSize::covering(198, 3),
            SnapshotArenaSize {
                intervals: 198,
                states: 24
            }
        );
        assert_eq!(
            SnapshotArenaSize::covering(6, 3),
            SnapshotArenaSize::dense(6)
        );
    }

    #[test]
    fn a_yielded_state_is_reassigned_only_when_none_is_free() {
        let pool = ClaimPool::new(vec![1u32, 2]);
        let first = pool.take().expect("a free state");
        let second = pool.take().expect("the other free state");
        assert!(pool.take().is_none(), "both are held");
        assert_eq!(pool.obtainable(), 0);

        first
            .yielded
            .store(true, std::sync::atomic::Ordering::Release);
        assert_eq!(pool.obtainable(), 1);
        assert!(first.present(), "yielding does not give the state up");

        let third = pool.take().expect("the yielded state is reassigned");
        assert!(!first.present());
        assert!(second.present() && third.present());
    }

    #[test]
    fn the_least_recently_used_yielded_state_goes_first() {
        let pool = ClaimPool::new(vec![1u32, 2, 3]);
        let claims: Vec<_> = (0..3).map(|_| pool.take().expect("free")).collect();
        for claim in &claims {
            claim
                .yielded
                .store(true, std::sync::atomic::Ordering::Release);
        }
        // Resuming from the oldest holds it again and makes it the newest.
        assert!(pool.hold(&claims[0]));
        let _fourth = pool.take().expect("a yielded state");
        assert!(claims[0].present() && !claims[1].present() && claims[2].present());
    }

    #[test]
    fn a_resumed_state_outlives_a_later_yield() {
        let pool = ClaimPool::new(vec![1u32]);
        let first = pool.take().expect("free");
        assert!(pool.hold(&first));
        first
            .yielded
            .store(true, std::sync::atomic::Ordering::Release);
        assert!(pool.take().is_none(), "a fork point is not reassigned");
        assert!(first.present());
    }

    #[test]
    fn a_state_released_under_a_waiting_request_goes_when_it_settles() {
        let pool = ClaimPool::new(vec![1u32]);
        let claim = pool.take().expect("free");
        assert!(pool.hold(&claim));
        claim.release();
        assert!(claim.present(), "a queued request still has to restore it");
        assert!(!pool.hold(&claim), "but no new request may start waiting");
        claim.settle();
        assert!(!claim.present());
        assert_eq!(pool.pool.available(), 1);
    }

    #[test]
    fn a_state_that_has_gone_cannot_be_held() {
        let pool = ClaimPool::new(vec![1u32]);
        let first = pool.take().expect("free");
        first
            .yielded
            .store(true, std::sync::atomic::Ordering::Release);
        let _second = pool.take().expect("reassigned");
        assert!(!pool.hold(&first));
    }

    #[test]
    fn the_claim_list_never_outgrows_the_pool() {
        let pool = ClaimPool::new(vec![1u32, 2]);
        for _ in 0..100 {
            // One long-lived state and a stream of short-lived ones, as a
            // sequence's fork point and its passing boundaries are.
            drop(pool.take().expect("free"));
        }
        let kept = pool.take().expect("free");
        for _ in 0..100 {
            let passing = pool.take().expect("free");
            passing.release();
        }
        assert!(kept.present());
        assert!(pool.claims.lock().unwrap().len() <= 3);
    }

    #[test]
    fn dropping_a_snapshot_returns_its_state() {
        let pool = ClaimPool::new(vec![1u32]);
        drop(pool.take().expect("free"));
        assert_eq!(pool.pool.available(), 1);
        assert!(pool.take().is_some());
    }

    #[test]
    fn a_lease_returns_its_slot_only_after_drop() {
        let pool = LeasePool::new(vec![7u32]);
        let lease = pool.checkout().expect("one slot available");
        assert!(pool.checkout().is_none());
        assert_eq!(*lease, 7);
        drop(lease);
        assert_eq!(*pool.checkout().expect("slot returned"), 7);
    }
}
