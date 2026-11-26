//! Launchers for the `inkling2_loop` kernels (Inkling 2 Mini).
//!
//! Every kernel here is Rust (`device-kernels/src/inkling.rs`); the
//! contractions run on [`super::hgemm`]. The CPU reference is
//! `llmcuda_kernels::inkling`, and each method names the piece of it that it
//! computes. Shapes are row-major `[tokens][width]` unless said otherwise.
//!
//! Nothing here sizes a launch from routed data: grids come from the pass's
//! token count and the dispatch capacity, and a tile no expert owns exits on
//! the device ([`InkDispatch`]).

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};

/// Slots per dispatch tile: the token tile of the fp16 GEMM.
pub const INK_TILE: usize = super::hgemm::HGEMM_BM;

/// The most residual states [`InklingKernels::mix`] blends.
pub const MIX_MAX_STATES: usize = 8;
/// The most queries [`InklingKernels::mix`] answers in one launch.
pub const MIX_MAX_QUERIES: usize = 4;

/// Errors from the Inkling kernels.
#[derive(Debug)]
pub enum InklingError {
    /// The driver refused a load or a launch.
    Driver(DriverError),
    /// A buffer is shorter than the launch shape reads or writes.
    Length {
        /// Which operand.
        what: &'static str,
        /// Elements required.
        need: usize,
        /// Elements supplied.
        got: usize,
    },
    /// A shape the kernels are not built for.
    Shape(&'static str),
}

impl core::fmt::Display for InklingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Driver(e) => write!(f, "driver: {e}"),
            Self::Length { what, need, got } => {
                write!(f, "{what}: {got} elements, the launch needs {need}")
            }
            Self::Shape(s) => write!(f, "unsupported shape: {s}"),
        }
    }
}

impl core::error::Error for InklingError {}

impl From<DriverError> for InklingError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

fn need(what: &'static str, got: usize, need: usize) -> Result<(), InklingError> {
    if got < need {
        return Err(InklingError::Length { what, need, got });
    }
    Ok(())
}

fn cfg(grid: (usize, usize, usize), block: u32, shared: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (grid.0 as u32, grid.1 as u32, grid.2 as u32),
        block_dim: (block, 1, 1),
        shared_mem_bytes: shared,
    }
}

/// Dynamic shared memory of the block reductions: one f32 per warp.
const REDUCE_SHARED: u32 = 8 * 4;

/// Device-side grouping of `(token, expert)` pairs into expert-owned tiles.
///
/// `sorted` lists pair ids by expert, each expert's run padded to whole
/// [`INK_TILE`]-slot tiles with the sentinel `pairs`; `owners[tile]` is the
/// expert a tile belongs to, or negative. `slot_of[pair]` is where the pair
/// landed. The capacity is the natural tile count plus one padded run per
/// expert, so no routing can overflow it.
pub struct InkDispatch {
    counts: CudaSlice<i32>,
    starts: CudaSlice<i32>,
    sorted: CudaSlice<i32>,
    owners: CudaSlice<i32>,
    slot_of: CudaSlice<i32>,
    tokens: usize,
    topk: usize,
    experts: usize,
}

impl InkDispatch {
    /// Allocate for `tokens` rows routed to `topk` of `experts` experts.
    pub fn new(
        stream: &Arc<CudaStream>,
        tokens: usize,
        topk: usize,
        experts: usize,
    ) -> Result<Self, InklingError> {
        // `k2_dispatch_prefix` runs one thread an expert in a 128-thread block.
        if tokens == 0 || topk == 0 || experts == 0 || experts > 128 || topk > experts {
            return Err(InklingError::Shape("dispatch needs 1..=128 experts"));
        }
        let capacity = (tokens * topk).div_ceil(INK_TILE) + experts;
        Ok(Self {
            counts: stream.alloc_zeros::<i32>(experts * 128)?,
            starts: stream.alloc_zeros::<i32>(experts)?,
            sorted: stream.alloc_zeros::<i32>(capacity * INK_TILE)?,
            owners: stream.alloc_zeros::<i32>(capacity)?,
            slot_of: stream.alloc_zeros::<i32>(tokens * topk)?,
            tokens,
            topk,
            experts,
        })
    }

    /// Tiles the launch grids cover.
    pub fn tiles(&self) -> usize {
        self.owners.len()
    }

    /// Slots: `tiles * INK_TILE`.
    pub fn slots(&self) -> usize {
        self.sorted.len()
    }

    /// The expert each tile belongs to, negative where none.
    pub fn owners(&self) -> &CudaSlice<i32> {
        &self.owners
    }

    /// Pair ids by slot.
    pub fn sorted(&self) -> &CudaSlice<i32> {
        &self.sorted
    }

    /// Slot by pair id.
    pub fn slot_of(&self) -> &CudaSlice<i32> {
        &self.slot_of
    }
}

/// The compiled kernels.
pub struct InklingKernels {
    embed: CudaFunction,
    linear: CudaFunction,
    mix: CudaFunction,
    ctrl_pair: CudaFunction,
    ctrl_apply: CudaFunction,
    conv: CudaFunction,
    clip: CudaFunction,
    rms_groups: CudaFunction,
    rel_table: CudaFunction,
    attn: CudaFunction,
    gather_half: CudaFunction,
    situ_half: CudaFunction,
    moe_combine: CudaFunction,
    softcap: CudaFunction,
    dispatch_count: CudaFunction,
    dispatch_prefix: CudaFunction,
    dispatch_gather: CudaFunction,
}

impl InklingKernels {
    /// Load the entry points.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, InklingError> {
        let m = ctx.load_module(cudarc::nvrtc::Ptx::from_src(super::RUST_KERNELS_PTX))?;
        Ok(Self {
            embed: m.load_function("ink_embed")?,
            linear: m.load_function("ink_linear")?,
            mix: m.load_function("ink_mix")?,
            ctrl_pair: m.load_function("ink_ctrl_pair")?,
            ctrl_apply: m.load_function("ink_ctrl_apply")?,
            conv: m.load_function("ink_conv")?,
            clip: m.load_function("ink_clip")?,
            rms_groups: m.load_function("ink_rms_groups")?,
            rel_table: m.load_function("ink_rel_table")?,
            attn: m.load_function("ink_attn")?,
            gather_half: m.load_function("ink_gather_half")?,
            situ_half: m.load_function("ink_situ_half")?,
            moe_combine: m.load_function("ink_moe_combine")?,
            softcap: m.load_function("ink_softcap")?,
            dispatch_count: m.load_function("k2_dispatch_count")?,
            dispatch_prefix: m.load_function("k2_dispatch_prefix")?,
            dispatch_gather: m.load_function("k2_dispatch_gather")?,
        })
    }

    /// Embedding rows of a Q8_0 or bf16 table as fp32.
    #[allow(clippy::too_many_arguments)]
    pub fn embed(
        &self,
        stream: &Arc<CudaStream>,
        table: &CudaSlice<u8>,
        ids: &CudaSlice<i32>,
        out: &mut CudaSlice<f32>,
        hidden: usize,
        tokens: usize,
        bf16: bool,
    ) -> Result<(), InklingError> {
        need("embed ids", ids.len(), tokens)?;
        need("embed out", out.len(), tokens * hidden)?;
        let (h, f) = (hidden as i32, i32::from(bf16));
        let mut b = stream.launch_builder(&self.embed);
        b.arg(table).arg(ids).arg(&mut *out).arg(&h).arg(&f);
        // SAFETY: lengths checked; the caller's ids name rows of the table.
        unsafe { b.launch(cfg((hidden.div_ceil(256), tokens, 1), 256, 0)) }?;
        Ok(())
    }

    /// `out[t][j] = w[j] . x[t]` in fp32 (`inkling::project`).
    #[allow(clippy::too_many_arguments)]
    pub fn linear(
        &self,
        stream: &Arc<CudaStream>,
        w: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        k: usize,
        ldx: usize,
        tokens: usize,
    ) -> Result<(), InklingError> {
        need("linear w", w.len(), rows * k)?;
        need("linear x", x.len(), tokens.saturating_sub(1) * ldx + k)?;
        need("linear out", out.len(), tokens * rows)?;
        let (r, k_i, l) = (rows as i32, k as i32, ldx as i32);
        let mut b = stream.launch_builder(&self.linear);
        b.arg(w).arg(x).arg(&mut *out).arg(&r).arg(&k_i).arg(&l);
        // SAFETY: lengths checked against the grid.
        unsafe { b.launch(cfg((rows.div_ceil(128), tokens, 1), 128, 0)) }?;
        Ok(())
    }

    /// Depth mixing (`inkling::mix`): `stack` is `[states][tokens][n]`,
    /// `queries` `[n_queries][n]`, `out` `[n_queries][tokens][n]`.
    #[allow(clippy::too_many_arguments)]
    pub fn mix(
        &self,
        stream: &Arc<CudaStream>,
        stack: &CudaSlice<f32>,
        queries: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n: usize,
        states: usize,
        n_queries: usize,
        tokens: usize,
    ) -> Result<(), InklingError> {
        if states == 0 || states > MIX_MAX_STATES || n_queries == 0 || n_queries > MIX_MAX_QUERIES {
            return Err(InklingError::Shape(
                "mix takes 1..=8 states and 1..=4 queries",
            ));
        }
        need("mix stack", stack.len(), states * tokens * n)?;
        need("mix queries", queries.len(), n_queries * n)?;
        need("mix out", out.len(), n_queries * tokens * n)?;
        let (n_i, s, q, t) = (n as i32, states as i32, n_queries as i32, tokens as i32);
        let mut b = stream.launch_builder(&self.mix);
        b.arg(stack)
            .arg(queries)
            .arg(&mut *out)
            .arg(&n_i)
            .arg(&s)
            .arg(&q)
            .arg(&t);
        // SAFETY: lengths checked; 256 threads and one f32 of dynamic shared
        // memory per warp, as the reductions require.
        unsafe { b.launch(cfg((tokens, 1, 1), 256, REDUCE_SHARED)) }?;
        Ok(())
    }

    /// A controller's gate input: normalized `cur` and `msg` side by side,
    /// binary16, `[tokens][2 * n]`.
    pub fn ctrl_pair(
        &self,
        stream: &Arc<CudaStream>,
        cur: &CudaSlice<f32>,
        msg: &CudaSlice<f32>,
        out: &mut CudaSlice<u16>,
        n: usize,
        tokens: usize,
    ) -> Result<(), InklingError> {
        need("ctrl_pair cur", cur.len(), tokens * n)?;
        need("ctrl_pair msg", msg.len(), tokens * n)?;
        need("ctrl_pair out", out.len(), tokens * 2 * n)?;
        let n_i = n as i32;
        let mut b = stream.launch_builder(&self.ctrl_pair);
        b.arg(cur).arg(msg).arg(&mut *out).arg(&n_i);
        // SAFETY: lengths checked; reduction launch contract as in `mix`.
        unsafe { b.launch(cfg((tokens, 1, 1), 256, REDUCE_SHARED)) }?;
        Ok(())
    }

    /// `out = cur - tanh(forget) * cur + tanh(gain) * msg` (`inkling::control`).
    #[allow(clippy::too_many_arguments)]
    pub fn ctrl_apply(
        &self,
        stream: &Arc<CudaStream>,
        cur: &CudaSlice<f32>,
        msg: &CudaSlice<f32>,
        gain: &CudaSlice<f32>,
        forget: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        count: usize,
    ) -> Result<(), InklingError> {
        for (what, len) in [
            ("ctrl_apply cur", cur.len()),
            ("ctrl_apply msg", msg.len()),
            ("ctrl_apply gain", gain.len()),
            ("ctrl_apply forget", forget.len()),
            ("ctrl_apply out", out.len()),
        ] {
            need(what, len, count)?;
        }
        let c = count as i64;
        let mut b = stream.launch_builder(&self.ctrl_apply);
        b.arg(cur)
            .arg(msg)
            .arg(gain)
            .arg(forget)
            .arg(&mut *out)
            .arg(&c);
        // SAFETY: lengths checked.
        unsafe { b.launch(cfg((count.div_ceil(256), 1, 1), 256, 0)) }?;
        Ok(())
    }

    /// One sequence's short convolution over `rows` consecutive positions,
    /// advancing `state` (`[channels][taps - 1]`) past them
    /// (`inkling::convolve`). `dynamic` is `[rows][heads][taps]`.
    #[allow(clippy::too_many_arguments)]
    pub fn conv(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        dynamic: Option<(&CudaSlice<f32>, usize)>,
        state: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        channels: usize,
        taps: usize,
        rows: usize,
    ) -> Result<(), InklingError> {
        need("conv x", x.len(), rows * channels)?;
        need("conv out", out.len(), rows * channels)?;
        need("conv weight", weight.len(), channels * taps)?;
        need("conv state", state.len(), channels * (taps - 1))?;
        let heads = dynamic.map_or(1, |d| d.1);
        if let Some((d, heads)) = dynamic {
            need("conv dynamic", d.len(), rows * heads * taps)?;
            if !channels.is_multiple_of(heads) {
                return Err(InklingError::Shape("conv channels must divide into heads"));
            }
        }
        let norm = 1.0 / (((if dynamic.is_some() { 2 } else { 1 }) * taps + 1) as f32).sqrt();
        let (c, t, h, r) = (channels as i32, taps as i32, heads as i32, rows as i32);
        let null = 0u64;
        let mut b = stream.launch_builder(&self.conv);
        b.arg(x).arg(weight);
        match dynamic {
            Some((d, _)) => b.arg(d),
            None => b.arg(&null),
        };
        b.arg(&mut *state)
            .arg(&mut *out)
            .arg(&c)
            .arg(&t)
            .arg(&h)
            .arg(&r)
            .arg(&norm);
        // SAFETY: lengths checked; a thread owns one channel of x, out and
        // state, and x is distinct from out by the borrow.
        unsafe { b.launch(cfg((channels.div_ceil(128), 1, 1), 128, 0)) }?;
        Ok(())
    }

    /// Clip each `width`-long group of every row to an L2 norm of
    /// `threshold` after the optional per-channel `scale`
    /// (`inkling::clip`); with `log = Some((alpha, floor))` also apply the
    /// row's logarithmic position scale, positions starting at `pos[0]`.
    #[allow(clippy::too_many_arguments)]
    pub fn clip(
        &self,
        stream: &Arc<CudaStream>,
        x: &mut CudaSlice<f32>,
        scale: Option<&CudaSlice<f32>>,
        pos: &CudaSlice<i32>,
        width: usize,
        groups: usize,
        threshold: f32,
        log: Option<(f32, f32)>,
        tokens: usize,
    ) -> Result<(), InklingError> {
        if groups > 32 {
            return Err(InklingError::Shape("clip takes at most 32 groups a row"));
        }
        need("clip x", x.len(), tokens * groups * width)?;
        need("clip pos", pos.len(), 1)?;
        if let Some(s) = scale {
            need("clip scale", s.len(), groups * width)?;
        }
        let (w, g) = (width as i32, groups as i32);
        let (alpha, floor) = log.unwrap_or((-1.0, 1.0));
        let null = 0u64;
        let mut b = stream.launch_builder(&self.clip);
        b.arg(&mut *x);
        match scale {
            Some(s) => b.arg(s),
            None => b.arg(&null),
        };
        b.arg(pos)
            .arg(&w)
            .arg(&g)
            .arg(&threshold)
            .arg(&alpha)
            .arg(&floor);
        // SAFETY: lengths checked; one thread a group.
        unsafe { b.launch(cfg((tokens, 1, 1), 32, 0)) }?;
        Ok(())
    }

    /// RMS-normalize each `width`-long group in place, no weight.
    pub fn rms_groups(
        &self,
        stream: &Arc<CudaStream>,
        x: &mut CudaSlice<f32>,
        width: usize,
        eps: f32,
        groups: usize,
    ) -> Result<(), InklingError> {
        need("rms_groups x", x.len(), groups * width)?;
        let (w, g) = (width as i32, groups as i64);
        let mut b = stream.launch_builder(&self.rms_groups);
        b.arg(&mut *x).arg(&w).arg(&eps).arg(&g);
        // SAFETY: lengths checked.
        unsafe { b.launch(cfg((groups.div_ceil(64), 1, 1), 64, 0)) }?;
        Ok(())
    }

    /// The bias of every distance per row and head, sink subtracted:
    /// `table` is `[tokens][heads][extent + 1]`.
    #[allow(clippy::too_many_arguments)]
    pub fn rel_table(
        &self,
        stream: &Arc<CudaStream>,
        bias: &CudaSlice<f32>,
        r: &CudaSlice<f32>,
        sink: &CudaSlice<f32>,
        table: &mut CudaSlice<f32>,
        n_rel: usize,
        extent: usize,
        heads: usize,
        tokens: usize,
    ) -> Result<(), InklingError> {
        need("rel_table bias", bias.len(), extent * n_rel)?;
        need("rel_table r", r.len(), tokens * heads * n_rel)?;
        need("rel_table sink", sink.len(), tokens * heads)?;
        need(
            "rel_table table",
            table.len(),
            tokens * heads * (extent + 1),
        )?;
        let (n, e) = (n_rel as i32, extent as i32);
        let mut b = stream.launch_builder(&self.rel_table);
        b.arg(bias)
            .arg(r)
            .arg(sink)
            .arg(&mut *table)
            .arg(&n)
            .arg(&e);
        // SAFETY: lengths checked against the grid.
        unsafe { b.launch(cfg(((extent + 1).div_ceil(256), heads, tokens), 256, 0)) }?;
        Ok(())
    }

    /// Attention of `tokens` consecutive rows over their sequence's cache
    /// (the attention loop of `inkling::hidden_states`): the rows sit at
    /// positions `pos[0]..`, and the caches already hold them. `window == 0`
    /// is full attention.
    #[allow(clippy::too_many_arguments)]
    pub fn attn(
        &self,
        stream: &Arc<CudaStream>,
        q: &CudaSlice<f32>,
        k_cache: &CudaSlice<u16>,
        v_cache: &CudaSlice<u16>,
        table: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        pos: &CudaSlice<i32>,
        window: usize,
        extent: usize,
        heads: usize,
        kv_heads: usize,
        dim: usize,
        tokens: usize,
    ) -> Result<(), InklingError> {
        if dim > 128 || !heads.is_multiple_of(kv_heads) {
            return Err(InklingError::Shape("attention head geometry"));
        }
        need("attn q", q.len(), tokens * heads * dim)?;
        need("attn out", out.len(), tokens * heads * dim)?;
        need("attn table", table.len(), tokens * heads * (extent + 1))?;
        need("attn pos", pos.len(), 1)?;
        let (w, e, kv, d) = (window as i32, extent as i32, kv_heads as i32, dim as i32);
        let scale = 1.0 / (dim as f32).sqrt();
        let mut b = stream.launch_builder(&self.attn);
        b.arg(q)
            .arg(k_cache)
            .arg(v_cache)
            .arg(table)
            .arg(&mut *out)
            .arg(pos)
            .arg(&w)
            .arg(&e)
            .arg(&kv)
            .arg(&d)
            .arg(&scale);
        // SAFETY: lengths checked; the caller's caches cover every position
        // up to the last row's, which only it can know.
        unsafe { b.launch(cfg((heads, tokens, 1), 256, 0)) }?;
        Ok(())
    }

    /// Group the routed pairs by expert. `ids` is `[tokens][topk]`.
    pub fn dispatch(
        &self,
        stream: &Arc<CudaStream>,
        ids: &CudaSlice<i32>,
        d: &mut InkDispatch,
    ) -> Result<(), InklingError> {
        need("dispatch ids", ids.len(), d.tokens * d.topk)?;
        let pairs = (d.tokens * d.topk) as i32;
        let (experts, capacity, tile) = (d.experts as i32, d.owners.len() as i32, INK_TILE as i32);
        let per_expert = cfg((d.experts, 1, 1), 128, 0);
        // SAFETY: one block an expert with disjoint counts and padded runs;
        // the capacity includes one padded run per expert.
        unsafe {
            stream
                .launch_builder(&self.dispatch_count)
                .arg(ids)
                .arg(&mut d.counts)
                .arg(&pairs)
                .launch(per_expert)?;
            stream
                .launch_builder(&self.dispatch_prefix)
                .arg(&d.counts)
                .arg(&mut d.starts)
                .arg(&mut d.owners)
                .arg(&experts)
                .arg(&capacity)
                .arg(&tile)
                .launch(cfg((1, 1, 1), 128, 0))?;
            stream
                .launch_builder(&self.dispatch_gather)
                .arg(ids)
                .arg(&d.counts)
                .arg(&d.starts)
                .arg(&mut d.sorted)
                .arg(&pairs)
                .arg(&tile)
                .launch(per_expert)?;
        }
        Ok(())
    }

    /// Expert-sorted binary16 activations, `[slots][n]`, and `slot_of`.
    pub fn gather_half(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        d: &mut InkDispatch,
        xs: &mut CudaSlice<u16>,
        n: usize,
    ) -> Result<(), InklingError> {
        need("gather x", x.len(), d.tokens * n)?;
        need("gather xs", xs.len(), d.slots() * n)?;
        let (n_i, topk) = (n as i32, d.topk as i32);
        let (pairs, tile) = ((d.tokens * d.topk) as i32, INK_TILE as i32);
        let grid = cfg((n.div_ceil(256), d.slots(), 1), 256, 0);
        let mut b = stream.launch_builder(&self.gather_half);
        b.arg(x)
            .arg(&d.sorted)
            .arg(&d.owners)
            .arg(&mut *xs)
            .arg(&mut d.slot_of)
            .arg(&n_i)
            .arg(&topk)
            .arg(&pairs)
            .arg(&tile);
        // SAFETY: lengths checked; live slots hold pair ids below `pairs` or
        // the sentinel, and each pair id occupies exactly one slot.
        unsafe { b.launch(grid) }?;
        Ok(())
    }

    /// SiTU over every live slot's merged gate/up row, binary16 out
    /// (`inkling::situ`).
    #[allow(clippy::too_many_arguments)]
    pub fn situ_half(
        &self,
        stream: &Arc<CudaStream>,
        gate_up: &CudaSlice<f32>,
        d: &InkDispatch,
        out: &mut CudaSlice<u16>,
        width: usize,
        beta: f32,
        linear_beta: f32,
    ) -> Result<(), InklingError> {
        need("situ gate_up", gate_up.len(), d.slots() * 2 * width)?;
        need("situ out", out.len(), d.slots() * width)?;
        let (w, tile) = (width as i32, INK_TILE as i32);
        let mut b = stream.launch_builder(&self.situ_half);
        b.arg(gate_up)
            .arg(&d.owners)
            .arg(&mut *out)
            .arg(&w)
            .arg(&tile)
            .arg(&beta)
            .arg(&linear_beta);
        // SAFETY: lengths checked against the grid.
        unsafe { b.launch(cfg((width.div_ceil(256), d.slots(), 1), 256, 0)) }?;
        Ok(())
    }

    /// `out[t] = sum_j weights[t][j] * y[slot_of[t][j]]`, ascending `j`.
    pub fn moe_combine(
        &self,
        stream: &Arc<CudaStream>,
        y: &CudaSlice<f32>,
        weights: &CudaSlice<f32>,
        d: &InkDispatch,
        out: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), InklingError> {
        need("combine y", y.len(), d.slots() * n)?;
        need("combine weights", weights.len(), d.tokens * d.topk)?;
        need("combine out", out.len(), d.tokens * n)?;
        let (n_i, topk) = (n as i32, d.topk as i32);
        let mut b = stream.launch_builder(&self.moe_combine);
        b.arg(y)
            .arg(weights)
            .arg(&d.slot_of)
            .arg(&mut *out)
            .arg(&n_i)
            .arg(&topk);
        // SAFETY: lengths checked; `slot_of` was written for every pair by
        // `gather_half` over the same dispatch.
        unsafe { b.launch(cfg((n.div_ceil(256), d.tokens, 1), 256, 0)) }?;
        Ok(())
    }

    /// `x = cap * tanh(x / cap)` over the first `count` values.
    pub fn softcap(
        &self,
        stream: &Arc<CudaStream>,
        x: &mut CudaSlice<f32>,
        count: usize,
        cap: f32,
    ) -> Result<(), InklingError> {
        need("softcap x", x.len(), count)?;
        let c = count as i64;
        let mut b = stream.launch_builder(&self.softcap);
        b.arg(&mut *x).arg(&c).arg(&cap);
        // SAFETY: length checked.
        unsafe { b.launch(cfg((count.div_ceil(256), 1, 1), 256, 0)) }?;
        Ok(())
    }
}
