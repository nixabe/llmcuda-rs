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

/// Token columns one decode-GEMM launch contracts.
pub const DG_MAX_ROWS: usize = 8;
/// Weight rows a decode-GEMM block owns: two warps of sixteen.
pub const DG_BLOCK_ROWS: usize = 32;

const DG_SRC: &str = r#"
extern "C" {

__device__ __forceinline__ void dg_mma(float* d, unsigned a0, unsigned a1, unsigned b0) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(b0));
}

// Two int8 (the low half of `q`) to one fp16 pair times one fp16 scale (both
// halves of `s2`): each byte is biased to unsigned and spliced under 0x64,
// which is 1024 + (q + 128) exactly; subtracting 1152 leaves q, and the
// multiply is the one rounding. As hg_dequant4 in the fp16 GEMM.
__device__ __forceinline__ unsigned dg_q8x2(unsigned q, unsigned s2) {
    unsigned v;
    q ^= 0x8080u;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(v) : "r"(q), "r"(0x64646464u), "r"(0x5150u));
    asm("sub.rn.f16x2 %0, %0, %1;" : "+r"(v) : "r"(0x64806480u));
    asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(v) : "r"(s2));
    return v;
}

// Two bf16 weights to one fp16 pair, each times `scale` (a power of two).
__device__ __forceinline__ unsigned dg_bf16x2(unsigned v, float scale) {
    float lo = __uint_as_float(v << 16) * scale, hi = __uint_as_float(v & 0xffff0000u) * scale;
    unsigned short l, h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(l) : "f"(lo));
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(hi));
    return (unsigned)l | ((unsigned)h << 16);
}

// One E2M1 nibble (sign in bit 3) to the fp16 bits of its doubled value.
__device__ __forceinline__ unsigned dg_fp4(unsigned nib) {
    const unsigned m = nib & 7u;
    const unsigned v = m < 2u ? m * 0x3C00u : (((15u + (m >> 1)) << 10) | ((m & 1u) << 9));
    return v | ((nib & 8u) << 12);
}

// An unsigned E4M3 sub-block scale, halved, as fp16 bits in both halves.
__device__ __forceinline__ unsigned dg_ue4m3x2(unsigned b) {
    const unsigned e = (b >> 3) & 15u, man = b & 7u;
    unsigned short sub;
    const float f = (float)man * 0.0009765625f;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(sub) : "f"(f));
    unsigned v = e ? (((e + 7u) << 10) | (man << 7)) : (unsigned)sub;
    v = (b == 0u || b == 0x7Fu) ? 0u : v;
    return v | (v << 16);
}

__device__ __forceinline__ float dg_situ(float gate, float up, float beta, float lb) {
    const float act = beta * tanhf(gate / beta) / (1.0f + expf(-gate));
    return act * (lb > 0.0f ? lb * tanhf(up / lb) : up);
}

}  // extern "C"

// A weight-only GEMM for at most eight token columns, the decode shape:
// out[col][row] = alpha * sum_k half(x[col][k]) * half(w[row][k]), fp32
// accumulate.
//
// No shared-memory staging. A warp owns sixteen weight rows and one
// m16n8k8 fragment set; each lane builds its two A registers (rows g and
// g + 8, k pair t) straight from the stored weights and reads its B register
// (token g, k pair t) straight from the binary16 activations. Per k8 step
// that is one activation load, two weight loads and one MMA for 128 weights
// against all eight columns.
//
// FMT 0: split Q8_0 (`w` int8 [rows][K], `ws` fp16 [rows][K/32]).
// FMT 1: bf16 [rows][K], staged at `wscale`.
// FMT 2: NVFP4 as stored. A byte holds elements b and b + 8 of its
//   sub-block, so the contraction walks k in *byte order*: the activations
//   must be laid out the same way (pair p of a sub-block holds elements p
//   and p + 8), which the kernels that write them do when asked. One table
//   load turns a byte into the fp16 pair.
//
// MODE 0, dense: columns are rows 0..ntok of the activation plane that
//   weight rows [0, b1), [b1, b2), [b2, ..) read (planes 0, 1, 2, `K`
//   apart per row, `plane` elements apart per plane). Output stride `ldo`.
// MODE 1, gate/up: one (token, expert) pair a block row. The warp's rows g
//   are gate rows and g + 8 the matching up rows; the epilogue is SiTU and
//   the result is binary16, in byte order when `nv_out`.
// MODE 2, down: one pair a block row; the result is scaled by the pair's
//   router weight and written fp32.
template <int FMT, int MODE>
__device__ __forceinline__ void dg_gemm(
    const unsigned char* __restrict__ w, const unsigned short* __restrict__ ws,
    const unsigned short* __restrict__ x, void* __restrict__ out,
    const int* __restrict__ ids, const float* __restrict__ escale,
    const float* __restrict__ pair_w,
    int rows, int K, int ntok, int ldo, long long plane, int b1, int b2,
    long long wstride, long long wsstride, int topk, int nv_out,
    float alpha, float wscale, float beta, float lb
) {
    __shared__ unsigned lut[256];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    if (FMT == 2) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned b = tid * 4 + i;
            lut[b] = dg_fp4(b & 15u) | (dg_fp4(b >> 4) << 16);
        }
        __syncthreads();
    }

    const unsigned short* xcol = nullptr;
    int row_a, row_b;
    if (MODE == 0) {
        const int r0 = blockIdx.x * 32 + warp * 16;
        row_a = r0 + g;
        row_b = r0 + 8 + g;
        const int p = (r0 >= b1) + (r0 >= b2);
        if (g < ntok) xcol = x + p * plane + (long long)g * K;
    } else {
        const int pair = blockIdx.y;
        const int e = ids[pair];
        w += (long long)e * wstride;
        ws += (long long)e * wsstride;
        if (escale) alpha *= escale[e];
        if (g == 0) xcol = x + (long long)(pair / topk) * K;
        if (MODE == 1) {
            const int i0 = blockIdx.x * 16 + warp * 8;
            row_a = i0 + g;
            row_b = rows / 2 + i0 + g;
        } else {
            const int r0 = blockIdx.x * 32 + warp * 16;
            row_a = r0 + g;
            row_b = r0 + 8 + g;
        }
    }
    const bool ok_a = row_a < rows, ok_b = row_b < rows;
    const int kblocks = K >> 5;

    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    unsigned sa = 0, sb = 0;
    const int steps = K >> 3;
    for (int kk = 0; kk < steps; ++kk) {
        unsigned a0 = 0, a1 = 0;
        if (FMT == 0) {
            if ((kk & 3) == 0) {
                const int kb = kk >> 2;
                sa = ok_a ? ws[(long long)row_a * kblocks + kb] : 0;
                sb = ok_b ? ws[(long long)row_b * kblocks + kb] : 0;
                sa |= sa << 16;
                sb |= sb << 16;
            }
            const long long col = kk * 8 + 2 * t;
            if (ok_a) a0 = dg_q8x2(*(const unsigned short*)(w + (long long)row_a * K + col), sa);
            if (ok_b) a1 = dg_q8x2(*(const unsigned short*)(w + (long long)row_b * K + col), sb);
        } else if (FMT == 1) {
            const long long col = (kk * 8 + 2 * t) * 2;
            if (ok_a) a0 = dg_bf16x2(*(const unsigned*)(w + (long long)row_a * K * 2 + col), wscale);
            if (ok_b) a1 = dg_bf16x2(*(const unsigned*)(w + (long long)row_b * K * 2 + col), wscale);
        } else {
            const long long blk = (long long)(kk >> 3) * 36;
            const int sub = (kk >> 1) & 3;
            const unsigned char* pa = w + (long long)row_a * (K >> 6) * 36 + blk;
            const unsigned char* pb = w + (long long)row_b * (K >> 6) * 36 + blk;
            if ((kk & 1) == 0) {
                sa = ok_a ? dg_ue4m3x2(pa[sub]) : 0;
                sb = ok_b ? dg_ue4m3x2(pb[sub]) : 0;
            }
            const int at = 4 + sub * 8 + (kk & 1) * 4 + t;
            if (ok_a) {
                a0 = lut[pa[at]];
                asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(a0) : "r"(sa));
            }
            if (ok_b) {
                a1 = lut[pb[at]];
                asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(a1) : "r"(sb));
            }
        }
        const unsigned b = xcol ? *(const unsigned*)(xcol + kk * 8 + 2 * t) : 0u;
        dg_mma(c, a0, a1, b);
    }

    // c[0], c[1]: row_a at token columns 2t, 2t + 1; c[2], c[3]: row_b.
    if (MODE == 0) {
        float* o = (float*)out;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int col = 2 * t + j;
            if (col >= ntok) continue;
            if (ok_a) o[(long long)col * ldo + row_a] = c[j] * alpha;
            if (ok_b) o[(long long)col * ldo + row_b] = c[2 + j] * alpha;
        }
    } else if (t == 0) {
        const int pair = blockIdx.y;
        if (MODE == 1) {
            if (ok_b) {
                const int half_rows = rows / 2;
                const float v = dg_situ(c[0] * alpha, c[2] * alpha, beta, lb);
                unsigned short h;
                asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v));
                // Byte order: element e of a 16-element sub-block sits at
                // pair (e & 7), half (e >> 3).
                const int e16 = row_a & 15;
                const int at = nv_out ? (row_a & ~15) + ((e16 & 7) << 1) + (e16 >> 3) : row_a;
                ((unsigned short*)out)[(long long)pair * half_rows + at] = h;
            }
        } else {
            float* o = (float*)out + (long long)pair * rows;
            const float s = alpha * pair_w[pair];
            if (ok_a) o[row_a] = c[0] * s;
            if (ok_b) o[row_b] = c[2] * s;
        }
    }
}

#define DG_ENTRY(NAME, FMT, MODE)                                                        \
    extern "C" __global__ void __launch_bounds__(64) NAME(                               \
        const unsigned char* __restrict__ w, const unsigned short* __restrict__ ws,      \
        const unsigned short* __restrict__ x, void* __restrict__ out,                    \
        const int* __restrict__ ids, const float* __restrict__ escale,                   \
        const float* __restrict__ pair_w, int rows, int K, int ntok, int ldo,            \
        long long plane, int b1, int b2, long long wstride, long long wsstride,          \
        int topk, int nv_out, float alpha, float wscale, float beta, float lb) {         \
        dg_gemm<FMT, MODE>(w, ws, x, out, ids, escale, pair_w, rows, K, ntok, ldo,       \
                           plane, b1, b2, wstride, wsstride, topk, nv_out, alpha,        \
                           wscale, beta, lb);                                            \
    }

DG_ENTRY(dg_q8_dense, 0, 0)
DG_ENTRY(dg_bf16_dense, 1, 0)
DG_ENTRY(dg_q8_gate_up, 0, 1)
DG_ENTRY(dg_bf16_gate_up, 1, 1)
DG_ENTRY(dg_nv_gate_up, 2, 1)
DG_ENTRY(dg_q8_down, 0, 2)
DG_ENTRY(dg_bf16_down, 1, 2)
DG_ENTRY(dg_nv_down, 2, 2)
"#;

use super::hgemm::HalfWeight;

/// The decode-shaped GEMM: a weight-only contraction for at most
/// [`DG_MAX_ROWS`] token rows that builds its tensor-core operands in
/// registers, with no shared-memory staging.
///
/// `out[row][n] = sum_k half(x[row][k]) * half(w[n][k])`, fp32 accumulate —
/// the arithmetic class of [`super::hgemm::HgemmKernels`], in a different
/// summation order. NVFP4 weights are contracted in *byte order*: their
/// activations must be written pair-permuted (pair `p` of each 16-element
/// group holds elements `p` and `p + 8`), which
/// [`InklingKernels`]'s decode kernels do on request.
pub struct DecodeGemm {
    dense: [CudaFunction; 2],
    gate_up: [CudaFunction; 3],
    down: [CudaFunction; 3],
}

/// Where the format-specific operands go: `(format, w bytes?, alpha, wscale,
/// per-matrix byte stride, per-matrix scale stride)`.
fn dg_format(w: &HalfWeight<'_>, rows: usize, k: usize) -> (usize, f32, f32, i64, i64) {
    match w {
        HalfWeight::Q8_0 { .. } => (0, 1.0, 1.0, (rows * k) as i64, (rows * k / 32) as i64),
        HalfWeight::Bf16 { exponent, .. } => (
            1,
            2f32.powi(-exponent),
            2f32.powi(*exponent),
            (rows * k * 2) as i64,
            0,
        ),
        HalfWeight::Nvfp4 { scale, .. } => (2, *scale, 1.0, (rows * k / 64 * 36) as i64, 0),
    }
}

impl DecodeGemm {
    /// Compile for the context's device.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, InklingError> {
        let ptx = super::compile(DG_SRC, "inkling decode gemm")
            .map_err(|_| InklingError::Shape("the decode GEMM failed to compile"))?;
        let m = ctx.load_module(ptx)?;
        Ok(Self {
            dense: [
                m.load_function("dg_q8_dense")?,
                m.load_function("dg_bf16_dense")?,
            ],
            gate_up: [
                m.load_function("dg_q8_gate_up")?,
                m.load_function("dg_bf16_gate_up")?,
                m.load_function("dg_nv_gate_up")?,
            ],
            down: [
                m.load_function("dg_q8_down")?,
                m.load_function("dg_bf16_down")?,
                m.load_function("dg_nv_down")?,
            ],
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        stream: &Arc<CudaStream>,
        func: &CudaFunction,
        w: HalfWeight<'_>,
        x: &CudaSlice<u16>,
        out: u64,
        ids: Option<&CudaSlice<i32>>,
        escale: Option<&CudaSlice<f32>>,
        pair_w: Option<&CudaSlice<f32>>,
        ints: [i32; 8],
        plane: i64,
        strides: (i64, i64),
        floats: [f32; 4],
        grid: (usize, usize),
    ) -> Result<(), InklingError> {
        let null = 0u64;
        let [rows, k, ntok, ldo, b1, b2, topk, nv_out] = ints;
        let [alpha, wscale, beta, lb] = floats;
        let mut b = stream.launch_builder(func);
        match w {
            HalfWeight::Q8_0 { q, scales } => b.arg(q).arg(scales),
            // Only the Q8_0 entry points read `ws`.
            HalfWeight::Bf16 { bytes, .. } | HalfWeight::Nvfp4 { bytes, .. } => b.arg(bytes).arg(x),
        };
        b.arg(x).arg(&out);
        match ids {
            Some(s) => b.arg(s),
            None => b.arg(&null),
        };
        match escale {
            Some(s) => b.arg(s),
            None => b.arg(&null),
        };
        match pair_w {
            Some(s) => b.arg(s),
            None => b.arg(&null),
        };
        b.arg(&rows)
            .arg(&k)
            .arg(&ntok)
            .arg(&ldo)
            .arg(&plane)
            .arg(&b1)
            .arg(&b2)
            .arg(&strides.0)
            .arg(&strides.1)
            .arg(&topk)
            .arg(&nv_out)
            .arg(&alpha)
            .arg(&wscale)
            .arg(&beta)
            .arg(&lb);
        // SAFETY: every caller length-checks its operands against the grid.
        unsafe { b.launch(cfg((grid.0, grid.1, 1), 64, 0)) }?;
        Ok(())
    }

    /// `out[t][n] = x_p[t] . w[n]` for `t < tokens`, where weight rows
    /// `[0, b1)`, `[b1, b2)` and `[b2, rows)` read activation planes 0, 1
    /// and 2 of `x` (`[planes][plane_rows][k]`). `b1` and `b2` must be
    /// multiples of [`DG_BLOCK_ROWS`]. Output rows are `ldo` apart.
    #[allow(clippy::too_many_arguments)]
    pub fn dense(
        &self,
        stream: &Arc<CudaStream>,
        w: HalfWeight<'_>,
        x: &CudaSlice<u16>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        k: usize,
        tokens: usize,
        ldo: usize,
        plane_rows: usize,
        b1: usize,
        b2: usize,
    ) -> Result<(), InklingError> {
        use cudarc::driver::DevicePtrMut;
        if tokens == 0 || tokens > DG_MAX_ROWS || !k.is_multiple_of(32) {
            return Err(InklingError::Shape(
                "decode GEMM takes 1..=8 rows, K a multiple of 32",
            ));
        }
        if !b1.is_multiple_of(DG_BLOCK_ROWS) || !b2.is_multiple_of(DG_BLOCK_ROWS) || b1 > b2 {
            return Err(InklingError::Shape("plane boundaries must be whole blocks"));
        }
        let planes = 1 + usize::from(b1 < rows) + usize::from(b2 < rows);
        need("dg x", x.len(), (planes - 1) * plane_rows * k + tokens * k)?;
        need("dg out", out.len(), (tokens - 1) * ldo + rows)?;
        let (format, alpha, wscale, _, _) = dg_format(&w, rows, k);
        match w {
            HalfWeight::Q8_0 { q, scales } => {
                need("dg q8 weights", q.len(), rows * k)?;
                need("dg q8 scales", scales.len(), rows * k / 32)?;
            }
            HalfWeight::Bf16 { bytes, .. } => need("dg bf16 weights", bytes.len(), rows * k * 2)?,
            HalfWeight::Nvfp4 { .. } => {
                return Err(InklingError::Shape("no dense NVFP4 decode GEMM"));
            }
        }
        let (ptr, _guard) = out.device_ptr_mut(stream);
        self.launch(
            stream,
            &self.dense[format],
            w,
            x,
            ptr,
            None,
            None,
            None,
            [
                rows as i32,
                k as i32,
                tokens as i32,
                ldo as i32,
                b1 as i32,
                b2 as i32,
                1,
                0,
            ],
            (plane_rows * k) as i64,
            (0, 0),
            [alpha, wscale, 0.0, 0.0],
            (rows.div_ceil(DG_BLOCK_ROWS), 1),
        )
    }

    /// The merged gate/up projection of every `(token, expert)` pair with
    /// SiTU applied: `out[pair][i]`, binary16, `i < rows / 2`. `w` is the
    /// stack of `stack` experts (`rows` = gate then up rows each), `ids`
    /// the expert of each pair, `x` the token rows (`pairs / topk` of
    /// them). `nv_out` writes the result in byte order, for an NVFP4 down
    /// projection.
    #[allow(clippy::too_many_arguments)]
    pub fn gate_up(
        &self,
        stream: &Arc<CudaStream>,
        w: HalfWeight<'_>,
        stack: usize,
        ids: &CudaSlice<i32>,
        escale: Option<&CudaSlice<f32>>,
        x: &CudaSlice<u16>,
        out: &mut CudaSlice<u16>,
        rows: usize,
        k: usize,
        pairs: usize,
        topk: usize,
        nv_out: bool,
        beta: f32,
        linear_beta: f32,
    ) -> Result<(), InklingError> {
        use cudarc::driver::DevicePtrMut;
        need("dg gate_up ids", ids.len(), pairs)?;
        need("dg gate_up x", x.len(), pairs.div_ceil(topk) * k)?;
        need("dg gate_up out", out.len(), pairs * rows / 2)?;
        self.check_stack(&w, stack, rows, k)?;
        let (format, alpha, wscale, wstride, wsstride) = dg_format(&w, rows, k);
        let (ptr, _guard) = out.device_ptr_mut(stream);
        self.launch(
            stream,
            &self.gate_up[format],
            w,
            x,
            ptr,
            Some(ids),
            escale,
            None,
            [
                rows as i32,
                k as i32,
                1,
                0,
                0,
                0,
                topk as i32,
                i32::from(nv_out),
            ],
            0,
            (wstride, wsstride),
            [alpha, wscale, beta, linear_beta],
            ((rows / 2).div_ceil(16), pairs),
        )
    }

    /// The down projection of every pair, scaled by its router weight:
    /// `out[pair][n]`, fp32. `x` is `[pairs][k]`, in byte order when `w` is
    /// NVFP4.
    #[allow(clippy::too_many_arguments)]
    pub fn down(
        &self,
        stream: &Arc<CudaStream>,
        w: HalfWeight<'_>,
        stack: usize,
        ids: &CudaSlice<i32>,
        escale: Option<&CudaSlice<f32>>,
        x: &CudaSlice<u16>,
        pair_w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        k: usize,
        pairs: usize,
    ) -> Result<(), InklingError> {
        use cudarc::driver::DevicePtrMut;
        need("dg down ids", ids.len(), pairs)?;
        need("dg down weights", pair_w.len(), pairs)?;
        need("dg down x", x.len(), pairs * k)?;
        need("dg down out", out.len(), pairs * rows)?;
        self.check_stack(&w, stack, rows, k)?;
        let (format, alpha, wscale, wstride, wsstride) = dg_format(&w, rows, k);
        let (ptr, _guard) = out.device_ptr_mut(stream);
        self.launch(
            stream,
            &self.down[format],
            w,
            x,
            ptr,
            Some(ids),
            escale,
            Some(pair_w),
            [rows as i32, k as i32, 1, 0, 0, 0, 1, 0],
            0,
            (wstride, wsstride),
            [alpha, wscale, 0.0, 0.0],
            (rows.div_ceil(DG_BLOCK_ROWS), pairs),
        )
    }

    fn check_stack(
        &self,
        w: &HalfWeight<'_>,
        stack: usize,
        rows: usize,
        k: usize,
    ) -> Result<(), InklingError> {
        match w {
            HalfWeight::Q8_0 { q, scales } => {
                need("dg expert q8 weights", q.len(), stack * rows * k)?;
                need("dg expert q8 scales", scales.len(), stack * rows * k / 32)
            }
            HalfWeight::Bf16 { bytes, .. } => {
                need("dg expert bf16 weights", bytes.len(), stack * rows * k * 2)
            }
            HalfWeight::Nvfp4 { bytes, .. } => {
                if !k.is_multiple_of(64) {
                    return Err(InklingError::Shape(
                        "NVFP4 rows are whole 64-element blocks",
                    ));
                }
                need(
                    "dg expert nvfp4 weights",
                    bytes.len(),
                    stack * rows * k / 64 * 36,
                )
            }
        }
    }
}
