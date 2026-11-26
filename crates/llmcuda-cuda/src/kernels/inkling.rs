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
    attn_mma: CudaFunction,
    linear_tiled: CudaFunction,
    reference: core::sync::atomic::AtomicBool,
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
        let mma = ctx.load_module(
            super::compile(ATTN_MMA_SRC, "inkling attention")
                .map_err(|_| InklingError::Shape("the attention kernel failed to compile"))?,
        )?;
        Ok(Self {
            attn_mma: mma.load_function("ink_attn_mma")?,
            linear_tiled: mma.load_function("ink_linear_tiled")?,
            reference: core::sync::atomic::AtomicBool::new(false),
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

    /// `out[t][j] = w[j] . x[t]` in fp32 (`inkling::project`): ascending
    /// fused sums, the same bits from the tiled kernel as the reference.
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
        if !self.reference.load(core::sync::atomic::Ordering::Relaxed) {
            let n = tokens as i32;
            let mut b = stream.launch_builder(&self.linear_tiled);
            b.arg(w)
                .arg(x)
                .arg(&mut *out)
                .arg(&r)
                .arg(&k_i)
                .arg(&l)
                .arg(&n);
            // SAFETY: lengths checked against the grid.
            unsafe { b.launch(cfg((rows.div_ceil(128), tokens.div_ceil(8), 1), 128, 0)) }?;
            return Ok(());
        }
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

    /// Run prefill on the plain reference kernels: the scalar attention
    /// instead of the tensor-core one, the one-thread-a-sum fp32 projection
    /// instead of the tiled one. For A/B measurement and differential tests.
    pub fn use_reference_kernels(&self, on: bool) {
        self.reference
            .store(on, core::sync::atomic::Ordering::Relaxed);
    }

    /// Attention of `tokens` consecutive rows over their sequence's cache
    /// (the attention loop of `inkling::hidden_states`): the rows sit at
    /// positions `pos[0]..`, and the caches already hold them. `window == 0`
    /// is full attention. At head_dim 128 this is the tensor-core kernel:
    /// queries and probabilities as hi + lo binary16 pairs, fp32
    /// accumulation.
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
        if dim == 128 && !self.reference.load(core::sync::atomic::Ordering::Relaxed) {
            let (hd, n) = (heads as i32, tokens as i32);
            let mut b = stream.launch_builder(&self.attn_mma);
            let null = 0u64;
            b.arg(q)
                .arg(k_cache)
                .arg(v_cache)
                .arg(table)
                .arg(&null)
                .arg(&null)
                .arg(&null)
                .arg(&mut *out)
                .arg(pos)
                .arg(&w)
                .arg(&e)
                .arg(&hd)
                .arg(&kv)
                .arg(&n)
                .arg(&scale);
            // SAFETY: lengths checked; the caller's caches cover every
            // position up to the last row's, and the kernel reads no key
            // past it.
            unsafe { b.launch(cfg((heads, tokens.div_ceil(ATTN_MMA_ROWS), 1), 32, 0)) }?;
            return Ok(());
        }
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

    /// Whether [`Self::attn_fused`] serves this geometry: the tensor-core
    /// kernel with the bias computed in it, in place of
    /// [`Self::rel_table`] and [`Self::attn`].
    pub fn fuses_the_bias(&self, dim: usize, n_rel: usize) -> bool {
        dim == 128 && n_rel == 16 && !self.reference.load(core::sync::atomic::Ordering::Relaxed)
    }

    /// [`Self::attn`] with the relative-position bias computed in the
    /// kernel from the rows' features `r` (`[tokens][heads][16]`), their
    /// sink logits and the layer's `rel_bias` (`[extent][16]`), for the
    /// distances each row reaches. Same scores as the table gives.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_fused(
        &self,
        stream: &Arc<CudaStream>,
        q: &CudaSlice<f32>,
        k_cache: &CudaSlice<u16>,
        v_cache: &CudaSlice<u16>,
        r: &CudaSlice<f32>,
        sink: &CudaSlice<f32>,
        rel_bias: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        pos: &CudaSlice<i32>,
        window: usize,
        extent: usize,
        heads: usize,
        kv_heads: usize,
        tokens: usize,
    ) -> Result<(), InklingError> {
        let dim = 128;
        if !heads.is_multiple_of(kv_heads) {
            return Err(InklingError::Shape("attention head geometry"));
        }
        need("attn q", q.len(), tokens * heads * dim)?;
        need("attn out", out.len(), tokens * heads * dim)?;
        need("attn r", r.len(), tokens * heads * 16)?;
        need("attn sink", sink.len(), tokens * heads)?;
        need("attn rel_bias", rel_bias.len(), extent * 16)?;
        need("attn pos", pos.len(), 1)?;
        let (w, e, kv, hd, n) = (
            window as i32,
            extent as i32,
            kv_heads as i32,
            heads as i32,
            tokens as i32,
        );
        let scale = 1.0 / (dim as f32).sqrt();
        let null = 0u64;
        let mut b = stream.launch_builder(&self.attn_mma);
        b.arg(q)
            .arg(k_cache)
            .arg(v_cache)
            .arg(&null)
            .arg(r)
            .arg(sink)
            .arg(rel_bias)
            .arg(&mut *out)
            .arg(pos)
            .arg(&w)
            .arg(&e)
            .arg(&hd)
            .arg(&kv)
            .arg(&n)
            .arg(&scale);
        // SAFETY: lengths checked; the caller's caches cover every position
        // up to the last row's, and the kernel reads no key past it.
        unsafe { b.launch(cfg((heads, tokens.div_ceil(ATTN_MMA_ROWS), 1), 32, 0)) }?;
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
        let grid = cfg((d.slots(), n.div_ceil(256), 1), 256, 0);
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
        unsafe { b.launch(cfg((d.slots(), width.div_ceil(256), 1), 256, 0)) }?;
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

/// Query rows one tensor-core attention block covers.
pub const ATTN_MMA_ROWS: usize = 16;

const ATTN_MMA_SRC: &str = r#"
extern "C" {

__device__ __forceinline__ void am_mma(float* d, unsigned a0, unsigned a1, unsigned b0) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(b0));
}

// A pair of fp32 as two binary16 pairs, `hi` the nearest binary16 and `lo`
// the binary16 of what that rounding dropped: hi + lo carries about 22
// bits. One binary16 a query element moved a score by ~1e-3 (the logits
// reach +-45 before the scale), which the fp32 scalar kernel and llama.cpp
// do not do.
__device__ __forceinline__ void am_split2(float lo, float hi, unsigned& H, unsigned& L) {
    unsigned short l, h, rl, rh;
    float bl, bh;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(l) : "f"(lo));
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(hi));
    asm("cvt.f32.f16 %0, %1;" : "=f"(bl) : "h"(l));
    asm("cvt.f32.f16 %0, %1;" : "=f"(bh) : "h"(h));
    const float dl = lo - bl, dh = hi - bh;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(rl) : "f"(dl));
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(rh) : "f"(dh));
    H = (unsigned)l | ((unsigned)h << 16);
    L = (unsigned)rl | ((unsigned)rh << 16);
}

// inkling2_loop attention for sixteen consecutive query rows of one head, on
// the tensor cores: p_j = exp(s_j) / (1 + sum exp(s)),
// s_j = q . k_j * scale + bias(p - j) - sink, over cache rows j <= p (and
// j > p - window when window > 0), p = pos[0] + row. head_dim is 128.
//
// The bias comes one of two ways. With `table` (rows x heads x (extent + 1),
// the sink already subtracted, the last entry every distance past the
// extent) it is a lookup, as the scalar kernel. Without, it is computed for
// the distances a row actually reaches: bias(d) = rel_bias[d] . r[row][head]
// over sixteen features for d < extent and zero past it, the row's features
// held in registers. Filling the table for every row cost more than the
// attention it fed: rows x heads x 1025 entries a layer, 150 MB at 1536
// rows, most of them distances no key sits at.
//
// One warp a block, and nothing staged in shared memory. Eight keys a step:
//
//   scores  C[query 16][key 8] = Q K^T. The sixteen query rows live in
//           registers for the whole scan, as hi + lo binary16 pairs. The head dimension is
//           contracted in a permuted order - lane quarter t owns dims
//           [32t, 32t + 32) of both operands - so a lane's share of a key
//           row is one 64-byte run.
//   softmax in fp32, online: a running maximum per query row (it starts at
//           the sink's logit, zero), sums and accumulators rescaled when it
//           moves.
//   values  O[query 16][dim 8] += P V, sixteen times for the 128 dims. The
//           probabilities go in as hi + lo binary16 straight from the score
//           fragment, whose layout is the A operand's. The output dims are
//           permuted the same way (column n of product m is dim 16n + m),
//           so a lane's share of a value row is one 32-byte run.
//
// The scalar kernel this replaces computed each score as a 128-step dot in
// one thread: 6.7 ms a layer at 512 rows.
extern "C" __global__ void __launch_bounds__(32) ink_attn_mma(
    const float* __restrict__ q, const unsigned short* __restrict__ k_cache,
    const unsigned short* __restrict__ v_cache, const float* __restrict__ table,
    const float* __restrict__ r, const float* __restrict__ sink,
    const float* __restrict__ rel_bias,
    float* __restrict__ out, const int* __restrict__ pos,
    int window, int extent, int heads, int kv_heads, int tokens, float scale
) {
    const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    const int h = blockIdx.x, r0 = blockIdx.y * 16;
    const int kv = h / (heads / kv_heads);
    const long long stride = (long long)kv_heads * 128;
    const int p0 = pos[0];
    const int ra = r0 + g, rb = r0 + 8 + g;
    const bool oka = ra < tokens, okb = rb < tokens;
    const int pa = p0 + ra, pb = p0 + rb;
    const int p_first = p0 + r0;
    const int p_last = p0 + min(r0 + 15, tokens - 1);

    unsigned qa[16], qb[16], qal[16], qbl[16];
#pragma unroll
    for (int s = 0; s < 16; ++s) {
        qa[s] = 0u;
        qb[s] = 0u;
        qal[s] = 0u;
        qbl[s] = 0u;
    }
    if (oka) {
        const float* qr = q + ((long long)ra * heads + h) * 128 + 32 * t;
#pragma unroll
        for (int s = 0; s < 16; ++s) am_split2(qr[2 * s], qr[2 * s + 1], qa[s], qal[s]);
    }
    if (okb) {
        const float* qr = q + ((long long)rb * heads + h) * 128 + 32 * t;
#pragma unroll
        for (int s = 0; s < 16; ++s) am_split2(qr[2 * s], qr[2 * s + 1], qb[s], qbl[s]);
    }
    const float* tba = table + ((long long)ra * heads + h) * (extent + 1);
    const float* tbb = table + ((long long)rb * heads + h) * (extent + 1);
    float rfa[16], rfb[16];
    float sinka = 0.0f, sinkb = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        rfa[i] = 0.0f;
        rfb[i] = 0.0f;
    }
    if (!table) {
        if (oka) {
            const float* rr = r + ((long long)ra * heads + h) * 16;
#pragma unroll
            for (int i = 0; i < 16; ++i) rfa[i] = rr[i];
            sinka = sink[(long long)ra * heads + h];
        }
        if (okb) {
            const float* rr = r + ((long long)rb * heads + h) * 16;
#pragma unroll
            for (int i = 0; i < 16; ++i) rfb[i] = rr[i];
            sinkb = sink[(long long)rb * heads + h];
        }
    }

    float acc[16][4];
#pragma unroll
    for (int m = 0; m < 16; ++m)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[m][i] = 0.0f;
    // The sink: logit zero, attending to nothing.
    float ma = 0.0f, mb = 0.0f;
    float la = t == 0 ? 1.0f : 0.0f, lb = la;
    const float lowest = -1.0e30f;

    int lo = (window > 0 && p_first + 1 > window) ? p_first + 1 - window : 0;
    lo &= ~7;
    for (int j0 = lo; j0 <= p_last; j0 += 8) {
        float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const int jk = j0 + g;
        // A lane past the last key contracts zeros: the MMA is warp-wide, so
        // only the load may be conditional.
        uint4 k0 = make_uint4(0, 0, 0, 0), k1 = k0, k2 = k0, k3 = k0;
        if (jk <= p_last) {
            const uint4* kp = (const uint4*)(k_cache + (long long)jk * stride + kv * 128 + 32 * t);
            k0 = kp[0];
            k1 = kp[1];
            k2 = kp[2];
            k3 = kp[3];
        }
        const unsigned kw[16] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w,
                                 k2.x, k2.y, k2.z, k2.w, k3.x, k3.y, k3.z, k3.w};
        // The low parts first: the small terms into the accumulator before
        // the large ones.
#pragma unroll
        for (int s = 0; s < 16; ++s) am_mma(c, qal[s], qbl[s], kw[s]);
#pragma unroll
        for (int s = 0; s < 16; ++s) am_mma(c, qa[s], qb[s], kw[s]);

        // c[0], c[1]: row a against keys j0 + 2t, j0 + 2t + 1; c[2], c[3]: row b.
        const int jv = j0 + 2 * t;
        float sa[2], sb[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int j = jv + i;
            const bool va = oka && j <= pa && (window == 0 || j > pa - window);
            const bool vb = okb && j <= pb && (window == 0 || j > pb - window);
            float ba = 0.0f, bb = 0.0f;
            if (table) {
                if (va) ba = tba[min(pa - j, extent)];
                if (vb) bb = tbb[min(pb - j, extent)];
            } else {
                // The table's entry, computed: the same ascending fused sum,
                // then the sink.
                if (va && pa - j < extent) {
                    const float* br = rel_bias + (long long)(pa - j) * 16;
#pragma unroll
                    for (int f = 0; f < 16; ++f) ba = fmaf(br[f], rfa[f], ba);
                }
                if (vb && pb - j < extent) {
                    const float* br = rel_bias + (long long)(pb - j) * 16;
#pragma unroll
                    for (int f = 0; f < 16; ++f) bb = fmaf(br[f], rfb[f], bb);
                }
                ba -= sinka;
                bb -= sinkb;
            }
            sa[i] = va ? c[i] * scale + ba : lowest;
            sb[i] = vb ? c[2 + i] * scale + bb : lowest;
        }
        float xa = fmaxf(sa[0], sa[1]), xb = fmaxf(sb[0], sb[1]);
        xa = fmaxf(xa, __shfl_xor_sync(0xffffffff, xa, 1));
        xa = fmaxf(xa, __shfl_xor_sync(0xffffffff, xa, 2));
        xb = fmaxf(xb, __shfl_xor_sync(0xffffffff, xb, 1));
        xb = fmaxf(xb, __shfl_xor_sync(0xffffffff, xb, 2));
        const float na = fmaxf(ma, xa), nb = fmaxf(mb, xb);
        const float fa = __expf(ma - na), fb = __expf(mb - nb);
        ma = na;
        mb = nb;
        const float pa0 = __expf(sa[0] - na), pa1 = __expf(sa[1] - na);
        const float pb0 = __expf(sb[0] - nb), pb1 = __expf(sb[1] - nb);
        la = la * fa + (pa0 + pa1);
        lb = lb * fb + (pb0 + pb1);
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            acc[m][0] *= fa;
            acc[m][1] *= fa;
            acc[m][2] *= fb;
            acc[m][3] *= fb;
        }
        unsigned a0, a1, a0l, a1l;
        am_split2(pa0, pa1, a0, a0l);
        am_split2(pb0, pb1, a1, a1l);

        // Values of keys jv and jv + 1, dims [16g, 16g + 16).
        unsigned wa[8], wb[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            wa[i] = 0u;
            wb[i] = 0u;
        }
        if (jv <= p_last) {
            const uint4* vp = (const uint4*)(v_cache + (long long)jv * stride + kv * 128 + 16 * g);
            const uint4 v0 = vp[0], v1 = vp[1];
            wa[0] = v0.x; wa[1] = v0.y; wa[2] = v0.z; wa[3] = v0.w;
            wa[4] = v1.x; wa[5] = v1.y; wa[6] = v1.z; wa[7] = v1.w;
        }
        if (jv + 1 <= p_last) {
            const uint4* vp = (const uint4*)(v_cache + (long long)(jv + 1) * stride + kv * 128 + 16 * g);
            const uint4 v0 = vp[0], v1 = vp[1];
            wb[0] = v0.x; wb[1] = v0.y; wb[2] = v0.z; wb[3] = v0.w;
            wb[4] = v1.x; wb[5] = v1.y; wb[6] = v1.z; wb[7] = v1.w;
        }
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            const int sh = 16 * (m & 1);
            const unsigned b = ((wa[m >> 1] >> sh) & 0xffffu) | (((wb[m >> 1] >> sh) & 0xffffu) << 16);
            am_mma(acc[m], a0l, a1l, b);
            am_mma(acc[m], a0, a1, b);
        }
    }

    la += __shfl_xor_sync(0xffffffff, la, 1);
    la += __shfl_xor_sync(0xffffffff, la, 2);
    lb += __shfl_xor_sync(0xffffffff, lb, 1);
    lb += __shfl_xor_sync(0xffffffff, lb, 2);
    const float ia = 1.0f / la, ib = 1.0f / lb;
    // acc[m]: columns 2t and 2t + 1 of product m, which are dims 32t + m and
    // 32t + 16 + m.
    if (oka) {
        float* o = out + ((long long)ra * heads + h) * 128 + 32 * t;
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            o[m] = acc[m][0] * ia;
            o[16 + m] = acc[m][1] * ia;
        }
    }
    if (okb) {
        float* o = out + ((long long)rb * heads + h) * 128 + 32 * t;
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            o[m] = acc[m][2] * ib;
            o[16 + m] = acc[m][3] * ib;
        }
    }
}

// out[t][j] = sum_i w[j][i] * x[t][i] in fp32, ascending i, each product
// fused into the running sum: ink_linear's sums exactly, eight rows of x a
// block. A thread owns one output j and reads its weight row once for the
// eight; the rows of x come from a shared tile, 64 columns at a time.
// ink_linear read both operands from global memory for every product, in
// one thread a (row, output): 7.4 ms for the router at 1536 rows.
//
// grid: (ceil(rows / 128), ceil(tokens / 8)), 128 threads.
extern "C" __global__ void __launch_bounds__(128) ink_linear_tiled(
    const float* __restrict__ w, const float* __restrict__ x, float* __restrict__ out,
    int rows, int k, int ldx, int tokens
) {
    __shared__ float xs[64][8];
    const int tid = threadIdx.x;
    const int j = blockIdx.x * 128 + tid;
    const int t0 = blockIdx.y * 8;
    const float* wr = w + (long long)min(j, rows - 1) * k;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int c0 = 0; c0 < k; c0 += 64) {
        const int len = min(64, k - c0);
#pragma unroll
        for (int idx = tid; idx < 512; idx += 128) {
            const int kk = idx >> 3, tok = idx & 7;
            float v = 0.0f;
            if (kk < len && t0 + tok < tokens) v = x[(long long)(t0 + tok) * ldx + c0 + kk];
            xs[kk][tok] = v;
        }
        __syncthreads();
        for (int kk = 0; kk < len; ++kk) {
            const float wv = wr[c0 + kk];
#pragma unroll
            for (int tok = 0; tok < 8; ++tok) acc[tok] = fmaf(wv, xs[kk][tok], acc[tok]);
        }
        __syncthreads();
    }
    if (j < rows) {
#pragma unroll
        for (int tok = 0; tok < 8; ++tok)
            if (t0 + tok < tokens) out[(long long)(t0 + tok) * rows + j] = acc[tok];
    }
}

}  // extern "C"
"#;

/// Token columns one decode-GEMM launch contracts.
pub const DG_MAX_ROWS: usize = 8;
/// Weight rows a decode-GEMM block owns. Activation-plane boundaries must
/// be multiples of it.
pub const DG_BLOCK_ROWS: usize = 16;

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
// No shared-memory staging. A block owns sixteen weight rows; each of its
// eight warps contracts one eighth of K for them with one m16n8k8 fragment
// set, and the partial sums are added in warp order at the end. A lane
// builds its two A registers (rows g and g + 8, k pair t) straight from the
// stored weights and reads its B register (token g, k pair t) straight from
// the binary16 activations: per k8 step, one activation load, two weight
// loads and one MMA for 128 weights against all eight columns.
//
// Eight warps a tile rather than one because a warp walking its rows' whole
// K is bound by the latency of its own loads, and a 8,608-row launch then
// keeps under eight warps on an SM: measured 242 GB/s of weights for that
// pack with one warp a tile (bench_inkling_dg).
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
    __shared__ float part[8][32][4];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    if (FMT == 2) {
        lut[tid] = dg_fp4(tid & 15u) | (dg_fp4(tid >> 4) << 16);
        __syncthreads();
    }

    const unsigned short* xcol = nullptr;
    int row_a, row_b;
    if (MODE == 0) {
        const int r0 = blockIdx.x * 16;
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
            const int i0 = blockIdx.x * 8;
            row_a = i0 + g;
            row_b = rows / 2 + i0 + g;
        } else {
            const int r0 = blockIdx.x * 16;
            row_a = r0 + g;
            row_b = r0 + 8 + g;
        }
    }
    const bool ok_a = row_a < rows, ok_b = row_b < rows;
    const int kblocks = K >> 5;

    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    // A warp's share of K, in 64-element chunks. The contraction is taken
    // in a permuted k order: within a chunk, lane quarter t owns elements
    // [16t, 16t + 16), and MMA step s contracts its pair 2s, 2s + 1. Weights
    // and activations are permuted alike, so the sum is the same set of
    // products, and a lane's whole chunk is one or two wide loads a row
    // instead of eight narrow ones. Loading a pair at a time put sixteen
    // cache lines through the load pipeline every step and held the launch
    // near 250 GB/s of weights; staging the wide loads through shared
    // memory to keep the natural order was slower still (bench_inkling_dg).
    const int chunks = K >> 6;
    const int share = (chunks + 7) / 8;
    const int ch0 = warp * share, ch1 = min(ch0 + share, chunks);
    for (int ch = ch0; ch < ch1; ++ch) {
        unsigned a[2][8];
#pragma unroll
        for (int hb = 0; hb < 2; ++hb) {
            const int row = hb ? row_b : row_a;
            const bool ok = hb ? ok_b : ok_a;
            unsigned* h = a[hb];
#pragma unroll
            for (int i = 0; i < 8; ++i) h[i] = 0u;
            if (ok) {
                if (FMT == 0) {
                    const uint4 q = *(const uint4*)(w + (long long)row * K + ch * 64 + t * 16);
                    unsigned sc = ws[(long long)row * kblocks + ch * 2 + (t >> 1)];
                    sc |= sc << 16;
                    h[0] = dg_q8x2(q.x, sc);
                    h[1] = dg_q8x2(q.x >> 16, sc);
                    h[2] = dg_q8x2(q.y, sc);
                    h[3] = dg_q8x2(q.y >> 16, sc);
                    h[4] = dg_q8x2(q.z, sc);
                    h[5] = dg_q8x2(q.z >> 16, sc);
                    h[6] = dg_q8x2(q.w, sc);
                    h[7] = dg_q8x2(q.w >> 16, sc);
                } else if (FMT == 1) {
                    const uint4* p = (const uint4*)(w + ((long long)row * K + ch * 64 + t * 16) * 2);
                    const uint4 v0 = p[0], v1 = p[1];
                    h[0] = dg_bf16x2(v0.x, wscale);
                    h[1] = dg_bf16x2(v0.y, wscale);
                    h[2] = dg_bf16x2(v0.z, wscale);
                    h[3] = dg_bf16x2(v0.w, wscale);
                    h[4] = dg_bf16x2(v1.x, wscale);
                    h[5] = dg_bf16x2(v1.y, wscale);
                    h[6] = dg_bf16x2(v1.z, wscale);
                    h[7] = dg_bf16x2(v1.w, wscale);
                } else {
                    // Sub-block t of this row's block: its scale byte and
                    // its eight bytes, each byte one fp16 pair in byte order.
                    const unsigned char* blk = w + ((long long)row * chunks + ch) * 36;
                    const unsigned sc = dg_ue4m3x2(blk[t]);
                    const unsigned d0 = *(const unsigned*)(blk + 4 + t * 8);
                    const unsigned d1 = *(const unsigned*)(blk + 8 + t * 8);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        h[i] = lut[(d0 >> (8 * i)) & 255u];
                        h[4 + i] = lut[(d1 >> (8 * i)) & 255u];
                    }
#pragma unroll
                    for (int i = 0; i < 8; ++i)
                        asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(h[i]) : "r"(sc));
                }
            }
        }
        uint4 x0 = make_uint4(0, 0, 0, 0), x1 = x0;
        if (xcol) {
            const uint4* xp = (const uint4*)(xcol + ch * 64 + t * 16);
            x0 = xp[0];
            x1 = xp[1];
        }
        dg_mma(c, a[0][0], a[1][0], x0.x);
        dg_mma(c, a[0][1], a[1][1], x0.y);
        dg_mma(c, a[0][2], a[1][2], x0.z);
        dg_mma(c, a[0][3], a[1][3], x0.w);
        dg_mma(c, a[0][4], a[1][4], x1.x);
        dg_mma(c, a[0][5], a[1][5], x1.y);
        dg_mma(c, a[0][6], a[1][6], x1.z);
        dg_mma(c, a[0][7], a[1][7], x1.w);
    }

    // Partials, summed by the first warp in warp order.
#pragma unroll
    for (int i = 0; i < 4; ++i) part[warp][lane][i] = c[i];
    __syncthreads();
    if (warp != 0) return;
#pragma unroll
    for (int wv = 1; wv < 8; ++wv)
#pragma unroll
        for (int i = 0; i < 4; ++i) c[i] += part[wv][lane][i];

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
    extern "C" __global__ void __launch_bounds__(256) NAME(                               \
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
        unsafe { b.launch(cfg((grid.0, grid.1, 1), 256, 0)) }?;
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
        if tokens == 0 || tokens > DG_MAX_ROWS || !k.is_multiple_of(64) {
            return Err(InklingError::Shape(
                "decode GEMM takes 1..=8 rows, K a multiple of 64",
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
            ((rows / 2).div_ceil(8), pairs),
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

/// Per-row state pointers a decode kernel takes: key cache, value cache,
/// then the key, value, attention-output and feed-forward carries.
pub const DEC_ROW_POINTERS: usize = 6;

/// The fused decode glue (`device-kernels/src/inkling_decode.rs`): what a
/// layer does between its GEMMs when every row is the next position of its
/// own sequence. Each launch covers all rows; a row's own state is reached
/// through `ptrs`, `[rows][DEC_ROW_POINTERS]` device addresses.
pub struct InklingDecodeKernels {
    inputs: CudaFunction,
    kv: CudaFunction,
    attn: CudaFunction,
    ffn_in: CudaFunction,
    ffn_out: CudaFunction,
    gate: CudaFunction,
    ctrl: CudaFunction,
    norm: CudaFunction,
    router: CudaFunction,
    route: CudaFunction,
}

/// Where a layer's projections sit in a row of the projection pack's
/// output, in floats. A negative tap offset means the layer has none.
#[derive(Debug, Clone, Copy)]
pub struct ProjOffsets {
    /// Row stride of the pack output.
    pub ldo: i32,
    /// Query projection.
    pub q: i32,
    /// Relative-position projection.
    pub rel: i32,
    /// Sink projection.
    pub sink: i32,
    /// Output-gate projection.
    pub gate: i32,
    /// Key projection.
    pub k: i32,
    /// Value projection.
    pub v: i32,
    /// Key convolution taps.
    pub k_taps: i32,
    /// Value convolution taps.
    pub v_taps: i32,
}

/// A loop controller's fp32 weights: `down` `[rank][2n]`, `gain` and
/// `forget` `[n][rank]`; three of each, concatenated, for the Q, K and V
/// controllers of [`InklingDecodeKernels::inputs`].
#[derive(Clone, Copy)]
pub struct ControlWeights<'a> {
    /// Gate read.
    pub down: &'a CudaSlice<f32>,
    /// Gain write.
    pub gain: &'a CudaSlice<f32>,
    /// Forget write.
    pub forget: &'a CudaSlice<f32>,
    /// Controller rank; divides 256 and is at most 64.
    pub rank: usize,
}

fn opt<T>(s: Option<&CudaSlice<T>>, stream: &Arc<CudaStream>) -> u64 {
    use cudarc::driver::DevicePtr;
    s.map_or(0, |s| s.device_ptr(stream).0)
}

impl InklingDecodeKernels {
    /// Load the entry points.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, InklingError> {
        let m = ctx.load_module(cudarc::nvrtc::Ptx::from_src(super::RUST_KERNELS_PTX))?;
        Ok(Self {
            inputs: m.load_function("ink_dec_inputs")?,
            kv: m.load_function("ink_dec_kv")?,
            attn: m.load_function("ink_dec_attn")?,
            ffn_in: m.load_function("ink_dec_ffn_in")?,
            ffn_out: m.load_function("ink_dec_ffn_out")?,
            gate: m.load_function("ink_dec_gate")?,
            ctrl: m.load_function("ink_dec_ctrl")?,
            norm: m.load_function("ink_dec_norm")?,
            router: m.load_function("ink_dec_router")?,
            route: m.load_function("ink_dec_route")?,
        })
    }

    /// Attention inputs of every row: depth mixing over `stack` (`states`
    /// planes, `[rows][n]` each) and `cur` when `queries` is given, then
    /// the input norms into `out` (binary16 planes `[3][rows][n]`). With
    /// `msgs` (`[3][rows][n]`) the three inputs are instead left in `work`
    /// with their inverse RMS in `inv`, for [`Self::control`] to finish.
    /// Returns how many distinct planes there are.
    #[allow(clippy::too_many_arguments)]
    pub fn inputs(
        &self,
        stream: &Arc<CudaStream>,
        cur: &CudaSlice<f32>,
        stack: &CudaSlice<f32>,
        queries: Option<(&CudaSlice<f32>, usize)>,
        msgs: Option<&CudaSlice<f32>>,
        norms: &CudaSlice<f32>,
        work: &mut CudaSlice<f32>,
        out: &mut CudaSlice<u16>,
        inv: &mut CudaSlice<f32>,
        n: usize,
        states: usize,
        eps: f32,
        rows: usize,
    ) -> Result<usize, InklingError> {
        let n_queries = queries.map_or(0, |q| q.1);
        if states + 1 > MIX_MAX_STATES || n_queries > 3 {
            return Err(InklingError::Shape(
                "decode inputs: too many states or queries",
            ));
        }
        need("dec inputs cur", cur.len(), rows * n)?;
        need("dec inputs stack", stack.len(), states * rows * n)?;
        need("dec inputs work", work.len(), 3 * rows * n)?;
        need("dec inputs out", out.len(), 3 * rows * n)?;
        need("dec inputs inv", inv.len(), rows * 6)?;
        let n_norms = norms.len() / n;
        if n_norms == 0 {
            return Err(InklingError::Shape("decode inputs: no norm weight"));
        }
        if let Some(m) = msgs {
            need("dec inputs msgs", m.len(), 3 * rows * n)?;
        }
        if let Some((q, count)) = queries {
            need("dec inputs queries", q.len(), count * n)?;
        }
        let q_ptr = opt(queries.map(|q| q.0), stream);
        let m_ptr = opt(msgs, stream);
        let (n_i, s, q, nn) = (n as i32, states as i32, n_queries as i32, n_norms as i32);
        let mut b = stream.launch_builder(&self.inputs);
        b.arg(cur)
            .arg(stack)
            .arg(&q_ptr)
            .arg(&m_ptr)
            .arg(norms)
            .arg(&mut *work)
            .arg(&mut *out)
            .arg(&mut *inv)
            .arg(&n_i)
            .arg(&s)
            .arg(&q)
            .arg(&nn)
            .arg(&eps);
        // SAFETY: lengths checked; 256 threads with the reductions' dynamic
        // shared memory.
        unsafe { b.launch(cfg((rows, 1, 1), 256, REDUCE_SHARED)) }?;
        Ok(if msgs.is_some() { 3 } else { n_queries.max(1) })
    }

    /// Loop controllers and norms over `ctrls` planes of `x`
    /// (`[ctrls][rows][n]`, in place): the gate read, the gain/forget
    /// update against `msg`, then the RMS norm into `out` (binary16, in
    /// byte order when `byte_order`). `inv` is what [`Self::inputs`] or
    /// [`Self::ffn_in`] left; `gate` is `[rows][ctrls][rank]` scratch.
    #[allow(clippy::too_many_arguments)]
    pub fn control(
        &self,
        stream: &Arc<CudaStream>,
        x: &mut CudaSlice<f32>,
        msg: &CudaSlice<f32>,
        c: ControlWeights<'_>,
        norms: &CudaSlice<f32>,
        out: &mut CudaSlice<u16>,
        inv: &CudaSlice<f32>,
        gate: &mut CudaSlice<f32>,
        n: usize,
        ctrls: usize,
        eps: f32,
        byte_order: bool,
        rows: usize,
    ) -> Result<(), InklingError> {
        if c.rank == 0 || c.rank > 64 || ctrls == 0 {
            return Err(InklingError::Shape("controller rank must be 1..=64"));
        }
        let n_norms = norms.len() / n;
        if n_norms == 0 {
            return Err(InklingError::Shape("decode control: no norm weight"));
        }
        for (what, len, want) in [
            ("dec control x", x.len(), ctrls * rows * n),
            ("dec control msg", msg.len(), ctrls * rows * n),
            ("dec control out", out.len(), ctrls * rows * n),
            ("dec control inv", inv.len(), rows * ctrls * 2),
            ("dec control gate", gate.len(), rows * ctrls * c.rank),
            ("dec control down", c.down.len(), ctrls * c.rank * 2 * n),
            ("dec control gain", c.gain.len(), ctrls * n * c.rank),
            ("dec control forget", c.forget.len(), ctrls * n * c.rank),
        ] {
            need(what, len, want)?;
        }
        let (n_i, r, cc, nn, nv) = (
            n as i32,
            c.rank as i32,
            ctrls as i32,
            n_norms as i32,
            i32::from(byte_order),
        );
        let mut b = stream.launch_builder(&self.gate);
        b.arg(&*x)
            .arg(msg)
            .arg(inv)
            .arg(c.down)
            .arg(&mut *gate)
            .arg(&n_i)
            .arg(&r)
            .arg(&cc);
        // SAFETY: lengths checked.
        unsafe { b.launch(cfg((c.rank, rows * ctrls, 1), 256, REDUCE_SHARED)) }?;
        let mut b = stream.launch_builder(&self.ctrl);
        b.arg(&mut *x)
            .arg(msg)
            .arg(&*gate)
            .arg(c.gain)
            .arg(c.forget)
            .arg(&n_i)
            .arg(&r)
            .arg(&cc);
        // SAFETY: lengths checked.
        unsafe { b.launch(cfg((n.div_ceil(256), rows * ctrls, 1), 256, 0)) }?;
        let mut b = stream.launch_builder(&self.norm);
        b.arg(&mut *x)
            .arg(norms)
            .arg(&mut *out)
            .arg(&n_i)
            .arg(&nn)
            .arg(&eps)
            .arg(&nv);
        // SAFETY: lengths checked.
        unsafe { b.launch(cfg((ctrls, rows, 1), 256, REDUCE_SHARED)) }?;
        Ok(())
    }

    /// Sigmoid top-k routing of rows of `x` (`[rows][n]`, fp32): router
    /// logits into `logits` (`[rows][experts]`), then `ids` and `weights`
    /// (`[rows][topk]`), the weights normalized and times `weight_scale`.
    #[allow(clippy::too_many_arguments)]
    pub fn route(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        router: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        logits: &mut CudaSlice<f32>,
        ids: &mut CudaSlice<i32>,
        weights: &mut CudaSlice<f32>,
        n: usize,
        experts: usize,
        topk: usize,
        weight_scale: f32,
        rows: usize,
    ) -> Result<(), InklingError> {
        if experts == 0
            || experts > 256
            || !experts.is_multiple_of(8)
            || topk > 32
            || topk > experts
            || n < 32
        {
            return Err(InklingError::Shape(
                "decode routing: experts a multiple of 8 up to 256, top-k <= 32",
            ));
        }
        need("dec route x", x.len(), rows * n)?;
        need("dec route router", router.len(), experts * n)?;
        need("dec route bias", bias.len(), experts)?;
        need("dec route logits", logits.len(), rows * experts)?;
        need("dec route ids", ids.len(), rows * topk)?;
        need("dec route weights", weights.len(), rows * topk)?;
        let (n_i, e, k) = (n as i32, experts as i32, topk as i32);
        let mut b = stream.launch_builder(&self.router);
        b.arg(x).arg(router).arg(&mut *logits).arg(&n_i).arg(&e);
        // SAFETY: lengths checked.
        unsafe { b.launch(cfg((experts / 8, rows, 1), 256, 0)) }?;
        let mut b = stream.launch_builder(&self.route);
        b.arg(&*logits)
            .arg(bias)
            .arg(&mut *ids)
            .arg(&mut *weights)
            .arg(&e)
            .arg(&k)
            .arg(&weight_scale);
        // SAFETY: lengths checked.
        unsafe { b.launch(cfg((rows, 1, 1), 256, 0)) }?;
        Ok(())
    }

    /// Keys and values of every row: convolution over the row's carry, key
    /// clip or value norm, and the append to the row's cache at `pos`.
    #[allow(clippy::too_many_arguments)]
    pub fn kv(
        &self,
        stream: &Arc<CudaStream>,
        proj: &CudaSlice<f32>,
        ptrs: &CudaSlice<u64>,
        pos: &CudaSlice<i32>,
        k_conv: &CudaSlice<f32>,
        v_conv: &CudaSlice<f32>,
        o: ProjOffsets,
        kv_heads: usize,
        dim: usize,
        taps: usize,
        clip: f32,
        eps: f32,
        rows: usize,
    ) -> Result<(), InklingError> {
        if !dim.is_multiple_of(32) || dim > 128 {
            return Err(InklingError::Shape(
                "decode kv: head width must be 32..=128 in warps",
            ));
        }
        need("dec kv proj", proj.len(), rows * o.ldo as usize)?;
        need("dec kv ptrs", ptrs.len(), rows * DEC_ROW_POINTERS)?;
        need("dec kv pos", pos.len(), rows)?;
        need("dec kv k_conv", k_conv.len(), kv_heads * dim * taps)?;
        need("dec kv v_conv", v_conv.len(), kv_heads * dim * taps)?;
        let (kv, t) = (kv_heads as i32, taps as i32);
        let mut b = stream.launch_builder(&self.kv);
        b.arg(proj)
            .arg(ptrs)
            .arg(pos)
            .arg(k_conv)
            .arg(v_conv)
            .arg(&o.ldo)
            .arg(&o.k)
            .arg(&o.v)
            .arg(&o.k_taps)
            .arg(&o.v_taps)
            .arg(&kv)
            .arg(&t)
            .arg(&clip)
            .arg(&eps);
        // SAFETY: lengths checked; the pointer table is the caller's promise
        // that each row's caches and carries are live and large enough.
        unsafe { b.launch(cfg((2 * kv_heads, rows, 1), dim as u32, 0)) }?;
        Ok(())
    }

    /// Attention of every row over its cache, clips and output gate
    /// included; `out` is binary16 `[rows][heads][dim]`. `log` is the
    /// logarithmic position scale of a full-attention layer.
    #[allow(clippy::too_many_arguments)]
    pub fn attn(
        &self,
        stream: &Arc<CudaStream>,
        proj: &CudaSlice<f32>,
        ptrs: &CudaSlice<u64>,
        pos: &CudaSlice<i32>,
        q_clip: &CudaSlice<f32>,
        rel_clip: &CudaSlice<f32>,
        rel_bias: &CudaSlice<f32>,
        out: &mut CudaSlice<u16>,
        o: ProjOffsets,
        heads: usize,
        kv_heads: usize,
        dim: usize,
        n_rel: usize,
        window: usize,
        extent: usize,
        qk_clip: f32,
        rel_bound: f32,
        log: Option<(f32, f32)>,
        rows: usize,
    ) -> Result<(), InklingError> {
        if dim > 128 || !dim.is_multiple_of(4) || n_rel > 32 || !heads.is_multiple_of(kv_heads) {
            return Err(InklingError::Shape("decode attention head geometry"));
        }
        need("dec attn proj", proj.len(), rows * o.ldo as usize)?;
        need("dec attn ptrs", ptrs.len(), rows * DEC_ROW_POINTERS)?;
        need("dec attn pos", pos.len(), rows)?;
        need("dec attn q_clip", q_clip.len(), heads * dim)?;
        need("dec attn rel_clip", rel_clip.len(), heads * n_rel)?;
        need("dec attn rel_bias", rel_bias.len(), extent * n_rel)?;
        need("dec attn out", out.len(), rows * heads * dim)?;
        let (kv, d, nr, w, e) = (
            kv_heads as i32,
            dim as i32,
            n_rel as i32,
            window as i32,
            extent as i32,
        );
        let (alpha, floor) = log.unwrap_or((-1.0, 1.0));
        let mut b = stream.launch_builder(&self.attn);
        b.arg(proj)
            .arg(ptrs)
            .arg(pos)
            .arg(q_clip)
            .arg(rel_clip)
            .arg(rel_bias)
            .arg(&mut *out)
            .arg(&o.ldo)
            .arg(&o.q)
            .arg(&o.rel)
            .arg(&o.sink)
            .arg(&o.gate)
            .arg(&kv)
            .arg(&d)
            .arg(&nr)
            .arg(&w)
            .arg(&e)
            .arg(&qk_clip)
            .arg(&rel_bound)
            .arg(&alpha)
            .arg(&floor);
        // SAFETY: lengths checked; caches reached through the pointer table.
        unsafe { b.launch(cfg((heads, rows, 1), 256, 0)) }?;
        Ok(())
    }

    /// From the attention output projection to the feed-forward input:
    /// output convolution and residual (or a new residual block when
    /// `opens`, whose input becomes stack plane `states`), depth mixing
    /// into `work`, then the norm (`work` in place and `out`, binary16, in
    /// byte order when `byte_order`). With `msg` the norm is left to
    /// [`Self::control`] and `inv` is written for it.
    #[allow(clippy::too_many_arguments)]
    pub fn ffn_in(
        &self,
        stream: &Arc<CudaStream>,
        o_raw: &CudaSlice<f32>,
        input: &CudaSlice<f32>,
        conv_w: &CudaSlice<f32>,
        ptrs: &CudaSlice<u64>,
        stack: &mut CudaSlice<f32>,
        query: &CudaSlice<f32>,
        msg: Option<&CudaSlice<f32>>,
        norm_w: &CudaSlice<f32>,
        ffn_inp: &mut CudaSlice<f32>,
        work: &mut CudaSlice<f32>,
        out: &mut CudaSlice<u16>,
        inv: &mut CudaSlice<f32>,
        n: usize,
        states: usize,
        opens: bool,
        taps: usize,
        byte_order: bool,
        eps: f32,
        rows: usize,
    ) -> Result<(), InklingError> {
        if states + usize::from(opens) + 1 > MIX_MAX_STATES {
            return Err(InklingError::Shape("decode ffn: too many residual blocks"));
        }
        for (what, len, want) in [
            ("dec ffn o_raw", o_raw.len(), rows * n),
            ("dec ffn input", input.len(), rows * n),
            ("dec ffn conv", conv_w.len(), n * taps),
            (
                "dec ffn stack",
                stack.len(),
                (states + usize::from(opens)) * rows * n,
            ),
            ("dec ffn query", query.len(), n),
            ("dec ffn norm", norm_w.len(), n),
            ("dec ffn ffn_inp", ffn_inp.len(), rows * n),
            ("dec ffn work", work.len(), rows * n),
            ("dec ffn ptrs", ptrs.len(), rows * DEC_ROW_POINTERS),
            ("dec ffn inv", inv.len(), rows * 2),
        ] {
            need(what, len, want)?;
        }
        need("dec ffn out", out.len(), rows * n)?;
        if let Some(m) = msg {
            need("dec ffn msg", m.len(), rows * n)?;
        }
        let m_ptr = opt(msg, stream);
        let ints = [
            n as i32,
            states as i32,
            i32::from(opens),
            taps as i32,
            i32::from(byte_order),
        ];
        let mut b = stream.launch_builder(&self.ffn_in);
        b.arg(o_raw)
            .arg(input)
            .arg(conv_w)
            .arg(ptrs)
            .arg(&mut *stack)
            .arg(query)
            .arg(&m_ptr)
            .arg(norm_w)
            .arg(&mut *ffn_inp)
            .arg(&mut *work)
            .arg(&mut *out)
            .arg(&mut *inv);
        for v in &ints {
            b.arg(v);
        }
        b.arg(&eps);
        // SAFETY: lengths checked; carries reached through the pointer table.
        unsafe { b.launch(cfg((rows, 1, 1), 256, REDUCE_SHARED)) }?;
        Ok(())
    }

    /// The end of a layer: `h = ffn_inp + conv(sum of the row's weighted
    /// expert outputs)`. `partial` is `[rows * topk][n]`.
    #[allow(clippy::too_many_arguments)]
    pub fn ffn_out(
        &self,
        stream: &Arc<CudaStream>,
        partial: &CudaSlice<f32>,
        ffn_inp: &CudaSlice<f32>,
        conv_w: &CudaSlice<f32>,
        ptrs: &CudaSlice<u64>,
        h: &mut CudaSlice<f32>,
        n: usize,
        topk: usize,
        taps: usize,
        rows: usize,
    ) -> Result<(), InklingError> {
        need("dec out partial", partial.len(), rows * topk * n)?;
        need("dec out ffn_inp", ffn_inp.len(), rows * n)?;
        need("dec out conv", conv_w.len(), n * taps)?;
        need("dec out ptrs", ptrs.len(), rows * DEC_ROW_POINTERS)?;
        need("dec out h", h.len(), rows * n)?;
        let (n_i, k, t) = (n as i32, topk as i32, taps as i32);
        let mut b = stream.launch_builder(&self.ffn_out);
        b.arg(partial)
            .arg(ffn_inp)
            .arg(conv_w)
            .arg(ptrs)
            .arg(&mut *h)
            .arg(&n_i)
            .arg(&k)
            .arg(&t);
        // SAFETY: lengths checked; carries reached through the pointer table.
        unsafe { b.launch(cfg((n.div_ceil(256), rows, 1), 256, 0)) }?;
        Ok(())
    }
}
