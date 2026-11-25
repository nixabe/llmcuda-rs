//! Phase 1.1 of the TensorRT trunk plan: TensorRT's GEMMs against the
//! engine's, at every Qwen3.6-35B-A3B Q8_0 projection shape.
//!
//! ```sh
//! export TENSORRT_ROOT=/path/to/tensorrt-11 CUDA_ROOT=/path/to/cuda-12
//! export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:$CUDA_ROOT/lib:$LD_LIBRARY_PATH
//! CUDA_VISIBLE_DEVICES=<idle card> \
//!     cargo run --release -p llmcuda-engine --features tensorrt --bin bench_trt_gemm
//! ```
//!
//! Every arm takes the same fp32 activations to an fp32 output, including
//! its own activation narrowing, because each is a cost the others do not
//! pay:
//!
//! - `int8`: `quantize_rows` + `q8_0_proj_split_half`, the path these
//!   projections take today. The incumbent.
//! - `hgemm`: `to_half` + the engine's own fp16 GEMM, for reference.
//! - `trt-exact`: TensorRT, fp32 in, Cast, fp16 GEMM, Cast, fp32 out, one
//!   exact-shape profile per token count, as the plan specifies.
//! - `trt-dyn`: the same network with one dynamic profile, the only form
//!   whose rows did not depend on the row count in `trt_probe` U6.
//! - `trt-q8dq`: INT8 codes with a fp16 scale per 32 (a lossless Q8_0)
//!   dequantized in-engine, dynamic profile.
//! - `trt-int4`: INT4 with one scale per output channel, requantized from
//!   the Q8_0 weight, dynamic profile. Lossy, so never a candidate under the
//!   plan's gates; measured to say what INT4 buys in TensorRT on this card.
//!
//! Weights are synthetic Q8_0 blocks at the real shapes (timing does not
//! depend on values); activations are uniform in [-1, 1].
//!
//! Correctness, before any timing:
//!
//! - The fp16 arms against the contract `hgemm_differential.rs` gates the
//!   engine's own fp16 GEMM with: `sum half(x) * half(w)` in f64, bound
//!   `2 k |sum| 2^-24 + 1e-6`. TensorRT's MatMul output is fp16 in a
//!   strongly typed network, so its result is rounded to fp16 before the
//!   Cast back; the distance to the fp16-rounded oracle is reported beside
//!   the bound, which says whether output rounding is all the error there is.
//! - Every arm against the exact product `sum x * w` in f64, for scale.
//! - Batch identity: the leading 512 rows must be bit-identical whatever
//!   the pass's row count.
//!
//! Timing: CUDA events around 10 calls after 3 warm-ups; per token count,
//! `LLMCUDA_TRT_ROUNDS` (default 5) rounds with the arm order rotated each
//! round, so drift lands on every arm. Medians and spreads are reported,
//! and a per-pass total weighted by how many of each matrix one forward pass
//! has.

use std::sync::Arc;

use cudarc::driver::sys::{CUevent_flags, CUstream};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr};
use half::f16;
use llmcuda_cuda::kernels::hgemm::{HalfWeight, HgemmKernels};
use llmcuda_cuda::kernels::mma::MmaKernels;
use llmcuda_kernels::quant::quantize_q8_0;
use llmcuda_kernels::rng::Xorshift64Star;
use llmcuda_trt::{
    AllocationStrategy, Builder, DataType, Dims, Engine, ExecutionContext, HostMemory, MatrixOp,
    Runtime,
};
use tracing::{error, info};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// `(name, output rows, contraction, matrices of this shape per forward
/// pass)`, from the Q8_0 tensors of `Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf`.
/// The fp32 router and GDN alpha/beta gates are left out: rounding the
/// router to fp16 can change which experts are chosen.
const SHAPES: [(&str, usize, usize, usize); 4] = [
    ("attn_qkv, attn_q", 8192, 2048, 40),
    ("attn_gate", 4096, 2048, 30),
    ("ssm_out, attn_output", 2048, 4096, 40),
    ("attn_k, attn_v", 512, 2048, 20),
];
const TOKENS: [usize; 3] = [512, 2048, 6144];
/// Leading rows every pass shares, for the batch-identity check.
const LEAD: usize = 512;
/// Rows checked against the f64 oracles, all inside the shared lead.
const SAMPLE: [usize; 8] = [0, 1, 2, 3, 100, 255, 256, 511];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arm {
    Int8,
    Hgemm,
    TrtExact,
    TrtDyn,
    TrtQ8Dq,
    TrtInt4,
}

const ARMS: [Arm; 6] = [
    Arm::Int8,
    Arm::Hgemm,
    Arm::TrtExact,
    Arm::TrtDyn,
    Arm::TrtQ8Dq,
    Arm::TrtInt4,
];

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Self::Int8 => "int8",
            Self::Hgemm => "hgemm",
            Self::TrtExact => "trt-exact",
            Self::TrtDyn => "trt-dyn",
            Self::TrtQ8Dq => "trt-q8dq",
            Self::TrtInt4 => "trt-int4",
        }
    }

    /// Whether the arm's contract is `sum half(x) * half(w)`.
    fn is_half_contract(self) -> bool {
        matches!(
            self,
            Self::Hgemm | Self::TrtExact | Self::TrtDyn | Self::TrtQ8Dq
        )
    }
}

/// One projection's weight in every form the arms read.
struct Weights {
    /// GGUF Q8_0 blocks, `[rows][k / 32]` of 34 bytes.
    q8_0: Vec<u8>,
    /// `[rows][k]` codes and `[rows][k / 32]` scales of the same blocks.
    codes: Vec<i8>,
    scales: Vec<f16>,
    /// `q * d`, exact in f32, and its fp16 rounding.
    exact: Vec<f32>,
    half: Vec<f16>,
    /// Per-row INT4 requantization: `[rows][k]` codes, `[rows]` scales.
    int4: Vec<i8>,
    int4_scales: Vec<f16>,
}

impl Weights {
    fn new(rows: usize, k: usize, seed: u64) -> Self {
        let mut rng = Xorshift64Star::new(seed);
        let mut w = Self {
            q8_0: Vec::with_capacity(rows * k / 32 * 34),
            codes: Vec::with_capacity(rows * k),
            scales: Vec::with_capacity(rows * k / 32),
            exact: Vec::with_capacity(rows * k),
            half: Vec::with_capacity(rows * k),
            int4: Vec::with_capacity(rows * k),
            int4_scales: Vec::with_capacity(rows),
        };
        for r in 0..rows {
            // Block magnitudes vary by row, as `hgemm_differential.rs` does,
            // so scales span several binades.
            let scale = 0.01 + (r % 7) as f32 * 0.03;
            for _ in 0..k / 32 {
                let mut block = [0f32; 32];
                for v in &mut block {
                    *v = rng.next_f32_range(-scale, scale);
                }
                let q = quantize_q8_0(&block);
                w.q8_0.extend_from_slice(&q.d.to_bits().to_le_bytes());
                w.q8_0.extend(q.qs.iter().map(|&v| v as u8));
                w.codes.extend_from_slice(&q.qs);
                w.scales.push(q.d);
                let d = q.d.to_f32();
                for &c in &q.qs {
                    let e = f32::from(c) * d;
                    w.exact.push(e);
                    w.half.push(f16::from_f32(e));
                }
            }
            let row = &w.exact[r * k..(r + 1) * k];
            let amax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
            let s = f16::from_f32((amax / 7.0).max(f32::MIN_POSITIVE));
            let sf = s.to_f32();
            w.int4_scales.push(s);
            w.int4
                .extend(row.iter().map(|&v| (v / sf).round().clamp(-8.0, 7.0) as i8));
        }
        w
    }
}

fn half_bytes(v: &[f16]) -> Vec<u8> {
    v.iter().flat_map(|h| h.to_bits().to_le_bytes()).collect()
}

/// TensorRT's INT4 packing, low nibble first (`trt_probe` U3: the other
/// order computes garbage).
fn pack_int4(codes: &[i8]) -> Vec<u8> {
    codes
        .chunks(2)
        .map(|p| (p[0] as u8 & 0x0F) | (p.get(1).map_or(0, |&v| v as u8 & 0x0F) << 4))
        .collect()
}

/// `x [-1, k] fp32 -> Cast -> x @ W^T -> Cast -> y fp32`, with W in the
/// arm's form, and `profiles` as `(min, opt, max)` token counts.
fn build_plan(
    arm: Arm,
    w: &Weights,
    rows: usize,
    k: usize,
    profiles: &[(i64, i64, i64)],
) -> Result<HostMemory> {
    let (rows_i, k_i) = (rows as i64, k as i64);
    let builder = Builder::new()?;
    let network = builder.network()?;
    let x = network.input("x", DataType::Float, [-1, k_i])?;
    let x16 = network.cast(x, DataType::Half)?;
    let weight = match arm {
        Arm::TrtExact | Arm::TrtDyn => {
            network.constant([rows_i, k_i], DataType::Half, half_bytes(&w.half), None)?
        }
        Arm::TrtQ8Dq => {
            let q = network.constant(
                [rows_i, k_i],
                DataType::Int8,
                w.codes.iter().map(|&c| c as u8).collect(),
                None,
            )?;
            let s = network.constant(
                [rows_i, k_i / 32],
                DataType::Half,
                half_bytes(&w.scales),
                None,
            )?;
            network.dequantize(q, s, DataType::Half, None, Some([1, 32].into()))?
        }
        Arm::TrtInt4 => {
            let q = network.constant([rows_i, k_i], DataType::Int4, pack_int4(&w.int4), None)?;
            let s = network.constant([rows_i], DataType::Half, half_bytes(&w.int4_scales), None)?;
            network.dequantize(q, s, DataType::Half, Some(0), None)?
        }
        Arm::Int8 | Arm::Hgemm => unreachable!("not a TensorRT arm"),
    };
    let y16 = network.matmul(x16, MatrixOp::None, weight, MatrixOp::Transpose)?;
    let y = network.cast(y16, DataType::Float)?;
    y.set_name("y")?;
    network.mark_output(y)?;
    let config = builder.config()?;
    config.set_max_aux_streams(0)?;
    for &(min, opt, max) in profiles {
        let d = |t: i64| Dims::from([t, k_i]);
        config.add_profile(&[("x", d(min), d(opt), d(max))])?;
    }
    let started = std::time::Instant::now();
    let plan = builder.build(&network, &config)?;
    info!(
        "  {:9} plan {:.1} MiB, built in {:.1} s",
        arm.name(),
        plan.len() as f64 / (1 << 20) as f64,
        started.elapsed().as_secs_f64()
    );
    Ok(plan)
}

/// One context per token count for a TensorRT arm, so switching costs
/// nothing in the timed loop. With exact profiles, context `i` runs profile
/// `i`.
fn contexts<'e>(
    engine: &'e Engine<'_>,
    exact: bool,
    stream: CUstream,
) -> Result<Vec<ExecutionContext<'e>>> {
    (0..TOKENS.len())
        .map(|i| {
            let c = engine.context(AllocationStrategy::Static)?;
            if exact {
                c.set_profile(i as i32, stream)?;
            }
            Ok(c)
        })
        .collect()
}

struct Bench {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    mma: MmaKernels,
    hg: HgemmKernels,
}

impl Bench {
    fn ptr<T>(&self, s: &CudaSlice<T>) -> u64 {
        s.device_ptr(&self.stream).0
    }

    /// Mean milliseconds per call of `body`, by CUDA events.
    fn measure(&self, mut body: impl FnMut() -> Result<()>) -> Result<f32> {
        for _ in 0..3 {
            body()?;
        }
        self.stream.synchronize()?;
        let start = self.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        let stop = self.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        start.record(&self.stream)?;
        for _ in 0..10 {
            body()?;
        }
        stop.record(&self.stream)?;
        stop.synchronize()?;
        Ok(start.elapsed_ms(&stop)? / 10.0)
    }
}

/// Device state for one shape.
struct Shape<'e> {
    rows: usize,
    k: usize,
    x: CudaSlice<f32>,
    xq: CudaSlice<i8>,
    xs: CudaSlice<f32>,
    xh: CudaSlice<u16>,
    wq: CudaSlice<i8>,
    ws: CudaSlice<u16>,
    workspace: CudaSlice<f32>,
    /// Per arm, per token count.
    outs: Vec<Vec<CudaSlice<f32>>>,
    /// Per arm: its contexts, if it is a TensorRT arm that built.
    trt: Vec<Option<Vec<ExecutionContext<'e>>>>,
}

impl Shape<'_> {
    fn arm_index(arm: Arm) -> usize {
        ARMS.iter().position(|&a| a == arm).expect("listed")
    }

    /// Enqueue one call of `arm` at `TOKENS[ti]`.
    fn run(&mut self, b: &Bench, arm: Arm, ti: usize) -> Result<()> {
        let t = TOKENS[ti];
        let (rows, k) = (self.rows, self.k);
        let ai = Self::arm_index(arm);
        match arm {
            Arm::Int8 => {
                b.mma
                    .quantize_rows(&b.stream, &self.x, &mut self.xq, &mut self.xs, t, k)?;
                let out = &mut self.outs[ai][ti];
                b.mma.q8_0_proj_split_half(
                    &b.stream, &self.wq, &self.ws, &self.xq, &self.xs, out, k, rows, t,
                )?;
            }
            Arm::Hgemm => {
                b.hg.to_half(&b.stream, &self.x, &mut self.xh, t * k)?;
                let out = &mut self.outs[ai][ti];
                b.hg.gemm(
                    &b.stream,
                    HalfWeight::Q8_0 {
                        q: &self.wq,
                        scales: &self.ws,
                    },
                    &self.xh,
                    out,
                    Some(&mut self.workspace),
                    t,
                    rows,
                    k,
                    rows,
                    false,
                )?;
            }
            _ => {
                let contexts = self.trt[ai].as_ref().ok_or("arm did not build")?;
                // SAFETY: addresses were bound in `bind`, to buffers this
                // shape owns; the work is ordered on the bench's stream.
                unsafe { contexts[ti].enqueue(b.stream.cu_stream())? };
            }
        }
        Ok(())
    }

    /// Bind each TensorRT context's input shape and addresses once.
    fn bind(&self, b: &Bench) -> Result<()> {
        for (ai, contexts) in self.trt.iter().enumerate() {
            let Some(contexts) = contexts else { continue };
            for (ti, &t) in TOKENS.iter().enumerate() {
                let c = &contexts[ti];
                c.set_input_shape("x", [t as i64, self.k as i64])?;
                // SAFETY: `x` holds the widest pass and `outs[ai][ti]` holds
                // `t` rows; both live as long as the contexts.
                unsafe {
                    c.set_tensor_address("x", b.ptr(&self.x))?;
                    c.set_tensor_address("y", b.ptr(&self.outs[ai][ti]))?;
                }
            }
        }
        Ok(())
    }
}

/// Bit-level differences between two equal-length slices.
fn bit_diff(a: &[f32], b: &[f32]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

struct Oracle {
    /// `sum half(x) * half(w)` and `sum x * w`, f64, at [`SAMPLE`] rows.
    half: Vec<f64>,
    exact: Vec<f64>,
    /// The hgemm contract's bound.
    bound: f64,
}

fn oracle(x: &[f32], w: &Weights, rows: usize, k: usize) -> Oracle {
    let mut half = Vec::with_capacity(SAMPLE.len() * rows);
    let mut exact = Vec::with_capacity(SAMPLE.len() * rows);
    let mut magnitude = 0f64;
    for &t in &SAMPLE {
        let xr = &x[t * k..(t + 1) * k];
        let xh: Vec<f64> = xr
            .iter()
            .map(|&v| f64::from(f16::from_f32(v).to_f32()))
            .collect();
        for r in 0..rows {
            let (mut s, mut m, mut e) = (0f64, 0f64, 0f64);
            for i in 0..k {
                let p = xh[i] * f64::from(w.half[r * k + i].to_f32());
                s += p;
                m += p.abs();
                e += f64::from(xr[i]) * f64::from(w.exact[r * k + i]);
            }
            magnitude = magnitude.max(m);
            half.push(s);
            exact.push(e);
        }
    }
    Oracle {
        half,
        exact,
        bound: 2.0 * (k as f64) * magnitude * 2f64.powi(-24) + 1e-6,
    }
}

/// `(max |got - want|, cosine)` over the sampled rows of a lead-rows dump.
fn against(lead: &[f32], want: &[f64], rows: usize, round_want_to_half: bool) -> (f64, f64) {
    let (mut worst, mut dot, mut gg, mut ww) = (0f64, 0f64, 0f64, 0f64);
    for (si, &t) in SAMPLE.iter().enumerate() {
        for r in 0..rows {
            let g = f64::from(lead[t * rows + r]);
            let mut w = want[si * rows + r];
            if round_want_to_half {
                w = f64::from(f16::from_f64(w).to_f32());
            }
            worst = worst.max((g - w).abs());
            dot += g * w;
            gg += g * g;
            ww += w * w;
        }
    }
    (worst, dot / (gg.sqrt() * ww.sqrt()).max(f64::MIN_POSITIVE))
}

fn stats(v: &[f32]) -> (f32, f32, f32) {
    let mut s = v.to_vec();
    s.sort_by(f32::total_cmp);
    (s[0], s[s.len() / 2], s[s.len() - 1])
}

fn run() -> Result<()> {
    let rounds: usize = std::env::var("LLMCUDA_TRT_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let ctx = CudaContext::new(0)?;
    // SAFETY: one stream, every buffer kept alive until after the last
    // synchronize; TensorRT reads and writes them through raw addresses.
    unsafe { ctx.disable_event_tracking() };
    // A created stream: TensorRT synchronizes extra on the legacy default.
    let stream = ctx.new_stream()?;
    let bench = Bench {
        mma: MmaKernels::new(&ctx)?,
        hg: HgemmKernels::new(&ctx)?,
        ctx,
        stream,
    };
    info!(
        "TensorRT {:?}; tokens {TOKENS:?}; {rounds} rounds per token count",
        llmcuda_trt::library_version()
    );
    let runtime = Runtime::new()?;
    let max_t = *TOKENS.iter().max().expect("non-empty");
    // Per arm, per token count: ms summed over one forward pass's matrices.
    let mut pass_ms = vec![vec![0f64; TOKENS.len()]; ARMS.len()];
    let mut verdicts: Vec<String> = Vec::new();

    for (si, &(name, rows, k, count)) in SHAPES.iter().enumerate() {
        info!("\n== {name}: {rows} x {k}, {count} per pass");
        let w = Weights::new(rows, k, 0x7a11 + si as u64);
        let mut rng = Xorshift64Star::new(0x7a77 + si as u64);
        let x = rng.vec_f32(max_t * k, -1.0, 1.0);

        let s = &bench.stream;
        let src = s.clone_htod(&w.q8_0)?;
        let mut wq = s.alloc_zeros::<i8>(rows * k)?;
        let mut ws = s.alloc_zeros::<u16>(rows * k / 32)?;
        bench
            .mma
            .repack_q8_0_half(s, &src, &mut wq, &mut ws, rows * k)?;
        drop(src);

        let mut engines = Vec::new();
        for arm in ARMS {
            let profiles: &[(i64, i64, i64)] = match arm {
                Arm::Int8 | Arm::Hgemm => {
                    engines.push(None);
                    continue;
                }
                Arm::TrtExact => &[(512, 512, 512), (2048, 2048, 2048), (6144, 6144, 6144)],
                _ => &[(1, 2048, max_t as i64)],
            };
            match build_plan(arm, &w, rows, k, profiles)
                .and_then(|plan| Ok(runtime.deserialize(&plan)?))
            {
                Ok(e) => engines.push(Some(e)),
                Err(e) => {
                    error!("  {} did not build: {e}", arm.name());
                    engines.push(None);
                }
            }
        }
        let mut trt = Vec::new();
        for (arm, engine) in ARMS.iter().zip(&engines) {
            trt.push(match engine {
                Some(e) => Some(contexts(e, *arm == Arm::TrtExact, s.cu_stream())?),
                None => None,
            });
        }

        let mut shape = Shape {
            rows,
            k,
            x: s.clone_htod(&x)?,
            xq: s.alloc_zeros::<i8>(max_t * k)?,
            xs: s.alloc_zeros::<f32>(max_t * k / 32)?,
            xh: s.alloc_zeros::<u16>(max_t * k)?,
            wq,
            ws,
            workspace: s.alloc_zeros::<f32>(bench.hg.workspace_elems(max_t, rows, k).max(1))?,
            outs: ARMS
                .iter()
                .map(|_| {
                    TOKENS
                        .iter()
                        .map(|&t| s.alloc_zeros::<f32>(t * rows))
                        .collect()
                })
                .collect::<std::result::Result<_, _>>()?,
            trt,
        };
        shape.bind(&bench)?;

        // Correctness and batch identity first.
        let o = oracle(&x, &w, rows, k);
        for (ai, &arm) in ARMS.iter().enumerate() {
            if ai >= 2 && shape.trt[ai].is_none() {
                continue;
            }
            let mut leads = Vec::new();
            for ti in 0..TOKENS.len() {
                shape.run(&bench, arm, ti)?;
                leads.push(s.clone_dtoh(&shape.outs[ai][ti].slice(0..LEAD * rows))?);
            }
            s.synchronize()?;
            let identity: Vec<usize> = leads[1..].iter().map(|l| bit_diff(&leads[0], l)).collect();
            let identical = identity.iter().all(|&d| d == 0);
            // Worst over every token count: rows that are not identical
            // across passes can be wrong in only one of them.
            let worst = |want: &[f64], round: bool| {
                leads
                    .iter()
                    .map(|l| against(l, want, rows, round))
                    .fold((0f64, 1f64), |a, b| (a.0.max(b.0), a.1.min(b.1)))
            };
            let (ex_abs, ex_cos) = worst(&o.exact, false);
            let contract = if arm.is_half_contract() {
                let (h_abs, _) = worst(&o.half, false);
                let (r_abs, _) = worst(&o.half, true);
                let ok = h_abs <= o.bound;
                verdicts.push(format!(
                    "{name:22} {:9} contract {} (max_abs {h_abs:.3e} vs bound {:.3e}; {r_abs:.3e} \
                     from the fp16-rounded oracle); rows identical across T: {identical}",
                    arm.name(),
                    if ok { "PASS" } else { "FAIL" },
                    o.bound,
                ));
                format!(
                    "half contract max_abs {h_abs:.3e} (bound {:.3e}, {}), vs fp16-rounded oracle \
                     {r_abs:.3e}; ",
                    o.bound,
                    if ok { "pass" } else { "FAIL" }
                )
            } else {
                verdicts.push(format!(
                    "{name:22} {:9} rows identical across T: {identical}",
                    arm.name()
                ));
                String::new()
            };
            info!(
                "  {:9} {contract}vs exact product max_abs {ex_abs:.3e} cosine {ex_cos:.7}; lead-row \
                 bit differences 512->2048 {}, 512->6144 {}",
                arm.name(),
                identity[0],
                identity[1]
            );
        }

        // Timing: rounds with the arm order rotated, per token count.
        let live: Vec<Arm> = ARMS
            .iter()
            .enumerate()
            .filter(|&(ai, _)| ai < 2 || shape.trt[ai].is_some())
            .map(|(_, &a)| a)
            .collect();
        for (ti, &t) in TOKENS.iter().enumerate() {
            let mut ms = vec![Vec::new(); ARMS.len()];
            for r in 0..rounds {
                for j in 0..live.len() {
                    let arm = live[(j + r) % live.len()];
                    let ai = Shape::arm_index(arm);
                    ms[ai].push(bench.measure(|| shape.run(&bench, arm, ti))?);
                }
            }
            let flop = 2.0 * (t * rows * k) as f64;
            for &arm in &live {
                let ai = Shape::arm_index(arm);
                let (lo, med, hi) = stats(&ms[ai]);
                pass_ms[ai][ti] += f64::from(med) * count as f64;
                let ratios: Vec<f32> = ms[0].iter().zip(&ms[ai]).map(|(a, b)| a / b).collect();
                let (rlo, rmed, rhi) = stats(&ratios);
                info!(
                    "  T={t:5} {:9} {med:8.3} ms [{lo:.3}, {hi:.3}]  {:6.1} TFLOP/s  vs int8 x{rmed:.3} \
                     [{rlo:.3}, {rhi:.3}]",
                    arm.name(),
                    flop / (f64::from(med) * 1e9),
                );
            }
        }
    }

    info!("\n== per forward pass (sum over the 130 Q8_0 projections, median ms)");
    for (ti, &t) in TOKENS.iter().enumerate() {
        let base = pass_ms[0][ti];
        let row: Vec<String> = ARMS
            .iter()
            .enumerate()
            .filter(|&(ai, _)| pass_ms[ai][ti] > 0.0)
            .map(|(ai, a)| {
                format!(
                    "{} {:.2} ms (x{:.3})",
                    a.name(),
                    pass_ms[ai][ti],
                    base / pass_ms[ai][ti]
                )
            })
            .collect();
        info!("  T={t:5}: {}", row.join(", "));
    }
    info!("\n== gates");
    for v in &verdicts {
        info!("  {v}");
    }
    Ok(())
}

fn main() {
    llmcuda_log::init_from_args();
    if let Err(e) = run() {
        error!("bench_trt_gemm failed: {e}");
        std::process::exit(1);
    }
}
