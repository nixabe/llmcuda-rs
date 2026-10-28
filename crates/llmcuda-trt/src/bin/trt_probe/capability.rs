//! P6: capability probes. Each answers a question the plan left open by
//! running it, not by reading documentation. These are recorded, not gated:
//! `Pass` means "yes, demonstrated", `Fail` means "no", with the reason
//! TensorRT gave.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use half::f16;
use llmcuda_cuda::CheckOutcome;
use llmcuda_kernels::attention::causal_attention_naive;
use llmcuda_kernels::compare::{ComparisonResult, compare};
use llmcuda_kernels::norm::residual_add;
use llmcuda_kernels::quant::{
    QK4_0, QK8_0, dequantize_q4_0, dequantize_q8_0, quantize_q4_0, quantize_q8_0,
};
use llmcuda_kernels::rng::Xorshift64Star;
use llmcuda_trt::{
    AllocationStrategy, Builder, BuilderFlag, CausalMask, DataType, DebugListener, Dims,
    ElementWise, MatrixOp, OnnxParser, ProfilingVerbosity, Runtime, TensorLocation, TimingEntry,
};
use tracing::info;

use crate::env::{Env, Error};
use crate::gate::{Check, bit_diff, build, config, exact_operands, half_bytes};
use crate::onnx::{self, Attr, Elem, Graph, Initializer, Node};
use crate::plugins::{self, NAMESPACE};

pub fn run_all(env: &Env, selected: &dyn Fn(&str) -> bool) -> Vec<(&'static str, CheckOutcome)> {
    let checks: [Check; 7] = [
        ("U2 INT8 block-32 DQ -> MatMul (Q8_0)", u2),
        ("U3 INT4 block-32 weight-only (Q4_0)", u3),
        ("U4 stripped plan + device refit", u4),
        ("U5 ONNX custom op -> IPluginV3 creator", u5),
        ("U6 GEMM rows identical across profiles", u6),
        ("U8 markDebug waypoint capture", u8),
        ("IAttention fp16 GQA 16:2 hd256 causal", attention),
    ];
    checks
        .into_iter()
        .filter(|(label, _)| selected(label))
        .map(|(label, check)| {
            info!("{label}");
            let outcome = check(env).unwrap_or_else(|e| CheckOutcome::Fail(format!("no: {e}")));
            info!("  -> {outcome}");
            (label, outcome)
        })
        .collect()
}

fn fmt_cmp(c: &ComparisonResult) -> String {
    format!(
        "max_abs {:.3e}, cosine {:.7}",
        c.max_abs_error, c.cosine_similarity
    )
}

/// `name (type)` of each engine layer, from the inspector's JSON, without a
/// JSON parser.
///
/// A detailed-verbosity engine lists layer objects that open with
/// `"Name": ..., "LayerType": ...`; tensor objects inside them carry a
/// `"Name"` too but no type, which is how they are told apart. The raw JSON
/// goes to `debug!` for anything this misses.
fn layer_names(json: &str) -> Vec<String> {
    tracing::debug!("engine information: {json}");
    fn quoted(s: &str) -> Option<(&str, &str)> {
        let s = s
            .trim_start()
            .strip_prefix(':')?
            .trim_start()
            .strip_prefix('"')?;
        let end = s.find('"')?;
        Some((&s[..end], &s[end + 1..]))
    }
    json.split("\"Name\"")
        .skip(1)
        .filter_map(|s| {
            let (name, rest) = quoted(s)?;
            let rest = rest.trim_start().strip_prefix(',')?.trim_start();
            let (kind, _) = quoted(rest.strip_prefix("\"LayerType\"")?)?;
            Some(format!("{name} ({kind})"))
        })
        .collect()
}

/// The weight layouts a block-quantized projection could take.
#[derive(Clone, Copy, Debug)]
enum Layout {
    /// `W [N, K]`, blocks along K (the last axis), `X @ W^T`. GGUF's layout.
    RowsNK,
    /// `W [K, N]`, blocks along K (the first axis), `X @ W`.
    RowsKN,
}

/// Build and run `x16 @ dequant(w)` for a block-quantized `w`, in both
/// layouts. `codes` are `[N][K]` signed integers, `scales` `[N][K/32]`.
fn block_dq_matmul(
    env: &Env,
    qtype: DataType,
    problem: &QuantProblem,
    pack: impl Fn(&[i8]) -> Vec<u8>,
) -> Result<String, Error> {
    let (m, k, n) = (QM, QK, QN);
    let QuantProblem {
        codes,
        scales,
        x,
        reference,
    } = problem;
    let mut notes = Vec::new();
    for layout in [Layout::RowsNK, Layout::RowsKN] {
        let builder = Builder::new()?;
        let network = builder.network()?;
        let input = network.input("x", DataType::Float, [m, k])?;
        let x16 = network.cast(input, DataType::Half)?;
        let blocks = k / 32;
        let (wdims, sdims, block, op, wcodes, wscales): (
            Dims,
            Dims,
            Dims,
            MatrixOp,
            Vec<i8>,
            Vec<f16>,
        ) = match layout {
            Layout::RowsNK => (
                [n, k].into(),
                [n, blocks].into(),
                [1, 32].into(),
                MatrixOp::Transpose,
                codes.to_vec(),
                scales.to_vec(),
            ),
            Layout::RowsKN => {
                let mut wt = vec![0i8; codes.len()];
                let mut st = vec![f16::ZERO; scales.len()];
                for r in 0..n as usize {
                    for c in 0..k as usize {
                        wt[c * n as usize + r] = codes[r * k as usize + c];
                    }
                    for bi in 0..blocks as usize {
                        st[bi * n as usize + r] = scales[r * blocks as usize + bi];
                    }
                }
                (
                    [k, n].into(),
                    [blocks, n].into(),
                    [32, 1].into(),
                    MatrixOp::None,
                    wt,
                    st,
                )
            }
        };
        let built = (|| -> Result<(llmcuda_trt::HostMemory, String), Error> {
            let w = network.constant(wdims, qtype, pack(&wcodes), None)?;
            let s = network.constant(sdims, DataType::Half, half_bytes(&wscales), None)?;
            let wdq = network.dequantize(w, s, DataType::Half, None, Some(block))?;
            let y16 = network.matmul(x16, MatrixOp::None, wdq, op)?;
            let y = network.cast(y16, DataType::Float)?;
            y.set_name("y")?;
            network.mark_output(y)?;
            let config = config(&builder)?;
            config.set_profiling_verbosity(ProfilingVerbosity::Detailed);
            let plan = build(&builder, &network, &config)?;
            let runtime = Runtime::new()?;
            let engine = runtime.deserialize(&plan)?;
            let layers = layer_names(&engine.information()?);
            let context = engine.context(AllocationStrategy::Static)?;
            let dx = env.upload(x)?;
            let dy = env.zeros::<f32>((m * n) as usize)?;
            // SAFETY: buffers sized to the network's static shapes.
            unsafe {
                context.set_tensor_address("x", env.ptr(&dx))?;
                context.set_tensor_address("y", env.ptr(&dy))?;
                context.enqueue(env.raw_stream())?;
            }
            let got = env.download(&dy)?;
            let c = compare(&got, reference);
            Ok((
                plan,
                format!(
                    "{} engine layers {layers:?}; vs dequantized CPU: {}",
                    layers.len(),
                    fmt_cmp(&c)
                ),
            ))
        })();
        notes.push(match built {
            Ok((_, s)) => format!("{layout:?}: builds; {s}"),
            Err(e) => format!("{layout:?}: {e}"),
        });
    }
    Ok(notes.join(" | "))
}

/// A `[QM, QK] x [QN, QK]^T` projection with a block-quantized weight.
struct QuantProblem {
    /// `[QN][QK]` signed codes.
    codes: Vec<i8>,
    /// `[QN][QK / 32]`.
    scales: Vec<f16>,
    /// `[QM][QK]` activations.
    x: Vec<f32>,
    /// fp16-rounded activations times the dequantized weight, in f64.
    reference: Vec<f32>,
}

/// Random weights quantized with the engine's own reference quantizer, so
/// the TensorRT form is a lossless re-expression of a GGUF block.
fn quantized_problem(four_bit: bool) -> QuantProblem {
    let (m, k, n) = (QM, QK, QN);
    let mut rng = Xorshift64Star::new(if four_bit { 0x6403 } else { 0x6402 });
    let (k_us, n_us) = (k as usize, n as usize);
    let mut codes = vec![0i8; n_us * k_us];
    let mut scales = vec![f16::ZERO; n_us * k_us / 32];
    let mut dequant = vec![0f32; n_us * k_us];
    for r in 0..n_us {
        for b in 0..k_us / 32 {
            let mut v = [0f32; 32];
            for x in &mut v {
                *x = rng.next_f32_range(-0.08, 0.08);
            }
            let at = r * k_us + b * 32;
            if four_bit {
                let block = quantize_q4_0(&v);
                let d = dequantize_q4_0(&block);
                for j in 0..QK4_0 / 2 {
                    codes[at + j] = (block.qs[j] & 0x0F) as i8 - 8;
                    codes[at + j + QK4_0 / 2] = (block.qs[j] >> 4) as i8 - 8;
                }
                dequant[at..at + 32].copy_from_slice(&d);
                scales[r * (k_us / 32) + b] = block.d;
            } else {
                let block = quantize_q8_0(&v);
                let d = dequantize_q8_0(&block);
                codes[at..at + QK8_0].copy_from_slice(&block.qs);
                dequant[at..at + 32].copy_from_slice(&d);
                scales[r * (k_us / 32) + b] = block.d;
            }
        }
    }
    let x = rng.vec_f32((m * k) as usize, -1.0, 1.0);
    let mut reference = vec![0f32; (m * n) as usize];
    for i in 0..m as usize {
        for j in 0..n_us {
            let s: f64 = (0..k_us)
                .map(|c| {
                    f64::from(f16::from_f32(x[i * k_us + c]).to_f32())
                        * f64::from(dequant[j * k_us + c])
                })
                .sum();
            reference[i * n_us + j] = s as f32;
        }
    }
    QuantProblem {
        codes,
        scales,
        x,
        reference,
    }
}

const QM: i64 = 64;
const QK: i64 = 2048;
const QN: i64 = 1024;

/// U2: can a Q8_0 weight be expressed losslessly (INT8 codes, one fp16
/// scale per 32) and fused into the GEMM on sm_75?
fn u2(env: &Env) -> Result<CheckOutcome, Error> {
    let problem = quantized_problem(false);
    let pack = |c: &[i8]| c.iter().map(|&v| v as u8).collect();
    let notes = block_dq_matmul(env, DataType::Int8, &problem, pack)?;
    Ok(answer(&notes, "builds"))
}

/// U3: the same with Q4_0's 4-bit codes. TensorRT's INT4 packs two codes
/// per byte; the nibble order is not in the headers, so both are tried and
/// the one that matches the reference is reported.
fn u3(env: &Env) -> Result<CheckOutcome, Error> {
    let problem = quantized_problem(true);
    let mut notes = Vec::new();
    for (order, low_first) in [("low nibble first", true), ("high nibble first", false)] {
        let pack = move |c: &[i8]| -> Vec<u8> {
            c.chunks(2)
                .map(|p| {
                    let (lo, hi) = (p[0] as u8 & 0x0F, p.get(1).map_or(0, |&v| v as u8 & 0x0F));
                    if low_first {
                        lo | hi << 4
                    } else {
                        hi | lo << 4
                    }
                })
                .collect()
        };
        let n = block_dq_matmul(env, DataType::Int4, &problem, pack)?;
        notes.push(format!("[{order}] {n}"));
    }
    Ok(answer(&notes.join(" || "), "builds"))
}

fn answer(notes: &str, yes_marker: &str) -> CheckOutcome {
    if notes.contains(yes_marker) {
        CheckOutcome::Pass(notes.to_owned())
    } else {
        CheckOutcome::Fail(notes.to_owned())
    }
}

/// U4: build a weight-stripped, refit-identical plan — from the network API
/// and from ONNX — then supply the stripped weight from *device* memory and
/// check the output. The stripped and full plan sizes show the weights were
/// really left out.
fn u4(env: &Env) -> Result<CheckOutcome, Error> {
    const M: i64 = 64;
    const K: i64 = 512;
    const N: i64 = 256;
    const NAME: &str = "probe.w";
    let mut rng = Xorshift64Star::new(0x6404);
    let w: Vec<f16> = (0..K * N)
        .map(|_| f16::from_f32(rng.next_f32_range(-0.1, 0.1)))
        .collect();
    let x = rng.vec_f32((M * K) as usize, -1.0, 1.0);
    let mut reference = vec![0f32; (M * N) as usize];
    for i in 0..M as usize {
        for j in 0..N as usize {
            reference[i * N as usize + j] = (0..K as usize)
                .map(|c| {
                    f64::from(f16::from_f32(x[i * K as usize + c]).to_f32())
                        * f64::from(w[c * N as usize + j].to_f32())
                })
                .sum::<f64>() as f32;
        }
    }
    let dw = env.upload(&w.iter().map(|h| h.to_bits()).collect::<Vec<u16>>())?;
    let wbytes = half_bytes(&w);

    let onnx_model = onnx::model(&Graph {
        nodes: &[
            Node {
                op_type: "Cast",
                domain: "",
                inputs: &["x"],
                outputs: &["x16"],
                attrs: &[("to", Attr::Int(Elem::Float16 as i64))],
            },
            Node {
                op_type: "MatMul",
                domain: "",
                inputs: &["x16", NAME],
                outputs: &["y16"],
                attrs: &[],
            },
            Node {
                op_type: "Cast",
                domain: "",
                inputs: &["y16"],
                outputs: &["y"],
                attrs: &[("to", Attr::Int(Elem::Float as i64))],
            },
        ],
        inputs: &[("x", Elem::Float, &[M, K])],
        outputs: &[("y", Elem::Float, &[M, N])],
        initializers: &[Initializer {
            name: NAME,
            elem: Elem::Float16,
            dims: &[K, N],
            raw: &wbytes,
        }],
        custom_domains: &[],
    });

    let mut notes = Vec::new();
    let mut all = true;
    for source in ["network API", "ONNX"] {
        let mut sizes = Vec::new();
        let mut result = String::new();
        for strip in [false, true] {
            let builder = Builder::new()?;
            let network = builder.network()?;
            let parser;
            if source == "ONNX" {
                parser = OnnxParser::new(&network)?;
                parser.parse(&onnx_model)?;
            } else {
                let input = network.input("x", DataType::Float, [M, K])?;
                let x16 = network.cast(input, DataType::Half)?;
                let wt = network.constant([K, N], DataType::Half, wbytes.clone(), Some(NAME))?;
                let y = network.cast(
                    network.matmul(x16, MatrixOp::None, wt, MatrixOp::None)?,
                    DataType::Float,
                )?;
                y.set_name("y")?;
                network.mark_output(y)?;
            }
            let config = config(&builder)?;
            if strip {
                config.set_flag(BuilderFlag::StripPlan, true);
                config.set_flag(BuilderFlag::RefitIdentical, true);
            }
            let plan = build(&builder, &network, &config)?;
            sizes.push(plan.len());
            if !strip {
                continue;
            }
            let runtime = Runtime::new()?;
            let engine = runtime.deserialize(&plan)?;
            let context = engine.context(AllocationStrategy::Static)?;
            let dx = env.upload(&x)?;
            let mut dy = env.zeros::<f32>((M * N) as usize)?;
            // SAFETY: static shapes; buffers outlive every download below.
            unsafe {
                context.set_tensor_address("x", env.ptr(&dx))?;
                context.set_tensor_address("y", env.ptr(&dy))?;
            }
            // Before the refit: the stripped weight is not in the plan, so
            // this must either refuse to run or produce something else.
            // SAFETY: as above.
            let before = match unsafe { context.enqueue(env.raw_stream()) } {
                Err(e) => format!("enqueue refused ({e})"),
                Ok(()) => format!(
                    "enqueue ran, output {}",
                    fmt_cmp(&compare(&env.download(&dy)?, &reference))
                ),
            };
            // Clear it, so stale output cannot pass for the refit's.
            env.stream.memset_zeros(&mut dy)?;
            let refitter = engine.refitter()?;
            let refittable = refitter.all();
            // SAFETY: `dw` holds K*N halves on the device and outlives the
            // refit, which is ordered on the stream before the enqueue.
            unsafe {
                refitter.set_named_weights(
                    NAME,
                    DataType::Half,
                    env.ptr(&dw) as *const std::ffi::c_void,
                    K * N,
                    TensorLocation::Device,
                )?;
            }
            refitter.refit(env.raw_stream())?;
            let missing = refitter.missing();
            // SAFETY: as above.
            unsafe { context.enqueue(env.raw_stream())? };
            let c = compare(&env.download(&dy)?, &reference);
            let ok = refittable.iter().any(|m| m == NAME)
                && missing.is_empty()
                && c.cosine_similarity > 0.9999;
            all &= ok;
            result = format!(
                "refittable {refittable:?}; before refit: {before}; after device refit (missing {missing:?}): output {}",
                fmt_cmp(&c)
            );
        }
        notes.push(format!(
            "{source}: plan {} B full vs {} B stripped; {result}",
            sizes[0], sizes[1]
        ));
    }
    let notes = notes.join(" | ");
    Ok(if all {
        CheckOutcome::Pass(notes)
    } else {
        CheckOutcome::Fail(notes)
    })
}

/// U5: an ONNX node in domain `llmcuda` resolves to the registered
/// IPluginV3 creator, its integer attribute arrives as a plugin field, and
/// the result is exact.
fn u5(env: &Env) -> Result<CheckOutcome, Error> {
    const N: i64 = 4099;
    const TAG: i64 = 4242;
    let mut notes = Vec::new();
    let mut ok = false;
    for with_namespace in [true, false] {
        let mut attrs = vec![
            ("probe_tag", Attr::Int(TAG)),
            ("plugin_version", Attr::Str("1")),
        ];
        if with_namespace {
            attrs.push(("plugin_namespace", Attr::Str(NAMESPACE)));
        }
        let model = onnx::model(&Graph {
            nodes: &[Node {
                op_type: "ProbeAdd",
                domain: NAMESPACE,
                inputs: &["a", "b"],
                outputs: &["out"],
                attrs: &attrs,
            }],
            inputs: &[("a", Elem::Float, &[N]), ("b", Elem::Float, &[N])],
            outputs: &[("out", Elem::Float, &[N])],
            initializers: &[],
            custom_domains: &[NAMESPACE],
        });
        let label = if with_namespace {
            "with plugin_namespace attribute"
        } else {
            "without plugin_namespace"
        };
        plugins::LAST_TAG.store(-1, Ordering::SeqCst);
        let run = (|| -> Result<String, Error> {
            let builder = Builder::new()?;
            let network = builder.network()?;
            let parser = OnnxParser::new(&network)?;
            parser.parse(&model)?;
            let tag = plugins::LAST_TAG.load(Ordering::SeqCst);
            let plan = build(&builder, &network, &config(&builder)?)?;
            drop(parser);
            let runtime = Runtime::new()?;
            let engine = runtime.deserialize(&plan)?;
            let context = engine.context(AllocationStrategy::Static)?;
            let (ha, hb) = exact_operands(N as usize, 0x6505, true);
            let (da, db) = (env.upload(&ha)?, env.upload(&hb)?);
            let dout = env.zeros::<f32>(N as usize)?;
            // SAFETY: static shapes; buffers outlive the download.
            unsafe {
                context.set_tensor_address("a", env.ptr(&da))?;
                context.set_tensor_address("b", env.ptr(&db))?;
                context.set_tensor_address("out", env.ptr(&dout))?;
                context.enqueue(env.raw_stream())?;
            }
            let (bad, _) = bit_diff(&env.download(&dout)?, &residual_add(&ha, &hb));
            if tag != TAG || bad != 0 {
                return Err(format!(
                    "creator got probe_tag {tag} (sent {TAG}); {bad} wrong outputs"
                )
                .into());
            }
            Ok(format!(
                "parsed, built, ran; probe_tag {tag} arrived as a field; output exact"
            ))
        })();
        match run {
            Ok(s) => {
                ok |= with_namespace;
                notes.push(format!("{label}: {s}"));
            }
            Err(e) => notes.push(format!("{label}: {e}")),
        }
    }
    let notes = notes.join(" | ");
    Ok(if ok {
        CheckOutcome::Pass(notes)
    } else {
        CheckOutcome::Fail(notes)
    })
}

const U6_K: i64 = 2048;
const U6_N: i64 = 8192;
const U6_MS: [i64; 3] = [512, 2048, 6144];

/// What one U6 build produced.
struct U6Run {
    /// Differing elements: rows 0..512 between profiles 0 and 1, 0 and 2;
    /// rows 0..2048 between 1 and 2.
    diffs: [usize; 3],
    entries: Vec<([u8; 16], TimingEntry)>,
    cache_blob: Vec<u8>,
    layers: Vec<String>,
    first_rows: Vec<f32>,
}

/// Build the U6 GEMM from `cache_blob` (empty for a fresh cache),
/// optionally rewriting every non-zero tactic in it to `pin` first, then
/// run each of [`U6_MS`] on the same leading rows. `exact` builds one
/// exact-shape profile per row count, as the plan specifies; otherwise one
/// dynamic profile covers all three.
fn u6_run(
    env: &Env,
    w: &[f16],
    x: &[f32],
    cache_blob: &[u8],
    pin: Option<u64>,
    exact: bool,
) -> Result<U6Run, Error> {
    let (k, n) = (U6_K, U6_N);
    let builder = Builder::new()?;
    let network = builder.network()?;
    let input = network.input("x", DataType::Float, [-1, k])?;
    let x16 = network.cast(input, DataType::Half)?;
    let wt = network.constant([k, n], DataType::Half, half_bytes(w), None)?;
    let y = network.cast(
        network.matmul(x16, MatrixOp::None, wt, MatrixOp::None)?,
        DataType::Float,
    )?;
    y.set_name("y")?;
    network.mark_output(y)?;
    let config = config(&builder)?;
    config.set_flag(BuilderFlag::EditableTimingCache, true);
    config.set_profiling_verbosity(ProfilingVerbosity::Detailed);
    let cache = config.timing_cache(cache_blob)?;
    if let Some(tactic) = pin {
        for key in cache.keys() {
            if let Some(e) = cache.query(&key).filter(|e| e.tactic != 0) {
                cache.update(&key, TimingEntry { tactic, ms: e.ms })?;
            }
        }
    }
    config.set_timing_cache(&cache, false)?;
    if exact {
        for m in U6_MS {
            config.add_exact_profile(&[("x", [m, k].into())])?;
        }
    } else {
        let d = |m: i64| Dims::from([m, k]);
        config.add_profile(&[("x", d(1), d(U6_MS[1]), d(U6_MS[2]))])?;
    }
    let plan = build(&builder, &network, &config)?;
    let entries = cache
        .keys()
        .into_iter()
        .filter_map(|key| cache.query(&key).map(|e| (key, e)))
        .collect();
    let cache_blob = cache.serialize()?.to_vec();

    let runtime = Runtime::new()?;
    let engine = runtime.deserialize(&plan)?;
    let layers = layer_names(&engine.information()?);
    let dx = env.upload(x)?;
    let mut outs = Vec::new();
    for (p, m) in U6_MS.into_iter().enumerate() {
        let context = engine.context(AllocationStrategy::Static)?;
        context.set_profile(if exact { p as i32 } else { 0 }, env.raw_stream())?;
        context.set_input_shape("x", [m, k])?;
        let dy = env.zeros::<f32>((m * n) as usize)?;
        // SAFETY: `dx` holds 6144 rows, the widest profile; `dy` holds m rows.
        unsafe {
            context.set_tensor_address("x", env.ptr(&dx))?;
            context.set_tensor_address("y", env.ptr(&dy))?;
            context.enqueue(env.raw_stream())?;
        }
        outs.push(env.download(&dy)?);
    }
    let lead = (U6_MS[0] * n) as usize;
    let mid = (U6_MS[1] * n) as usize;
    Ok(U6Run {
        diffs: [
            bit_diff(&outs[0][..lead], &outs[1][..lead]).0,
            bit_diff(&outs[0][..lead], &outs[2][..lead]).0,
            bit_diff(&outs[1][..mid], &outs[2][..mid]).0,
        ],
        entries,
        cache_blob,
        layers,
        first_rows: outs[0][..4 * n as usize].to_vec(),
    })
}

/// U6: one fp16 GEMM, the real Qwen3.6 GDN `attn_qkv` shape (2048 -> 8192),
/// three exact-shape profiles. The same leading rows go through each; any
/// bit difference means a row's result depends on how many rows it shared
/// the pass with, which breaks the batched-equals-alone contract.
///
/// If they differ, the editable timing cache is used to try to pin every
/// timed tactic to one value, each candidate in turn, and the rows are
/// compared again. One dynamic profile is also tried, for comparison.
fn u6(env: &Env) -> Result<CheckOutcome, Error> {
    let (k, n) = (U6_K, U6_N);
    let mut rng = Xorshift64Star::new(0x6606);
    let w: Vec<f16> = (0..k * n)
        .map(|_| f16::from_f32(rng.next_f32_range(-0.03, 0.03)))
        .collect();
    let x = rng.vec_f32((U6_MS[2] * k) as usize, -1.0, 1.0);

    let base = u6_run(env, &w, &x, &[], None, true)?;
    let rows = 4usize;
    let reference: Vec<f32> = (0..rows * n as usize)
        .map(|i| {
            let (r, c) = (i / n as usize, i % n as usize);
            (0..k as usize)
                .map(|kk| {
                    f64::from(f16::from_f32(x[r * k as usize + kk]).to_f32())
                        * f64::from(w[kk * n as usize + c].to_f32())
                })
                .sum::<f64>() as f32
        })
        .collect();
    let sanity = compare(&base.first_rows, &reference);
    let tactics: Vec<String> = base
        .entries
        .iter()
        .map(|(_, e)| format!("{:x}", e.tactic))
        .collect();
    let summary = format!(
        "M = {U6_MS:?}: rows 0..512 differ in {} (512 vs 2048) and {} (512 vs 6144) elements, \
         rows 0..2048 in {} (2048 vs 6144); timing cache {} keys, tactics {tactics:?}; \
         engine layers {:?}; first 4 rows vs fp16-rounded CPU: {}",
        base.diffs[0],
        base.diffs[1],
        base.diffs[2],
        base.entries.len(),
        base.layers,
        fmt_cmp(&sanity)
    );
    // Not the plan's design, but the obvious alternative if exact profiles
    // fail: one dynamic profile, so one tactic for every row count.
    let dynamic = match u6_run(env, &w, &x, &[], None, false) {
        Ok(r) => format!(
            "one dynamic profile [1, 6144] instead: diffs {:?}, layers {:?}",
            r.diffs, r.layers
        ),
        Err(e) => format!("one dynamic profile [1, 6144] instead: {e}"),
    };
    let summary = format!("{summary}; {dynamic}");
    if base.diffs == [0, 0, 0] {
        return Ok(CheckOutcome::Pass(format!(
            "identical without pinning; {summary}"
        )));
    }

    let mut candidates: Vec<u64> = base
        .entries
        .iter()
        .map(|(_, e)| e.tactic)
        .filter(|&t| t != 0)
        .collect();
    candidates.sort_unstable();
    candidates.dedup();
    let mut pins = Vec::new();
    let mut pinned_ok = None;
    for tactic in candidates {
        match u6_run(env, &w, &x, &base.cache_blob, Some(tactic), true) {
            Ok(r) => {
                let used: Vec<String> = r
                    .entries
                    .iter()
                    .map(|(_, e)| format!("{:x}", e.tactic))
                    .collect();
                pins.push(format!(
                    "pin {tactic:x}: diffs {:?}, cache after {used:?}",
                    r.diffs
                ));
                if r.diffs == [0, 0, 0] {
                    pinned_ok = Some(tactic);
                    break;
                }
            }
            Err(e) => pins.push(format!("pin {tactic:x}: {e}")),
        }
    }
    let pins = pins.join("; ");
    // A pin that works in one build is not an answer: tactic choice is
    // timed, so the next build may pick differently. Only an unpinned
    // identity passes; a pin's success is reported for the record.
    let pinned = match pinned_ok {
        Some(t) => format!("pinning every timed tactic to {t:x} made them identical in this build"),
        None => "no single pinned tactic made them identical".to_owned(),
    };
    Ok(CheckOutcome::Fail(format!(
        "rows depend on the profile; {pinned}; {summary}; {pins}"
    )))
}

/// U8: mark an intermediate as a debug tensor, capture it through a
/// listener (copied on the stream inside the callback, since TensorRT may
/// reuse the address), and check it and the final output exactly.
fn u8(env: &Env) -> Result<CheckOutcome, Error> {
    const N: i64 = 65_539;
    let builder = Builder::new()?;
    let network = builder.network()?;
    let a = network.input("a", DataType::Float, [N])?;
    let b = network.input("b", DataType::Float, [N])?;
    let t = network.elementwise(a, b, ElementWise::Sum)?;
    t.set_name("waypoint")?;
    network.mark_debug(t)?;
    let out = network.plugin(&[t, b], "ProbeAdd", "1", NAMESPACE, &[], 1)?[0];
    out.set_name("out")?;
    network.mark_output(out)?;
    let plan = build(&builder, &network, &config(&builder)?)?;
    let runtime = Runtime::new()?;
    let engine = runtime.deserialize(&plan)?;
    let mut context = engine.context(AllocationStrategy::Static)?;

    let (ha, hb) = exact_operands(N as usize, 0x6808, false);
    let (da, db) = (env.upload(&ha)?, env.upload(&hb)?);
    let dout = env.zeros::<f32>(N as usize)?;
    let dcopy = env.zeros::<f32>(N as usize)?;
    let copy_ptr = env.ptr(&dcopy);
    let seen: Arc<Mutex<Vec<String>>> = Arc::default();
    let seen_cb = seen.clone();
    let listener = DebugListener::new(move |t| {
        let n = t.shape.volume().unwrap_or(0) as usize;
        seen_cb.lock().unwrap().push(format!(
            "{} {:?} {:?} {:?}",
            t.name,
            t.shape.as_slice(),
            t.data_type,
            t.location
        ));
        if t.name != "waypoint" || n != N as usize {
            return false;
        }
        // SAFETY: `copy_ptr` holds N floats and outlives the enqueue; the
        // copy is ordered on TensorRT's stream before any reuse.
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(copy_ptr, t.address as u64, n * 4, t.stream)
        }
        .is_ok()
    })?;
    context.set_debug_listener(listener)?;
    context.set_tensor_debug_state("waypoint", true)?;
    // SAFETY: static shapes; buffers outlive the download.
    unsafe {
        context.set_tensor_address("a", env.ptr(&da))?;
        context.set_tensor_address("b", env.ptr(&db))?;
        context.set_tensor_address("out", env.ptr(&dout))?;
        context.enqueue(env.raw_stream())?;
    }
    let waypoint = env.download(&dcopy)?;
    let got = env.download(&dout)?;
    let sum = residual_add(&ha, &hb);
    let (wbad, _) = bit_diff(&waypoint, &sum);
    let (obad, _) = bit_diff(&got, &residual_add(&sum, &hb));
    let seen = seen.lock().unwrap().clone();
    let summary = format!(
        "listener calls {seen:?}; waypoint {wbad} of {N} differ from a+b; output {obad} differ"
    );
    Ok(if seen.len() == 1 && wbad == 0 && obad == 0 {
        CheckOutcome::Pass(summary)
    } else {
        CheckOutcome::Fail(summary)
    })
}

/// IAttention, as the plan would use it after a KV layout migration: fp16,
/// 16 query heads over 2 KV heads, head dim 256, causal from position 0,
/// padded BHND. Built without `setDecomposable` first — the question — and
/// with it only if that fails, for context. Compared against the engine's
/// CPU reference, `causal_attention_naive`, which applies 1/sqrt(256)
/// itself; TensorRT's IAttention has no scale, so Q is pre-scaled by 1/16,
/// which is exact in fp16.
fn attention(env: &Env) -> Result<CheckOutcome, Error> {
    const HQ: i64 = 16;
    const HKV: i64 = 2;
    const D: i64 = 256;
    const S: i64 = 512;
    let mut rng = Xorshift64Star::new(0x6a77);
    let sample = |n: i64, rng: &mut Xorshift64Star| -> Vec<f16> {
        (0..n)
            .map(|_| f16::from_f32(rng.next_f32_range(-1.0, 1.0)))
            .collect()
    };
    let q = sample(HQ * S * D, &mut rng);
    let k = sample(HKV * S * D, &mut rng);
    let v = sample(HKV * S * D, &mut rng);
    let q_scaled: Vec<f16> = q.iter().map(|h| f16::from_f32(h.to_f32() / 16.0)).collect();

    let mut notes = Vec::new();
    for decomposable in [false, true] {
        let builder = Builder::new()?;
        let network = builder.network()?;
        let tq = network.input("q", DataType::Half, [1, HQ, S, D])?;
        let tk = network.input("k", DataType::Half, [1, HKV, S, D])?;
        let tv = network.input("v", DataType::Half, [1, HKV, S, D])?;
        let built = (|| -> Result<String, Error> {
            let o = network.attention(tq, tk, tv, CausalMask::UpperLeft, decomposable)?;
            o.set_name("o")?;
            network.mark_output(o)?;
            let config = config(&builder)?;
            config.set_profiling_verbosity(ProfilingVerbosity::Detailed);
            let plan = build(&builder, &network, &config)?;
            let runtime = Runtime::new()?;
            let engine = runtime.deserialize(&plan)?;
            let layers = layer_names(&engine.information()?);
            let context = engine.context(AllocationStrategy::Static)?;
            let bits = |h: &[f16]| h.iter().map(|x| x.to_bits()).collect::<Vec<u16>>();
            let (dq, dk, dv) = (
                env.upload(&bits(&q_scaled))?,
                env.upload(&bits(&k))?,
                env.upload(&bits(&v))?,
            );
            let dout = env.zeros::<u16>((HQ * S * D) as usize)?;
            // SAFETY: static shapes; buffers outlive the download.
            unsafe {
                context.set_tensor_address("q", env.ptr(&dq))?;
                context.set_tensor_address("k", env.ptr(&dk))?;
                context.set_tensor_address("v", env.ptr(&dv))?;
                context.set_tensor_address("o", env.ptr(&dout))?;
                context.enqueue(env.raw_stream())?;
            }
            let got: Vec<f32> = env
                .download(&dout)?
                .into_iter()
                .map(|b| f16::from_bits(b).to_f32())
                .collect();
            let mut reference = Vec::with_capacity(got.len());
            let rows = |src: &[f16], h: i64| -> Vec<Vec<f32>> {
                (0..S)
                    .map(|s| {
                        let at = ((h * S + s) * D) as usize;
                        src[at..at + D as usize]
                            .iter()
                            .map(|x| x.to_f32())
                            .collect()
                    })
                    .collect()
            };
            for h in 0..HQ {
                let kvh = h / (HQ / HKV);
                for row in causal_attention_naive(&rows(&q, h), &rows(&k, kvh), &rows(&v, kvh)) {
                    reference.extend(row);
                }
            }
            let c = compare(&got, &reference);
            Ok(format!(
                "builds; {} layers {layers:?}; S={S} vs CPU reference: {}",
                layers.len(),
                fmt_cmp(&c)
            ))
        })();
        let label = if decomposable {
            "decomposable"
        } else {
            "not decomposable"
        };
        match built {
            Ok(s) => {
                notes.push(format!("{label}: {s}"));
                if !decomposable {
                    break;
                }
            }
            Err(e) => notes.push(format!("{label}: {e}")),
        }
    }
    let notes = notes.join(" | ");
    Ok(if notes.starts_with("not decomposable: builds") {
        CheckOutcome::Pass(notes)
    } else {
        CheckOutcome::Fail(notes)
    })
}
