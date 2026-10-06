//! The device decision head against the scalar oracle,
//! `llmcuda_kernels::decision::joint_schema_head`.
//!
//! Synthetic: a small random head (hidden 256, width 128, four heads,
//! feed-forward 256, two routing and two joint layers) written into an
//! in-memory GGUF with mixed storage formats, so `DecisionHead::host_weights`
//! is exercised on the same path the real file takes, then scored on several
//! prompt shapes and every LM-head format the lexical gather reads.
//!
//! Real weights (opt-in): `LLMCUDA_CLEF_MODEL=/path/Clef-Flash-*.gguf` loads
//! the real head and output embedding, feeds random RMS-normed hidden
//! states, compares against the oracle, and times `forward` with CUDA events
//! at 2048 and 8192 positions.
//!
//! # Tolerances
//!
//! The oracle is f64 throughout; the device is fp32 with fp32 accumulation
//! over the same (already dequantized) weights, so the whole disagreement is
//! fp32 rounding. A K-long fp32 dot product of O(1) terms carries a relative
//! error of order `sqrt(K) * 2^-24` (about 4e-6 at K = 4096), and a score
//! passes through about thirty such reductions. Observed while writing the
//! test: at most 7.2e-7 absolute on synthetic scores (|score| <= 2), 1.2e-6
//! on real scores (|score| <= 4.9, at 200 and 600 positions), and on the
//! memory projection 2.9e-6 (synthetic, K = 256) and 7.2e-6 (real,
//! K = 4096). Every gate is 8x or more above the worst observed value and far
//! below a meaningful change: scaling one softmax temperature by 0.9 moved a
//! synthetic score by 6.2e-3 and failed, and a wrong layout, transposed
//! weight, dropped bias or mis-indexed span moves scores by O(0.1).

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use llmcuda_cuda::device::driver_available;
use llmcuda_cuda::kernels::lm_head::{HeadFormat, HeadTensor};
use llmcuda_engine::block::decision::{DecisionHead, DecisionLimits};
use llmcuda_gguf::GgufFile;
use llmcuda_kernels::decision::{
    Attention, HeadWeights, LayerNorm, Linear, OptionSpan, QuestionSpan, QuestionType,
    joint_schema_head,
};
use llmcuda_kernels::quant::*;
use llmcuda_kernels::rng::Xorshift64Star;
use llmcuda_model::{DecisionConfig, ModelConfig};

const HIDDEN: usize = 256;
const WIDTH: usize = 128;
const HEADS: usize = 4;
const FF: usize = 256;
const ROUTING: usize = 2;
const JOINT: usize = 2;
const VOCAB: usize = 64;
const EPS: f32 = 1e-5;

/// Scores, synthetic and real: |gpu - ref| <= ABS + REL * |ref|.
const SCORE_ABS: f32 = 1e-5;
const SCORE_REL: f32 = 1e-5;
/// The memory projection, synthetic (K = 256).
const MEMORY_ABS: f32 = 3e-5;
/// The memory projection, real (K = 4096).
const REAL_MEMORY_ABS: f32 = 1e-4;

// ---------------------------------------------------------------- formats --

#[derive(Clone, Copy, Debug, PartialEq)]
enum Fmt {
    F32,
    F16,
    Bf16,
    Q8_0,
    Q4_0,
    Q4K,
    Q5K,
    Q6K,
}

impl Fmt {
    fn ggml(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q8_0 => 8,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Bf16 => 30,
        }
    }

    fn head(self) -> HeadFormat {
        match self {
            Self::F16 => HeadFormat::F16,
            Self::Bf16 => HeadFormat::Bf16,
            Self::Q8_0 => HeadFormat::Q8_0,
            Self::Q4_0 => HeadFormat::Q4_0,
            Self::Q4K => HeadFormat::Q4K,
            Self::Q5K => HeadFormat::Q5K,
            Self::Q6K => HeadFormat::Q6K,
            Self::F32 => unreachable!("no f32 LM head"),
        }
    }

    /// Encode `x` (whole blocks) in this format.
    fn encode(self, x: &[f32]) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            Self::F32 => x.iter().for_each(|v| out.extend(v.to_le_bytes())),
            Self::F16 => x
                .iter()
                .for_each(|&v| out.extend(half::f16::from_f32(v).to_le_bytes())),
            Self::Bf16 => x
                .iter()
                .for_each(|&v| out.extend(half::bf16::from_f32(v).to_le_bytes())),
            Self::Q8_0 => x
                .as_chunks::<32>()
                .0
                .iter()
                .for_each(|b| out.extend(quantize_q8_0(b).to_bytes())),
            Self::Q4_0 => x
                .as_chunks::<32>()
                .0
                .iter()
                .for_each(|b| out.extend(quantize_q4_0(b).to_bytes())),
            Self::Q4K => x
                .as_chunks::<256>()
                .0
                .iter()
                .for_each(|b| out.extend(quantize_q4_k(b).to_bytes())),
            Self::Q5K => x
                .as_chunks::<256>()
                .0
                .iter()
                .for_each(|b| out.extend(quantize_q5_k(b).to_bytes())),
            Self::Q6K => x
                .as_chunks::<256>()
                .0
                .iter()
                .for_each(|b| out.extend(quantize_q6_k(b).to_bytes())),
        }
        out
    }

    /// Decode bytes back with the scalar reference.
    fn decode(self, b: &[u8]) -> Vec<f32> {
        match self {
            Self::F32 => b
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
            Self::F16 => dequantize_row_f16(b),
            Self::Bf16 => dequantize_row_bf16(b),
            Self::Q8_0 => dequantize_row_q8_0(b).unwrap(),
            Self::Q4_0 => dequantize_row_q4_0(b).unwrap(),
            Self::Q4K => dequantize_row_q4_k(b).unwrap(),
            Self::Q5K => dequantize_row_q5_k(b).unwrap(),
            Self::Q6K => dequantize_row_q6_k(b).unwrap(),
        }
    }
}

// ------------------------------------------------------- synthetic weights --

/// One tensor of the synthetic file: name, GGUF dims (innermost first),
/// format and the f32 values it was generated from.
struct Tensor {
    name: String,
    dims: Vec<u64>,
    fmt: Fmt,
    values: Vec<f32>,
}

struct Builder {
    rng: Xorshift64Star,
    tensors: Vec<Tensor>,
}

impl Builder {
    fn add(&mut self, name: &str, dims: &[usize], fmt: Fmt, values: Vec<f32>) {
        self.tensors.push(Tensor {
            name: name.into(),
            dims: dims.iter().map(|&d| d as u64).collect(),
            fmt,
            values,
        });
    }

    fn norm(&mut self, name: &str, n: usize) {
        let w = (0..n).map(|_| self.rng.next_f32_range(0.6, 1.4)).collect();
        let b = self.rng.vec_f32(n, -0.2, 0.2);
        self.add(&format!("{name}.weight"), &[n], Fmt::F32, w);
        self.add(&format!("{name}.bias"), &[n], Fmt::F32, b);
    }

    /// `[out][inp]`, entries U(-a, a) with a = sqrt(3 / inp) so outputs keep
    /// roughly the input's scale.
    fn linear(&mut self, name: &str, inp: usize, out: usize, bias: bool, fmt: Fmt) {
        let a = (3.0 / inp as f32).sqrt();
        let w = self.rng.vec_f32(inp * out, -a, a);
        self.add(&format!("{name}.weight"), &[inp, out], fmt, w);
        if bias {
            let b = self.rng.vec_f32(out, -0.1, 0.1);
            self.add(&format!("{name}.bias"), &[out], Fmt::F32, b);
        }
    }

    fn attention(&mut self, prefix: &str) {
        for p in ["q", "k", "v", "o"] {
            self.linear(&format!("{prefix}_{p}"), WIDTH, WIDTH, true, Fmt::Q8_0);
        }
    }
}

/// A GGUF v3 image holding `tensors` and no metadata.
fn gguf(tensors: &[Tensor]) -> (Vec<u8>, Vec<Vec<u8>>) {
    fn string(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }
    let payloads: Vec<Vec<u8>> = tensors.iter().map(|t| t.fmt.encode(&t.values)).collect();
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend((tensors.len() as u64).to_le_bytes());
    out.extend(0u64.to_le_bytes());
    let mut offset = 0u64;
    for (t, p) in tensors.iter().zip(&payloads) {
        string(&mut out, &t.name);
        out.extend((t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            out.extend(d.to_le_bytes());
        }
        out.extend(t.fmt.ggml().to_le_bytes());
        out.extend(offset.to_le_bytes());
        offset += (p.len() as u64).next_multiple_of(32);
    }
    out.resize(out.len().next_multiple_of(32), 0);
    for p in &payloads {
        out.extend(p);
        out.resize(out.len().next_multiple_of(32), 0);
    }
    (out, payloads)
}

/// The synthetic file, and the weights it should load as (each tensor
/// decoded from the bytes written).
fn synthetic_file(seed: u64) -> (GgufFile, ModelConfig, Vec<(String, Vec<f32>)>) {
    let mut b = Builder {
        rng: Xorshift64Star::new(seed),
        tensors: Vec::new(),
    };
    b.norm("decision.hidden_norm", HIDDEN);
    // Every format host_weights dequantizes appears at least once.
    b.linear("decision.proj_memory", HIDDEN, WIDTH, false, Fmt::Q8_0);
    b.linear("decision.proj_question", HIDDEN, WIDTH, false, Fmt::Q6K);
    b.linear(
        "decision.proj_option_question",
        HIDDEN,
        WIDTH,
        false,
        Fmt::Q4_0,
    );
    b.linear("decision.proj_global", HIDDEN, WIDTH, false, Fmt::Q5K);
    b.linear(
        "decision.proj_option_context",
        HIDDEN,
        WIDTH,
        false,
        Fmt::F16,
    );
    b.linear(
        "decision.proj_option_lexical",
        HIDDEN,
        WIDTH,
        false,
        Fmt::Q4K,
    );
    let types = b.rng.vec_f32(3 * WIDTH, -0.5, 0.5);
    b.add("token_types.weight", &[WIDTH, 3], Fmt::F32, types);
    for il in 0..ROUTING + JOINT {
        let p = format!("dec.blk.{il}");
        if il >= ROUTING {
            b.norm(&format!("{p}.attn_norm"), WIDTH);
            b.attention(&format!("{p}.attn"));
        }
        b.norm(&format!("{p}.cross_attn_norm"), WIDTH);
        if il < ROUTING {
            b.norm(&format!("{p}.cross_attn_norm_kv"), WIDTH);
        }
        b.attention(&format!("{p}.cross_attn"));
        b.norm(&format!("{p}.ffn_norm"), WIDTH);
        b.linear(&format!("{p}.ffn_up"), WIDTH, FF, true, Fmt::Q8_0);
        b.linear(&format!("{p}.ffn_down"), FF, WIDTH, true, Fmt::Q8_0);
    }
    b.norm("decision.option_summary_norm", WIDTH);
    b.norm("decision.field_norm", WIDTH);
    b.norm("decision.option_norm", WIDTH);
    b.linear("decision.scorer", 4 * WIDTH, WIDTH, true, Fmt::Q8_0);
    let out_w = b.rng.vec_f32(WIDTH, -0.15, 0.15);
    b.add("decision.scorer_out.weight", &[WIDTH], Fmt::Bf16, out_w);
    b.add("decision.scorer_out.bias", &[1], Fmt::F32, vec![0.05]);
    b.add("decision.scales", &[3], Fmt::F32, vec![1.7, 2.3, 0.62]);

    let (bytes, payloads) = gguf(&b.tensors);
    let decoded = b
        .tensors
        .iter()
        .zip(&payloads)
        .map(|(t, p)| (t.name.clone(), t.fmt.decode(p)))
        .collect();
    let config = ModelConfig {
        hidden_size: HIDDEN as u32,
        decision: Some(DecisionConfig {
            routing_layers: ROUTING as u32,
            joint_layers: JOINT as u32,
            heads: HEADS as u32,
            layer_norm_eps_bits: EPS.to_bits(),
        }),
        ..ModelConfig::qwen3_8_27b()
    };
    (GgufFile::from_bytes(bytes).unwrap(), config, decoded)
}

/// Every tensor host_weights returned, by GGUF name, for comparison with
/// what was written.
fn flatten(w: &HeadWeights) -> Vec<(String, Vec<f32>)> {
    let mut out = Vec::new();
    let mut norm = |name: &str, n: &LayerNorm| {
        out.push((format!("{name}.weight"), n.weight.clone()));
        out.push((format!("{name}.bias"), n.bias.clone()));
    };
    norm("decision.hidden_norm", &w.hidden_norm);
    norm("decision.option_summary_norm", &w.option_summary_norm);
    norm("decision.field_norm", &w.field_norm);
    norm("decision.option_norm", &w.option_norm);
    for (il, l) in w.layers.iter().enumerate() {
        let p = format!("dec.blk.{il}");
        if let Some((n, _)) = &l.self_attn {
            norm(&format!("{p}.attn_norm"), n);
        }
        norm(&format!("{p}.cross_attn_norm"), &l.cross_norm);
        if let Some(n) = &l.cross_norm_kv {
            norm(&format!("{p}.cross_attn_norm_kv"), n);
        }
        norm(&format!("{p}.ffn_norm"), &l.ffn_norm);
    }
    let mut lin = |name: &str, l: &Linear| {
        out.push((format!("{name}.weight"), l.weight.clone()));
        if let Some(b) = &l.bias {
            out.push((format!("{name}.bias"), b.clone()));
        }
    };
    lin("decision.proj_memory", &w.proj_memory);
    lin("decision.proj_question", &w.proj_question);
    lin("decision.proj_option_question", &w.proj_option_question);
    lin("decision.proj_global", &w.proj_global);
    lin("decision.proj_option_context", &w.proj_option_context);
    lin("decision.proj_option_lexical", &w.proj_option_lexical);
    lin("decision.scorer", &w.scorer);
    lin("decision.scorer_out", &w.scorer_out);
    let mut attn = |prefix: &str, a: &Attention| {
        lin(&format!("{prefix}_q"), &a.q);
        lin(&format!("{prefix}_k"), &a.k);
        lin(&format!("{prefix}_v"), &a.v);
        lin(&format!("{prefix}_o"), &a.o);
    };
    for (il, l) in w.layers.iter().enumerate() {
        let p = format!("dec.blk.{il}");
        if let Some((_, a)) = &l.self_attn {
            attn(&format!("{p}.attn"), a);
        }
        attn(&format!("{p}.cross_attn"), &l.cross_attn);
    }
    for (il, l) in w.layers.iter().enumerate() {
        let p = format!("dec.blk.{il}");
        lin(&format!("{p}.ffn_up"), &l.ffn_up);
        lin(&format!("{p}.ffn_down"), &l.ffn_down);
    }
    out.push(("token_types.weight".into(), w.type_embd.clone()));
    out.push(("decision.scales".into(), w.scales.to_vec()));
    out
}

// ------------------------------------------------------------------ cases --

struct Case {
    name: &'static str,
    positions: usize,
    questions: Vec<QuestionSpan>,
    options: Vec<OptionSpan>,
}

fn q(kind: QuestionType, start: usize, end: usize) -> QuestionSpan {
    QuestionSpan { kind, start, end }
}

fn o(question: usize, start: usize, end: usize) -> OptionSpan {
    OptionSpan {
        question,
        start,
        end,
    }
}

fn cases() -> Vec<Case> {
    use QuestionType::{Choice, Noul, Score};
    // Three questions with 1, 2 and 7 options over a prompt of `p`, spans
    // single- and multi-token, touching position 0 and the last position.
    let three = |p: usize| {
        let mut options = vec![o(0, p - 1, p)];
        options.extend([o(1, 0, 1), o(1, 5, 9)]);
        options.extend((0..7).map(|i| {
            let s = 12 + 3 * i;
            o(2, s, if i % 2 == 0 { s + 1 } else { s + 3 })
        }));
        Case {
            name: "",
            positions: p,
            questions: vec![q(Noul, 0, 4), q(Choice, 2, 11), q(Score, 9, p)],
            options,
        }
    };
    vec![
        Case {
            name: "1 position, 1 question, 1 option",
            positions: 1,
            questions: vec![q(Choice, 0, 1)],
            options: vec![o(0, 0, 1)],
        },
        Case {
            name: "2 positions, 1 question, 2 options",
            positions: 2,
            questions: vec![q(Noul, 0, 1)],
            options: vec![o(0, 1, 2), o(0, 0, 2)],
        },
        Case {
            name: "37 positions, 1 question, 7 options",
            positions: 37,
            questions: vec![q(Choice, 0, 6)],
            options: (0..7)
                .map(|i| {
                    let s = 6 + 4 * i;
                    o(0, s, if i == 6 { 37 } else { s + 1 + i % 3 })
                })
                .collect(),
        },
        Case {
            name: "37 positions, 3 questions, 1/2/7 options",
            ..three(37)
        },
        Case {
            name: "300 positions, 3 questions, 2 options each",
            positions: 300,
            questions: vec![q(Score, 0, 40), q(Choice, 100, 101), q(Noul, 250, 300)],
            options: vec![
                o(0, 40, 45),
                o(0, 45, 46),
                o(1, 101, 120),
                o(1, 120, 121),
                o(2, 0, 1),
                o(2, 299, 300),
            ],
        },
        Case {
            name: "300 positions, 3 questions, 1/2/7 options",
            ..three(300)
        },
        Case {
            name: "700 positions, 3 questions, 1/2/7 options (wide GEMM tiles)",
            ..three(700)
        },
        Case {
            name: "300 positions, 1 question, 40 options (three query tiles)",
            positions: 300,
            questions: vec![q(Choice, 0, 8)],
            options: (0..40).map(|i| o(0, 8 + 7 * i, 10 + 7 * i)).collect(),
        },
    ]
}

/// RMS-normalized random rows, scaled per feature like a final RMSNorm.
fn hidden_states(
    rng: &mut Xorshift64Star,
    positions: usize,
    width: usize,
    gain: &[f32],
) -> Vec<f32> {
    let mut h = Vec::with_capacity(positions * width);
    for _ in 0..positions {
        // A mean offset per row keeps LayerNorm's centering honest.
        let shift = rng.next_f32_range(-0.5, 0.5);
        let row: Vec<f32> = (0..width)
            .map(|_| {
                // Sum of uniforms: roughly normal, heavier than uniform.
                (0..4).map(|_| rng.next_f32_range(-1.0, 1.0)).sum::<f32>() + shift
            })
            .collect();
        let rms = (row.iter().map(|v| v * v).sum::<f32>() / width as f32 + 1e-6).sqrt();
        h.extend(row.iter().zip(gain).map(|(v, g)| v / rms * g));
    }
    h
}

/// `proj_memory(LayerNorm(hidden))` in f64, the head's first intermediate.
fn memory_reference(w: &HeadWeights, hidden: &[f32]) -> Vec<f32> {
    let mut out = Vec::new();
    for row in hidden.chunks_exact(w.hidden) {
        let n = row.len() as f64;
        let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        let var = row
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / n;
        let inv = 1.0 / (var + f64::from(w.eps)).sqrt();
        let normed: Vec<f64> = row
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                (f64::from(v) - mean) * inv * f64::from(w.hidden_norm.weight[i])
                    + f64::from(w.hidden_norm.bias[i])
            })
            .collect();
        let p = &w.proj_memory;
        for r in 0..p.out {
            let wr = &p.weight[r * p.inp..(r + 1) * p.inp];
            out.push(
                wr.iter()
                    .zip(&normed)
                    .map(|(&a, b)| f64::from(a) * b)
                    .sum::<f64>() as f32,
            );
        }
    }
    out
}

struct Errors {
    max_abs: f32,
    max_rel: f32,
    worst_excess: f32,
}

/// Max-abs and relative error of `got` against `want`, and the largest
/// ratio of an error to its `abs + rel * |want|` budget (> 1 fails).
fn errors(got: &[f32], want: &[f32], abs: f32, rel: f32) -> Errors {
    assert_eq!(got.len(), want.len());
    let mut e = Errors {
        max_abs: 0.0,
        max_rel: 0.0,
        worst_excess: 0.0,
    };
    for (&g, &w) in got.iter().zip(want) {
        assert!(g.is_finite(), "non-finite device value {g} (want {w})");
        let d = (g - w).abs();
        e.max_abs = e.max_abs.max(d);
        e.max_rel = e.max_rel.max(d / w.abs().max(1e-6));
        e.worst_excess = e.worst_excess.max(d / (abs + rel * w.abs()));
    }
    e
}

fn device() -> Option<(Arc<CudaContext>, Arc<CudaStream>)> {
    if !driver_available() {
        return None;
    }
    let ctx = CudaContext::new(0).ok()?;
    let stream = ctx.default_stream();
    Some((ctx, stream))
}

#[test]
fn synthetic_head_matches_the_scalar_reference() {
    let Some((ctx, stream)) = device() else {
        println!("SKIPPED: no CUDA device");
        return;
    };
    let (file, config, written) = synthetic_file(0x5eed_c1ef);
    let host = DecisionHead::host_weights(&file, &config).expect("synthetic head loads");

    // host_weights must hand back exactly what was written, tensor by
    // tensor: a swapped name or transposed matrix fails here, before any
    // device arithmetic can agree with it by sharing the mistake.
    let mut loaded = flatten(&host);
    loaded.sort_by(|a, b| a.0.cmp(&b.0));
    let mut written = written;
    written.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        loaded.iter().map(|t| &t.0).collect::<Vec<_>>(),
        written.iter().map(|t| &t.0).collect::<Vec<_>>(),
        "host_weights must read every written tensor and nothing else"
    );
    for ((name, got), (_, want)) in loaded.iter().zip(&written) {
        assert_eq!(got.len(), want.len(), "{name}");
        assert!(
            got.iter()
                .zip(want)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{name}: host_weights does not match the bytes written"
        );
    }
    println!(
        "host_weights: {} tensors bit-identical to the file",
        loaded.len()
    );

    let limits = DecisionLimits {
        max_positions: 1024,
        max_questions: 4,
        max_options: 64,
    };
    let mut head = DecisionHead::from_weights(&ctx, &stream, &host, limits).expect("head loads");
    let mut rng = Xorshift64Star::new(0xdec1_5105);
    let gain: Vec<f32> = (0..HIDDEN).map(|_| rng.next_f32_range(0.5, 2.0)).collect();
    let table: Vec<f32> = rng.vec_f32(VOCAB * HIDDEN, -0.08, 0.08);
    let mut scores = Vec::with_capacity(limits.max_options);
    let mut worst = 0.0f32;

    // Every LM-head format on one shape, then every shape on Q8_0.
    let mut runs: Vec<(Fmt, Case)> = [Fmt::Q6K, Fmt::Q4K, Fmt::Q5K, Fmt::Q4_0, Fmt::Bf16, Fmt::F16]
        .into_iter()
        .map(|f| (f, cases().swap_remove(3)))
        .collect();
    runs.extend(cases().into_iter().map(|c| (Fmt::Q8_0, c)));

    for (fmt, case) in runs {
        let lm_bytes = fmt.encode(&table);
        let row_bytes = lm_bytes.len() / VOCAB;
        let lm_dev: CudaSlice<u8> = stream.clone_htod(&lm_bytes).unwrap();
        let p = case.positions;
        let tokens: Vec<i32> = (0..p)
            .map(|_| rng.next_u32_below(VOCAB as u32) as i32)
            .collect();
        let hidden = hidden_states(&mut rng, p, HIDDEN, &gain);
        let hidden_dev = stream.clone_htod(&hidden).unwrap();

        let want = joint_schema_head(
            &host,
            &hidden,
            &tokens,
            &case.questions,
            &case.options,
            |t| {
                let at = t as usize * row_bytes;
                fmt.decode(&lm_bytes[at..at + row_bytes])
            },
        );
        head.forward(
            &stream,
            &hidden_dev,
            p,
            &tokens,
            &case.questions,
            &case.options,
            HeadTensor {
                bytes: &lm_dev,
                format: fmt.head(),
            },
            &mut scores,
        )
        .expect("forward");
        assert_eq!(scores.len(), case.options.len());
        let e = errors(&scores, &want, SCORE_ABS, SCORE_REL);
        let mem = head.read_memory(&stream, p).unwrap();
        let m = errors(&mem, &memory_reference(&host, &hidden), MEMORY_ABS, 1e-5);
        println!(
            "{:<62} lm {:<5} scores max_abs {:.3e} max_rel {:.3e} (|ref| <= {:.2}) | memory max_abs {:.3e}",
            if case.name.is_empty() { "?" } else { case.name },
            format!("{fmt:?}"),
            e.max_abs,
            e.max_rel,
            want.iter().fold(0.0f32, |a, v| a.max(v.abs())),
            m.max_abs,
        );
        assert!(
            m.worst_excess <= 1.0,
            "{}: memory projection off by {:.3e}",
            case.name,
            m.max_abs
        );
        assert!(
            e.worst_excess <= 1.0,
            "{} ({fmt:?}): scores {scores:?} want {want:?}",
            case.name
        );
        worst = worst.max(e.worst_excess);
    }
    println!("worst score error / budget: {worst:.3}");
}

#[test]
fn requests_outside_the_limits_are_refused() {
    let Some((ctx, stream)) = device() else {
        println!("SKIPPED: no CUDA device");
        return;
    };
    let (file, config, _) = synthetic_file(7);
    let host = DecisionHead::host_weights(&file, &config).unwrap();
    let limits = DecisionLimits {
        max_positions: 16,
        max_questions: 2,
        max_options: 3,
    };
    let mut head = DecisionHead::from_weights(&ctx, &stream, &host, limits).unwrap();
    let lm: CudaSlice<u8> = stream
        .clone_htod(&Fmt::Q8_0.encode(&vec![0.01; VOCAB * HIDDEN]))
        .unwrap();
    let lm_head = HeadTensor::q8_0(&lm);
    let hidden = stream.clone_htod(&vec![0.5f32; 32 * HIDDEN]).unwrap();
    let tokens = [1i32; 32];
    let qs = [q(QuestionType::Choice, 0, 2)];
    let os = [o(0, 2, 4)];
    let mut scores = Vec::with_capacity(4);
    let refused = |r: Result<(), _>| {
        matches!(
            r,
            Err(llmcuda_engine::block::decision::DecisionHeadError::Request(
                _
            ))
        )
    };
    // Too many positions.
    assert!(refused(head.forward(
        &stream,
        &hidden,
        17,
        &tokens[..17],
        &qs,
        &os,
        lm_head,
        &mut scores
    )));
    // An option past the prompt.
    assert!(refused(head.forward(
        &stream,
        &hidden,
        4,
        &tokens[..4],
        &qs,
        &[o(0, 3, 5)],
        lm_head,
        &mut scores
    )));
    // A question without options.
    assert!(refused(head.forward(
        &stream,
        &hidden,
        8,
        &tokens[..8],
        &[qs[0], q(QuestionType::Noul, 4, 6)],
        &os,
        lm_head,
        &mut scores
    )));
    // A token outside the output embedding.
    let mut bad = tokens[..8].to_vec();
    bad[3] = VOCAB as i32;
    assert!(refused(head.forward(
        &stream,
        &hidden,
        8,
        &bad,
        &qs,
        &os,
        lm_head,
        &mut scores
    )));
    // And a valid request still runs after all that.
    head.forward(
        &stream,
        &hidden,
        8,
        &tokens[..8],
        &qs,
        &os,
        lm_head,
        &mut scores,
    )
    .unwrap();
    assert_eq!(scores.len(), 1);
    assert!(scores[0].is_finite());
}

// ------------------------------------------------------------ real weights --

#[test]
fn real_head_matches_the_scalar_reference() {
    let Ok(path) = std::env::var("LLMCUDA_CLEF_MODEL") else {
        println!("SKIPPED: LLMCUDA_CLEF_MODEL is not set");
        return;
    };
    let Some((ctx, stream)) = device() else {
        println!("SKIPPED: no CUDA device");
        return;
    };
    let file = GgufFile::open(&path).expect("open the model");
    let config = ModelConfig::from_gguf(&file).expect("config");
    let host = DecisionHead::host_weights(&file, &config).expect("real head loads");
    let hidden_size = host.hidden;
    println!(
        "real head: hidden {} width {} heads {} ff {} layers {} ({} routing)",
        host.hidden,
        host.width,
        host.heads,
        host.layers[0].ffn_up.out,
        host.layers.len(),
        host.routing_layers
    );
    let info = file.tensor("output.weight").expect("output.weight");
    let format = HeadFormat::from_ggml(info.ggml_type.name()).expect("supported LM head format");
    let vocab = info.dims[1] as usize;
    let out_bytes = file.tensor_bytes("output.weight").unwrap();
    let row_bytes = out_bytes.len() / vocab;
    let lm_dev: CudaSlice<u8> = stream.clone_htod(out_bytes).unwrap();
    let gain = file
        .tensor_bytes("output_norm.weight")
        .map(dequantize_row_f32)
        .unwrap_or_else(|| vec![1.0; hidden_size]);
    let decode_row = |t: i32| {
        let at = t as usize * row_bytes;
        let b = &out_bytes[at..at + row_bytes];
        match format {
            HeadFormat::Q8_0 => dequantize_row_q8_0(b).unwrap(),
            HeadFormat::Q6K => dequantize_row_q6_k(b).unwrap(),
            HeadFormat::Q4K => dequantize_row_q4_k(b).unwrap(),
            HeadFormat::Q5K => dequantize_row_q5_k(b).unwrap(),
            HeadFormat::Q4_0 => dequantize_row_q4_0(b).unwrap(),
            HeadFormat::Bf16 => dequantize_row_bf16(b),
            HeadFormat::F16 => dequantize_row_f16(b),
        }
    };

    let limits = DecisionLimits {
        max_positions: 8192,
        max_questions: 8,
        max_options: 64,
    };
    let mib = |b: usize| b as f64 / (1 << 20) as f64;
    {
        // The runtime's defaults: DEFAULT_DECISION_MAX_POSITIONS,
        // DECISION_MAX_QUESTIONS, DECISION_MAX_OPTIONS.
        let defaults = DecisionLimits {
            max_positions: llmcuda_engine::runtime::DEFAULT_DECISION_MAX_POSITIONS,
            max_questions: llmcuda_engine::runtime::DECISION_MAX_QUESTIONS,
            max_options: llmcuda_engine::runtime::DECISION_MAX_OPTIONS,
        };
        let head = DecisionHead::from_weights(&ctx, &stream, &host, defaults).expect("head");
        let (weights, total) = head.device_bytes();
        println!(
            "device memory at {defaults:?}: weights {:.0} MiB + workspace {:.0} MiB",
            mib(weights),
            mib(total - weights)
        );
    }
    let mut head = DecisionHead::from_weights(&ctx, &stream, &host, limits).expect("head");
    let (weights, total) = head.device_bytes();
    println!(
        "device memory at {limits:?}: weights {:.0} MiB + workspace {:.0} MiB",
        mib(weights),
        mib(total - weights)
    );
    let mut rng = Xorshift64Star::new(0xc1ef_f1a5);
    let mut scores = Vec::with_capacity(limits.max_options);

    use QuestionType::{Choice, Noul, Score};
    let questions = vec![q(Noul, 0, 12), q(Choice, 40, 61), q(Score, 120, 133)];
    let span_options = |p: usize| {
        vec![
            o(0, 12, 13),
            o(0, 13, 14),
            o(1, 61, 64),
            o(1, 64, 70),
            o(1, 70, 71),
            o(2, 133, 134),
            o(2, 134, 135),
            o(2, 135, 136),
            o(2, p - 10, p),
        ]
    };
    // 200 positions run the 64x64 GEMM everywhere; 600 puts the memory and
    // K/V projections on the 128x128 tiles, with a ragged last row tile.
    for p in [200usize, 600] {
        let options = span_options(p);
        let tokens: Vec<i32> = (0..p)
            .map(|_| rng.next_u32_below(vocab as u32) as i32)
            .collect();
        let hidden = hidden_states(&mut rng, p, hidden_size, &gain);
        let hidden_dev = stream.clone_htod(&hidden).unwrap();
        let started = std::time::Instant::now();
        let want = joint_schema_head(&host, &hidden, &tokens, &questions, &options, decode_row);
        println!(
            "scalar reference: {:.1} s for {p} positions",
            started.elapsed().as_secs_f64()
        );
        head.forward(
            &stream,
            &hidden_dev,
            p,
            &tokens,
            &questions,
            &options,
            HeadTensor {
                bytes: &lm_dev,
                format,
            },
            &mut scores,
        )
        .expect("forward");
        let e = errors(&scores, &want, SCORE_ABS, SCORE_REL);
        let mem = head.read_memory(&stream, p).unwrap();
        let m = errors(
            &mem,
            &memory_reference(&host, &hidden),
            REAL_MEMORY_ABS,
            1e-5,
        );
        println!("device scores    {scores:?}");
        println!("reference scores {want:?}");
        println!(
            "real head, {p} positions, {} options: scores max_abs {:.3e} max_rel {:.3e}; memory max_abs {:.3e}",
            options.len(),
            e.max_abs,
            e.max_rel,
            m.max_abs,
        );
        assert!(
            m.worst_excess <= 1.0,
            "memory projection off by {:.3e}",
            m.max_abs
        );
        assert!(e.worst_excess <= 1.0, "scores off by {:.3e}", e.max_abs);
    }
    let options = span_options(200);

    // Timing: event-bracketed `forward` (it synchronizes internally, so the
    // bracket is the whole queued head plus its metadata upload and score
    // download), median of five after one warm-up.
    for p in [2048usize, 8192] {
        let tokens: Vec<i32> = (0..p)
            .map(|_| rng.next_u32_below(vocab as u32) as i32)
            .collect();
        let hidden = stream
            .clone_htod(&hidden_states(&mut rng, p, hidden_size, &gain))
            .unwrap();
        let questions = vec![q(Noul, 0, 12), q(Choice, 40, 61), q(Score, 120, 133)];
        let options = options.clone();
        let mut times = Vec::new();
        for i in 0..6 {
            let start = ctx
                .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                .unwrap();
            let end = ctx
                .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                .unwrap();
            start.record(&stream).unwrap();
            head.forward(
                &stream,
                &hidden,
                p,
                &tokens,
                &questions,
                &options,
                HeadTensor {
                    bytes: &lm_dev,
                    format,
                },
                &mut scores,
            )
            .unwrap();
            end.record(&stream).unwrap();
            let ms = start.elapsed_ms(&end).unwrap();
            if i > 0 {
                times.push(ms);
            }
        }
        times.sort_by(f32::total_cmp);
        println!(
            "forward at {p} positions, {} questions, {} options: median {:.2} ms (min {:.2}, max {:.2}) over {} runs",
            questions.len(),
            options.len(),
            times[times.len() / 2],
            times[0],
            times[times.len() - 1],
            times.len()
        );
    }
}

fn dequantize_row_f32(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}
