//! Manual agent-conversation benchmark for the test-only slot prototype.
//!
//! Run with `cargo test --release -p llmcuda-engine --lib
//! resident_continuation::conversations::agent_conversations -- --ignored
//! --nocapture --test-threads=1`, pinning an idle GPU first. All inference
//! shapes, GPU slots, history capacity and pinned snapshots are allocated
//! before timing. The production router is not involved.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaStream, PinnedHostSlice};
use llmcuda_gguf::GgufFile;
use llmcuda_model::ModelConfig;
use llmcuda_model::weights::WeightSchema;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::{BPE, Vocab};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{AddedToken, SplitDelimiterBehavior, Tokenizer};

use super::{ResidentSlots, Scope};
use crate::DeviceWeights;
use crate::forward::{Forward, arena_holds_model_entry};
use crate::state::{SequenceSnapshot, SequenceState, SnapshotArena};

const CAPACITY: usize = 16_384;
const MAX_REPLY: usize = 256;
const MAX_CHUNK: usize = 512;
const ASSISTANT: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";

/// Same GGUF BPE/pre-tokenizer algorithm as llmcuda-server's from_gguf. Kept in
/// this manual fixture so the engine does not acquire a server dependency.
fn tokenizer(file: &GgufFile) -> Tokenizer {
    assert_eq!(file.get_str("tokenizer.ggml.model"), Some("gpt2"));
    assert_eq!(file.get_str("tokenizer.ggml.pre"), Some("qwen35"));
    let tokens = file.get_string_array("tokenizer.ggml.tokens").unwrap();
    let types = file.get_i32_array("tokenizer.ggml.token_type").unwrap();
    assert_eq!(tokens.len(), types.len());
    assert!(types.iter().all(|kind| (1..=6).contains(kind)));
    let vocab: Vocab = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), i as u32))
        .collect();
    let merges = file
        .get_string_array("tokenizer.ggml.merges")
        .unwrap()
        .iter()
        .map(|m| {
            let (a, b) = m.split_once(' ').unwrap();
            (a.to_owned(), b.to_owned())
        })
        .collect();
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .unwrap();
    let pattern = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    let split = Split::new(
        SplitPattern::Regex(pattern.into()),
        SplitDelimiterBehavior::Isolated,
        true,
    )
    .unwrap();
    let mut result = Tokenizer::new(bpe);
    result.with_pre_tokenizer(Some(Sequence::new(vec![
        split.into(),
        ByteLevel::new(false, false, false).into(),
    ])));
    result.with_decoder(Some(ByteLevelDecoder::new(false, false, false)));
    result.add_special_tokens(
        &tokens
            .iter()
            .zip(types)
            .filter(|(_, kind)| (2..=5).contains(*kind))
            .map(|(token, _)| AddedToken::from(token.clone(), true))
            .collect::<Vec<_>>(),
    );
    result
}

fn append(tokenizer: &Tokenizer, history: &mut Vec<i32>, text: &str) {
    let encoded = tokenizer.encode(text, false).unwrap();
    assert!(history.len() + encoded.len() <= CAPACITY);
    history.extend(encoded.get_ids().iter().map(|&id| id as i32));
}

struct Checkpoint {
    scope: Scope,
    tokens: Vec<i32>,
    snapshot: Option<SequenceSnapshot>,
}

struct Turn {
    prompt: usize,
    reused: usize,
    resident: usize,
    restore_bytes: usize,
    snapshot_bytes: usize,
    token_position_bytes: usize,
    ttft_ms: f64,
    ids: Vec<u32>,
    first_logits: Vec<f32>,
    response: String,
}

struct Conversation {
    turns: Vec<Turn>,
    wall_ms: f64,
}

struct Harness {
    stream: Arc<CudaStream>,
    passes: Vec<Forward>,
    slots: ResidentSlots<SequenceState>,
    arena: SnapshotArena,
    sampled: PinnedHostSlice<i32>,
}

fn sample(pass: &mut Forward, stream: &Arc<CudaStream>, host: &mut PinnedHostSlice<i32>) -> i32 {
    pass.launch_argmax(stream).unwrap();
    stream.memcpy_dtoh(&pass.argmax_out, host).unwrap();
    stream.synchronize().unwrap();
    host.as_slice().unwrap()[0]
}

impl Harness {
    fn turn(
        &mut self,
        tok: &Tokenizer,
        scope: Scope,
        history: &mut Vec<i32>,
        checkpoint: &mut Checkpoint,
        resident: bool,
        validate: bool,
    ) -> Turn {
        let prompt = history.len();
        let mut ids = Vec::with_capacity(MAX_REPLY);
        let started = Instant::now();
        let mut lease = self
            .slots
            .begin(scope, history, prompt + MAX_REPLY, 1, resident)
            .unwrap();
        let resident_prefix = lease.reused;
        let mut reused = resident_prefix;
        let mut restore_bytes = 0;
        lease.prepare(|state| {
            if checkpoint.scope == scope
                && !checkpoint.tokens.is_empty()
                && checkpoint.tokens.len() < history.len()
                && history.starts_with(&checkpoint.tokens)
                && let Some(snapshot) = &checkpoint.snapshot
            {
                assert_eq!(snapshot.position(), checkpoint.tokens.len());
                state.restore(&self.stream, snapshot).unwrap();
                reused = snapshot.position();
                restore_bytes = snapshot.attention_bytes() + snapshot.gdn_bytes();
            } else {
                state.reset(&self.stream).unwrap();
            }
        });
        let state = lease.state();
        assert_eq!(state.position(), reused);
        let mut position = reused;
        let mut last = 0;
        let mut calls = 0;
        while position < prompt {
            let remaining = (prompt - position).min(MAX_CHUNK);
            last = remaining.ilog2() as usize;
            let count = 1 << last;
            self.passes[last]
                .run(
                    &self.stream,
                    state,
                    &history[position..position + count],
                    |_, _| {},
                )
                .unwrap();
            position += count;
            calls += 1;
        }
        let mut next = sample(&mut self.passes[last], &self.stream, &mut self.sampled);
        let ttft_ms = started.elapsed().as_secs_f64() * 1e3;
        let first_logits = if validate {
            self.stream.clone_dtoh(self.passes[last].logits()).unwrap()
        } else {
            Vec::new()
        };
        let eos = tok.token_to_id("<|im_end|>").unwrap();
        let end = tok.token_to_id("<|endoftext|>").unwrap();
        for index in 0..MAX_REPLY {
            ids.push(next as u32);
            history.push(next);
            if next as u32 == eos || next as u32 == end {
                break;
            }
            if index + 1 < MAX_REPLY {
                self.passes[0]
                    .run(&self.stream, state, &[next], |_, _| {})
                    .unwrap();
                next = sample(&mut self.passes[0], &self.stream, &mut self.sampled);
                calls += 1;
            }
        }
        assert!(
            matches!(ids.last(), Some(&id) if id == eos || id == end),
            "reply exceeded {MAX_REPLY} tokens: {}",
            tok.decode(&ids, false).unwrap()
        );
        // The last predicted token (normally EOS) was emitted but not consumed.
        let consumed = state.position();
        assert_eq!(consumed + 1, history.len());
        let snapshot = state.snapshot(&self.stream, &self.arena, None).unwrap();
        let snapshot_bytes = snapshot.attention_bytes() + snapshot.gdn_bytes();
        checkpoint.snapshot = Some(snapshot);
        checkpoint.tokens.clear();
        checkpoint.tokens.extend_from_slice(&history[..consumed]);
        checkpoint.scope = scope;
        lease.commit(&history[..consumed], consumed);
        Turn {
            prompt,
            reused,
            resident: resident_prefix,
            restore_bytes,
            snapshot_bytes,
            // Each pass uploads its input token IDs and the two i32 positions.
            // Snapshot restore additionally publishes those two positions.
            token_position_bytes: (consumed - reused) * 4
                + calls * 8
                + usize::from(restore_bytes != 0) * 8,
            ttft_ms,
            response: tok.decode(&ids, true).unwrap(),
            ids,
            first_logits,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn conversation(
        &mut self,
        tok: &Tokenizer,
        scope: Scope,
        root: &Path,
        task: usize,
        resident: bool,
        validate: bool,
    ) -> Conversation {
        assert!(self.slots.evict(0));
        let files = if task == 0 {
            ["Cargo.toml", "CONTRIBUTING.md"]
        } else {
            ["docs/MODEL.md", "docs/CACHE.md"]
        };
        let question = if task == 0 {
            "Explain how a contributor should validate and commit a CUDA inference kernel change."
        } else {
            "Explain why attention and recurrent state use different cache geometry, and how the dense model differs."
        };
        let system = format!(
            "You are a repository assistant using a read-only file tool. Before answering, read {} and {} in that order, once each. To call the tool, output one or more READ lines, each containing READ followed by one space and one file path, then end your turn. Tool replies contain the file. After reading both, output FINAL followed by a concise answer of at most 80 words. On a later user follow-up, answer with FINAL directly using the files already read. Do not use Markdown fences around tool calls. Do not think aloud.",
            files[0], files[1]
        );
        let mut history = Vec::with_capacity(CAPACITY);
        append(
            tok,
            &mut history,
            &format!(
                "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n{ASSISTANT}"
            ),
        );
        let mut checkpoint = Checkpoint {
            scope,
            tokens: Vec::with_capacity(CAPACITY),
            snapshot: None,
        };
        let mut turns = Vec::with_capacity(8);
        let mut read = [false; 2];
        let mut followed_up = false;
        let started = Instant::now();
        for _ in 0..8 {
            let turn = self.turn(
                tok,
                scope,
                &mut history,
                &mut checkpoint,
                resident,
                validate,
            );
            let response = turn.response.trim().to_owned();
            turns.push(turn);
            if response.starts_with("READ ") {
                let mut results = String::from("\n<|im_start|>user");
                for line in response.lines().filter(|line| !line.trim().is_empty()) {
                    let path = line
                        .strip_prefix("READ ")
                        .expect("invalid tool-call line")
                        .trim();
                    let index = files
                        .iter()
                        .position(|candidate| *candidate == path)
                        .unwrap_or_else(|| {
                            panic!("model requested a non-whitelisted file: {path:?}")
                        });
                    assert!(!read[index], "model repeated a file read");
                    read[index] = true;
                    let content = std::fs::read_to_string(root.join(path)).unwrap();
                    results.push_str(&format!(
                        "\n<tool_response>\nFile {path}:\n{content}\n</tool_response>"
                    ));
                }
                results.push_str(&format!("<|im_end|>\n{ASSISTANT}"));
                append(tok, &mut history, &results);
            } else {
                if !read.iter().all(|&done| done) {
                    let missing = files
                        .iter()
                        .zip(read)
                        .filter(|(_, done)| !done)
                        .map(|(path, _)| format!("READ {path}"))
                        .collect::<Vec<_>>()
                        .join("\n");
                    append(
                        tok,
                        &mut history,
                        &format!(
                            "\n<|im_start|>user\nYou must inspect the files before answering. Call the file tool now by outputting these lines, with no answer yet:\n{missing}<|im_end|>\n{ASSISTANT}"
                        ),
                    );
                    continue;
                }
                if followed_up {
                    return Conversation {
                        turns,
                        wall_ms: started.elapsed().as_secs_f64() * 1e3,
                    };
                }
                followed_up = true;
                append(
                    tok,
                    &mut history,
                    &format!(
                        "\n<|im_start|>user\nGive the single most important constraint from those files, in one sentence.<|im_end|>\n{ASSISTANT}"
                    ),
                );
            }
        }
        panic!(
            "agent did not finish the conversation within eight turns: {:?}",
            turns.iter().map(|turn| &turn.response).collect::<Vec<_>>()
        );
    }
}

#[test]
#[ignore = "manual GPU benchmark with model-generated file-tool conversations"]
fn agent_conversations() {
    let path = std::env::var_os("LLMCUDA_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf"
            ))
        });
    let gpu =
        std::env::var("CUDA_VISIBLE_DEVICES").expect("pin one idle GPU for this manual benchmark");
    assert!(!gpu.contains(','), "benchmark one GPU at a time");
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let ctx = CudaContext::new(0).expect("manual benchmark requires a CUDA device");
    let stream = ctx.new_stream().unwrap();
    // SAFETY: this harness owns and uses exactly one stream for all passes.
    unsafe { ctx.disable_event_tracking() };
    let file = GgufFile::open(path).unwrap();
    let config = ModelConfig::from_gguf(&file).unwrap();
    let target = ModelConfig::qwen3_6_35b_a3b();
    assert_eq!(config.architecture, target.architecture);
    assert_eq!(config.num_layers, target.num_layers);
    assert_eq!(config.hidden_size, target.hidden_size);
    assert_eq!(config.vocab_size, target.vocab_size);
    assert_eq!(config.gdn, target.gdn);
    assert_eq!(config.attention, target.attention);
    assert_eq!(config.ffn, target.ffn);
    assert!(config.native_context as usize >= CAPACITY);
    let tok = tokenizer(&file);
    let schema = WeightSchema::new(&config);
    let directory = schema.resolve(&file).unwrap();
    let (weights, _) =
        DeviceWeights::load_where_entry(&ctx, &stream, &file, &directory, |role, ty| {
            arena_holds_model_entry(&config, role, ty)
        })
        .unwrap();
    let mut passes = vec![
        Forward::new(
            &ctx,
            &stream,
            &file,
            &directory,
            &weights,
            config.clone(),
            1,
        )
        .unwrap(),
    ];
    for power in 1..=MAX_CHUNK.ilog2() {
        passes.push(
            passes[0]
                .reshape(&ctx, &stream, &file, &directory, &weights, 1 << power)
                .unwrap(),
        );
    }
    // All shapes have a fixed live-token count. Publish each scalar before
    // timing; subsequent forwards elide the unchanged counter upload.
    for pass in &mut passes {
        pass.moe.publish_tokens(&stream, pass.tokens).unwrap();
    }
    let state = passes[0].new_state(&stream, CAPACITY).unwrap();
    let state_bytes = state.bytes();
    let arena = SnapshotArena::new(&ctx, &config, CAPACITY, 2).unwrap();
    println!(
        "ALLOCATION gpu_state_bytes={state_bytes} history_bytes={} pinned_bytes={} shapes=1,2,4,8,16,32,64,128,256,512",
        CAPACITY * 4,
        arena.bytes_per_slot() * arena.capacity()
    );
    // SAFETY: sample writes the complete one-token buffer before reading it.
    let sampled = unsafe { ctx.alloc_pinned::<i32>(1).unwrap() };
    let mut harness = Harness {
        stream,
        passes,
        slots: ResidentSlots::new(vec![state], CAPACITY),
        arena,
        sampled,
    };
    let base = Scope {
        model: &weights as *const _ as usize,
        tokenizer: &tok as *const _ as usize,
        rope: 0,
        session: 0,
    };
    for task in 0..2 {
        let scope = Scope {
            session: task as u64,
            ..base
        };
        // Discarded validation/warmup conversations also verify the complete
        // first-token logits and every emitted token, not plausible text.
        let a = harness.conversation(&tok, scope, &root, task, false, true);
        let b = harness.conversation(&tok, scope, &root, task, true, true);
        assert_eq!(a.turns.len(), b.turns.len());
        for (index, (a, b)) in a.turns.iter().zip(&b.turns).enumerate() {
            assert_eq!(a.ids, b.ids, "task {task}, turn {index} emitted tokens");
            assert_eq!(a.first_logits.len(), config.vocab_size as usize);
            assert!(
                a.first_logits
                    .iter()
                    .zip(&b.first_logits)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "task {task}, turn {index} logits"
            );
            println!(
                "TRANSCRIPT task={task} turn={index} {}",
                a.response.replace('\n', "\\n")
            );
        }
        println!(
            "VALIDATED task={task} turns={} all_generated_ids_and_first_logits_bit_identical",
            a.turns.len()
        );
        if std::env::var_os("LLMCUDA_CONTINUATION_VALIDATE_ONLY").is_some() {
            continue;
        }
        for (iteration, arm) in "ABBAAB".chars().enumerate() {
            let status = std::process::Command::new("nvidia-smi")
                .args([
                    "-i",
                    &gpu,
                    "--query-gpu=uuid,temperature.gpu,clocks.sm,clocks.mem,memory.used",
                    "--format=csv,noheader",
                ])
                .output()
                .unwrap();
            assert!(status.status.success());
            println!(
                "GPU task={task} iteration={} arm={arm} {}",
                iteration + 1,
                String::from_utf8(status.stdout).unwrap().trim()
            );
            let result = harness.conversation(&tok, scope, &root, task, arm == 'B', false);
            assert_eq!(result.turns.len(), a.turns.len());
            for (turn, (measured, reference)) in result.turns.iter().zip(&a.turns).enumerate() {
                assert_eq!(
                    measured.ids, reference.ids,
                    "timed conversation changed emitted tokens"
                );
                println!(
                    "TURN task={task} iteration={} arm={arm} turn={turn} prompt={} reused={} resident={} restore_bytes={} snapshot_bytes={} token_position_bytes={} ttft_ms={:.6} emitted={}",
                    iteration + 1,
                    measured.prompt,
                    measured.reused,
                    measured.resident,
                    measured.restore_bytes,
                    measured.snapshot_bytes,
                    measured.token_position_bytes,
                    measured.ttft_ms,
                    measured.ids.len()
                );
            }
            println!(
                "CONVERSATION task={task} iteration={} arm={arm} wall_ms={:.6}",
                iteration + 1,
                result.wall_ms
            );
        }
    }
}
