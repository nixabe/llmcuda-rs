# Testing and numerics

## The risk this exists for

Numerics drift is the highest-likelihood risk in this project. A subtly wrong
kernel does not crash and does not produce garbage — the model stays fluent
and gets quietly worse. No throughput benchmark catches it, and reading
generated text and judging it plausible catches it least of all.

So: **a kernel without a passing differential test is not done, regardless of
how fast it runs.**

## The structure

Every kernel has a scalar fp32 CPU reference in
[`llmcuda-kernels`](../crates/llmcuda-kernels), written for obvious correctness
rather than speed. These are the oracle. A clever reference that is subtly
wrong is worse than no reference at all, so they are deliberately naive.

`llmcuda-kernels` depends on no CUDA and no device. That is a design property,
not an accident: the correctness work must be runnable and debuggable on a
laptop, and must never be blocked on GPU access.

| Module | Reference |
| --- | --- |
| `gdn::recurrent` | Gated DeltaNet, recurrent (decode) form |
| `gdn::chunked` | Gated DeltaNet, chunked parallel (prefill) form |
| `gdn::tri` | Lower-triangular inverse by forward substitution |
| `moe::router` | Softmax + top-k over 256 experts, deterministic ties |
| `moe::dispatch` | `moe_align_block_size` equivalent |
| `moe::gemm` | Grouped MoE GEMM with SwiGLU and top-k reduction |
| `attention` | GQA 16:2, head dim 256, plus online-softmax form |
| `rope` | Partial rotary — 64 of 256 dims |
| `norm` | RMSNorm, SwiGLU, residual |
| `quant` | Q6_K and Q8_0 dequantization |
| `compare` | The differential harness |
| `rng` | Seeded xorshift, for reproducible inputs |

## Metrics: report all of them

`compare()` returns a `ComparisonResult` carrying:

| Metric | Catches | Misses |
| --- | --- | --- |
| `max_abs_error` (+ index) | Magnitude errors | Direction errors on small values |
| `max_rel_error` (+ index) | Proportional errors on small values | Errors on near-zero references |
| `cosine_similarity` | Direction errors | **Uniform scaling** |
| `non_finite_count` | NaN/Inf | Everything else |

**Cosine alone is not sufficient.** A kernel returning exactly half the correct
output has cosine similarity 1.0. `compare.rs` carries a test named
`cosine_alone_hides_a_magnitude_error_but_max_abs_catches_it` that pins this,
and every equivalence assertion in the crate gates on the full tolerance via
`assert_matches`, not on cosine.

Results report **which element** was worst and at what index.
`worst_abs_pair()` returns the offending `(candidate, reference)` values.
"Cosine 0.994" is not actionable; "worst element at index 3117, expected 0.42,
got 0.31" is.

## Tolerances

| Preset | Bounds | For |
| --- | --- | --- |
| `exact()` | 0, 0, cosine 1.0 | Bit-identical paths |
| `tight_fp32()` | 1e-3 abs/rel, cosine 1−1e-5 | Two fp32 implementations of the same formula differing only in operation order |
| `gdn_chunk_vs_recurrent()` | 5e-2 abs/rel, cosine 1−1e-3 | The milestone-01 gate |
| `reduced_precision_gpu()` | looser | fp16/tf32 device kernels vs fp32 reference |

fp32 summation is not associative, so exact equality is the wrong bar for a
reformulation. These are tight enough to catch a real formulation bug while
absorbing reassociation noise. Each carries its rationale in a doc comment.

Inputs come from a seeded xorshift generator (`rng::Xorshift64Star`) so a
failure reproduces exactly. No `rand` dependency.

## Controls: a known-good case in every differential

A differential test says "the device agrees with the reference." It does not
say "the device agrees with the reference *about the thing the test is named
after*." Those come apart, and when they do the test still passes.

So a differential over several cases should include at least one whose answer
is already known independently, and **a failing control is more informative
than a failing case**. It means the harness is measuring something other than
what its name claims, and every green result beside it is suspect.

The example that earned this section no longer exists, which is the reason to
write the lesson down rather than leave it in the test. A
`moe_quant_formats_differential.rs` covered the community-quant expert
prologues (Q4_K, Q5_K, Q4_0) and included Q6_K and Q8_0 as a control, because
those two are already covered on real weights by `moe_differential.rs`. On
first run the three new formats passed at cosine 1.000000 and **the control
failed** at cosine 0.999989, max_abs 5.5e-4. The feature was later reverted
for an unrelated performance defect (`40f7fc6`, and see `docs/MODEL.md`), and
the test went with it.

Neither number was wrong. Q6_K and Q8_0 have int8 tensor-core bodies and took
that path; the three new formats have none and took the fp32 one. The int8
path quantizes the *activations* as well, so it sits further from an fp32
reference — a property of that path, not of Q6_K's unpacking. The harness was
comparing two different kernels against one reference and reporting the gap as
a fact about a format. Fixed by calling `disable_tensor_cores()` so every case
goes down the path actually under test.

Without the control, three passing tests would have shipped, measuring a
narrower claim than their names made, and nothing would have said so.

The surviving instance of the same pattern is
`lm_head_formats_differential.rs`, which checks each added head format against
the file's own `output.weight` requantized to it, with Q8_0 — the format the
file actually stores — as the case whose answer is known in advance.

## The GDN equivalence test

This is the most valuable check in the crate: it validates the recurrent
and chunked formulations independently of captured model activations.

The chunked parallel form (prefill) and the recurrent form (decode) compute the
same function by very different routes — the chunked form requires inverting a
lower-triangular matrix per chunk. **They must agree on identical input.**

That equivalence is what will later reveal whether a GPU chunked kernel is
wrong, because it is checkable without any reference activations from
`transformers`.

Current status: passes at the **real Qwen3.6 shape** — head dim 128, chunk
length 64, taken from `ModelConfig` rather than hardcoded — over a 197-token
sequence spanning three full chunks and a partial one, gated on the full
tolerance, comparing both the per-token outputs and the final recurrent state.
Also covered: `chunk_len = 1`, `chunk_len = seq_len`, nonzero initial state,
and chunk-boundary independence.

## Provenance: ported versus derived

Formulations are cited in doc comments, distinguishing what was ported from
upstream from what was inferred. That distinction is the honest measure of how
much a reference should be trusted.

**Ported and cross-checked.** The GDN *recurrent* form was derived from two
independent upstream implementations that were compared line by line and found
to agree exactly:

- llama.cpp, `ggml_compute_forward_gated_delta_net_one_chunk`
  (`ggml/src/ggml-cpu/ops.cpp`)
- vLLM, `fused_recurrent_gated_delta_rule_fwd_kernel`, which is what
  `Qwen3NextGatedDeltaNetAttention` — the module this model family actually
  instantiates — calls on CPU

Both confirm L2-normalized q/k with eps 1e-6, a **scalar per-head** decay
(confirmed from `A_log`/`dt_bias` having shape `[num_v_heads]`, not the
per-channel variant llama.cpp also supports but Qwen does not use),
decay-then-correction-then-output ordering, and an output scale of
`1/sqrt(head_dim)`.

Two independently written implementations converging exactly is strong
evidence. It is **not** verification against a captured Qwen3.6 activation, and
it does not rule out Qwen3.6 having changed the gating relative to the
Qwen3-Next/3.5 code path they share.

**Derived, not ported.** No scalar chunked reference exists upstream — vLLM's
is Triton-JIT and llama.cpp has none — so the chunked form was derived from the
verified recurrence via the WY representation. The derivation yields
`(I + A)^{-1}`; `(I − A)^{-1}` is valid under the opposite sign convention,
and the equivalence test is
what actually validates it.

Other provenance: `moe::dispatch` follows vLLM's `moe_align_block_size` output
contract, verified against the CUDA source for padding and sentinel semantics.
Q6_K/Q8_0 *dequantization* is transcribed from `ggml-quants.c`; the
*quantizer* is a simple encoder written for round-trip testing only and is
explicitly **not** llama.cpp's least-squares quantizer. RoPE's NEOX pairing was
confirmed from `ggml_compute_forward_rope_flt`.

**Vision tower: ported and validated against upstream execution.** The
`vision` reference (patch conv, position-grid interpolation, 27 SigLIP
blocks, `qwen3vl_merger` projector) is ported from
`tools/mtmd/models/qwen3vl.cpp` and `tools/mtmd/clip.cpp`, and — unusually
for this crate — validated not just by construction but against llama.cpp
*running the same mmproj file*: `tests/vision_golden.rs` replays
`llama-mtmd-debug`'s synthetic images through the reference and compares the
final embeddings value-for-value against that tool's tensor dumps. Captured
logs are never committed; the test reads them from
`LLMCUDA_MTMD_GOLDEN_DIR` and skips without it. Generate with, for `IMG` in
`gray`/`red`/`cb` and `N` in `64`/`96`:

```sh
llama-mtmd-debug -m <model.gguf> --mmproj <mmproj-F16.gguf> \
  -p encode --image $IMG -n $N --no-mmproj-offload -ngl 0 --no-warmup \
  > $LLMCUDA_MTMD_GOLDEN_DIR/golden-$IMG-$N.log 2>&1
```

Agreement at last check: ≤ 3.2e-4 on output mass, ≤ 5e-3 per printed value —
the budget covers llama.cpp's f16 GELU table (`GGML_GELU_FP16`) against the
reference's exact tanh form. A Q8_0 mmproj has no golden of its own; the
loader is checked against the F16 export of the same checkpoint instead
(`llmcuda-engine/tests/vision_differential.rs`, `q8_0_mmproj_tracks_its_f16_export`,
behind `LLMCUDA_MMPROJ_Q8_0` and `LLMCUDA_MMPROJ_F16`): every dequantized
matrix within 0.6 Q8_0 steps of its F16 twin, then the two device towers on
one image. Text-side M-RoPE (`mrope`) is ported from
`ggml_mrope_cache_init`'s `IMROPE` branch and carries a bit-exactness test
that scalar positions collapse to `rope::apply_rope` — the same reduction
`llmcuda-engine`'s text path depends on.

The second preprocessing path, `preprocess_hf` (the Hugging Face
processor's, which `/v1/systemone` uses), is checked against
the code it ports, not against llama.cpp: `tests/hf_image_golden.rs`
replays transformers' own `smart_resize` over 4,980 size and bound
combinations (248 of them refused for aspect ratio) and PyTorch's CPU uint8
antialiased bicubic over 84 resizes and 34M values, and requires every size
and every byte to match. Corrupting the port's tie rounding or its weight
precision fails the matching gate. Generate with
`python tools/oracle/hf_image.py $LLMCUDA_HF_IMAGE_GOLDEN_DIR` (torch,
numpy, transformers); the test SKIPS without the variable.

## The dense architecture (`qwen35`)

Qwen3.8-27B shares every kernel with Qwen3.6-35B-A3B except the feed-forward
block and the prefill projections, and the shared ones are gated by the tests
above at Qwen3.6's widths. What is new is gated by these:

| Test | What it asserts | Needs |
| --- | --- | --- |
| `llmcuda-model/tests/real_dense_model_weights.rs` | `WeightSchema` resolves against all 866 tensors of the real file with none left unclaimed; every derived hyperparameter (the GDN head split above all) agrees with the file's own metadata; the file carries no routed tensor; and the *routed* schema refuses it by architecture rather than by a wall of shapes | the file |
| `llmcuda-engine/tests/dense_ffn_differential.rs` | The whole dense block against `expert_mlp` + `rms_norm` on real Q8_0 weights, at all four of its kernel paths | the file, a device |
| `llmcuda-engine/tests/hgemm_differential.rs` | The fp16 weight-only GEMM every dense prefill projection runs on, against an f64 oracle that rounds exactly where the kernel says it rounds (activations to fp16 once, each weight to the fp16 of its stored value, nothing else but the fp32 sum): Q8_0 and bf16 weights, both tiles, unsplit and split-K two to four ways, tails on both axes, a wide output stride, accumulation into a residual | a device |
| `llmcuda-engine/tests/prefix_pass.rs` | A prompt piece run as the prefix of a wider pass is the pass of its own width **to the bit** — final hidden states and last-row logits — cold and continued through a carried state, at and above the tensor-core tile; below it, the documented cross-kernel difference only | a dense file, a device |

The differential is the interesting one, because the dense block runs on the
MoE crate's shared-expert kernels and so what is genuinely new is the *block*:
that the post-mixer RMSNorm feeds the MLP, that the MLP has no gate of its
own, and that the residual added back is the block's input rather than its
normalized form. Each of those three has a wrong version that still generates
fluent text.

It also gates all four kernel paths against one reference, in one process
(`DenseFfnBlock::with_prefill_path` picks the prefill one), which is what says
they are the same computation rather than four:

| Path | `max_tokens` | `max_abs` | cosine |
| --- | ---: | ---: | ---: |
| Q8_0, fp32 `moe_shared_ffn` | 16 | 3.20e-4 | 1.000000 |
| split int8, fp16 `HgemmKernels` — prefill, the default | 128 | 4.62e-3 | 1.000000 |
| split int8, `shared_expert_mma` — prefill, `LLMCUDA_HALF_GEMM=0` | 128 | 6.45e-2 | 0.999997 |
| split int8, `dense_proj_split_t3` — decode | 3 | 4.18e-3 | 1.000000 |

On a tensor whose own max magnitude is 82.3. The int8 GEMM is 14x looser than
the rest because it is the only path that quantizes *activations* to int8,
against a 32-value absmax; the fp16 GEMM and the decode GEMV both round them
to fp16 instead, and land within 10% of each other.

`max_rel_error` is excluded from both gates for the floor reason the routed
MoE differential records, and the test asserts the element driving it really
is small against the tensor's own scale — so the exclusion fails loudly if it
stops being justified.

### End to end, against llama.cpp

The `forward_pass` golden is a Qwen3.6 capture and there is no dense
equivalent, so the end-to-end check is greedy agreement with
`llama-completion` at `--temp 0 --top-k 1` on the same file: three prompts,
160 tokens each, both files. Three of the six runs match character for
character. Each of the other three diverges at a token where llama.cpp's own
top two logits are 0.0004–0.387 apart, against 0.13–0.43 max-abs between the
engines' logits on the same ids. These are near-ties flipped, not trajectories
drifting, and one of them llama.cpp breaks both ways itself. The prompts,
positions and logits are in [BENCHMARKS.md](BENCHMARKS.md) ("Greedy agreement
with llama.cpp").

A divergence is not, by itself, evidence about accuracy. Where they split,
our int8 prefill path tends to land on llama.cpp's token, because both round
activations to int8 the same way. The fp16 path is the one with 14× less error
against the CPU reference.

## The decision model (`clef`)

Clef-Flash is a `qwen35` trunk under a decision head, so the trunk is gated by
everything above. The head, and the path that feeds it, are gated by these:

| Test | What it asserts | Needs |
| --- | --- | --- |
| `llmcuda-model/tests/gguf_config.rs` (`clef_*`) | The `clef` architecture reads as a dense trunk plus its decision-head geometry; a file missing any of the head's five keys, or naming a decision type other than the one the head implements, is refused | nothing |
| `llmcuda-kernels` `decision` unit tests | The reference's GELU is the exact erf form, not the tanh approximation, and its `erf` matches known values | nothing |
| `llmcuda-engine/tests/decision_differential.rs` | The device head against the f64 scalar oracle (`llmcuda_kernels::decision::joint_schema_head`): a synthetic head written into an in-memory GGUF in mixed storage formats, scored on several prompt shapes and through every LM-head format the lexical gather reads, gated at 8x the worst observed error; requests outside the limits refused before anything is launched; and, behind `LLMCUDA_CLEF_MODEL`, the real head on random normed hidden states | a device; the file for the real-head case, which SKIPS without the variable |
| `llmcuda-engine/tests/prefix_pass.rs` | The row above in the dense section. Its default model is Clef, because a decision prompt is the only thing the runtime runs as a prefix of a wider pass | a dense file, a device |
| `llmcuda-server` `http::systemone` tests | `images` must be `data:` URLs; `videos`, and `media_kwargs` beside images, are refused. Behind `LLMCUDA_CLEF_MODEL`, on the real vocabulary: the media span is `encode_record`'s (one marker per image, each pad expanded, then a newline), it sits between the prefix and the state, every question and option span moves by its length, and a short limit truncates the state but never the images | nothing; the file for the layout case, which SKIPS without the variable |

### End to end, against the reference and llama.cpp

There is no captured golden for Clef. The end-to-end check is the served
answers against Cloudflare's PyTorch reference (`joint_schema_model.py`) on
the same requests, with llama-server beside it as the other GGUF
implementation. Over the 24 requests of the benchmark sets (99 questions):

- the prompt token count matches the reference tokenizer's on every request
  (the prompt is the GGUF's `systemone` template, tokenized piece by piece,
  as llama.cpp's server builds it);
- both servers pick the reference's option on every question;
- our option probabilities are nearer the reference's than llama-server's on
  median, p99 and max |Δp|, against both its fp16 and its bf16 runs.

The figures are in [BENCHMARKS.md](BENCHMARKS.md#clef-flash-clef). The
servers report probabilities to four decimals, so differences under 5e-5
are rounding, not a measurement.

### Images, against the reference

Same method, with images: Cloudflare's `systemone` in fp16 on one RTX 8000,
its processor built from `Qwen3VLProcessor` and the slow PIL
`Qwen2VLImageProcessor` (`AutoProcessor` demands torchvision for the video
processor). One request, four questions and eleven reported probabilities,
over rendered invoice images whose printed status is PAID or OVERDUE, served
from `Clef-Flash-Q8_0.gguf` with the Q8_0 mmproj. Token counts match on
every request at a sufficient `--image-max-tokens`. Max |Δp| against the
reference:

| Images | max \|Δp\| | Same choices |
| --- | --- | --- |
| none (the backbone's own gap) | 0.012 | yes |
| one, 512×384: no resize on either side | 0.001, 0.005 | yes |
| two, 512×384, in both orders and repeated | 0.014, 0.035, 0.002 | yes |
| one, 500×300: stretched to 512×288 | 0.002, 0.004 | yes |
| 512×384, then 500×300 with the other status | 0.022 | yes |
| one, 240×180: grown to the 64-token floor (70 tokens) | 0.002 | yes |
| one, 1600×1200 (1,875 tokens) at `--image-max-tokens 2048` | 0.006 | yes |

Without an image the model is near even on the printed status; with one it
puts 0.99 on the right one, as the reference does. At the default
`--image-max-tokens 1024` the 1600×1200 image is shrunk to fit — 1,382
prompt tokens against the reference's 2,310 — and still reads 0.004 here,
but that is this image, not a bound. Before `/v1/systemone` used the
processor's resize, the fifth row read 0.23 with the choice flipped: a
prompt whose images contradict each other sits where the resize moves the
answer most. The reference's processor here is transformers' PIL backend,
because torchvision is absent; its default torchvision backend calls the
PyTorch kernel ported here (torchvision itself was not available to confirm
it takes the uint8 path). PIL and that kernel differ by at most 2 levels on
0.35% of values over random images, 1.3% at worst on noise.

## Serving acceptance status

Stated plainly, because a gap you know about is manageable and one you assume
away is not.

What has passed on the three local RTX 8000s: `worker_smoke`, `engine_smoke`
and `cross_worker_restore`. The engine run serves nine sequences, balances
three onto each worker, and exercises a mixed two-decode/one-prefill step on
every card; the cross-worker restore emits exactly the same token ids as the
cold path. Nine simultaneous HTTP `/v1/completions` requests are served by one
server bound to all three cards, every response producing the same
continuation.

The HTTP surface itself is checked by hand against a live server, because the
engine below it needs a GPU and a 30 GiB model and so cannot be stood up in a
`cargo test`. What that check has covered, on one RTX 8000:

- All four generation endpoints, streaming and not, plus `/v1/models` and
  `/v1/messages/count_tokens`.
- API-key authentication: accepted via both header spellings, refused with
  each dialect's own `401` envelope, and open when no key is set.
- Every refusal in [API.md](API.md): `n > 1`, a forcing `tool_choice`, a
  late system message, an image content part, `previous_response_id`,
  malformed JSON.
- Sampling: `temperature: 0` still greedy and exact; equal seeds replaying
  byte-identical completions on chat and raw completions; different seeds
  diverging; out-of-range `temperature` refused.
- Tool calling in all three chat dialects, streaming and not: calls parsed
  with schema-typed arguments, `tool_calls` / `tool_use` / `function_call`
  wire shapes, and tool-result round-trips answered from the result.
- Three concurrent streams on one worker, each returning its own correct
  answer.
- Disconnect cancellation: a prompt that runs 42 s to `max_tokens` leaves the
  card idle within 3 s of the client vanishing.
- A 4000-token generation, which crosses a GDN retention boundary — the case
  that used to fail the whole scheduler step.
- **Prefix reuse through generated tokens, against a cold control.** A
  14-token prompt generated 2,600 tokens; a second request whose prompt was
  that prompt plus that output reused 2,048 tokens, none of which the first
  request's prompt could have named. The resumed continuation was compared
  against the identical request on a freshly started server that had never
  seen the text — byte-identical, which is the check that matters, because a
  chain naming the wrong prefix produces fluent output rather than an error.
  Four interleaved pairs on unique prompts put the saving at 2.1× on time to
  first token for a 2,621-token prompt, and every pair's resumed answer
  matched its cold arm.

Everything below the wire format is covered by the unit tests in
`crates/llmcuda-server/src/http/`: the ChatML rendering is pinned string by
string, and stop-sequence hold-back, header parsing, and constant-time key
comparison have their own tests.

What none of that covers: overload behaviour, and any of it under sustained
load rather than by hand.

The scheduler-driven path measures within 0.7% of the isolated kernel path, so
runtime plumbing is not where throughput goes. Serving numbers belong in
`BENCHMARKS.md`, not here.

## Running

```sh
cargo test --workspace --release -- --test-threads=1 # serialize GPU allocations
cargo test --release -p llmcuda-kernels                 # references and harness
cargo test --release -p llmcuda-kernels gdn             # the critical path
cargo run -p llmcuda-cuda --bin probe    # device gate and milestone-00 spike
```

### Live llama.cpp parity

Use the same GGUF in both servers. The golden checks one short cold prompt;
`tools/serving/check_correctness.py` adds live known-answer checks, identical
raw prompts, Unicode, a long-context lookup, cache reuse in mixed concurrent
batches, and tool calls plus result replay in all three chat dialects. It also
checks each dialect's streaming tool response. Results go outside the repository:

```sh
python tools/serving/check_correctness.py \
  --candidate http://127.0.0.1:18080 --reference http://127.0.0.1:18081 \
  --output /tmp/llmcuda-quality.json
```

Start the candidate fresh for the first long prompt to be a cold control. The
harness asserts known answers and cache identity, and reports free-form greedy
text agreement separately. Different reduction/activation arithmetic can flip
near ties; equal seeds also do not imply identical sampled text because the RNGs
differ. This is an acceptance check, not a performance benchmark or proof of
equivalence on arbitrary prompts, models, or context lengths. It does not test
vision or execute real tools.

For another checkpoint such as Ornith 1.5, set `LLMCUDA_MODEL` for the live
template test and capture a separate numerical golden from that exact GGUF
using [the oracle procedure](ORACLE.md). Never pair Ornith weights with the
Qwen3.6 golden. Matching architecture and vocabulary do not establish equal
templates or numerical behavior; the Qwen3.6 thresholds must not be widened
just to clear another checkpoint. To localize an attention-layer failure,
extend the capture filter's `(3|39)` group with that layer. `attention_block`
keeps blocks 3 and 39 mandatory and checks every additional captured attention
block with the same gates, reporting numerical failures together so a
reference discrepancy does not hide a device-versus-CPU discrepancy.

Prefill partitioning is another source of numerical variation: a scheduler
grant can end inside a GDN chunk, so concurrent admission or a different token
budget can change a later greedy choice. This also occurs in the original
engine, independently of the prompt renderer and host sampler. Exact batched
decode agreement from an identical initial state does not establish identical
states after different prefill partitions. Keep the prompt and partitioning
fixed for an exact comparison; investigate changed continuations with numerical
oracles rather than assuming either corruption or harmlessness from the prose.

The prompt/tokenization oracle catches shared mistakes that comparing our two
renderers cannot. Start llama-server with `--jinja --no-prefill-assistant` and
the same model, then run (requires `curl`):

```sh
LLMCUDA_LLAMA_URL=http://127.0.0.1:18081 \
  cargo test --release -p llmcuda-server \
  the_model_template_matches_llama_cpp_when_requested -- --nocapture
```

It compares rendered bytes and token IDs for both thinking modes, tool loops,
parallel calls, omitted tool fields, and typed/Unicode arguments. Minijinja's
HTML-oriented JSON filter and scalar string spelling differ from minja's;
prompt filters must match the latter, including JSON separator spaces and
Python `True`/`None` in replayed arguments. `--no-prefill-assistant` aligns the
reference with this server's next-turn behavior: llama.cpp otherwise continues
a final assistant message automatically, while this server appends a new turn.

### A green run is not the same as a run

Three ways this suite reports success without having tested anything, all of
them found the hard way. They compound: a run can hit all three at once and
look identical to a clean one.

**A skipped test passes.** Tests that need a device, a model or a capture
print `SKIPPED: ...` and return `ok` — which is the right behaviour, since a
checkout without 30 GB of weights should not fail — but cargo captures stdout
for passing tests, so the message is invisible unless you pass `--nocapture`.
`forward_pass` reports **8 passed in 0.36 s** when it cannot find the golden,
and 0.36 s is the only visible sign that the workspace's most important gate
did not execute. **From a `git worktree` it never finds it**: `.golden/` is
in `.gitignore`, so the capture does not come along. Point it at the one in
the main checkout:

```sh
LLMCUDA_GOLDEN=/home/nixabe/llmcuda-rs/.golden/qwen36-golden.bin \
  cargo test --release -p llmcuda-engine --test forward_pass -- --nocapture
```

The real thing takes ~21 s and prints its cosine against llama.cpp's captured
logits. If it finished instantly, it did not run.

**A piped exit status is the pipe's.** `cargo test ... | grep ...` reports
grep's status, not cargo's, so a run killed by a signal or the OOM killer —
three workers and a 30 GB model make that ordinary here — exits zero as long
as the filter matched something earlier. Write the raw output to a file, take
cargo's own status as the verdict, and keep the filter as the summary you
read rather than the thing you believe.

**A truncated run looks like a short one.** Count what reported against what
exists: `cargo test --workspace --release --no-run` prints one `Executable`
line per test binary — 62 of them at the time of writing, against 36
`tests/*.rs` files, the rest being lib, bin and doc targets, and a full
`--workspace --release` run reports 70 `test result:` lines once the eight
doc-test targets are counted. Every one of those numbers grows when a test
file lands, so **count them rather than quoting these**; the previous pair had
drifted by two before anyone checked, and the sentence you are reading was
written saying 69 and corrected by the very run that verified it. A sweep that stops halfway is otherwise
indistinguishable from one that passed, because every target it did reach was
green.

### The widths a bench produces are not the widths that exist

Anything that picks a kernel, a tile or a launch config from the token count
has to be gated on `serving_speculative_identity` before anything else. The
benchmarks produce N=1, N=3 and prefill chunk widths; the scheduler produces
those **and a two-token verify step**, and width 2 is the only shape in this
workspace that no benchmark reaches.

The router's expert tile was swept at N=3, confirmed under interleaved pairs
at N=3, and checked for bit-identity at N=1 and N=3. It sized a launch for one
kernel's dynamic shared memory and dispatched a different kernel at exactly
`max_tokens == 2`, and it shipped. Nothing had asked which widths *exist*:

```sh
CUDA_VISIBLE_DEVICES=1 \
  cargo test --release -p llmcuda-engine --test serving_speculative_identity
```

About 180 s. It failed with `CUDA_ERROR_ILLEGAL_ADDRESS`, and no bench, no
differential and no bit-pattern diff did — because every one of them was
scoped by what the change *was* rather than by which entry points it landed in
and which shapes the scheduler puts through them.

**Choose a differential's widths from the launch's structure, not from the
widths that seem interesting.** The same afternoon, from the other direction:
the fused GDN gate kernel's differential covered tokens 1 and 8, which are one
block on the untiled path and exactly one full tile on the tiled one. Neither
covered a multi-block launch and neither covered a partial tail, so the two
cases that looked like the extremes were both interior. Widened to
`[1, 2, 7, 8, 9, 17]` — one, two and seven blocks untiled; one tile, a tile
plus a one-row tail, and two tiles plus a tail — and 2 and 9 were the ones
missing. The kernel was correct; the coverage was thin in a way its case list
concealed. The general form: enumerate one, several, exactly-a-tile,
tile-plus-tail, and several-tiles-plus-tail, because those are the shapes a
grid computation can get wrong.

Serving acceptance uses the scheduler-driven runtime rather than the isolated
forward benchmarks:

```sh
# One visible card, N=1 or N=3 through Worker::step_device.
CUDA_VISIBLE_DEVICES=0 LLMCUDA_BATCH_N=3 \
  cargo run --release -p llmcuda-engine --bin worker_smoke

# Three visible cards, nine requests, including a 2-decode + 1-prefill step
# on every worker.
CUDA_VISIBLE_DEVICES=0,1,2 \
  cargo run --release -p llmcuda-engine --bin engine_smoke

# Two separate CUDA contexts: cold 2K prefill versus pinned-host KV/GDN
# restore, compared by exact emitted token ids.
CUDA_VISIBLE_DEVICES=0,1 \
  cargo run --release -p llmcuda-engine --bin cross_worker_restore
```

These commands are acceptance checks, not benchmarks. Run the interleaved
benchmark commands from `BENCHMARKS.md` before making a throughput claim, and
follow its measurement discipline — a single pair on this host proves nothing.

Tests needing the 32 GB model file or a GPU **skip and say so**. A skipped test
is not a passing test — do not read a green run on a GPU-less machine as
validation of device work. See
[CONTRIBUTING.md](../CONTRIBUTING.md#reporting-results).

The corollary bites on a machine where the cards *are* free: `cargo test
--workspace --release` runs each integration binary concurrently, and the
device ones each load the whole model, so `batch_decode` and `batch_prefill`
together exhaust a 48 GiB card and fail with `CUDA_ERROR_OUT_OF_MEMORY`. That
is contention, not a regression — the same tests pass serialized:

```sh
CUDA_VISIBLE_DEVICES=0 cargo test --release -p llmcuda-engine --test batch_decode  -- --test-threads=1
CUDA_VISIBLE_DEVICES=0 cargo test --release -p llmcuda-engine --test batch_prefill -- --test-threads=1
cargo test --workspace --release --lib --bins
```

Read an OOM in a whole-workspace run as "run them one at a time", and check
before concluding a device kernel broke.

## K2 projection and routing gates

`k2_differential` checks grouped norms, sigmoid routing, selected values and
softplus gates against `llmcuda-kernels::k2`. Q8_0 controls accompany Q4_K
and Q6_K projection cases. Dense four- and eight-token tiles and expert-grouped
value projections must also match the scalar GPU contraction **bit for bit**,
including ragged token tiles, inactive experts and padded runs. Router cases
include ties, 512 experts, saturated sigmoid scores and the normalization floor.

The K2 integer projection gate separately compares eight-, twelve- and
sixteen-bit activation references, including Q4 affine correction. It requires
identical bits across one/three-token GEMV and eight/eleven/32/65-token GEMM,
normal and flat routed inputs, dense matrices, inactive experts and a 135-row
tail that crosses both row-tile sizes. A separate 2,560-wide dense case checks
both Q4 and Q6 specialized address paths against the CPU and exact GEMV/GEMM results. The
routed cases cross both 32-token subtiles of each fixed 64-slot dispatch run.
Explicit activation reuse must preserve
those results, and stale generations must be rejected. Rotary reuse is checked
against the CPU and direct GPU kernel, including tails, independent sequence
positions up to 524,280, coefficient refresh and f16 cache writes. Production
uses sixteen bits. K2 attention has CPU gates for compensated prefill and
decode; batched scalar and compensated tensor-core attention must match
serial calls exactly over independent windows up to 32K, at both tensor-core
warp counts and with one, three and eight independent cache slots.

The mixed Q4_K/Q6_K FFN cases in `moe_quant_formats_differential` check both
format orders against the scalar reference. Dispatch pads the end of each run;
the community kernels may skip an empty tail tile, but must retain every live
row and its original contribution order.

For both real K2 quants, `k2_model` with `LLMCUDA_K2_MODEL` and
`LLMCUDA_K2_GOLDEN` checks all 48 block boundaries, final grouped normalization,
logits and argmax against the publisher's capture. It also checks chunking,
batched prefill and decode, and graph replay, including a repeated prefix
beyond the tiny-context attention path and a full-width versus chunked
prefill comparison that exercises the tensor-core query tiles. A 512-token
prefix checks the production tensor-core decode selection and batched graph
replay against serial decode. The predicted
packed-weight residency must equal the runtime weight report. See
[MODEL.md](MODEL.md#k2-horizon) for capture and invocation instructions. These publisher thresholds were
calibrated on that six-token prompt; they are not a broader quality evaluation.
The six-token oracle's N=3 prefill exercises the tiled and grouped kernels at
eighteen physical rows.

`bench_k2` measures actual GGUF query and value-expert tensors using CUDA events,
including device dispatch in the grouped timing. Run it with `LLMCUDA_MODEL`
and optional `LLMCUDA_K2_N=1,3,128,512`. `LLMCUDA_K2_INTEGER_ONLY=1` skips
the scalar controls; use the same setting in both alternating binaries.
`LLMCUDA_K2_TENSORS` selects comma-separated tensor names; expert measurements
use deterministic top-4 routes. Its repeated reads have a different cache
state from a full pass; accept changes only after alternating
`bench_forward` and `bench_decode_batch` runs confirm them in the model.

`bench_k2_attention` compares scalar batch, serial tensor-core and batched
tensor-core decode at 512, 2K, 8K and 32K, using independent caches and
CUDA events. Each case alternates the paths for three rounds.
`LLMCUDA_DEC_MMA_SPLITS` overrides the fixed tensor-core split count.
