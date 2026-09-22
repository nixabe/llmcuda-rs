# Inference optimization campaign

Status: the original baseline, profiling, and capture-tool validation are
complete. No inference optimization has been accepted or rejected in this
campaign yet.

## Scope and provenance

- Original inference implementation:
  `d398f1940e1a9adc270661201465349fa4857545`.
- Target: Qwen3.6-35B-A3B, `UD-Q6_K_XL`, three concurrent sequences.
- Measurements use GPU 1, a Quadro RTX 8000, with other compute processes
  stopped. Benchmarks, builds, and correctness runs are serialized.
- Device UUID: `GPU-431c478e-c355-bfd5-5c82-43d12887b125`. GPU boost
  clocks remain automatic; no clock or power policy is changed. Each process
  records its starting temperature, clocks, memory usage, and executable hash.
- CUDA toolkit: 12.4; driver: 595.84; Nsight Systems: 2023.4.4.
- Default CUDA C++ kernels; the optional `rust-kernels` feature is disabled.
- Original benchmark executables were rebuilt through Cargo and copied with
  their SHA-256 hashes before any inference changes. Local raw logs and
  traces live under the ignored `bench-results/campaign-d398f19/` directory.

The documentation cleanup is commit `0e92f0c`. It changes descriptions,
example messages, test names, and repository hygiene rules; it changes no
inference arithmetic or correctness threshold.

The current comparison with llama.cpp remains in
[BENCHMARKS.md](BENCHMARKS.md#current-standing). This campaign compares
candidates with the original engine implementation; those comparisons do not
replace a same-hour run against llama.cpp.

## Measurement protocol

The original prefill ladder uses N=3, one discarded warmup, and one timed
repetition per process and depth, repeated in three separate rounds. The
512-token point uses 1,536 physical rows; the longer points use 6,138 physical
rows, or 2,046 tokens per sequence. Their exact lengths are 2,046, 8,184,
32,736, 65,472, and 130,944. Report the three individual rates and their sample
standard deviation, not a standard deviation over averages.

Decode uses `bench_decode_batch`, 32 timed steps after four warmup steps,
at 2,048 and 32,768 tokens for N=1 and N=3, and 122,880 tokens for N=3 if it
fits. Each configuration runs in three separate rounds. The harness's
`single_stream` row is kept separately from its `batch 1` row.

The commands for one original round, using the preserved executables, are:

```sh
export CUDA_VISIBLE_DEVICES=1
export LLMXABE_MODEL=/home/nixabe/llmxabe/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf
LLMXABE_PREFILL_SEQUENCES=3 LLMXABE_BENCH_CHUNK=1536 \
  LLMXABE_BENCH_N=512 LLMXABE_BENCH_REPS=1 \
  bench-results/campaign-d398f19/original/bench_forward
LLMXABE_PREFILL_SEQUENCES=3 LLMXABE_BENCH_CHUNK=6138 \
  LLMXABE_BENCH_N=2046,8184,32736,65472,130944 LLMXABE_BENCH_REPS=1 \
  bench-results/campaign-d398f19/original/bench_forward
LLMXABE_BATCH_N=1,3 bench-results/campaign-d398f19/original/bench_decode_batch 2048 32
LLMXABE_BATCH_N=1,3 bench-results/campaign-d398f19/original/bench_decode_batch 32768 32
LLMXABE_BATCH_N=3 bench-results/campaign-d398f19/original/bench_decode_batch 122880 32
```

For candidates, use at least three alternating pairs with a reversal:
`A B, B A, A B`. Run correctness checks outside the timing window. Narrow
kernel timings use CUDA events. Whole-model harnesses measure completed
prefill or decode work, including their documented synchronization and
sampling costs.

Nsight traces are for attribution, not throughput comparisons. Decode traces
must enable `--cuda-graph-trace=node`. Exclude model initialization, prefill
setup, and discarded warmups from a timed-decode breakdown. Fused kernels
must be reported as combined operations unless a separate measurement can
identify their contributions. Kernel-duration sums can exceed elapsed time
when streams overlap; distinguish the sum from the union of GPU intervals.
The layer attribution follows the 40 ordered MoE-combine boundaries in each
pass. A projection entry point alone does not identify its owner: attention
projections reuse LM-head kernels, and tensor-core projections are shared
between GDN, attention, and the shared expert. Check the expected layer and
embedding counts before accepting a trace summary.

For bounded capture, set `LLMXABE_PROFILE_TIMED=1` and pass
`-c cudaProfilerApi --capture-range-end=repeat` to Nsight. Decode brackets
the timed step loop. Prefill brackets each timed chunk and synchronizes its
stream before ending the capture; discarded warmups remain untraced.
The capture calls can take seconds on this host, so their displayed rates
must not enter the baseline or any A/B comparison. Sum each chunk's
category intervals to reconstruct a prompt's execution breakdown; separate
capture clocks are not a single continuous timeline.

## Original baseline

All rates below are aggregate tokens/s from GPU 1 and the original
implementation. Each row's three observations come from separate process
runs. Sample SD is computed over those three rates.

### Prefill, N=3

| Tokens per sequence | Run 1 | Run 2 | Run 3 | Mean | Sample SD |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 3252.48 | 3177.32 | 3171.02 | 3200.27 | 45.32 |
| 2,046 | 3662.11 | 3613.35 | 3611.58 | 3629.01 | 28.68 |
| 8,184 | 3389.45 | 3355.97 | 3351.82 | 3365.75 | 20.63 |
| 32,736 | 2649.44 | 2634.23 | 2633.07 | 2638.91 | 9.13 |
| 65,472 | 2072.63 | 2071.86 | 2072.36 | 2072.28 | 0.39 |
| 130,944 | 1454.36 | 1454.55 | 1454.47 | 1454.46 | 0.10 |

The 512-token process reported 32.229 GiB peak VRAM. The longer ladder
reported 42.104 GiB with all three states allocated for 130,944 positions.
The shorter points varied more across rounds than the deep points. A low
SD at 128K does not establish the noise floor for a short decode benchmark.

### Decode

| Context | Shape | Run 1 | Run 2 | Run 3 | Mean | Sample SD |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2,048 | single stream | 113.6 | 113.7 | 113.6 | 113.63 | 0.06 |
| 2,048 | batch 1 | 114.3 | 114.5 | 114.3 | 114.37 | 0.12 |
| 2,048 | batch 3 | 215.9 | 216.0 | 214.6 | 215.50 | 0.78 |
| 32,768 | single stream | 97.0 | 97.6 | 97.0 | 97.20 | 0.35 |
| 32,768 | batch 1 | 97.4 | 97.4 | 97.6 | 97.47 | 0.12 |
| 32,768 | batch 3 | 166.5 | 167.1 | 167.0 | 166.87 | 0.32 |
| 122,880 | single stream | 65.9 | 66.0 | 65.9 | 65.93 | 0.06 |
| 122,880 | batch 3 | 101.2 | 101.7 | 101.5 | 101.47 | 0.25 |

Peak reported N=3 VRAM is 31.805, 33.555, and 38.711 GiB at those three
depths. Decode rates are printed to one decimal place by the existing
harness; the extra digits in the mean and SD describe that recorded data,
not additional timing precision.

## Original correctness checks

Before changing inference code, the following completed on GPU 1:

```sh
cargo fmt --all
cargo clippy --workspace --all-targets
CUDA_VISIBLE_DEVICES=1 \
  LLMXABE_GOLDEN=/home/nixabe/llmxabe/.golden/qwen36-golden.bin \
  cargo test --workspace --release -- --test-threads=1 --nocapture
```

Cargo reported 87 completed test targets and 974 successful test returns.
The log contains 12 explicit skip messages; a returned `ok` for one of those
is not numerical validation. Skips cover the opt-in prefill/twin audits,
the absent dense-model and DFlash fixtures, external vision goldens, and live
llama.cpp template parity. The installed Qwen3.6 forward golden, attention,
GDN, MoE, LM-head, batch prefill/decode, and available speculative identity
checks ran. The forward golden covered 19 tokens and 40 blocks and matched
the captured argmax.

## Original execution profile

The N=3 decode profile covers 32 timed steps at each depth, after four
discarded warmup steps. Each step contains **659 kernel launches**. The
2K trace was filtered after collection; the 32K trace uses the opt-in
`LLMXABE_PROFILE_TIMED` capture boundary. Nsight failed to import the
original long 32K trace with an event-order error, so the benchmark tools
now support bounded captures through `-c cudaProfilerApi`. No inference
kernel was changed for this instrumentation.

The table reports the union of each category's GPU intervals, in
milliseconds per step. Three attention streams overlap; adding their
individual kernel durations would overstate attention's elapsed cost.
The category unions sum to the overall active GPU interval here.

| Component | 2K | 32K |
| --- | ---: | ---: |
| GDN projections, including fused residual add | 2.699 | 2.706 |
| GDN recurrence and Q/K normalization | 0.723 | 0.725 |
| GDN alpha/beta gates | 0.181 | 0.181 |
| Attention projections | 0.831 | 0.814 |
| Attention and split-result combine | 0.751 | 4.663 |
| MoE router projection | 0.355 | 0.355 |
| MoE route and dispatch | 0.365 | 0.366 |
| Routed and shared experts, fused | 5.656 | 5.673 |
| MoE gate, combine, and residual | 0.164 | 0.164 |
| Standalone RMSNorm | 0.313 | 0.313 |
| Fused RMSNorm/SwiGLU | 0.089 | 0.089 |
| Other elementwise, convolution, rotary, and append | 0.307 | 0.306 |
| LM head | 0.958 | 0.961 |
| Greedy sampling | 0.020 | 0.020 |
| Embedding | 0.005 | 0.005 |
| **Active GPU interval** | **13.416** | **17.342** |
| GPU span, including gaps | 14.019 | 17.952 |
| Sum of individual kernel durations | 14.868 | 26.622 |

The routed and shared expert bodies occupy disjoint block ranges within the
same decode launches. Their actual costs are jointly measured; a separate
time for each is not identifiable from those launches. Prefill uses separate
expert kernels and can distinguish them.

CPU `cuGraphLaunch` calls average 0.578 ms at 2K and 0.676 ms at 32K in
these traces. They overlap device execution. The respective GPU gaps are
0.603 and 0.610 ms per step and include copies, scheduling, and launch
effects; neither quantity is a pure graph-overhead measurement. The large
CPU duration of `cuMemcpyDtoHAsync_v2` includes waiting for completed greedy
sampling, rather than time spent transferring a few token IDs.

The target model's alpha/beta gates average **6.03 us per call**, or
0.181 ms for its 30 layers. That is only 1.35% of active GPU time at 2K.
The gate experiment therefore needs a quick measurement gate; the older
approximately 1 ms estimate is not the current target's measured cost.

### Small decode launches

The 2K trace identifies these repeated kernels below 10 us. Rows are sorted
by total duration per step; this is kernel time, not a promise that combining
launches saves that entire time. Geometry and dependency points must match.

| Operation | Calls/step | Mean us/call | Total ms/step |
| --- | ---: | ---: | ---: |
| MoE route/dispatch | 40 | 9.129 | 0.365 |
| MoE router | 40 | 8.887 | 0.355 |
| RMSNorm | 101 | 3.095 | 0.313 |
| GDN alpha/beta | 30 | 6.027 | 0.181 |
| GDN convolution/SiLU/split | 30 | 5.893 | 0.177 |
| MoE combine | 40 | 4.090 | 0.164 |
| RMSNorm/SwiGLU | 30 | 2.963 | 0.089 |
| GDN Q/K normalization | 30 | 2.473 | 0.074 |
| Rotary/append | 10 | 6.821 | 0.068 |
| Attention sigmoid gate | 10 | 2.491 | 0.025 |
| Residual add | 10 | 1.861 | 0.019 |
| Query/gate split | 10 | 1.811 | 0.018 |

The gate and convolution kernels both use 128 threads per block and no
shared memory. They consume already-produced inputs and write disjoint
outputs, making a combined grid a candidate for the later launch experiment.

### Prefill breakdown, N=3

Each column covers a complete timed prompt batch, assembled from 1, 4, 16,
or 64 bounded chunk captures. All 85 captures contain one embedding launch
and 40 MoE-combine boundaries. Each chunk has 1,390 kernel launches.
Values are **seconds of active GPU intervals per category**, summed across
chunks. There is small overlap between rotary/append work and attention
on the other streams, so category rows are not strictly additive.

| Component | 2,046 | 8,184 | 32,736 | 130,944 |
| --- | ---: | ---: | ---: | ---: |
| Attention | 0.042 | 0.622 | 9.700 | 157.189 |
| Routed experts | 0.564 | 2.311 | 9.421 | 37.863 |
| GDN projections | 0.273 | 1.127 | 4.612 | 18.555 |
| GDN recurrence and Q/K normalization | 0.238 | 0.982 | 4.008 | 16.159 |
| Other elementwise, convolution, rotary, and preparation | 0.117 | 0.473 | 1.909 | 7.648 |
| MoE router | 0.094 | 0.386 | 1.577 | 6.352 |
| Attention projections | 0.075 | 0.307 | 1.250 | 5.017 |
| MoE gate/combine/residual | 0.077 | 0.307 | 1.226 | 4.906 |
| Shared expert | 0.050 | 0.204 | 0.833 | 3.344 |
| Standalone RMSNorm | 0.022 | 0.091 | 0.369 | 1.483 |
| MoE route/dispatch | 0.021 | 0.085 | 0.348 | 1.399 |
| Fused RMSNorm/SwiGLU | 0.022 | 0.087 | 0.349 | 1.395 |
| GDN alpha/beta gates | 0.017 | 0.070 | 0.284 | 1.142 |
| LM head | 0.001 | 0.004 | 0.016 | 0.064 |
| Embedding | <0.001 | 0.001 | 0.002 | 0.009 |
| Greedy sampling | <0.001 | <0.001 | <0.001 | 0.001 |
| **Active GPU interval** | **1.612** | **7.055** | **35.900** | **262.512** |
| Sum of chunk GPU spans, including gaps | 1.643 | 7.182 | 36.409 | 264.549 |

Attention occupies 2.6%, 8.8%, 27.0%, and **59.9%** of active GPU time
across the ladder. This confirms the deep-attention priority. The gap
between the sum of chunk spans and active GPU intervals is 31.7 ms at 2K
and 2.04 s at 128K; it includes device scheduling and launch effects but
excludes the profiler's pauses between captures.

### Event calibrations

`profile_forward` independently bracketed the original single-sequence
cold forward at one and 2,048 tokens, with two warmups and three timed
passes. At 2,048 tokens its GPU spans total 636.75 ms: GDN mixers 235.78 ms,
attention mixers 56.63 ms, all MoE stages 343.27 ms, and LM head 0.91 ms.
Event instrumentation adds 1.33 ms against the preceding uninstrumented
635.70 ms mean; the following uninstrumented mean is 636.76 ms. These are
cold single-sequence stage measurements, not the N=3 serving baseline.

The original `bench_attention` event sweep uses three concurrent queries
and separate KV caches. These are single calibration runs, not an A/B:

| Existing keys | Prefill, 2,046 new tokens/sequence (ms) | Decode, one token/sequence (ms) |
| ---: | ---: | ---: |
| 0 | 3.383 | 0.036 |
| 2,048 | 10.315 | 0.086 |
| 8,192 | 30.427 | 0.282 |
| 32,768 | 113.314 | 0.481 |
| 65,536 | 234.931 | 0.952 |
| 98,304 | 352.164 | 1.420 |
| 131,072 | 469.133 | 1.874 |

## Initial resource inspection

CUDA 12.4 `nvcc -arch=sm_75 -cubin -Xptxas=-v` on the extracted, unchanged
attention source reports 252 registers and zero spill loads, spill stores,
or stack bytes for `attn_flash_causal_mma`. Its dynamic shared allocation at
head dimension 256 is 13,952 bytes. With 256 threads and Turing's register
allocation granularity, this permits one block per SM, or eight resident
warps out of 32. This is a resource bound, not a throughput measurement.
The actual NVRTC-to-driver path reports **255 registers and zero local
bytes** through `cuFuncGetAttribute`; Nsight confirms 255 registers.
NVRTC itself is 12.4.127, so the distinction is between offline code
generation and the installed driver's JIT. Candidates must be inspected
through both paths rather than assuming the offline count is the runtime
allocation.

The same compilation reports 64 registers for `gdn_alpha_beta_gates_t1`
and 22 for each of `rms_norm_rows` and `rms_norm_swiglu_rows`, all without
stack allocation or spills. The decode trace reports the same register
counts. Alpha/beta uses no shared memory; standalone RMSNorm uses 128
bytes at hidden width 2,048 and 32 bytes at head width 256.

The key loop has two block barriers per eight-key tile. Its two softmax
fences are already warp-local because one warp owns each query head. Q
fragments and the output accumulator remain in registers across the key
loop; K/V staging and the score/softmax handoffs use shared memory.

The upstream checkouts named in the development guide are absent on this
host. The attention comparison uses the official llama.cpp source at
[`7ab4ee7baad2d920464cbacfad4f4b07cf111fd2`](https://github.com/ggml-org/llama.cpp/blob/7ab4ee7baad2d920464cbacfad4f4b07cf111fd2/ggml/src/ggml-cuda/fattn-mma-f16.cuh).
Its Turing path keeps Q in registers and softmax state in MMA fragments;
the asynchronous-copy pipeline is disabled when `CP_ASYNC_AVAILABLE` is
absent. Tile sizes and precision choices must be compared separately from
the ownership of intermediate values.

The capture hooks passed formatting, workspace Clippy, and a fresh GPU 1
release workspace run: 87 completed targets, 974 successful test returns,
zero failures, and the same 12 explicitly reported skips. Bounded Nsight
captures were validated against the expected chunk, layer, and kernel counts.

## Results still required

1. Investigate deep-prefill attention, exact-order GDN gates, consumer
   normalization, additional launch fusion, and one persistent fragment,
   in that order unless the profile establishes a different priority.
2. Prototype or implement safe same-slot continuation and measure multi-turn
   time to first token, transferred bytes, reuse, and conversation time.
3. Re-evaluate the available speculative drafters after the kernel work.
4. Record each experiment's individual pairs, register/shared-memory changes,
   correctness checks, and keep/reject decision.
5. Compare the final tree with the original implementation and report the
   remaining bottlenecks and justified next experiments.
