# Inference optimization campaign

Status: the original baseline, profiling, capture-tool validation, and
deep-prefill attention phase are complete. The corrected fragment-softmax
implementation is accepted. The exact-order gate experiments are complete and
rejected. Consumer-side normalization is accepted after paired model measurements
and the complete release suite.

| Experiment | Kernel throughput change | Prefill throughput change | Decode throughput change | Correct | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| Fragment softmax with implicit normalizer FMA | Not timed | Not timed | Not timed | No: forward golden | Reject arithmetic form |
| Fragment softmax with explicit original rounding | +21.69–22.77% at 128K | +12.09–12.11% at 128K, N=3 | Not separately measured | Yes; full suite and new regression | Keep |
| Cooperative gate loading | Cache dependent; N=3 resident loss | Not measured | Not measured | Bit-exact isolated probe | Reject |
| One head per gate block | Cache dependent | Not measured | −0.09–0.14% at 2K, N=3 | Yes; targeted model checks | Reject |
| Consumer RMSNorm plus GDN gates | About +21% resident / +39% rotating weights, N=3 | Not separately measured | +0.14–0.32% at 2K; +0.42–0.48% at 32K, N=3 | Yes; full suite and CPU/GPU regression | Keep |

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

## Deep-prefill attention experiment

Candidate: keep score, probability, maximum, and normalizer values in the
`m16n8k8` fragment layout. Each lane holds two keys from each of two query
rows. Reduce even and odd key subsequences separately, add their totals, and
broadcast the quad leader's sum. This reproduces the former eight-lane
XOR(4,2,1) sum at the lane that wrote the normalizer. Explicitly round the
score scaling product before the softmax subtraction, preserving the
rounding previously imposed by the shared store.

The query tile, eight-key tile, Q residency, output ownership, and K/V
prefetch depth remain unchanged. This differs from the previously rejected
decode warp-softmax experiment: it does not enlarge or duplicate the output
accumulator. The comparison with llama.cpp's `flash_attn_ext_f16_iter` and
`flash_attn_ext_f16_process_tile` informed fragment ownership; its tile
sizes, softmax grouping, and numeric choices were not copied.

| Resource | Original | Candidate |
| --- | ---: | ---: |
| Offline registers/thread | 252 | 255 |
| Runtime registers/thread | 255 | 255 |
| Offline spill stores/loads and stack bytes | 0 | 0 |
| Runtime local bytes/thread | 0 | 0 |
| Dynamic shared bytes/block | 13,952 | 8,320 |
| Threads/block | 256 | 256 |
| Register-limited blocks/SM | 1 | 1 |
| Block barriers/eight keys | 2 | 2 |

At source level, both softmax warp fences disappear with their shared
handoffs. Offline SASS has two `BAR.SYNC` sites and one `WARPSYNC` site in
both kernels; source fence removal must not be described as a measured
reduction in hardware barrier instructions. Static shared-load/store sites
fall from 144/23 to 64/3 and shuffle sites from 24 to 14. MMA sites remain
64 and no local-load/store sites appear. These counts explain the candidate;
they do not establish a performance result.

The first implementation passed all ten attention differentials, batch
decode, and batch prefill but failed the forward golden at rank 4: the
captured pair is separated by 0.153701, beyond the 0.088937 measured error
on the shared top logits. Performance evaluation stopped. Direct original
versus candidate GPU comparisons differed by up to 2.384e-7; the CPU
attention tolerances alone did not expose the model consequence.

PTX identified the arithmetic change: NVRTC moved one normalizer addition
past the following MMA control flow, leaving a separately rounded multiply
and add where the original shared-state update used FMA. The correction is
an explicit `__fmaf_rn`, preserving the original operation. It produces
bit-identical output against the original in four direct GPU comparisons:
19/32 cold queries and 19/64 queries at offsets 61/256 (548,864 elements).
Registers and local-memory usage are unchanged. The corrected version
passes all eight forward-golden tests, returning the original argmax 25358
and logit 19.950254. The corrected version also passes workspace Clippy and
the full GPU 1 release workspace suite: 87 completed targets, 974 successful
test returns, zero failures, and the same 12 explicit skips listed above.
The repeated attention, batch, GDN, MoE, LM-head, and available speculative
identity checks ran. The timing results follow below.

### Isolated attention pairs

Three N=3 queries use separate KV caches and 2,046 new tokens each. Each
process discards two warmups and measures five repetitions using CUDA
events. Values below are milliseconds; A is the preserved original and B
is the corrected candidate. Columns retain execution order, including the
reversed middle pair. Throughput change is `A_ms / B_ms - 1`.

| Existing keys | A1 | B1 | B2 | A2 | A3 | B3 | Paired throughput change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 4.481 | 3.713 | 3.713 | 4.606 | 4.608 | 3.710 | +20.68% / +24.05% / +24.20% |
| 2,048 | 12.421 | 10.432 | 8.443 | 10.845 | 12.632 | 10.281 | +19.07% / +28.45% / +22.87% |
| 8,192 | 29.957 | 24.610 | 24.771 | 30.617 | 30.827 | 25.132 | +21.73% / +23.60% / +22.66% |
| 32,768 | 111.396 | 90.917 | 91.178 | 113.548 | 113.944 | 92.686 | +22.52% / +24.53% / +22.94% |
| 65,536 | 232.395 | 190.520 | 190.773 | 236.391 | 237.269 | 193.856 | +21.98% / +23.91% / +22.39% |
| 98,304 | 346.364 | 283.970 | 285.826 | 353.563 | 355.089 | 290.371 | +21.97% / +23.70% / +22.29% |
| 131,072 | 460.313 | 378.254 | 382.427 | 469.514 | 472.848 | 386.069 | +21.69% / +22.77% / +22.48% |

The candidate wins all three pairs at every offset. At 131,072 existing
keys the throughput change is +21.69% to +22.77%. Short points have a larger
spread, particularly offset 2,048, so their isolated percentages are not a
model-level claim. The full prefill ladder was measured separately below.

### Full-model prefill pairs

Three pairs use the original protocol at every depth: N=3, one discarded
warmup and one timed pass per process and depth. A uses the original
inference kernels with the same benchmark capture hooks as B; capture is
disabled in both. The 512-token point uses 1,536 physical rows; the ladder
uses 6,138 rows with states allocated for 130,944 positions. All rates are
aggregate tokens/s. No build, test, or other GPU job ran alongside these
pairs. Columns retain the actual `A B, B A, A B` execution order.

| Tokens/sequence | A1 | B1 | B2 | A2 | A3 | B3 | Paired throughput change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 512 | 3211.210 | 3184.750 | 3187.230 | 3183.970 | 3184.170 | 3186.330 | -0.82% / +0.10% / +0.07% |
| 2,046 | 3642.280 | 3631.270 | 3641.570 | 3617.000 | 3619.730 | 3635.720 | -0.30% / +0.68% / +0.44% |
| 8,184 | 3379.600 | 3414.880 | 3416.000 | 3360.920 | 3361.280 | 3418.490 | +1.04% / +1.64% / +1.70% |
| 32,736 | 2644.980 | 2782.300 | 2781.250 | 2642.710 | 2646.250 | 2785.440 | +5.19% / +5.24% / +5.26% |
| 65,472 | 2080.270 | 2255.790 | 2256.610 | 2079.690 | 2079.530 | 2255.900 | +8.44% / +8.51% / +8.48% |
| 130,944 | 1460.730 | 1637.610 | 1636.970 | 1460.380 | 1461.130 | 1637.730 | +12.11% / +12.09% / +12.09% |

The process-level means and sample standard deviations are:

| Tokens/sequence | A mean ± sample SD | B mean ± sample SD |
| ---: | ---: | ---: |
| 512 | 3193.12 ± 15.67 | 3186.10 ± 1.26 |
| 2,046 | 3626.34 ± 13.87 | 3636.19 ± 5.17 |
| 8,184 | 3367.27 ± 10.68 | 3416.46 ± 1.85 |
| 32,736 | 2644.65 ± 1.79 | 2783.00 ± 2.18 |
| 65,472 | 2079.83 ± 0.39 | 2256.10 ± 0.45 |
| 130,944 | 1460.75 ± 0.38 | 1637.44 ± 0.41 |

The 8K, 32K, 65K, and 128K comparisons win in every pair. At 128K the
full-model gain is **12.09–12.11%**, following a **21.69–22.77%** isolated
attention gain at that offset. The 512 and 2K results change sign across
pairs and lie within the observed short-run drift; no throughput improvement
or regression is established there. Peak reported VRAM remains 32.229 GiB
for 512 and 42.104 GiB for the long ladder. Decode throughput was not
separately attributed to this prefill-only experiment.

The performance gate is satisfied. A new fixture-free regression gives all
query rows of a head an identical live prefix while forcing the running
maximum to change. It checks their output against the scalar CPU oracle and
requires bit identity across equivalent rows, including different halves
of an MMA fragment. The corrected kernel passes; temporarily replacing the
explicit FMA with the original source expression makes row 8, head 0 differ
by one bit and fails the test. The correct source was restored afterward.
Final comment cleanup and removal of an unused helper produce byte-identical
PTX to the measured version. The fresh full GPU 1 release workspace run,
including this regression, completed 87 targets with 975 successful test
returns, zero failures, and the same 12 explicit skips. Formatting and
workspace Clippy are clean. The corrected implementation is kept.

## Exact-order GDN gate experiments

The current target's alpha/beta gates cost about 0.181 ms per shallow N=3
decode step, 1.35% of active GPU time. The older 1 ms estimate describes a
different measurement and does not size the remaining opportunity here.

The loading prototype gives each (head, token) a 256-thread block. Threads
cooperatively load both weight rows and the input row into 24 KiB of shared
memory; only the first warp performs the original ascending lane-strided
FMA loop and XOR(16,8,4,2,1) reduction. This changes loading and grid
coverage, without splitting the arithmetic among additional warps. A
control runs the unchanged kernel with one head per block instead of four.

All five gate outputs are bit-identical to the original for N=1, N=3, and
N=7 on deterministic random inputs, 1,760 output values in total. Runtime
and offline register counts are 64 for the original/control and 45 for
cooperative loading; neither has spills, stack allocation, or runtime local
memory. The original/control use no shared memory. At N=3 their grids
contain 24/96 blocks of 128/32 threads; the prototype has 96 blocks of 256
threads and 24,576 dynamic shared bytes per block. The resource ceiling is
32 resident warps/SM for the original, 16 for the one-warp control (block
count limit), and 16 for cooperative loading (two shared-memory blocks).
Only one warp per cooperative block continues into the contraction.

The isolated probe captures 300 launches per CUDA graph and times ten graph
replays with CUDA events after three warmups. It compares repeated use of
one 512 KiB weight pair with rotation through 30 pairs (15 MiB, beyond L2),
to expose cache sensitivity. A is the original four-head grid, B cooperative
loading, and C the one-head control. Values are microseconds per launch;
columns preserve the actual alternating execution order. All measurements
ran alone on GPU 1, with the campaign's NVRTC options.

| N | Weight reuse | A1 | B1 | C1 | C2 | B2 | A2 | A3 | B3 | C3 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | resident | 5.2425 | 4.9335 | 4.7192 | 4.7180 | 4.9524 | 5.2601 | 5.2595 | 4.9534 | 4.7179 |
| 1 | 30 layers | 5.9747 | 4.9299 | 4.8345 | 4.8325 | 4.9317 | 5.3166 | 5.3194 | 4.9329 | 4.8193 |
| 3 | resident | 3.8567 | 4.2375 | 4.3241 | 4.3209 | 4.2454 | 3.8694 | 3.8685 | 4.2448 | 4.3181 |
| 3 | 30 layers | 5.2412 | 5.0085 | 4.9360 | 4.9332 | 5.0332 | 5.2731 | 5.2719 | 5.0415 | 4.9377 |
| 7 | resident | 4.8539 | 6.7925 | 5.7315 | 5.7312 | 6.8086 | 4.9046 | 4.9028 | 6.8595 | 5.7364 |
| 7 | 30 layers | 5.7141 | 7.8752 | 6.3345 | 6.3380 | 7.8943 | 5.7294 | 5.7296 | 7.8958 | 6.3229 |

Cooperative loading does not establish a cache-independent win. At N=3 it
loses with resident weights and improves when rotating weights; the
one-head control is faster in the rotating case. Both alternatives lose
at N=7, so the model experiment restricts the control to the target's
2,048-wide, 32-head F32 gates at at most three tokens. Wider batches and
other geometries keep their original dispatch. The first rotating N=1
baseline is visibly slower than its later repetitions; it remains in the
table, and no conclusion rests on that pair alone.


### Gate-grid full-model pairs

Model B uses the one-head control. Three pairs at context 2,048 use the
accepted attention kernel in both arms, N=1/N=3, four discarded warmup
steps, and 128 timed steps. Rates include greedy sampling. No compilation,
test, or other GPU work overlaps the pairs. Values are aggregate tokens/s.

| Shape | A1 | B1 | B2 | A2 | A3 | B3 | Paired change | A mean ± SD | B mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| single_stream | 114.1 | 114.3 | 113.9 | 113.6 | 113.6 | 113.9 | +0.18% / +0.26% / +0.26% | 113.77 ± 0.29 | 114.03 ± 0.23 |
| batch 1 | 114.8 | 114.8 | 114.7 | 114.3 | 114.3 | 114.4 | +0.00% / +0.35% / +0.09% | 114.47 ± 0.29 | 114.63 ± 0.21 |
| batch 3 | 216.8 | 216.6 | 215.8 | 216.1 | 215.7 | 215.5 | -0.09% / -0.14% / -0.09% | 216.20 ± 0.56 | 215.97 ± 0.57 |

N=3 loses all three pairs, by 0.09–0.14%. Single-stream N=1 changes by
+0.18–0.26%, while batch N=1 includes one tie; these small changes are below
the observed 0.4–0.5% drift across baseline repetitions. They do not justify
a new N=1-only dispatch. The one-head grid is reverted; no deeper-context
or prefill timing is claimed. Cooperative staging is also rejected at the
isolated gate: it loses at N=3 with resident weights and trails the simpler
control when rotating weights. It was not integrated into a full model.

The model control passed GDN differentials, exact Q6_K/F32 gate comparisons,
the forward golden, batch decode, and batch prefill: five targets, 19
successful returns, no failures, and one explicit opt-in audit skip. The
restored inference source is identical to the attention phase's fully tested
tree (87 targets, 975 successful returns, zero failures, 12 explicit skips).
Formatting and Clippy are checked again for this documentation-only result.

## Consumer-side RMSNorm experiment

The GDN alpha/beta projections each consume a complete normalized row. One
256-thread block per (head, token) combines that consumer
with normalization, using four virtual thread bands to reproduce the
existing 1,024-thread norm reduction. Each virtual thread accumulates its
original square sequence, each warp uses the original down-shuffle tree,
and thread zero adds the 32 warp partials in their original order. The
first warp then performs the unchanged gate contraction and nonlinearities.
Head zero writes the normalized row needed by QKV and output-gate
projections; other heads only keep their own normalized row in shared
memory. Kernel completion supplies the dependency for those projections.

This is consumer-side fusion with duplicated statistics, rather than a
producer-side cross-block reduction. It removes one launch per GDN layer
when selected, but still materializes the normalized row for other
consumers. The dispatch covers target F32 gates at one through three
tokens; larger shapes and other geometries/formats retain their existing
normalization and gate launches. Gate calculation moves ahead of the QKV
projection, so its isolated saving requires a whole-model cache-state check.

The integrated and isolated kernels both report 57 registers, zero local
bytes, no offline stack/spills, and 8,320 dynamic shared bytes per block.
The original norm uses 22 registers, 1,024 threads and 128 shared bytes;
the gate uses 64 registers, 128 threads and no shared memory. All three
resource ceilings permit 32 resident warps/SM. The fused block can have
four resident blocks/SM, though only its first warp performs the contraction
after normalization. These are resource ceilings, not measured occupancy.

N=1 and N=3 isolated checks compare the normalized rows and all five gate
outputs, 8,832 floats total: every value matches the standalone GPU paths
bit for bit. The integrated trial also passes the forward golden, batch
decode, carried and wide batch prefill, GDN differential, and gate-format
identity checks: five targets, 19 successful returns, zero failures, one
explicit opt-in audit skip. No tolerance changes are made.

### Normalization fragment timings

A is standalone RMSNorm followed by the original gate launch; B is the fused
consumer. Each graph contains 300 complete fragments and each CUDA-event
measurement covers ten replays. Rates below are microseconds per fragment,
including both launches in A. A pilot used three warmup replays and exposed
clock ramp-up in its N=1 points; all pilot values are retained here. The
repeat uses 200 warmup replays per measurement to make the short workload
sustain GPU activity before timing. No clock policy was changed.

| Warmup replays | N | Weight reuse | A1 | B1 | B2 | A2 | A3 | B3 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 1 | resident | 8.4062 | 6.1850 | 6.1851 | 8.1962 | 8.1953 | 6.0634 |
| 3 | 1 | 30 layers | 10.3629 | 7.2684 | 7.2649 | 8.3784 | 8.3081 | 5.8001 |
| 3 | 3 | resident | 6.2719 | 5.1865 | 5.1890 | 6.2466 | 6.2482 | 5.1851 |
| 3 | 3 | 30 layers | 8.2262 | 5.9208 | 5.9197 | 8.2215 | 8.2232 | 5.9187 |
| 200 | 1 | resident | 6.0867 | 4.4822 | 4.4823 | 6.0836 | 6.0839 | 4.4845 |
| 200 | 1 | 30 layers | 8.2603 | 5.7900 | 5.8186 | 8.2657 | 8.2627 | 5.8539 |
| 200 | 3 | resident | 6.2893 | 5.2159 | 5.2161 | 6.2902 | 6.2865 | 5.2155 |
| 200 | 3 | 30 layers | 8.2565 | 5.9453 | 5.9440 | 8.2452 | 8.2507 | 5.9539 |

With sustained warmup, N=3 fragment throughput improves about 21% with
resident weights and 39% when rotating 30 weight pairs. This qualifies the
consumer for model measurements; it is not an inference throughput claim.
The full-model pairs follow below.

### Normalization full-model pairs

A is the accepted attention implementation with standalone input norms and
gates. B adds only this consumer fusion. Each process discards four warmup
steps and times 128 decode steps, including greedy sampling. All three
execution shapes are reported separately. Columns preserve the reversed
middle pair; values are aggregate tokens/s.

| Context | Shape | A1 | B1 | B2 | A2 | A3 | B3 | Paired change | A mean ± SD | B mean ± SD |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 2,048 | single_stream | 114.3 | 114.8 | 114.2 | 113.5 | 113.4 | 114.1 | +0.44% / +0.62% / +0.62% | 113.73 ± 0.49 | 114.37 ± 0.38 |
| 2,048 | batch 1 | 114.6 | 115.2 | 115.1 | 114.2 | 114.1 | 114.9 | +0.52% / +0.79% / +0.70% | 114.30 ± 0.26 | 115.07 ± 0.15 |
| 2,048 | batch 3 | 216.3 | 216.7 | 216.1 | 215.8 | 215.5 | 216.2 | +0.18% / +0.14% / +0.32% | 215.87 ± 0.40 | 216.33 ± 0.32 |
| 32,768 | single_stream | 97.5 | 97.7 | 97.6 | 97.0 | 96.8 | 97.6 | +0.21% / +0.62% / +0.83% | 97.10 ± 0.36 | 97.63 ± 0.06 |
| 32,768 | batch 1 | 97.6 | 98.1 | 98.0 | 97.4 | 97.1 | 97.9 | +0.51% / +0.62% / +0.82% | 97.37 ± 0.25 | 98.00 ± 0.10 |
| 32,768 | batch 3 | 166.0 | 166.7 | 166.8 | 166.0 | 165.6 | 166.4 | +0.42% / +0.48% / +0.48% | 165.87 ± 0.23 | 166.63 ± 0.21 |

All three pairs favor B at both contexts and all measured execution shapes.
The small N=3 margins satisfy the directional criterion; they do not imply
large decode headroom. Peak reported VRAM is unchanged: 31.805 GiB at 2K
and 33.555 GiB at 32K for N=3. No separate prefill gain is claimed: the
ordinary prefill chunks do not select this fusion. The change is accepted.
The complete GPU 1 release suite passes: 87 targets, 976 successful returns,
zero failures, and 12 explicit skips for the same unavailable fixtures and
opt-in audits listed above. Formatting and workspace all-target Clippy are
clean. The fixture-free regression passes: at N=1/2/3 and the N=4 fallback, it compares every
normalized value and gate output in bits against the separate GPU kernels
and against the existing CPU RMSNorm, GEMV, softplus, and sigmoid references.
It includes a zero input row and biases on both sides of the softplus
passthrough threshold. The existing RMSNorm and anchored GDN elementwise
tolerances are reused unchanged.

A bounded Nsight attribution capture at 2K, N=3, 32 timed steps confirms
659 → 629 kernel calls per step. Standalone RMSNorm calls fall from 101 to
71; the 30 separate gate calls become 30 combined calls. The combined
kernel reports the same 57 registers and 8,320 shared bytes in the trace.
These instrumented captures are not throughput evidence: active kernel
intervals total 13.2660 ms/step in A and 13.2718 in B, while clocks and
profiling overhead are uncontrolled between them. The unprofiled reversed
pairs above determine acceptance. Final source formatting produces
byte-identical NVRTC PTX to the measured implementation.

## Results still required

1. Investigate additional launch fusion and one persistent fragment, in
   that order unless the profile establishes a different priority.
2. Prototype or implement safe same-slot continuation and measure multi-turn
   time to first token, transferred bytes, reuse, and conversation time.
3. Re-evaluate the available speculative drafters after the kernel work.
4. Record each experiment's individual pairs, register/shared-memory changes,
   correctness checks, and keep/reject decision.
5. Compare the final tree with the original implementation and report the
   remaining bottlenecks and justified next experiments.
