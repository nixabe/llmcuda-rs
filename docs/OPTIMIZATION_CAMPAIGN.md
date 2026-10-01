# Inference optimization campaign

All eight phases are complete and committed. Fresh comparisons of the final
implementation with the original show consistent prefill gains from 8K through
128K, reaching 12.05–12.17% at 128K. N=3 decode improves 0.19–0.84% at 2K;
the 32K and 120K pairs change direction and do not establish a reliable gain.
Reported peak VRAM is unchanged. The complete release suite passed with the
fixture and audit skips documented below; no numerical threshold was relaxed.

The retained production changes are fragment-local prefill softmax with
original rounding, consumer-side normalization, a partial-copy repair, and
wide-verification correctness repairs. Resident continuation remains a
test-only prototype: it preserves exact output and reduces follow-up latency
in the two measured conversations. Production router integration is future
work. Rejected experiments are documented, and plain decode remains the
general serving default.

| Experiment | Kernel throughput change | Prefill result | Decode or serving result | Correct | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| Fragment softmax with implicit normalizer FMA | Not timed | Not timed | Not timed | No: forward golden | Reject arithmetic form |
| Fragment softmax with explicit original rounding | +21.69–22.77% at 128K | +12.09–12.11% at 128K, N=3 | Not separately measured | Yes; full suite and new regression | Keep |
| Cooperative gate loading | Cache dependent; N=3 resident loss | Not measured | Not measured | Bit-exact isolated probe | Reject |
| One head per gate block | Cache dependent | Not measured | −0.09–0.14% at 2K, N=3 | Yes; targeted model checks | Reject |
| Consumer RMSNorm plus GDN gates | About +21% resident / +39% rotating weights, N=3 | Not separately measured | +0.14–0.32% at 2K; +0.42–0.48% at 32K, N=3 | Yes; full suite and CPU/GPU regression | Keep |
| Query/key head norm grid seam | +72.9–73.8%, N=3 | Not measured | +0.09–0.28% at 2K; −0.24% to +0.24% at 32K, N=3 | Yes; targeted model and CPU/GPU checks | Reject: depth result changes sign |
| Block-persistent GDN normalization and recurrence | −1.20–1.66% N=3 resident; +2.01–2.35% rotating states; N=1 loses | Not measured | Not measured | Bit-exact isolated probe | Reject |
| Exact-prefix resident continuation | No kernel change | Follow-up TTFT −8.34–18.31% | Conversation wall time −0.27–0.83%; cold tok/s not measured | Bit-exact logits and generated IDs; ownership and copy regressions | Keep test-only prototype |
| Wide-verification residency and arithmetic | No new CUDA kernel | Ordinary prefill unchanged | No speed claim against broken verification | New bit-exact GDN and full-model acceptance regressions | Keep correctness repair |
| Existing speculative policies | No kernel change | MTP loses at depth; all observations below | Map: +10.37–10.43% shallow N=1, +1.29–1.49% at 120K N=3 with full acceptance; active policies lose shallow/32K N=3 | Corrected verify path and fixed-width guard | Keep plain serving default; record conditional crossover |

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

Each phase has its own commit:

| Phase | Commit | Outcome |
| --- | --- | --- |
| 1: baseline and profiling | `957c02b` | Timed capture support and measured attribution |
| 2: deep-prefill attention | `3366970` | Retain fragment-local softmax with original rounding |
| 3: exact-order GDN gates | `b87ae7f` | Record rejected loading and grid experiments |
| 4: consumer normalization | `335e6c3` | Retain RMSNorm plus GDN-gate fusion |
| 5: further launch fusion | `5570127` | Reject query/key normalization seam |
| 6: persistent fragment | `acdf717` | Reject block-persistent GDN prototype |
| 7: resident continuation | `949c4a9` | Retain test-only prototype and partial-copy repair |
| 8: speculative re-evaluation | `bca1014` | Retain exact verification repairs; measure policy crossover |

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
export LLMCUDA_MODEL=./models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf
LLMCUDA_PREFILL_SEQUENCES=3 LLMCUDA_BENCH_CHUNK=1536 \
  LLMCUDA_BENCH_N=512 LLMCUDA_BENCH_REPS=1 \
  bench-results/campaign-d398f19/original/bench_forward
LLMCUDA_PREFILL_SEQUENCES=3 LLMCUDA_BENCH_CHUNK=6138 \
  LLMCUDA_BENCH_N=2046,8184,32736,65472,130944 LLMCUDA_BENCH_REPS=1 \
  bench-results/campaign-d398f19/original/bench_forward
LLMCUDA_BATCH_N=1,3 bench-results/campaign-d398f19/original/bench_decode_batch 2048 32
LLMCUDA_BATCH_N=1,3 bench-results/campaign-d398f19/original/bench_decode_batch 32768 32
LLMCUDA_BATCH_N=3 bench-results/campaign-d398f19/original/bench_decode_batch 122880 32
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

For bounded capture, set `LLMCUDA_PROFILE_TIMED=1` and pass
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

## Final comparison with the original implementation

A is the preserved original implementation, `d398f1940e1a9adc270661201465349fa4857545`.
B is the validated final implementation, `bca1014c9708218282e361048ee7f265afe60796`.
These are fresh alternating runs, not a comparison with the immediately
preceding optimization or with the initial baseline taken hours earlier.
The order is A1 B1 B2 A2 A3 B3, on GPU 1 with the same model, physical chunk
sizes, contexts, warmups and timing windows specified above. Prefill and
decode run serially, with no concurrent builds, tests or other GPU work.

Frozen executable SHA-256 values:

| Executable | A | B |
| --- | --- | --- |
| `bench_forward` | `dfa82d7c9a5b20730889cb9448028cf6fbea0ff91c5296cf0580e1eb9041efec` | `b7e064c67b07a95fc43c8ee9b972a7d226f059d1960ae09690eadba4627fe98c` |
| `bench_decode_batch` | `da16972ba5163c88e1f1cbb4562a4c45960369f9507c3f82d42ba78dd529cc7c` | `3676b3b8508f8675d09f57e191137764dfed38d15ba10b3a7c1c817c7a300034` |

All rates are aggregate tokens/s. Means and sample SD use the three
observations per arm; the paired changes follow A1/B1, A2/B2 and A3/B3.

### Final prefill, N=3

| Workload / context / shape | A1 | B1 | B2 | A2 | A3 | B3 | A mean ± SD | B mean ± SD | Paired change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| prefill / 512 / batch 3 | 3252.95 | 3180.57 | 3180.37 | 3183.59 | 3180.34 | 3198.21 | 3205.63 ± 41.02 | 3186.38 ± 10.24 | -2.23% / -0.10% / +0.56% |
| prefill / 2,046 / batch 3 | 3666.56 | 3633.56 | 3633.14 | 3609.94 | 3609.63 | 3632.62 | 3628.71 ± 32.78 | 3633.11 ± 0.47 | -0.90% / +0.64% / +0.64% |
| prefill / 8,184 / batch 3 | 3405.45 | 3412.48 | 3415.18 | 3358.65 | 3364.24 | 3414.00 | 3376.11 ± 25.56 | 3413.89 ± 1.35 | +0.21% / +1.68% / +1.48% |
| prefill / 32,736 / batch 3 | 2659.34 | 2779.46 | 2783.07 | 2643.26 | 2641.22 | 2780.43 | 2647.94 ± 9.93 | 2780.99 ± 1.87 | +4.52% / +5.29% / +5.27% |
| prefill / 65,472 / batch 3 | 2082.88 | 2255.97 | 2257.05 | 2080.31 | 2079.05 | 2254.53 | 2080.75 ± 1.95 | 2255.85 ± 1.26 | +8.31% / +8.50% / +8.44% |
| prefill / 130,944 / batch 3 | 1460.12 | 1637.85 | 1637.48 | 1460.62 | 1460.27 | 1636.30 | 1460.34 ± 0.26 | 1637.21 ± 0.81 | +12.17% / +12.11% / +12.05% |

Every pair favors the final implementation from 8K through 128K. The
128K improvement is 12.05–12.17%, with original rates of 1,460.12–1,460.62
and final rates of 1,636.30–1,637.85 tokens/s. The two shallowest points
change sign across pairs: 512 tokens and 2K do not establish a consistent
gain or regression. At 512, the original arm itself spans 2.28%; its
higher first observation must remain in the table, not be discarded to
improve the reported result.

Every 512-token process reports 32.229 GiB peak VRAM, and every longer
ladder reports 42.104 GiB, for both original and final binaries. No increase
in reported prefill VRAM accompanies the throughput improvement.

### Final decode

| Workload / context / shape | A1 | B1 | B2 | A2 | A3 | B3 | A mean ± SD | B mean ± SD | Paired change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| decode / 2,048 / batch 1 | 114.20 | 115.00 | 115.10 | 114.40 | 114.50 | 115.40 | 114.37 ± 0.15 | 115.17 ± 0.21 | +0.70% / +0.61% / +0.79% |
| decode / 2,048 / batch 3 | 215.00 | 215.40 | 216.90 | 215.10 | 214.70 | 216.00 | 214.93 ± 0.21 | 216.10 ± 0.75 | +0.19% / +0.84% / +0.61% |
| decode / 2,048 / single_stream | 113.50 | 114.00 | 114.50 | 113.40 | 113.60 | 114.40 | 113.50 ± 0.10 | 114.30 ± 0.26 | +0.44% / +0.97% / +0.70% |
| decode / 32,768 / batch 1 | 97.40 | 97.80 | 98.30 | 97.90 | 98.10 | 97.90 | 97.80 ± 0.36 | 98.00 ± 0.26 | +0.41% / +0.41% / -0.20% |
| decode / 32,768 / batch 3 | 167.10 | 165.70 | 166.90 | 166.50 | 165.20 | 165.50 | 166.27 ± 0.97 | 166.03 ± 0.76 | -0.84% / +0.24% / +0.18% |
| decode / 32,768 / single_stream | 97.20 | 97.80 | 98.00 | 97.00 | 97.00 | 97.60 | 97.07 ± 0.12 | 97.80 ± 0.20 | +0.62% / +1.03% / +0.62% |
| decode / 122,880 / batch 3 | 101.70 | 101.20 | 100.50 | 101.30 | 101.10 | 101.40 | 101.37 ± 0.31 | 101.03 ± 0.47 | -0.49% / -0.79% / +0.30% |
| decode / 122,880 / single_stream | 65.80 | 66.00 | 66.40 | 65.50 | 65.80 | 66.70 | 65.70 ± 0.17 | 66.37 ± 0.35 | +0.30% / +1.37% / +1.37% |

Single-stream decode favors the final implementation in all pairs at all
three depths. Batch-1 decode gains 0.61–0.79% at 2K, but changes direction
at 32K. For the primary N=3 target, the 2K pairs gain 0.19–0.84%; the
32K pairs range from −0.84% to +0.24%, and the 120K pairs from −0.79%
to +0.30%. The two deeper N=3 means are slightly lower (−0.14% and
−0.33%). Their changes are smaller than the observed within-arm spread
and change sign across pairs: this series establishes neither a consistent
deep-batch gain nor the absence of a subpercent regression. The earlier
longer, targeted normalization comparisons do not replace these final
original-versus-final observations.

These are 32-step timing windows, and decode rates are recorded to one
decimal place. Additional digits in the means, SDs and changes summarize
those recorded observations; they do not add measurement precision.

Reported peak decode VRAM is identical in every original and final run:

| Context | Single stream | Batch 1 | Batch 3 |
| ---: | ---: | ---: | ---: |
| 2,048 | 31.586 GiB | 31.586 GiB | 31.805 GiB |
| 32,768 | 32.180 GiB | 32.180 GiB | 33.555 GiB |
| 122,880 | 33.898 GiB | Not measured | 38.711 GiB |

All 30 scheduled final-comparison processes completed successfully: 12
prefill processes and 18 decode processes, yielding six observations for
each of the 14 workload shapes. These measurements do not refresh the
separate llama.cpp head-to-head.

## Original correctness checks

Before changing inference code, the following completed on GPU 1:

```sh
cargo fmt --all
cargo clippy --workspace --all-targets
CUDA_VISIBLE_DEVICES=1 \
  LLMCUDA_GOLDEN=/home/nixabe/llmcuda-rs/.golden/qwen36-golden.bin \
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
`LLMCUDA_PROFILE_TIMED` capture boundary. Nsight failed to import the
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

## Query/key head normalization grid seam

The post-normalization 2K, N=3 trace identifies these sub-10-microsecond
kernels. Rows are ordered by total duration in the 32-step capture; launch
counts and averages describe that same capture. These durations include
instrumentation and select candidates, not performance claims.

| Kernel | Total ms | Calls | Mean us |
| --- | ---: | ---: | ---: |
| `moe_route` | 11.496 | 1280 | 8.981 |
| `moe_block_router_logits_t3` | 11.254 | 1280 | 8.792 |
| `gdn_norm_alpha_beta_gates` | 6.971 | 960 | 7.261 |
| `rms_norm_rows` | 6.759 | 2272 | 2.975 |
| `gdn_conv_silu_split_step_batch` | 5.889 | 960 | 6.135 |
| `moe_block_gate_and_combine` | 5.156 | 1280 | 4.028 |
| `rms_norm_swiglu_rows` | 2.772 | 960 | 2.888 |
| `attn_decode_rope_append_batch` | 2.196 | 320 | 6.861 |
| `gdn_normalize_qk` | 2.187 | 960 | 2.278 |
| `sigmoid_gate_mul` | 0.751 | 320 | 2.345 |
| `tensor_add` | 0.599 | 320 | 1.873 |
| `attn_split_query_gate` | 0.576 | 320 | 1.800 |
| `argmax_partial` | 0.443 | 96 | 4.617 |
| `argmax_final` | 0.200 | 96 | 2.079 |
| `fwd_embed_q8_0` | 0.150 | 32 | 4.679 |

Query and key per-head norms each use 256 threads and independent 256-float
rows. Once the key/value projections finish, both inputs are ready; assigning
the first grid range to query rows and the second to key rows can remove one
launch at each of ten attention layers. Their weights and outputs are
disjoint. The query norm moves later, so whole-model measurements must check
whether the altered order changes cache behavior. The trial only selects
hidden=2,048, 16 query heads, two KV heads, head width 256, and one through
three tokens. Other geometries and larger chunks keep the old order.

The isolated source retains the norm's exact tree and multiplication order.
N=1 and N=3 match all 18,432 output floats in bits. Both original and seam
use 22 registers, zero local bytes, 256 threads and 32 dynamic shared bytes;
the resource ceiling remains four blocks / 32 warps per SM. Each graph
contains 300 fragments, with 200 warmup replays and ten timed replays.
A has two launches per fragment; B has one. All values are us/fragment:

| N | A1 | B1 | B2 | A2 | A3 | B3 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.7091 | 1.7239 | 1.7159 | 3.2031 | 3.2031 | 1.7169 |
| 3 | 3.4961 | 2.0216 | 2.0215 | 3.5140 | 3.5142 | 2.0232 |

The first N=1 A measurement still exposes startup clock drift; it is
retained rather than discarded. N=3 improves in all pairs by 72.9–73.8%
in fragment throughput. The integrated kernel retains 22 registers, no
stack or spill traffic and no local bytes. Targeted release tests pass:
27 successful returns across batch decode, batch prefill, forward golden
and layer-op differential targets; zero failures and one opt-in audit skip.
The new N=1/2/3 regression compares separate and merged GPU outputs in bits
and checks both row sets against the existing CPU RMSNorm reference with
unchanged tolerances. The full-model measurements are below.

### Head-normalization full-model pairs

A is commit `335e6c3`'s measured inference implementation. B adds only the
query/key norm seam. The protocol uses four discarded warmup steps and
128 timed decode steps, greedy sampling included, with each process on
GPU 1 and no concurrent build/test/GPU work. Rates are aggregate tokens/s.

| Context | Shape | A1 | B1 | B2 | A2 | A3 | B3 | Paired change | A mean ± SD | B mean ± SD |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 2048 | single_stream | 114.8 | 114.6 | 114.3 | 114.3 | 114.2 | 114.4 | -0.17% / +0.00% / +0.18% | 114.43 ± 0.32 | 114.43 ± 0.15 |
| 2048 | batch 1 | 114.9 | 115.1 | 115.2 | 115.0 | 114.9 | 115.0 | +0.17% / +0.17% / +0.09% | 114.93 ± 0.06 | 115.10 ± 0.10 |
| 2048 | batch 3 | 216.3 | 216.5 | 216.7 | 216.1 | 216.0 | 216.3 | +0.09% / +0.28% / +0.14% | 216.13 ± 0.15 | 216.50 ± 0.20 |
| 32768 | single_stream | 97.9 | 97.8 | 97.6 | 97.5 | 97.5 | 97.7 | -0.10% / +0.10% / +0.21% | 97.63 ± 0.23 | 97.70 ± 0.10 |
| 32768 | batch 1 | 98.1 | 98.1 | 98.0 | 98.0 | 97.8 | 97.8 | +0.00% / +0.00% / +0.00% | 97.97 ± 0.15 | 97.97 ± 0.15 |
| 32768 | batch 3 | 166.7 | 166.5 | 166.6 | 166.2 | 166.6 | 166.2 | -0.12% / +0.24% / -0.24% | 166.50 ± 0.26 | 166.43 ± 0.21 |

At 2K, all three batch-1 and N=3 pairs favor the seam; single-stream N=1
changes sign and has identical means. At 32K, batch-1 ties in all three pairs
and N=3 loses two of three pairs. Its −0.12% / +0.24% / −0.24% changes are
smaller than the observed 0.30% spread between baseline N=3 runs. The deeper
result is inconclusive, with no stable improvement. These rounded,
sub-drift rates do not justify a context-specific dispatch. The candidate is
rejected and its implementation and added test are reverted. No prefill
benefit or regression is claimed. Peak N=3 VRAM remains 31.805 / 33.555 GiB
at 2K / 32K.

A separate bounded 32-step Nsight capture verifies 629 → 619 launches per
step. Standalone RMSNorm calls fall from 71 to 51, with ten paired calls
added. Thus the launch saving is real, but insufficient to establish the
required model gain. The restored inference source is byte-identical to
`335e6c3`, whose GPU 1 release suite completed 87 targets with 976 successful
returns, zero failures and 12 explicit skips. Formatting and all-target
workspace Clippy are checked again after restoration. No additional full
suite is attributed to the rejected trial.

## Block-persistent GDN fragment

This prototype assigns one 256-thread block to a (sequence, value head).
It normalizes that head's Q/K vectors into shared memory, then its eight
warps each advance 16 recurrent-state rows before leaving the block. It
combines two operations in one repeated layer fragment; it has no inter-block
handshake, cooperative-grid requirement, CPU synchronization between layers,
or whole-transformer persistent loop. Fixed pointers and launch geometry
remain CUDA-graph compatible. No production dispatch is installed.

The canonical 128-thread normalization uses four warp partials. The
prototype's other four warps contribute positive zeros after those partials;
the finite square sum keeps its original bits. State-row float4 loads,
decay, delta correction, FMA order and warp XOR reductions are copied from
`llmcuda-cuda`'s `gdn_recurrent_step`. Q/K remain resident in 1,024 shared bytes,
with another 32 bytes for reductions. The two value heads sharing one Q/K
head duplicate its normalization. Only one writes the normalized waypoint.
Every warp loops over its disjoint rows, with sequence selected by the same
fixed pointer-slot convention as the original kernel.

At N=1 and N=3, three consecutive updates compare all recurrent state,
normalized Q/K and output values against the original GPU fragment:
6,389,760 float comparisons, all bit-identical. Inputs differ by sequence
and include a zero Q/K head. This is direct GPU equivalence on the probe,
not a new CPU differential or a full-model test of the prototype. The
original GPU fragment's CPU differentials passed in the preceding complete
release suite; no tolerance is changed.

| Kernel | Threads/block | Registers/thread | Dynamic shared bytes | Local/stack/spill bytes |
| --- | ---: | ---: | ---: | ---: |
| Original Q/K normalization | 128 | 27 | 16 | 0 |
| Original recurrent step | 128 | 64 | 0 | 0 |
| Combined head pipeline | 256 | 62 | 1,056 | 0 |

Runtime JIT attributes and offline ptxas agree. Both the original state
kernel and pipeline have a 32-warp/SM resource ceiling; the pipeline has
four possible resident blocks instead of eight. More significantly, N=3
shrinks the state grid from 3,072 blocks to 96, only 1.33 blocks per SM on
average over this 72-SM card. At N=1 there are only 32 blocks. These are grid
and resource limits, not measured occupancy. Lower register use does not
compensate for the smaller grid in all tested cache states.

The CUDA-event probe captures 300 fragments per graph, discards 200 warmup
replays, and times ten replays per measurement. A has the separate normalize
and state launches; B has the combined block pipeline. One bank reuses the
same state; 30 banks rotate through 60 MiB (N=1) or 180 MiB (N=3) of state
per arm. Q/K, gates and value inputs are fixed in each case. Values are
us/complete fragment; every observation is retained, including the first
N=1 run's startup drift.

| N | State banks | A1 | B1 | B2 | A2 | A3 | B3 | Paired throughput change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1 | 6.3072 | 9.7387 | 9.8227 | 6.8122 | 6.6862 | 9.8336 | -35.24% / -30.65% / -32.01% |
| 1 | 30 | 11.7118 | 14.9606 | 14.9763 | 11.7221 | 11.7382 | 14.9796 | -21.72% / -21.73% / -21.64% |
| 3 | 1 | 20.2942 | 20.5397 | 20.6938 | 20.3767 | 20.3059 | 20.6479 | -1.20% / -1.53% / -1.66% |
| 3 | 30 | 28.0072 | 27.4030 | 27.4129 | 28.0583 | 27.9756 | 27.4241 | +2.20% / +2.35% / +2.01% |

Reject at the isolated gate. N=1 loses in every pair under both cache
conditions, and N=3 loses with one resident state bank. The rotating N=3
case does improve consistently, but its absolute saving is only
0.55–0.65 us/fragment and does not establish a robust replacement for the
independently specialized grids. No full-model integration, prefill or
decode-rate claim follows from this experiment. A future N=3-only study
would still need to establish the actual in-model cache condition and win
whole-model pairs; these timings alone do not justify that dispatch.

No production source changed. Its complete release-suite evidence remains
`335e6c3` (87 targets, 976 successful returns, zero failures, 12 explicit
skips); formatting and workspace all-target Clippy are checked for this
phase's documentation commit.

## Exact-prefix resident continuation

This phase keeps a **test-only prototype**, not a production router change.
`forward::resident_continuation` owns a fixed set of GPU states and
preallocated token histories. A mutable lease selects an unowned slot,
compares every consumed token, and checks model, tokenizer, rotary offset
and session identity. Those identities are tied to the live immutable model
and tokenizer instances in this harness. Reuse requires a strictly longer
prompt: an exact-length hit falls back because pending logits are not cached.
A miss must restore a verified host checkpoint or reset before exposing
state. Eviction clears validity; an abandoned pass stays invalid; capacity
exhaustion refuses the request without resizing. Only a completed pass can
publish its consumed prefix. The final emitted token has not yet entered
KV/recurrent state and must be consumed on the next turn.

The experiment exposed a prerequisite bug: the snapshot arena accepted a
partial retention interval, but attention copies asserted that the valid
prefix filled the entire pinned allocation. The production fix copies only
the valid host/device subranges while retaining the pinned allocation's
stream event guard. It adds no allocation or host barrier. Its GPU regression
covers empty, partial and full intervals, independent K/V payloads, untouched
tails, and capture/restore across two streams without an intervening host
synchronization. Four CPU tests exercise prefix/scope mismatches, capacity,
ownership, eviction, abandoned passes, and consumed-token bookkeeping.

The manual ignored test runs two model-generated file-tool conversations:

- Contributor task: read `Cargo.toml` and `CONTRIBUTING.md`, explain CUDA
  kernel validation/commits, then answer a one-sentence follow-up.
- Architecture task: read `docs/MODEL.md` and `docs/CACHE.md`, explain cache
  geometry and the dense model difference, then answer the same follow-up.

Both tasks take three turns: generated file calls, the file-informed answer,
and the follow-up. The tool executes only the two whitelisted local reads.
Raw token history is append-only, including emitted special tokens; generic
chat-template rerendering that rewrites earlier thinking is outside this
prototype. Such a prefix mismatch must fall back, not inherit resident state.

A restores a verified full-prefix checkpoint at each continuation; B retains
the same slot. **Both arms capture the same host checkpoint after every
turn**, so D2H snapshot time remains in conversation time. This isolates the
saved restoration, with the partial-copy fix present in both arms. Inference
shapes 1–512, one 16,384-token GPU state (401,408,000 bytes), 65,536 bytes of
slot history, two pinned snapshot slots (802,816,000 bytes together), and a
pinned argmax buffer are allocated before timing. The shapes share weights;
there is no per-token sampling allocation. Initial model/shape/pool setup is
excluded. TTFT starts at slot claim and ends after the first GPU argmax read;
conversation time also includes subsequent tokenization, file I/O, generation
and snapshots. These are one-slot measurements, not N=3 serving throughput.

GPU 1, same target model and automatic clocks as the campaign baseline.
Executable SHA-256:
`43717411cea77ea9022b60a391215c7b7902f353c249e8c9355b90ddd579066e`.
A discarded A/B pair validates every generated token and all 248,320
first-token logits per turn bit for bit, then six timed conversations run
A1 B1 B2 A2 A3 B3. Every timed generated sequence must also match the
validation reference. No logits readback is added to timed runs. All 12
timed conversations and both validation pairs passed.

All latency observations below are milliseconds. The means use the three
observations per arm; SD is the sample standard deviation.

| Task / turn | A1 | B1 | B2 | A2 | A3 | B3 | A mean ± SD | B mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Contributor / file calls | 204.785 | 205.415 | 205.180 | 205.825 | 202.944 | 206.744 | 204.518 ± 1.459 | 205.780 ± 0.844 |
| Contributor / answer | 1297.024 | 1290.043 | 1296.323 | 1306.767 | 1311.088 | 1305.161 | 1304.960 ± 7.204 | 1297.176 ± 7.595 |
| Contributor / follow-up | 153.423 | 138.672 | 140.374 | 153.152 | 153.105 | 138.236 | 153.227 ± 0.172 | 139.094 ± 1.130 |
| Architecture / file calls | 227.894 | 229.121 | 232.280 | 229.851 | 231.727 | 230.198 | 229.824 ± 1.917 | 230.533 ± 1.606 |
| Architecture / answer | 6674.592 | 6684.214 | 6715.171 | 6728.532 | 6734.715 | 6726.183 | 6712.613 ± 33.072 | 6708.523 ± 21.760 |
| Architecture / follow-up | 231.922 | 189.455 | 193.096 | 231.172 | 232.858 | 192.620 | 231.984 ± 0.844 | 191.724 ± 1.979 |
| Contributor / whole conversation | 3319.764 | 3305.409 | 3312.357 | 3339.999 | 3343.862 | 3331.891 | 3334.541 ± 12.943 | 3316.552 ± 13.730 |
| Architecture / whole conversation | 9701.178 | 9674.907 | 9718.282 | 9767.038 | 9776.315 | 9728.883 | 9748.177 ± 40.966 | 9707.358 ± 28.598 |

Follow-up TTFT falls in every pair: 9.62% / 8.34% / 9.71% on the
contributor task and 18.31% / 16.47% / 17.28% on architecture. The larger
file-prefill turn dominates conversation time, so whole-conversation savings
are much smaller: 0.43% / 0.83% / 0.36% and 0.27% / 0.50% / 0.49%,
respectively. All six paired conversation changes favor reuse, but their
small magnitude and these two deterministic tasks do not establish a
production latency distribution. The cold first turn has no consistent
change. The architecture file-prefill turn changes sign and is unresolved.

Transfer counts are payload bytes derived from the actual copied tensor
lengths. `reuse` is the same consumed prefix in A and B; it is restored in A
and already resident in B. B's state H2D payload is zero on all turns.
Common input-token and position uploads remain and are listed separately.

| Task / turn | Prompt tokens | Reused tokens | Emitted tokens | A state H2D bytes | D2H snapshot bytes, both | Token/position H2D bytes A / B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Contributor / file calls | 147 | 0 | 10 | 0 | 69,058,560 | 728 / 728 |
| Contributor / answer | 2,369 | 156 | 138 | 69,058,560 | 117,186,560 | 10,568 / 10,560 |
| Contributor / follow-up | 2,534 | 2,506 | 42 | 117,186,560 | 118,599,680 | 636 / 628 |
| Architecture / file calls | 154 | 0 | 13 | 0 | 69,263,360 | 792 / 792 |
| Architecture / answer | 11,997 | 166 | 174 | 69,263,360 | 315,105,280 | 49,632 / 49,624 |
| Architecture / follow-up | 12,198 | 12,170 | 49 | 315,105,280 | 316,661,760 | 720 / 712 |

Reproduce after checking the GPU is idle:

```sh
CUDA_VISIBLE_DEVICES=1 cargo test --release -p llmcuda-engine --lib \
  resident_continuation::conversations::agent_conversations -- \
  --ignored --nocapture --test-threads=1
```

The prototype justifies a future production routing experiment with multiple
sessions, eviction pressure, cancellation and real chat-template prefix
identity. It does not justify removing the host fallback or changing snapshot
retention. No CUDA kernel resources or arithmetic change in this phase.

The complete GPU-1 release workspace suite passed: 87 targets, 981 successful
returns, zero failures, 12 explicit fixture/audit skips, and one ignored
manual conversation test (executed successfully above). Formatting and
workspace all-target Clippy are clean. No numerical tolerance was changed.

## Speculative decoding after the kernel changes

Re-evaluation exposed two correctness problems in previously unexercised
wide verification, before a performance decision could be made. At three
49-row windows, GDN's prefill threshold discarded its split-layout handle
at 64 total rows. The fallback requested stored Q8_0 bytes that production
no longer keeps resident. Removing that threshold preserves the existing
four-row projection slices and their one-token FMA order; it does not add
another copy of the weights or switch verification to integer MMA.

A new full-model regression then caught a second failure: attention still
selected prefill arithmetic for a wide verify shape. The crafted draft
expected 24 accepted tokens but stopped at 10. Verification now explicitly
selects ordinary FP32 attention projections and per-query decode attention
at every window width, just as its MoE already selects an exact decode
regime. This setting is local to the verify pass. Ordinary prefill and
plain-decode dispatch remain the same. No CUDA source or numerical tolerance
changes in this repair; it selects existing kernels for the required
arithmetic, even where that costs additional launches.

Two regressions passed on GPU 1:

- A real-weight GDN differential removes all stored-format projection
  copies, then compares every output and zero/partial/full rollback state
  bit for bit against a single-token chain. Cases cover 1, 63, 64 and 65
  rows, plus three independent 49-row windows. All pass.
- A full-model three-sequence verify test crafts zero, 24-token and
  48-token acceptance, checks each bonus token immediately, and compares
  64 emitted tokens per sequence with plain greedy decode. It failed with
  only the residency fix and passed after selecting exact attention.

The benchmark also refuses a request that retires during setup or timing,
and refuses a changed decode width. A deliberate 40-token output budget at
N=3 triggers the guard before any throughput result is published. Per-sequence
emission counts and wall times are logged outside the timed window, exposing
both acceptance imbalance and context growth.

The frozen corrected benchmark executable has SHA-256
`d26308a1a1c2be7ffd82cdea91398564cb18e8d0a61be50ea05795bf8ab749a7`.
Measurements use the same target model and GPU 1 as the earlier phases.
Every policy in a configuration reserves the same output budget; the
393,216-token pool and 1% watermark are unchanged. The generated text is the
model's own greedy continuation of the existing deterministic synthetic
prompt, after every sequence has emitted at least 32 warmup tokens.

| Configuration | Initial context/sequence | N | Timed window | Reserved output/sequence | Policies and draft caps |
| --- | ---: | ---: | --- | ---: | --- |
| Shallow N=1 | 256 | 1 | At least 512 emitted tokens | 8,192 | none, map-k 48, MTP 3 |
| Sustained shallow N=3 | 256 | 3 | 512 scheduler steps | 32,768 | none, simple 48, map-k4v 8, MTP 3 |
| 32K N=3 | 32,768 | 3 | 64 scheduler steps | 8,192 | same four policies |
| 120K N=3 | 122,880 | 3 | 32 scheduler steps | 4,096 | same four policies, subject to allocation |

The N=1 token quota limits differences in generated length; an accepted
window can overshoot it, so actual counts are retained. N=3 uses fixed
scheduler steps to keep all three sequences active and avoid the old
slowest-sequence quota skew. This still permits differing context growth,
which must remain explicit: sustained shallow results are not rates at a
single fixed KV depth. Deep windows are short relative to their initial
context. The policy order is A B C, C B A, A B C, or A B C D, D C B A,
A B C D when there are three candidate policies. A is always `none`.
An allocation failure is recorded once rather than repeatedly timed.

Earlier exploratory runs used a 128-step shallow window, where the N=1 map
policy emitted no extra tokens per step, and then a 512-step window. The
latter activated wide N=3 verification and exposed the missing-weight error.
Those runs are retained locally with their original executable and logs;
wide pre-repair timings are not evidence for a correct implementation.
The final comparisons use the corrected executable above.

The narrow per-query attention dispatch already existed in the original
campaign commit. Consequently, a difference from older speculative tables
cannot be attributed solely to the campaign's prefill attention optimization.
The corrected verify path deliberately retains decode arithmetic at all
widths, so the prefill MMA improvement is not itself a verify-kernel win.
DFlash is unavailable because its model file is absent; it is not measured.

All 42 scheduled successful policy measurements completed. The one failed
allocation was simple at cap 48, N=3 and 122,880 tokens/sequence; its later
repetitions were skipped after the first explicit `CUDA_ERROR_OUT_OF_MEMORY`.
This is a memory rejection, not a zero-throughput observation.

On the matched shallow N=1 quota, map-k gains 10.37–10.43% and MTP gains
5.61–6.28%. Every measured active speculative policy loses during sustained
shallow N=3 generation and at 32K. Simple at 32K emits no extra tokens per
step; its near-plain rate is not an active-drafting win.

The deep result is different: map-k4v at cap 8 gains 1.29–1.49% in every pair,
with perfect acceptance (27 tokens/step). MTP also has perfect acceptance
(12 tokens/step), but loses 11.14–11.23%. Map's approximately 265 ms step
barely repays its ninefold output multiple against plain decode's 30 ms;
MTP's approximately 134 ms step cannot repay a fourfold multiple. These are
completed worker-step wall costs, not per-kernel CUDA-event timings. Neither
the small deep map win nor the shallow single-stream wins justify changing
the general serving default. The measured text and acceptance distribution
are part of each result.

Both correctness repairs are retained. No performance gain is claimed
against the broken wide-verification path, and no new speculation kernel or
policy is introduced. The default remains plain decode. The following
observations are tokens/s; repeated emission counts were identical across
all three runs of each configuration. `quota-shallow-n1` and
`long-shallow-n3` identify the two shallow windows in the method table above.

### Speculative prefill observations

| Configuration / policy (cap) | Run 1 | Run 2 | Run 3 | Mean ± sample SD |
| --- | ---: | ---: | ---: | ---: |
| quota-shallow-n1 / none (0) | 1615.9 | 1583.9 | 1600.8 | 1600.20 ± 16.01 |
| quota-shallow-n1 / ngram-map-k (48) | 1593.1 | 1581.5 | 1586.4 | 1587.00 ± 5.82 |
| quota-shallow-n1 / draft-mtp (3) | 1519.1 | 1518.6 | 1546.4 | 1528.03 ± 15.91 |
| long-shallow-n3 / none (0) | 1697.1 | 1740.3 | 1747.6 | 1728.33 ± 27.29 |
| long-shallow-n3 / ngram-simple (48) | 1675.3 | 1692.6 | 1738.3 | 1702.07 ± 32.55 |
| long-shallow-n3 / ngram-map-k4v (8) | 1704.4 | 1740.6 | 1728.8 | 1724.60 ± 18.46 |
| long-shallow-n3 / draft-mtp (3) | 1610.0 | 1659.2 | 1643.6 | 1637.60 ± 25.14 |
| 32k-n3 / none (0) | 2409.2 | 2411.0 | 2412.2 | 2410.80 ± 1.51 |
| 32k-n3 / ngram-simple (48) | 2408.5 | 2411.4 | 2411.9 | 2410.60 ± 1.84 |
| 32k-n3 / ngram-map-k4v (8) | 2408.9 | 2410.7 | 2410.6 | 2410.07 ± 1.01 |
| 32k-n3 / draft-mtp (3) | 2259.1 | 2258.7 | 2262.6 | 2260.13 ± 2.15 |
| 120k-n3 / none (0) | 1526.9 | 1528.3 | 1529.0 | 1528.07 ± 1.07 |
| 120k-n3 / ngram-map-k4v (8) | 1528.1 | 1527.4 | 1529.5 | 1528.33 ± 1.07 |
| 120k-n3 / draft-mtp (3) | 1415.7 | 1416.3 | 1417.8 | 1416.60 ± 1.08 |

### Speculative decode observations

| Configuration / policy (cap) | Run 1 | Run 2 | Run 3 | Mean ± sample SD |
| --- | ---: | ---: | ---: | ---: |
| quota-shallow-n1 / none (0) | 118.4 | 117.9 | 117.7 | 118.00 ± 0.36 |
| quota-shallow-n1 / ngram-map-k (48) | 130.7 | 130.2 | 129.9 | 130.27 ± 0.40 |
| quota-shallow-n1 / draft-mtp (3) | 125.4 | 125.3 | 124.3 | 125.00 ± 0.61 |
| long-shallow-n3 / none (0) | 219.7 | 219.4 | 219.6 | 219.57 ± 0.15 |
| long-shallow-n3 / ngram-simple (48) | 149.1 | 148.6 | 149.2 | 148.97 ± 0.32 |
| long-shallow-n3 / ngram-map-k4v (8) | 101.5 | 101.2 | 101.2 | 101.30 ± 0.17 |
| long-shallow-n3 / draft-mtp (3) | 168.8 | 168.4 | 168.8 | 168.67 ± 0.23 |
| 32k-n3 / none (0) | 165.8 | 166.0 | 165.7 | 165.83 ± 0.15 |
| 32k-n3 / ngram-simple (48) | 165.4 | 165.2 | 165.2 | 165.27 ± 0.12 |
| 32k-n3 / ngram-map-k4v (8) | 117.6 | 117.8 | 117.8 | 117.73 ± 0.12 |
| 32k-n3 / draft-mtp (3) | 127.4 | 127.4 | 127.7 | 127.50 ± 0.17 |
| 120k-n3 / none (0) | 100.5 | 100.6 | 100.7 | 100.60 ± 0.10 |
| 120k-n3 / ngram-map-k4v (8) | 102.0 | 101.9 | 102.0 | 101.97 ± 0.06 |
| 120k-n3 / draft-mtp (3) | 89.3 | 89.3 | 89.4 | 89.33 ± 0.06 |

### Generation windows

| Configuration / policy | Emitted before timing, by sequence | Emitted after timing, by sequence | Tokens/step | Paired decode change |
| --- | --- | --- | ---: | --- |
| quota-shallow-n1 / none | [32] | [544] | 1.000 | control |
| quota-shallow-n1 / ngram-map-k | [32] | [569] | 2.114 | +10.39% / +10.43% / +10.37% |
| quota-shallow-n1 / draft-mtp | [32] | [546] | 2.856 | +5.91% / +6.28% / +5.61% |
| long-shallow-n3 / none | [32, 32, 32] | [544, 544, 544] | 3.000 | control |
| long-shallow-n3 / ngram-simple | [32, 32, 32] | [736, 544, 544] | 3.375 | -32.13% / -32.27% / -32.06% |
| long-shallow-n3 / ngram-map-k4v | [47, 32, 104] | [2486, 544, 3737] | 12.859 | -53.80% / -53.87% / -53.92% |
| long-shallow-n3 / draft-mtp | [41, 32, 43] | [1885, 1783, 2089] | 11.018 | -23.17% / -23.25% / -23.13% |
| 32k-n3 / none | [34, 33, 32] | [98, 97, 96] | 3.000 | control |
| 32k-n3 / ngram-simple | [34, 33, 32] | [98, 97, 96] | 3.000 | -0.24% / -0.48% / -0.30% |
| 32k-n3 / ngram-map-k4v | [105, 32, 71] | [681, 96, 647] | 19.000 | -29.07% / -29.04% / -28.91% |
| 32k-n3 / draft-mtp | [64, 34, 50] | [320, 204, 306] | 10.656 | -23.16% / -23.25% / -22.93% |
| 120k-n3 / none | [34, 33, 32] | [66, 65, 64] | 3.000 | control |
| 120k-n3 / ngram-map-k4v | [62, 77, 36] | [350, 365, 324] | 27.000 | +1.49% / +1.29% / +1.29% |
| 120k-n3 / draft-mtp | [45, 51, 32] | [173, 179, 160] | 12.000 | -11.14% / -11.23% / -11.22% |

## Final correctness

The complete release workspace suite passed on GPU 1 after the wide-verify
repairs: 87 targets, 983 successful test returns, zero failures, 12 explicit
fixture/audit skips, and one ignored manual conversation benchmark. The
ignored benchmark was run separately in the resident-continuation phase;
its implementation and ordinary-forward dispatch are unchanged by the
verification repairs. Formatting and workspace all-target Clippy are clean.

```sh
cargo fmt --all
cargo clippy --workspace --all-targets
CUDA_VISIBLE_DEVICES=1 \
  LLMCUDA_GOLDEN=/home/nixabe/llmcuda-rs/.golden/qwen36-golden.bin \
  cargo test --workspace --release -- --test-threads=1 --nocapture
```

The executed GPU checks include the forward golden, batch decode and
prefill, graph replay, attention, GDN, MoE and LM-head differentials,
serving speculative identity, and both new wide-verification regressions.
No threshold was loosened. The 12 explicit skips cover absent dense-model,
DFlash, vision and live-template fixtures, plus opt-in audit runs; they are
not numerical passes. Dense-model GPU correctness and DFlash execution are
therefore not established by this campaign.

## Remaining bottlenecks and next experiments

The original traces identify two different limits. At shallow N=3 decode,
expert projections occupy about 5.66 ms of the 13.42 ms active GPU interval,
with GDN projections at 2.70 ms and the LM head at 0.96 ms. At 32K,
attention rises to 4.66 ms. These are the measured original attributions,
not a reconstructed profile of the final executable. The accepted decode
fusion removes 30 launches; it does not change those projection kernels.
The rejected head-normalization seam and persistent fragment demonstrate
why a lower launch count alone does not identify the next useful change.

Deep prefill remains an attention problem. The original 128K trace assigns
59.9% of active kernel time to attention. The fragment rewrite removes
shared-memory traffic while retaining the two block barriers per key octet,
the 255-register allocation, and the one-block occupancy ceiling. A further
experiment needs to shorten synchronization or fragment lifetimes without
growing that live register set; widening the existing tiles has already
failed. No post-change attribution percentage is inferred from throughput.

The next experiments justified by these results are:

1. Integrate resident continuation into a bounded production routing
   experiment, checking actual chat-template token prefixes, multiple
   sessions, cancellation and eviction pressure. Keep exclusive ownership
   and the ordinary host-checkpoint fallback. The two measured conversations
   justify this experiment, not a claim about production p99.
2. Investigate shared work across verification rows while preserving the
   one-token projection and attention arithmetic. Wide verification now
   deliberately pays for per-query decode launches. Reusing prefill MMA
   without a numerical argument would repeat the regression caught here.
3. Profile the expert integer-unpacking path and deep decode attention
   before selecting another kernel change. A memory-traffic reduction is
   not sufficient evidence when the integer pipe or softmax is the limit.
   Use a narrow probe, then the same alternating whole-model pairs.
4. Consider another consumer normalization only where its grid already
   covers a complete row and can reproduce the canonical reduction. The
   small GDN-gate win does not establish that duplicating the reduction in
   larger projection grids will help.
5. Refresh the separate llama.cpp standing with matched, alternating runs
   at its best settings when that checkout is available. The original-engine
   comparison here does not provide a new cross-engine ratio.

CUDA graph capture remains a design constraint, not the largest unmeasured
opportunity: its previously measured contribution does not justify ranking
it ahead of these costs. Per-group page geometry, independent recurrent
snapshot retention, and full-sequence admission are unchanged.
