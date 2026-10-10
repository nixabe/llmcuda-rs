# Benchmarks

Where `llmcuda-rs` stands against `llama.cpp` on the target host, the method
behind it, and two durable lists: **WHY** and **WHY NOT**.

## Setup

| | |
| --- | --- |
| Hardware | 3× Quadro RTX 8000, sm_75 (Turing), 47.3 GiB usable, 672 GB/s, 72 SMs, 6 MiB L2 |
| Host | 125 GB RAM, 12 vCPU, CUDA 12.4, driver 595.84 |
| llama.cpp | build 10456, commit `fd6863a69` |
| Model | `Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf` — 30.36 GiB, 35.51 B params |
| Second model | `Qwen3.8-27B-UD-Q8_K_XL.gguf` (`qwen35`, dense) — 29.30 GiB, 26.90 B params |

llama.cpp at its own best settings, three parallel sequences, one card:

```sh
llama-batched-bench -m Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf \
  -ngl 99 -sm none -fa on -b 4096 -ub 4096 -ctk f16 -ctv f16 \
  -npl 3 -npp <ctx> -ntg 32
```

The bar is the faster of `-ub 2048` (llama.cpp's better decode setting) and
`-ub 4096` (its better prefill setting at most depths). Its decode bars at
`-ub 2048`, N=3: **187.90 tok/s** at 2K, **153.79 tok/s** at 32K.

## Current standing

One card, N=3, llama.cpp at `-np 3 -b 4096 -ub 4096`; prefill on GPU 2,
decode on GPU 1. Not re-measured since the
[engine-only campaign](OPTIMIZATION_CAMPAIGN.md#final-comparison-with-the-original-implementation),
which has its own pairs and the single-sequence numbers.

| cell | llmcuda-rs tok/s | llama.cpp tok/s | margin |
| :--- | ---: | ---: | ---: |
| prefill 512 | 3,281.0 ± 8.6 | 3,007.5 | **+9.1%** |
| prefill 2K | 3,819.4 ± 23.4 | 3,201.5 | **+19.3%** |
| prefill 8K | 3,665.5 ± 16.8 | 3,201.8 | **+14.5%** |
| prefill 32K | 2,790.3 ± 5.8 | 2,655.1 | **+5.1%** |
| prefill 65K | 2,226.0 ± 0.2 | 2,142.8 | **+3.9%** |
| prefill 128K | 1,564.5 ± 0.3 | 1,551.6 | **+0.8%** |
| decode 2K | 217.3 / 217.1 / 216.6 | 184.5 / 185.9 / 183.3 | **+16.8–18.2%** |
| decode 32K | 165.7 / 166.5 / 165.2 | 151.0 / 150.4 / 150.8 | **+9.5–10.7%** |

The 512–32K prefill rows ran a binary predating the per-sequence prefill fork
(an effect orders of magnitude inside the margins); the 65K row ran while
gates used a neighbouring card, so its margin is a floor; 128K ran
uncontended. Both decode rows are three same-hour alternating pairs (middle
pair reversed) on a quiet box, listed pair by pair. The prefill 2K llama.cpp
column is not a same-hour re-run, so allow for llama.cpp's ~1%/day drift.

### Capability

| | |
| --- | --- |
| Max context, one card | 131,072 positions, peak ~35.6 GiB of 47.27 |
| KV cache | binary16, 20 KiB/token over 10 attention layers |
| Recurrent state | fixed ~2 MiB per GDN layer per sequence, independent of depth |
| Parallel sequences | batched decode and flattened batch prefill at N=1–8 |
| Speculative decode | seven drafters (`ngram`, `ngram-simple`, `ngram-mod`, `ngram-map-k`, `ngram-map-k4v`, `draft-mtp`, `spec-dflash`), each bit-exact against plain decode; off by default. On `qwen35` `draft-mtp` is now a win at N=1 **and** N=3; it stays opt-in because on the shipped `UD-Q8_K_XL` file its extra layer and per-sequence draft cache do not fit beside three 128K-capable caches. See WHY. |

### K2-Horizon on one card

GPU 0, IFM's Q4_K_M and Q6_K files, engine at `90b2a35`, three alternating
process pairs per cell on an idle box, against
[MBZUAI-IFM/llama.cpp `42adf019`](https://github.com/MBZUAI-IFM/llama.cpp/tree/42adf019f76013dac873b5b43950d54d5ab27216)
(branch `model/K2Horizon`, sm_75) at its faster of `-ub 2048` and `-ub 4096`,
`-sm none -fa on -b 4096`, f16 KV. Synthetic token streams: timings, not
quality.

Prefill: N=1, cold cache; ± is the SD of three process means.

| quant | prompt tokens | engine tok/s | publisher tok/s | publisher ubatch | margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4_K_M | 512 | 2663.20 ± 11.47 | 1783.62 ± 10.36 | 2048 | **+49.3%** |
| Q4_K_M | 2,048 | 2953.03 ± 4.86 | 2565.55 ± 14.19 | 2048 | **+15.1%** |
| Q6_K | 512 | 2137.19 ± 20.70 | 1627.75 ± 5.53 | 2048 | **+31.3%** |
| Q6_K | 2,048 | 2439.53 ± 19.25 | 2405.35 ± 10.80 | 2048 | **+1.4%** |

Decode: greedy, 64 timed steps per process; the engine replays its CUDA
graph, the reference runs
[`tools/oracle/bench_decode.cpp`](../tools/oracle/bench_decode.cpp).
Throughput is aggregate across N.

| quant | starting context | N | engine tok/s | publisher tok/s | publisher ubatch | margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4_K_M | 512 | 1 | 85.8 | 76.9 | 4096 | **+11.6%** |
| Q4_K_M | 512 | 3 | 169.5 | 144.4 | 2048 | **+17.4%** |
| Q4_K_M | 2,048 | 1 | 81.9 | 73.4 | 2048 | **+11.7%** |
| Q4_K_M | 2,048 | 3 | 142.0 | 133.9 | 4096 | **+6.1%** |
| Q6_K | 512 | 1 | 72.0 | 65.2 | 2048 | **+10.4%** |
| Q6_K | 512 | 3 | 135.7 | 125.0 | 2048 | **+8.6%** |
| Q6_K | 2,048 | 1 | 69.1 | 62.7 | 2048 | **+10.2%** |
| Q6_K | 2,048 | 3 | 119.7 | 114.7 | 4096 | **+4.4%** |

Latency in ms; mean ranges span the process means, p95 ranges each process's
64 steps:

| quant | starting context | N | engine mean (range) | engine p95 range | publisher mean (range) | publisher p95 range |
| --- | ---: | ---: | --- | --- | --- | --- |
| Q4_K_M | 512 | 1 | 11.66 (11.62–11.68) | 11.65–12.06 | 13.01 (13.00–13.02) | 13.04–13.07 |
| Q4_K_M | 512 | 3 | 17.69 (17.69–17.70) | 18.07–18.11 | 20.77 (20.73–20.83) | 20.98–21.10 |
| Q4_K_M | 2,048 | 1 | 12.21 (12.18–12.23) | 12.23–12.48 | 13.63 (13.62–13.64) | 13.64–13.68 |
| Q4_K_M | 2,048 | 3 | 21.13 (21.09–21.19) | 21.59–21.62 | 22.40 (22.37–22.43) | 22.71–22.75 |
| Q6_K | 512 | 1 | 13.89 (13.87–13.92) | 13.90–13.97 | 15.35 (15.32–15.37) | 15.50–15.57 |
| Q6_K | 512 | 3 | 22.10 (22.06–22.13) | 22.43–22.56 | 23.99 (23.97–24.02) | 24.40–24.46 |
| Q6_K | 2,048 | 1 | 14.48 (14.45–14.50) | 14.49–14.54 | 15.96 (15.95–15.96) | 15.98–16.12 |
| Q6_K | 2,048 | 3 | 25.06 (25.02–25.08) | 25.33–25.48 | 26.16 (26.15–26.17) | 26.35–26.41 |

Q6_K prefill at 2,048 won its pairs by only +2.1%, +1.5% and +0.7%: parity to
keep. The engine uses sixteen-bit activation codes, the publisher `q8_1`.
N=3 decode peaks at **22.65 GiB** (Q4_K_M) and **31.46 GiB** (Q6_K) at 2K.
Both quants pass the publisher capture's gates at all 48 block boundaries and
final logits ([TESTING.md](TESTING.md#k2-projection-and-routing-gates)). Not
measured: HTTP time to first token, mixed contention, three-card serving,
deeper contexts.

## How these numbers are measured

- **Same card, same hour, alternating processes.** llama.cpp drifts about a
  point a day, cold-to-warm drift is ~1.3%, and cards differ by ~1.5% (GPU 2
  fast, GPU 1 the house standard).
- **At least three interleaved pairs; report the pairs, not the mean.** A
  change must win every pair. Arms of 208.7/206.3, 207.9/201.4 and
  207.3/205.3 average +1.79% but read about +1.1%: the 201.4 is a bad
  baseline reading.
- **A sweep only chooses where to put pairs.** One run per point resolves
  nothing below roughly 1.5%. An `ET = 4` win of +1.31% read
  +0.24 / −0.19 / −0.05% against a baseline built from the same commit.
- **Numbers need the whole box, not just the card.** Never bench two GPUs at
  once (~6% host contention). Inside a pairs window nothing else runs
  anywhere, correctness runs and builds included.
- **Re-measure after a rebase.** A +1.1% branch landed on a `main` that had
  meanwhile gone −5.0%.
- **CUDA events, not `Instant`**, inside a pass.
- **Kernel-level first, then confirm in the model.** A narrow bench and a real
  step disagree about cache state (`ld.global.cs` in WHY NOT).
- **`ncu` does not work here** (`ERR_NVGPUCTRPERM`). Use `nsys`,
  `cuobjdump -sass`, roofline arithmetic from the GGUF tensor directory, and
  ablation.
- **Rebuild through Cargo.** Running `target/release/<bin>` directly has twice
  benchmarked a stale binary.
- **Compare against llama.cpp's best settings, never its defaults:** `-ub 4096`
  for prefill at depth, `-fa 1` and backend sampling for decode. See
  "The baseline was llama.cpp's default, not its best" in WHY NOT.

## A second `qwen35moe` checkpoint

`ornith-ai/Ornith-1.5-35B-A3B`, a different finetune requantized to the
shipped mix ([MODEL.md](MODEL.md#serving-a-second-qwen35moe-checkpoint)),
shows the standing belongs to the engine, not one file. Same method as
Current standing.

| cell | Ornith | sd | llama.cpp | margin |
| :--- | ---: | ---: | ---: | ---: |
| prefill 512 | 3,322.8 | 0.00% | 2,910.3 | **+14.2%** |
| prefill 2K | 3,869.9 | 0.04% | 3,251.4 | **+19.0%** |
| prefill 8K | 3,563.2 | 0.17% | 3,180.0 | **+12.0%** |
| prefill 32K | 2,803.4 | 0.06% | 2,649.3 | **+5.8%** |
| prefill 65K | 2,218.4 | 0.11% | 2,152.3 | **+3.1%** |
| prefill 128K | 1,567.1 | 0.20% | 1,555.9 | **+0.7%** |
| decode 2K | 207.8 | 0.05% | 187.3 | **+10.9%** |
| decode 32K | 158.6 | 0.22% | 151.6 | **+4.6%** |

Every cell won all three pairs, including 128K prefill: 1,567.6 v 1,555.9,
1,570.0 v 1,555.8, 1,563.7 v 1,553.0. They were measured before the current Qwen3.6
decode re-measurement, so their decode margins are not comparable with it.
The 512 margin is quoted against llama.cpp's best of three (2,770.8–2,910.3,
`-c` ruled out); against its mean it reads +16.6%.

## Qwen3.8-27B (`qwen35`)

**Not comparable to the Qwen3.6 standing**: the dense model reads 15× the
feed-forward weight per token (17.11 B parameters against 1.13 B) and 3.2×
the KV. GPU 1, alternating processes, three interleaved reps, llama.cpp
same-hour; our spread is under 1.2% per cell. **Every cell is clear of
llama.cpp or level with it, on both files.**

| cell | file | llmcuda-rs agg | per slot | llama.cpp agg | per slot | margin |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| prefill 512, N=1 | UD-Q8_K_XL | 846.2 | 846.2 | 674.1 | 674.1 | **+25.5%** |
| prefill 512, N=1 | Q8_0 | 841.7 | 841.7 | 760.7 | 760.7 | **+10.6%** |
| prefill 512, N=3 | UD-Q8_K_XL | 889.9 | 296.6 | 749.7 | 249.9 | **+18.7%** |
| prefill 512, N=3 | Q8_0 | 892.4 | 297.5 | 851.5 | 283.8 | **+4.8%** |
| decode, N=1 | UD-Q8_K_XL | 18.11 | 18.11 | 18.12 | 18.12 | parity |
| decode, N=1 | Q8_0 | 19.37 | 19.37 | 19.34 | 19.34 | +0.2% |
| decode, N=3 | UD-Q8_K_XL | 47.82 | 15.94 | 43.58 | 14.53 | **+9.8%** |
| decode, N=3 | Q8_0 | 50.31 | 16.77 | 48.20 | 16.07 | **+4.4%** |
| decode, N=1, `draft-mtp` | Q8_0 | 35.1 | 35.1 | 19.40 | 19.40 | **+81%** |
| decode, N=3, `draft-mtp` | Q8_0 | 50.0 | 16.7 | 48.15 | 16.05 | **+3.8%** |

- `UD-Q8_K_XL` is the shipped Unsloth file (Q8_0, with bf16 `output.weight`
  and `attn_q`/`attn_k`/`attn_v`); `Q8_0` is Unsloth's plain Q8_0 (27.05
  GiB), whose logits have not been compared against llama.cpp's.
- llama.cpp's N=1 prefill on the shipped file spread 18.8%, so that row is
  three further alternating pairs; that re-run also turned N=1 decode's +1.1%
  into parity. Per slot is aggregate / N by construction.
- The `draft-mtp` rows are opt-in, not re-run with the rest, and not
  like-for-like: the engine drafts with the file's next-token-prediction head,
  which llama.cpp discards
  (`model has unused tensor blk.64.nextn.eh_proj.weight -- ignoring`).
- **Prefill wins** because dense prefill projections run on the fp16 tensor
  cores ("Arithmetic: on a dense projection, the fp16 tensor cores beat the
  integer ones" in WHY). llama.cpp's int8 `mmq` (Q8_0) and fp32 (bf16) paths
  make its two files differ by 13% at N=1; ours differ by 0.5%. bf16 still
  costs decode its bytes: 18.11 against 19.37 tok/s, 6.5%.
- **N=1 decode is parity to keep**: both engines are at the streaming wall.
  llama.cpp's best on this model is neither file (see quantization below);
  do not quote these rows against the MoE standing.

llama.cpp `abeada3` ran
`-b 2048 -ub 2048 -fa on -ctk f16 -ctv f16 -sm none -npp 512 -ntg 64` with
`-npl 1` or `-npl 3`; ours
is `bench_forward`'s 512-token column (`LLMCUDA_PREFILL_SEQUENCES=3` for N=3)
and `bench_decode_batch 512 64`. Peak VRAM 31.42 GiB of 47.27 at N=3 prefill
on the shipped file, 30.42 on Q8_0.

### `-ub` is not a knob on this workload

Three reps each, `-npl 1,3 -npp 512 -ntg 64`:

| llama.cpp config | PP N=1 | TG N=1 | PP N=3 | TG N=3 |
| :--- | ---: | ---: | ---: | ---: |
| default `-c`, `-ub 2048` | 693.1 | 18.05 | **744.7** | **43.98** |
| `-c 6144 -ub 2048` | 692.6 | 18.05 | 741.6 | 42.33 |
| `-c 6144 -ub 4096` | 692.0 | 18.05 | 742.9 | 43.89 |
| `-c 6144 -ub 512` | 692.2 | 18.03 | 674.3 | 42.65 |

At `-npp 512 -npl 3` the batch is 1536 rows, under the 2048 cap, so
`-ub 4096` has nothing to raise (at the default `-c` its 4.04 GiB compute
buffer does not fit). `-ub 512` loses 9% of N=3 prefill; decode
ignores `-ub`. So `-b 2048 -ub 2048` is llama.cpp's best here.

### Speculative decode: what the width band was hiding

`bench_worker_spec`, `all-Q8_0`, GPU 1, context 256, 512 tokens per sequence:

| drafter | N=1 tok/s | tokens/step | N=3 tok/s | tokens/step |
| :--- | ---: | ---: | ---: | ---: |
| none | 19.2 | 1.000 | 49.1 | 3.000 |
| `ngram` | 20.2 | 1.101 | 23.5 | 4.152 |
| `draft-mtp` | **35.1** | 2.860 | **50.0** | 9.350 |

**`draft-mtp` is +83% at N=1** (+21% for the same class of change on
`qwen35moe`): a dense verify step reads the same 27 GB for one token or nine.
`ngram` loses at N=3: its twelve-row verify pass has almost no acceptance.
N=3 won once both kernel families served the verify shape, a few rows
against a deep window:

- **FFN:** the GEMV now covers up to 16 rows instead of falling to
  `shared_expert_mma` above four: 1.374 -> 0.974 ms a layer at twelve rows,
  N=3 from 42.2 to 48.2.
- **Attention:** at twelve query rows the GQA-shared prefill kernel launches
  only `kv_heads` (two) blocks, 433 us a layer. Routing those rows through
  the flash-decode split took N=1 from 32.0 to 35.5 and N=3 from 48.2 to 50.5.

**Off by default for memory, not speed.** On the shipped `UD-Q8_K_XL` (2.25
GiB larger than `all-Q8_0`) the head's extra layer and per-sequence draft
cache make N=3 die in the Gated DeltaNet state allocation, with 34.99 GiB
free after the weights. Enabling it needs admission (AGENTS.md rule 4) to
count the drafter's reservation: scheduler work, not kernel work.

### A whittled MoE is the structural version of requantizing

`Qwen3.8-Whittle-MoE-27B-A17.8B` (community) swaps the dense MLP for 64
experts (16 active) plus a shared expert: ~17.8 B of 26.9 B parameters read
per token. llama.cpp, same card, three interleaved reps, Q8_0 except
Qwen3.6:

| model | GiB | PP N=1 | TG N=1 | PP N=3 | TG N=3 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-27B dense (`all-Q8_0`) | 27.05 | 792.3 | 19.25 | 851.9 | 48.37 |
| Qwen3.8-Whittle-MoE A17.8B | 26.71 | 665.5 | **25.73** | 889.5 | 51.51 |
| Qwen3.6-35B-A3B (Q6_K) | 30.36 | 2101.7 | **102.57** | 2912.1 | **194.19** |

Whittle buys **+33.7% decode at N=1** for −16% prefill (+6.5% at N=3) by
reading ~32% fewer bytes, at lower efficiency (72% of peak against 79%).
**Qwen3.6-35B-A3B is 4x Whittle and 5.3x the dense model on decode, and 3x on
prefill.**

This engine cannot load Whittle, though every kernel it needs exists:
`general.architecture = qwen35moe` maps to the hardcoded Qwen3.6-35B-A3B geometry, and the schema fails on about a
thousand shapes. **`general.architecture` names a family, not a model.**
Reading the geometry from metadata would be a design change.

### What the quantization is worth, and where the ceiling is

Decode reads the whole model per token: **27.93 GiB = 29.99 GB**, a
**22.4 tok/s ceiling** at 672 GB/s. llama.cpp's 18.11 is 542 GB/s (80.7% of
peak), our 17.2 is 516 GB/s (76.8%). Requantizing is the only large decode
lever (llama.cpp, three interleaved reps per file):

| file | GiB | PP N=1 | TG N=1 | PP N=3 | TG N=3 | achieved BW at N=1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| UD-Q8_K_XL (shipped) | 29.30 | 689.7 | 18.11 | 746.2 | 44.09 | 542 GB/s |
| all-Q8_0 | 27.05 | 779.0 | 19.41 | 837.2 | 48.22 | 538 GB/s |
| Q6_K | 20.89 | 669.3 | 22.13 | 674.0 | 52.64 | 463 GB/s |
| Q4_K_M | 15.66 | 714.1 | **29.51** | 718.2 | **60.11** | 453 GB/s |

**+63% decode from Q4_K_M**, despite less efficient streaming (453 against
538 GB/s: k-quant unpacking is integer-pipe work). **We cannot load either
k-quant**: the dense FFN accepts Q6_K, but the embedding gather, the LM-head
GEMV (`HeadFormat` is Q8_0 or bf16) and the attention projections do not, and
the build refuses them by name. The fastest way to serve
Qwen3.8-27B on this card today is llama.cpp at Q4_K_M.

### Where the dense decode step goes, and what the card can actually stream

`bench_dense_ffn`'s calibration kernel, a coalesced `uint4` read of the
resident weights, streams **604.5 GB/s, 90.0% of the 672 GB/s pin rate**:
that is the ceiling. One `bench_decode` step, `all-Q8_0`, N=1, from
`nsys --cuda-graph-trace=node`, bytes from the tensor directory:

| stage | launches | ms | weight bytes | GB/s | of 604.5 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| dense FFN GEMVs | 192 | 31.70 | 18.18 GB | 573 | 95% |
| GDN projections | 144 | 11.07 | 5.85 GB | 528 | 87% |
| attention projections | 64 | 3.36 | 1.78 GB | 530 | 88% |
| LM head | 1 | 2.23 | 1.35 GB | 606 | **100%** |
| GDN alpha/beta gates | 48 | 1.02 | 0.03 GB | 25 | 4% |
| everything else | ~596 | 2.60 | — | — | — |
| **step** | ~1,045 | **51.98** | 27.19 GB | 523 | **87%** |

- **The LM head is at 100% of calibration**; the GEMV families are at 87–95%
  (97% weighted). The shortfall is the tail.
- **llama.cpp is at the same wall**: 19.40 tok/s is 528 GB/s, 87%. The 1.0%
  between us is 0.43 ms of launch tail, not bandwidth.
- **The tail** is 2.60 ms over ~596 launches (5.0%). The gates are 1.02 ms of
  it for 30 MB (48 warps on 72 SMs; attempts in WHY NOT). Tail fusions that
  worked were ~0.13 ms each.

Attention projections and the LM head share `LmHeadKernels::forward`; merged
they read 94%, and only a duration histogram separated them.

### What the FFN residency is worth

Both layouts are 271 MiB a layer; holding both did not fit
(`CUDA_ERROR_OUT_OF_MEMORY`), so the residency picks the kernel. All rows
predate removing the split layout's fp32 scales and are kept for their
ranking; the first two are single runs, the shipped row a three-pair figure.

| residency | prefill 512 | decode | note |
| :--- | ---: | ---: | :--- |
| Q8_0, fp32 `moe_shared_ffn` everywhere | 76.2 | 15.74 | 9.1× behind llama.cpp on prefill |
| split int8, `shared_expert_mma` everywhere | 321.3 | 9.17 | prefill 4.2×; decode −47% |
| split int8, GEMV at ≤ 16 tokens (**shipped**) | 317.5 | 17.26 | both |

`shared_expert_mma` stages a 64-token tile and discards 63/64 of it at one
token: 63.5 ms became 109.0. The shipped GEMV reuses `GdnBlock`'s
split-layout projection GEMV and is the accurate path (fp16 activations):
`max_abs` 4.18e-3, cosine 1.000000 against the CPU reference, versus 6.45e-2
and 0.999997 for the int8 GEMM, and 4.62e-3 for prefill's fp16 GEMM.

### Greedy agreement with llama.cpp

Greedy (`--temp 0 --top-k 1`) against `llama-completion`, same file, 160
tokens, no chat template. Prompts: "The capital of France is" (5 tokens);
"Explain how a B-tree stays balanced when keys are inserted and deleted, and
why databases use B-trees for indexes." (25); a 917-token operations log
ending in a request to summarize it (the only one wide enough for the fp16
GEMM).

| prompt | UD-Q8_K_XL | Q8_0 |
| :--- | :--- | :--- |
| capital, 5 tokens | identical, 690/690 chars | identical, 690/690 chars |
| B-tree, 25 tokens | diverges at char 190 of 711 | diverges at the first token |
| operations log, 917 tokens | identical, 690/690 chars | diverges at char 82 of 707 |

Every divergence is a near-tie in llama.cpp's own logits:

| divergence | llama.cpp's top two | gap | we chose |
| :--- | :--- | ---: | :--- |
| UD, B-tree, char 190 | " standard" 19.938, " backbone" 19.921 | 0.017 | " backbone" |
| Q8_0, B-tree, first token | "Here" 14.115, "\n\n" 13.728 | 0.387 | "\n\n" |
| Q8_0, operations log, char 82 | "," 23.6089, " and" 23.6085 | 0.0004 | "," |

The engines' last-token logits differ by 0.13–0.43 max-abs, so smaller gaps
flip on rounding; llama.cpp breaks the third tie both ways itself. Our int8
prefill path (`LLMCUDA_HALF_GEMM=0`) takes llama.cpp's token in all three,
the fp16 path in the third: agreement in rounding, not accuracy, since the
fp16 path's error against the CPU reference is 14× smaller (TESTING.md).

## Clef-Flash (`clef`)

A decision model: a 9.1B `qwen35` trunk plus a head that scores each
question's options, served on `/v1/systemone`. One prefill, no decode:
**nothing here is comparable to the tables above**. Baselines: `llama-server`
(the 27B's build) and Cloudflare's PyTorch reference (`joint_schema_model.py`,
with `fla` and `causal-conv1d`). GPU 1, same requests, prompt tokens/s
(median latency); `Clef-Flash-Q8_0.gguf` for the servers, HF safetensors for
PyTorch:

| set | conc. | llmcuda-rs | llama-server | PyTorch fp16 | vs llama-server | vs PyTorch |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| short, 16 × 658–1,094 tokens | 1 | **2,552** (342 ms) | 1,884 (377 ms) | 1,809 (482 ms) | **+35%** | **+41%** |
| short | 4 | **2,550** | 2,345 | 1,638 | **+8.7%** | **+56%** |
| long, 8 × 3,012–3,512 tokens | 1 | **2,451** (1,376 ms) | 1,481 (1,409 ms) | 1,730 (1,977 ms) | **+65%** | **+42%** |
| long | 4 | **2,446** | 2,406 | 1,791 | **+1.7%** | **+37%** |

Short rows and PyTorch: median of three rounds, engine order rotated. Long
server rows: median of three alternating pairs, each won. PyTorch's conc. 4
is its batch of four.

- **Read the long rows by latency and c4.** llama-server re-reserves its
  graph for each new longest prompt (3.0–3.8 s against 1.2–1.5 s), inflating
  the c1 margin. At steady state it is **2.3% on latency and 1.7% on
  throughput**: parity to keep.
- llama-server serializes decisions:
  `-ngl 99 -fa on -c 20480 -b 16384 -ub 4096 -np 1`.
- PyTorch's best on Turing is fp16; bf16 has no tensor-core path here
  (446–460 tokens/s at batch 1, a quarter of fp16's).

Both servers pick the reference's option on all 99 questions; ours sits
closer to it on every summary (servers round to four decimals):

| set | vs | engine | median \|Δp\| | p99 | max |
| :--- | :--- | :--- | ---: | ---: | ---: |
| short | PyTorch fp16 | llmcuda-rs | 4.6e-4 | 1.2e-2 | 1.6e-2 |
| short | PyTorch fp16 | llama-server | 8.1e-4 | 2.8e-2 | 3.7e-2 |
| short | PyTorch bf16 | llmcuda-rs | 6.6e-4 | 1.5e-2 | 1.6e-2 |
| short | PyTorch bf16 | llama-server | 7.7e-4 | 2.7e-2 | 2.8e-2 |
| long | PyTorch fp16 | llmcuda-rs | 5.8e-4 | 6.2e-3 | 1.1e-2 |
| long | PyTorch fp16 | llama-server | 1.4e-3 | 1.0e-2 | 2.2e-2 |
| long | PyTorch bf16 | llmcuda-rs | 8.9e-4 | 6.1e-3 | 8.8e-3 |
| long | PyTorch bf16 | llama-server | 1.4e-3 | 1.4e-2 | 2.5e-2 |

The trunk alone is 323 ms at 877 tokens and 1,321 ms at 3,350, against
served medians of 342 and 1,376 ms; head, template, tokenization and HTTP
are 4–6% of a request. The prefill is the request (WHY: "A decision prompt is
one pass, at the width it is").

## Serving concurrent sessions

End-to-end HTTP through `/v1/chat/completions`, streamed: **not comparable to
the standing table.** `bench_server.py` (see Reproducing) reads prompt
lengths from `usage.prompt_tokens` and gives each prompt a unique leading
token to bypass the prefix cache (WHY NOT has why). Prefill rate is prompt
tokens over time to the last first-token; decode per-slot is 1 / median
inter-token latency. Trials agree within 0.5% unless noted.
`--spec-type none -c 405504 -s 3 -tb 4096 -pc 2048`, three slots per card.

### One card, prefill

Aggregate / per-slot:

| depth | 1 session | 2 sessions | 3 sessions |
| :--- | ---: | ---: | ---: |
| 16K | 2,409 agg / 2,409 slot | 2,350 / 1,175 | 2,290 / 763 |
| 32K | 2,191 / 2,191 | 2,196 / 1,098 | 2,154 / 718 |
| 64K | 1,785 / 1,785 | 1,784 / 892 | 1,709 / 570 |
| 128K | 1,310 / 1,310 | — | 1,310 / 437 |

Aggregate is flat across sessions. The 16K three-session dip (−4.9%) is the
kernel's own cost of interleaving three sequences (2,795 → 2,656 tok/s
measured separately), not scheduling.

### One card, decode

| depth | 1 session | 2 sessions | 3 sessions |
| :--- | ---: | ---: | ---: |
| 16K | 63.6 agg / 64.1 slot | 95.8 / 48.6 | 117.0 / 39.7 |
| 32K | 69.8 / 70.5 | 94.9 / 48.4 | 113.4 / 38.7 |
| 64K | 61.9 / 62.7 | 79.3 / 41.5 | 96.1 / 33.3 |
| 128K | 51.6 / 52.4 | — | 71.5 / 26.1 |

Aggregate rises with sessions because they share one weight read per step.

### Three cards, the nine-session target

| depth | 3 sessions | 6 sessions | 9 sessions |
| :--- | ---: | ---: | ---: |
| 16K prefill | 6,803 agg / 2,268 slot | 7,029 / 1,171 | 6,101 / 678 |
| 64K prefill | 5,290 / 1,764 | 5,281 / 880 | 5,071 / 564 |
| 16K decode | 202.3 / 68.4 | 297.4 / 52.2 | 368.6 / 42.5 |
| 64K decode | 193.7 / 66.8 | 244.4 / 44.0 | 304.0 / 35.0 |

Nine-session trials spread 6.1% (16K prefill) and 3.9% (64K decode); read
them as the pair, not the digit. At 64K nine sessions reach **2.97x** one
card's three-session rate on 3x the cards, first tokens grouped by card at
[93.9, 99.3, 108.5 x5, 110.7 x2].

### A session beside busy neighbours

One session decodes 200 tokens while 64K prompts prefill on the other two
cards:

| decoding session | mean | median | p90 | p99 | max | wall spent stalled |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| fleet stepped in lockstep | 146.0 ms | 12.1 | 958 | 1,579 | 2,959 | 92% |
| workers on their own loops | 11.8 ms | 11.6 | 12.1 | 14.2 | 21.1 | 0% |
| the same fleet, idle | 11.7 ms | 11.6 | 11.8 | 12.4 | 19.8 | — |

The median, 12.1 ms in every row, hides the cost: in lockstep 21 of 199
tokens absorbed 26.9 s of the session's 29.0 s. With per-worker loops it
matches an idle fleet, and the neighbouring prefills finished no slower.

### Prefix cache: coverage decides whether it hits at all

Six-turn conversations, three at once on one card; reuse is the server's own
`reused_prefix_tokens`.

| arena | snapshots/worker | coverage per slot | prefix reuse | wall |
| :--- | ---: | ---: | ---: | ---: |
| old fixed default | 24 | 16,384 | **0% every request** | 319.1 s |
| `--cache-ram 12GiB` | 119 | 79,872 | 73% → 86% | 133.2 s |
| current default | 198 | 135,168 (100%) | 73% → 86% | 132.2 s |

- **An undersized arena caches nothing.** Publishing yields rather than
  evicts, so a worker that cannot hold its sessions' retention points
  publishes nothing. The default is derived from the serving configuration.
- **The ~85% plateau is the retention quantum** (43,008 of 49,879 tokens is
  21 × 2048). Against the reusable prefix, reuse is 97.1% → 98.4%; the
  remaining 552–692 tokens per turn (about `R/2`) need token-granular reuse
  (see WHY on llama.cpp's in-place slots).
- **Oversizing buys nothing.** `--cache-ram 64GiB` on three cards pinned
  63.86 GiB (106% of the context) and took 60 s to healthy against 34 s at
  19.88 GiB, with identical reuse. `RLIMIT_MEMLOCK` (15.72 GiB) does not bind;
  the NVIDIA driver pins outside it.

### What this cost to get right

Two driver-loop lock faults dominated. Before their fixes, four sessions on
one card got first tokens at 34, 71, 136 and 203 seconds, and three sessions at 64K were bimodal
at 1,734 or 826 tok/s (both in WHY). The scheduler's step sharing across
prefilling sessions is worth +3.0% at 16K and +3.4% at 64K, with worst-case
time to first token 3.5% lower.

## Correctness gates

Every throughput number here was taken with these green. A speedup that flips
a logit ranking or needs a loosened tolerance is a reject regardless of size.

| Gate | What it asserts |
| --- | --- |
| `forward_pass` golden | Reproduces llama.cpp's argmax token 25358 (`' Tokyo'`) and its logit ordering on the captured 19-token prompt |
| `batch_decode` | A sequence decodes **bit-identically** whether batched with others or run alone — `max_abs 0.000e0` on every full 248,320-logit row |
| `batch_prefill` | A 6,138-row flattened N=3 pass is bit-exact against three serial 2,046-row passes |
| `attention_differential` | Nine cases including a 128K window, at `GATE` 1e-5 for scalar paths and `MMA_GATE` (8 × binary16 half-ulp) for tensor-core paths |
| `moe_differential`, `gdn_*_differential`, `lm_head_differential` | Every kernel against its CPU reference, plus exact cross-kernel identity where two compiled kernels must agree |

`0.000e0` batch-vs-single is a **serving contract**, not a tolerance: a
sequence's output must not depend on which others are resident. A ~1e-5
reduction-order residual upstream was measured flipping expert selection
(top-8 routing) and spiking a layer's divergence 26×.

---

# WHY

Why the engine is shaped as it is: each entry is a mechanism that paid,
stated so it transfers to the next kernel.

## K2 integer projections: one contraction order across shapes

Weights are packed at upload; activations become sixteen-bit integer codes
with an fp32 scale per 32 values. Decode uses DP4A, prefill the int8 tensor
cores.

- **GEMV and GEMM agree bit for bit**, including incomplete tiles: integer
  dots are exact in any order, and group terms, windows and the ascending
  window sum are the same fp32 operations (`(t0+t2)+(t1+t3)` per 128-value
  window). This is a new contraction, not bit-equal to the old fp32
  projection; CPU differentials and model oracle gates cover it.
- **Formats.** Q4 stays nibbles; Q6 stays six bits (four low-nibble and two
  high-bit words per 32-value group, each aligned with an activation word).
  Records interleave each word across four rows. K2 has its own residency
  calculation; the Qwen constant would misreport free memory.
- **Decode reuse.** One 16-byte load gives a lane one word of four rows, and
  its activation codes serve all four; Q6 removes its bias of 32 with the
  quad's shared code sum. A lane per row had made L1 activation traffic about
  four times the weight traffic; reuse halved a 2560×4096 Q4 projection at one
  token.
- **Prefill.** One kernel for Q4 and Q6, dense and routed. Activations are
  quantized in fragment order; high and low code planes each take one m8n8k16
  pair, combined exactly as high×256+low in int32. Load offsets are fixed
  outside the window loop; recomputing them had been about a sixth of each
  window's instructions. Replacing the nibble-plane kernels raised cold Q4
  prefill about 40% at 512 tokens.
- **Magic-number int-to-float.** `I2F` issues at a quarter of the FP32 rate on
  sm_75. Accumulators seeded with the bit patterns of 1.5×2^31 and 1.5×2^23
  yield the float of 256h+l after one exact subtraction and one exact add.
  Splitting the conversions between pipes raised Q6 prefill about 2%.
- **Q6 de-phased staging.** Ablation at 2,048 tokens found no binding pipe
  (FP32 epilogue 16%, epilogue plus MMAs no more, window barriers 15%): the
  phases ran serially. Double-buffered activations (58,752 bytes of shared
  memory) let two token warps unpack the next window on opposite sides of
  their products, overlapping integer and tensor/FP32 work. Q4 loses with
  this arrangement (WHY NOT).
- **Shared quantization.** An unchanged input is quantized once per workspace
  generation, graph capture included. FFN-down quantizes `silu(gate)·up`
  inside the quantizer, so SwiGLU is never stored or launched. Routed and
  shared-expert gate/up share one quantization, and one pass adds routed,
  shared and residual with the separate launches' rounding.
- **Grouped q/gate/k.** One prefill grid: 72 row blocks per token tile fill
  the 72 SMs, where 32, 32 and 8 each ended on a partial wave.
- **Routing.** Count, prefix and an order-preserving gather run on device into
  runs padded to 64, in buffers and grids fixed at construction. Sigmoid ties
  go to the lowest expert id; combining follows route order.
- **Attention.** Prefill keeps fp32 query and softmax-weight residuals around
  half operands and stages 32 keys per barrier pair for two 16-query tiles,
  so each K/V element serves 32 queries; one key octet for 16 queries had cost
  about 6% of cold Q4 prefill at 2,048 tokens. Decode uses compensated
  tensor-core attention with 32 fixed splits from 512 visible keys.

`bench_k2` times real tensors with CUDA events; kernel timing does not
establish serving throughput ([Current standing](#current-standing)).

## Rust kernel authoring can keep the existing launcher

The opt-in `llmcuda-cuda/rust-kernels` feature swaps eight kernels for
cuda-oxide PTX behind the existing cudarc ABI, allocations and graph capture:
residual add, standalone SwiGLU, sigmoid gating, softplus, standalone RoPE,
production tiled RoPE, RMSNorm and fused RMSNorm/SwiGLU. Toolchain pin:
[TOOLCHAIN.md](TOOLCHAIN.md), [`experiments/cuda-oxide`](../experiments/cuda-oxide).
CUDA C++ stays the default: every model comparison below changes sign across
pairs. Deep contexts, the dense model and three-card serving were not
measured.

Kernel tables: six alternating CUDA-event pairs on GPU 1, order reversed on
odd pairs, 100 graph-captured launches per interval; small shapes are
cache-hot. Model tables: three alternating prebuilt process pairs on GPU 1,
only the switch different, middle pair reversed, box quiet; cells are
pair 0 / 1 / 2. Register counts are offline CUDA 12.4 `ptxas -arch=sm_75`,
not the driver's JIT.

Residual add: NVRTC unrolls the grid-stride loop and cuda-oxide does not
(36 against 16 registers, no spills). The port is bit-exact, including signed
zero, subnormals, in-place aliases and graph replay.

| Elements | NVRTC us/launch, min–max | Rust us/launch, min–max |
| ---: | ---: | ---: |
| 2,048 | 1.716–1.767 | 1.484–1.515 |
| 6,144 | 1.721–1.755 | 1.500–1.531 |
| 1,048,576 | 23.859–23.940 | 23.817–23.879 |
| 8,388,608 | 183.419–183.807 | 180.163–180.305 |

Only 40 standalone residuals run per step, so the model pairs are parity:

| N=3, 2K per sequence | NVRTC tok/s | Rust tok/s | Rust change across pairs |
| --- | --- | --- | --- |
| Decode, 64 steps after 4 warmup | 217.8 / 216.8 / 216.5 | 217.3 / 216.8 / 216.7 | −0.23% / 0.00% / +0.09% |
| Prefill, 1,536 total chunk rows, 5 timed repetitions | 3073.34 / 3055.24 / 3057.28 | 3062.50 / 3059.73 / 3054.73 | −0.35% / +0.15% / −0.08% |

Prefill within-process standard deviations: 6.88–12.42 tok/s NVRTC,
7.21–9.79 Rust.

### Activation ports

`swiglu_mul` and `sigmoid_gate_mul` pass the existing CPU oracles: max-abs
**2.861e-6** (SwiGLU, a million elements) and **4.768e-7** (sigmoid, 262,144),
cosine **1.000000000**. The 19-token, 40-block golden selects token **25358**.
22 and 26 registers, no spills; 256-thread grid capped at 1,024 blocks.

| Kernel / shape | Elements | NVRTC us/launch, min–max | Rust us/launch, min–max |
| --- | ---: | ---: | ---: |
| SwiGLU | 12,288 | 2.115–2.151 | 1.882–1.899 |
| SwiGLU | 262,145 | 5.693–5.713 | 4.424–4.459 |
| SwiGLU | 8,388,608 | 182.156–182.255 | 186.574–186.675 |
| Sigmoid, elementwise | 12,288 | 1.556–1.592 | 1.434–1.452 |
| Sigmoid, elementwise | 262,145 | 4.182–4.219 | 3.418–3.438 |
| Sigmoid, elementwise | 8,388,608 | 253.953–254.505 | 252.453–252.785 |
| Sigmoid, per row, width 2,048 | 12,288 | 1.638–1.672 | 1.619–1.720 |
| Sigmoid, per row, width 2,048 | 262,145 | 4.585–4.618 | 4.356–4.380 |
| Sigmoid, per row, width 2,048 | 8,388,608 | 132.924–133.251 | 142.222–142.510 |

SwiGLU is fused in the model, so only the standalone attention sigmoid gates
change. Activation ports alone:

| Qwen3.6, N=3, 2K per sequence | NVRTC tok/s | Rust activations tok/s | Rust change across pairs |
| --- | --- | --- | --- |
| Decode, 64 steps after 4 warmup | 217.6 / 216.8 / 217.0 | 217.4 / 217.5 / 217.0 | −0.09% / +0.32% / 0.00% |
| Prefill, 1,536 total chunk rows, 5 timed repetitions | 3080.91 / 3059.20 / 3060.56 | 3071.82 / 3065.06 / 3059.15 | −0.30% / +0.19% / −0.05% |

Prefill standard deviations: 8.49–10.13 tok/s NVRTC, 6.84–8.23 Rust. Large
standalone SwiGLU and broadcast sigmoid regress.

### Rust normalization and rotary kernels need separate kernel and model gates

Standalone softplus, SwiGLU and layer-ops RoPE are API ports, not production
hot paths; fused batched-decode rotary/append and multimodal IMRoPE remain
CUDA C++.

RMSNorm's second reduction is now a parallel shuffle over the warp partials,
removing a block barrier; the changed addition order makes it not
bit-identical to the old kernel. Target-width max-abs errors against CPU were
**3.099e-6**, **1.192e-6** and **9.537e-7** (hidden, attention-head,
GDN-head), cosines **1.000000000**, at unchanged tolerances. The 19-token
golden passes but does not validate deep context. 1,536 rows are one prefill
chunk:

| Kernel | Rows × width | NVRTC us/launch, min–max | Rust us/launch, min–max |
| --- | ---: | ---: | ---: |
| RMSNorm | 3 × 2,048 | 3.391–3.422 | 3.297–3.322 |
| RMSNorm | 96 × 128 | 2.569–2.601 | 2.478–2.517 |
| RMSNorm | 48 × 256 | 2.722–2.743 | 2.642–2.658 |
| RMSNorm | 1,536 × 2,048 | 53.054–61.022 | 53.520–59.764 |
| RMSNorm | 49,152 × 128 | 100.803–101.132 | 99.779–100.161 |
| RMSNorm | 24,576 × 256 | 101.814–102.522 | 101.039–101.499 |
| RMSNorm/SwiGLU | 3 × 2,048 | 3.856–3.912 | 4.219–4.234 |
| RMSNorm/SwiGLU | 96 × 128 | 2.867–2.909 | 2.601–2.625 |
| RMSNorm/SwiGLU | 48 × 256 | 2.898–2.928 | 2.744–2.753 |
| RMSNorm/SwiGLU | 1,536 × 2,048 | 93.456–93.825 | 94.901–95.349 |
| RMSNorm/SwiGLU | 49,152 × 128 | 182.724–182.995 | 182.378–182.496 |
| RMSNorm/SwiGLU | 24,576 × 256 | 182.834–182.926 | 182.565–182.701 |

Decode RMSNorm gained **2.4–3.8%**; hidden-width prefill spread **−0.9% to
+9.1%**, not a stable win. Fused GDN normalization gained **9.8–11.0%** at
96 × 128, about 0.26 us per GDN layer or roughly 8 us across 30 layers against
a 13.8 ms batch decode step, and **0.16–0.27%** at its prefill shape. Generic
fused width 2,048 still loses.

Production `attn_rope_partial_neox` already reuses frequencies across 16
tokens. Its Rust port keeps that and, in grids of at most 72 blocks (the SM
count), puts idle rotary threads on alternate tokens at 8-token reuse; larger
grids keep 16-token reuse, since duplicated frequency work loses on query
prefill. Angles stay double precision. Timing used frequency base 10,000,000,
positions from 2,048, width 256 with 64 rotated dimensions:

| RoPE API | Tokens × heads | NVRTC us/launch, min–max | Rust us/launch, min–max |
| --- | ---: | ---: | ---: |
| Standalone | 3 × 16 | 7.086–7.137 | 6.615–6.630 |
| Standalone | 3 × 2 | 7.066–7.096 | 6.572–6.590 |
| Standalone | 512 × 16 | 322.540–325.626 | 181.100–181.752 |
| Standalone | 512 × 2 | 43.909–44.200 | 23.987–24.085 |
| Standalone | 1,536 × 16 | 955.752–955.843 | 517.571–517.996 |
| Standalone | 1,536 × 2 | 130.367–130.934 | 73.280–73.626 |
| Production tiled | 3 × 16 | 6.697–6.717 | 6.199–6.223 |
| Production tiled | 3 × 2 | 6.697–6.758 | 6.185–6.410 |
| Production tiled | 512 × 16 | 66.801–66.990 | 67.261–67.372 |
| Production tiled | 512 × 2 | 19.313–19.367 | 12.861–12.902 |
| Production tiled | 1,536 × 16 | 209.058–210.680 | 211.395–213.299 |
| Production tiled | 1,536 × 2 | 26.941–27.400 | 26.255–26.352 |

Production key RoPE at 512 tokens gained **49.8–50.5%** (about **6.5 us** per
call) while query RoPE was **0.4–0.8% slower**; at 1,536 tokens key gained
**2.2–4.0%** and query lost **0.4–2.0%**. The three-token cases gained
**4.8–8.9%**, but batched decode uses the fused rotary/append path.

Softplus keeps libdevice exp/log and the `x > 20` guard; it is not on the
production GDN gate path:

| Softplus elements | NVRTC us/launch, min–max | Rust us/launch, min–max |
| ---: | ---: | ---: |
| 96 | 1.298–1.335 | 1.131–1.176 |
| 49,152 | 1.597–1.611 | 1.393–1.413 |
| 8,388,608 | 129.293–130.004 | 126.403–126.484 |

All eight ports together:

| Qwen3.6, N=3, 2K per sequence | CUDA C++ tok/s | Eight Rust ports tok/s | Rust change across pairs |
| --- | --- | --- | --- |
| Decode, 64 steps after 4 warmup | 217.8 / 217.7 / 216.9 | 217.7 / 217.6 / 217.0 | -0.05% / -0.05% / +0.05% |
| Prefill, 1,536 total chunk rows, 5 timed repetitions | 3085.54 / 3065.17 / 3064.11 | 3074.83 / 3067.78 / 3061.60 | -0.35% / +0.09% / -0.08% |

Prefill standard deviations: **7.78–9.42 tok/s** C++, **6.86–11.02** Rust. The
kernel gains are a small fraction of the pass.

## Compile-time formats free registers that ptxas then spends

A runtime format argument in a `__forceinline__` MoE prologue emitted every
arm at all nine call sites; that is why three community formats cost 5.05%
(WHY NOT). The format is now a template parameter, `dequant_tile_ct<Q>`, and
removing the branch *raised* register counts, because ptxas spends freed
budget on unrolling and loads in flight:

    moe_expert_ffn        80 -> 96 registers   24 -> 16 warps/SM
    moe_expert_ffn_gemv   44 -> 59             40 -> 32 (both cap at 32)
    moe_shared_ffn_gemv   42 -> 57             40 -> 32 (both cap at 32)
    moe_shared_ffn_gemv_t3  64 -> 66           32 -> 24

**64 registers per thread is sm_75's full-occupancy budget at any block
size**, so only `moe_expert_ffn` and `_t3` lost blocks.
`__launch_bounds__(GEMM_THREADS_CU, N)` at 3 and 4 blocks per SM recovered
both with **zero spill stores and zero spill loads**; module compile went
3.13 s -> 3.82 s. `bench_decode_batch 2048 32`, GPU 0, three interleaved pairs
(middle reversed), against the preceding commit, not llama.cpp:

    N=1   +6.98%  +2.86%  +6.92%   (worst +2.86%)
    N=2   +1.58%  +1.46%  +0.98%   (worst +0.98%)
    N=3   +1.36%  +1.51%  +1.21%   (worst +1.21%)

N=1 spreads widely; read it as +2.9%, the base-first pair, as
first-runner-wins predicts. `__launch_bounds__(T, 1)` is not a
no-op: it relaxes ptxas's heuristic and made `_t4` and `_t8` worse. A
register count means nothing until converted to blocks per SM (granularity 8
on Turing, then the warp ceiling).

## Arithmetic: prefill belongs on the tensor cores

On this card `mma.m16n8k8` fp32-accumulate measured **97.6 TFLOP/s** and
`mma.m8n8k16` s8→s32 **~198 TOP/s**, against fp32 FMA's 16.3 TFLOP/s;
llama.cpp's int8 `mmq` runs a 512-token pass at ~61% of fp32 peak, and a
perfectly tiled fp32 engine lands ~3.9× behind. Quantized weights reach the
tensor cores one of two ways, chosen by VRAM:

- **Repack** into split quant/scale arrays (dense projections, shared
  expert): 27 TOP/s against 2.1 TOP/s assembling operands from GGUF Q8_0's
  34-byte blocks, which never word-align a fragment.
- **Stage through shared memory** (routed experts: 10.7 G weights, no room
  for a second copy).

The integer gate is per tensor. A layer-wide `weights.formats.all_q8_0()`
sent Qwen3.8-27B's whole attention block to fp32 GEMV, because `UD-Q8_K_XL`
stores `attn_q`/`attn_k`/`attn_v` as bf16 (requantizing the file to Q8_0 was
**2.05× and 2.12×** prefill at N=1 and N=3). Per tensor, `attn_output`
qualifies on `qwen35`: **+13.7%** prefill (316.0 → 359.3 tok/s, won 3/3), at
**+0.50 GiB** resident and an *unexplained* **−0.5%** dense N=3 decode
(43.87 → 43.63 tok/s; decode launches are unchanged). On `qwen35moe` the
predicates coincide (prefill 2452.4 → 2446.3, spreads overlap; peak VRAM
byte-identical). `LLMCUDA_ATTN_INT8_ALL_OR_NOTHING=1` restores the old one.

## Arithmetic: on dense projections, fp16 tensor cores beat integer ones

On the dense model's shapes the int8 split path (`mma_q8_0_proj_split`)
reaches **~44 TOP/s**, 22% of its peak; cuBLASLt fp16 sustains **62–83
TFLOP/s**. The per-32 weight and activation scales sit inside the
contraction, so every 8×8 accumulator is rescaled after every two `m8n8k16`,
on the MMAs' issue slots. `hgemm` dequantizes each staged weight tile to fp16
once in shared memory, reuses it across the tile's tokens, and accumulates
the whole K in fp32: 50–58 TFLOP/s at the 27B's 512-token shapes, the rest
being load latency.

Activations keep fp16's 11-bit significand; each Q8_0 weight rounds once
(2^-11 relative, against an int8 code's up to 1/254 of its block max).
Last-token logits on one 512-token prompt against llama.cpp, which rounds
activations to int8:

| file | path | max-abs | 1 − cos | argmax, top-10 |
| :--- | :--- | ---: | ---: | :--- |
| UD-Q8_K_XL | fp16 | 0.2399 | 8.69e-5 | agree |
| UD-Q8_K_XL | int8 | 0.2461 | 7.29e-5 | agree |
| Q8_0 | fp16 | 0.3080 | 1.26e-4 | agree |
| Q8_0 | int8 | 0.2669 | 1.00e-4 | agree |

Neither is uniformly nearer; `tests/hgemm_differential.rs` holds the kernel
to its own rounding contract in f64. bf16 converts exactly after a per-tensor
power-of-two shift (`bf16_half_exponent`), for all but **12 of 2.57e9**
values on `UD-Q8_K_XL`, each under 1e-12 off. `bench_forward` in-process A/B
(`LLMCUDA_BENCH_AB=LLMCUDA_HALF_GEMM`), six alternating pairs, GPU 1:

| file | tokens | int8 path | fp16 path | pairs |
| :--- | ---: | ---: | ---: | :--- |
| UD-Q8_K_XL | 512 | 365.1 | **849.1** | 2.31–2.34× |
| UD-Q8_K_XL | 1,536 | 359.4 | **817.8** | 2.27–2.29× |
| Q8_0 | 512 | 653.2 | **839.9** | 1.28–1.29× |
| Q8_0 | 1,536 | 649.8 | **817.9** | 1.26× |

The shipped file gains most because three projections had no tensor-core
path. Dense model only: `qwen35moe` keeps int8 by default, and
`LLMCUDA_HALF_GEMM=1` there is unmeasured.

## Memory: find the index the operand does not depend on

About half of every large win was a grid re-reading bytes its operand does
not depend on:

- The GDN projection read its weights once per token, 512× at prefill width.
- `gdn_chunk_inter` re-read the recurrent state per token.
- The MoE router loaded 2.1 GB per layer to cover 6 MB.
- Prefill attention gridded on 16 query heads re-streamed 2 KV heads' K/V;
  gridding on the KV head cut traffic 4–8×.
- The GDN scan gives a warp four value columns sharing q/k, decay and beta.

A token tile amortizes the weight, a row band the activation; the optimum is
interior, not at the biggest tile.

## Prefill attention: keep softmax values in their owning warp

`Q K^T` scores already have the fragment layout `P V` consumes, so each query
head's warp keeps scores, probabilities, maxima and normalizers in registers
with no shared-memory handoff. Even and odd key subsequences reduce separately
before a quad-leader broadcast, preserving the old addition tree. The score
product rounds explicitly and the normalizer uses explicit FMA, because
relying on contraction changed a model ranking (WHY NOT). 255 registers, 8,320
dynamic shared bytes, one block per SM. Gain: the
[full-model prefill pairs](OPTIMIZATION_CAMPAIGN.md#full-model-prefill-pairs).

## GDN input normalization: consume the row before leaving the block

At decode widths one block per head computes the input norm and feeds the
alpha/beta projection from shared memory; head zero also writes the
normalized row for the other projections. Accumulation order and the gate's
FMA/XOR tree are preserved, so outputs are bit-identical to the separate
kernels. It duplicates statistics per head, so it is selected only for the
2,048-wide, 32-head F32 geometry at N≤3 (57 registers, no spills). The
[model pairs](OPTIMIZATION_CAMPAIGN.md#normalization-full-model-pairs) show a
small decode gain, smaller than the isolated kernel's.

## Residency: one copy per projection, in its reader's form

The arena held the file's layout of every GDN and attention projection, which
nothing read (GDN reads its split int8 repack, attention its own copies). The
load filter `arena_holds_entry` keeps attention projections out in every
format and Q8_0 GDN projections out, repacking those from a transient upload
one tensor at a time (build peak one 83 MiB tensor above steady state). The
golden caught one width (single sequence, 2..63 tokens) still reading the
stored layout. `bench_forward`, 512 tokens, three prompts, driver accounting:

| file | arena | peak VRAM |
| :--- | ---: | ---: |
| `Qwen3.8-27B-UD-Q8_K_XL` | 11.822 → 3.658 GiB | 39.387 → **31.262 GiB** |
| `Qwen3.6-35B-A3B-UD-Q6_K_XL` | 2.291 → 1.025 GiB | 33.479 → **32.229 GiB** |

Kernels read the same bytes, so throughput was not measured.

## Residency: a transient must not share the pool with what stays

cudarc allocates from the device's stream-ordered pool. K2's projection upload
copied each tensor raw, repacked it into a wider record, and freed the copy:
every free left a hole between resident weights that the next, larger record
could not reuse and `cuMemPoolTrimTo` could not return, because each hole
shared a chunk with live data. The pool held 34.81 GiB to serve 32.00 GiB on
Q6_K. Taking the transient from `cuMemAlloc` instead (`with_unpooled_upload`)
returns it to the driver on free. One idle card after load, nvidia-smi:

| file | before | after |
| :--- | ---: | ---: |
| K2 Q6_K | 35,868 MiB | **32,988 MiB** |
| K2 Q4_K_M | 27,292 MiB | **25,244 MiB** |

Read VRAM as the pool's `RESERVED_MEM_CURRENT` against `USED_MEM_CURRENT`
before reading code; nvidia-smi cannot tell fragmentation from allocation.
The Qwen3.8-27B int8 repack and the attention int8 repack upload the same
way and have not been measured.

## Residency: a shape nothing runs is still resident

A pass never straddles a retention boundary, so outside a decision model no
prefill pass is wider than the retention interval (2048). The default
`--prefill-chunk 4096` still built a 4096-token pass at startup that nothing
chose: on K2 Q6_K its activation scratch was 2.20 GiB, by the pool's counters
around `Forward::build`. The runtime now builds its prefill shape at the
narrower of the two, which took 2,276 MiB off an idle card on either K2 quant
and 1,414 MiB on Qwen3.6. K2 also stopped allocating four gated-attention buffers
its mixer never reads, 136 MiB at 2048 tokens.

K2 keeps its input embedding in pinned host memory mapped into the device
(`forward::host_holds`): a step gathers one row per token across PCIe, and
the arena loses 0.49 GiB (Q6_K) or 0.34 GiB (Q4_K_M). A decode step pays one
round trip, about 20 us: within ±0.2% on K2's 14.6 ms Q6_K step over three
pairs, but Qwen3.6 lost 0.2–0.4% at N=1 and 0.3% at N=3 in every pair, so
its table stays on the card (WHY NOT).

One idle card after load at default flags, nvidia-smi, these two changes with
K2's narrower Q4_K records (Layout, below):

| file | before | after |
| :--- | ---: | ---: |
| K2 Q4_K_M | 27,488 MiB | **23,292 MiB** |
| K2 Q6_K | 35,264 MiB | **32,316 MiB** |
| Qwen3.6-35B-A3B Q6_K_XL | 34,670 MiB | **33,256 MiB** |

## The instruction-count bug that looks like a bandwidth bug

Turing issues **4 load/store operations per SM per clock against 64 FMAs**, so
a kernel at one memory instruction per multiply-add runs at a sixteenth of the
arithmetic pipe at any L2 hit rate. Flash attention, the GDN state update and
solve, and the alpha/beta gates all looked bandwidth-bound; attention reached
2 TB/s effective and was still 16× off. A register tile fixed all four and
left them **bit-identical**, because it reassociates nothing.

## Layout: banks and sectors

- **A shared stride that is a multiple of 32 words is an 8-way bank
  conflict.** The MoE tensor-core kernels moved activation staging from
  128-byte to 144-byte rows (36 words, `36 mod 32 = 4`): **+13% of prefill**.
- **Q6_K's 210-byte superblock is padded to 224 on the device**, aligning
  fields for wide loads and superblocks on sectors (212 aligns only the
  loads). It costs 6.7% more sector traffic on two tensors, a trade prefill
  can afford and decode cannot.
- **K2's records are whole words, and Q6_K's are whole lines.** Q4_K records
  dropped the per-group code sums only the 8-bit activation experiment read:
  148 bytes per 256 values and row instead of 160, 1,408 MiB less on an idle
  Q4_K_M card and 2.5–2.7% faster decode at 2K. Q6_K's 224 is not slack: four interleaved rows make 896
  bytes, seven 128-byte lines, and the prefill tile's 384-byte windows land
  on three lines each. At 212 or 216 they straddle a fourth (WHY NOT).
- **Q8_0 puts a row's quants at byte `b * 34 + 2`**: fifteen warp reads in
  sixteen straddle a 32-byte sector, discarding half of every fetch. The split
  repack fixes it; reading in place, two aligned 16-bit loads recover four
  signed bytes.
- **The split repack's scale width is traffic, not precision.** fp32 scales
  cost 1.125 bytes per element against the file's 1.0625, though a Q8_0 scale
  is an fp16; on `qwen35`'s dense FFN that was **1.01 GiB of every decoded
  token**. Keeping the fp16 bits is worth 4.2% of the dense FFN block at one
  token and 1.4 GiB of resident VRAM.

`cuobjdump -sass` counting `LDG.E.128` against scalar `LDG.E` shows whether a
runtime-derived stride or a per-element predicate blocked vectorisation.

## Decode is a GEMV, so a GEMM's machinery is overhead

At one token each weight is read once, so activation staging and barriers are pure cost. The step is launch-latency-bound: added stream work (events, graph nodes) loses at N=1 whatever it overlaps.

- **Skip the all-padding tile pass.** A `bm == 0` early-continue: **+21% at N=3**.
- **Route pairs directly at small widths.** At N=3, 24 token/expert pairs land on 23.26 distinct experts in expectation. A fixed grid over `max_tokens * top_k`, gated by the device-side `valid_tokens` scalar, beats bucketing and skips both dispatch-table launches.
- **Give every prefill-widened tile a one-unit sibling.** `TT` 8 at one token wastes 7/8 of its work, hence `_t1` and `_t3`. Halving the router's expert tile `ET` was worth nothing measurable (WHY NOT).
- **Select the kernel by width, not residency.** Qwen3.8-27B's dense FFN as a GEMM at one token cost 109.0 ms against a GEMV's 63.5 on identical weight traffic.
- **Merge small launches into one grid, not onto side streams** (side streams lost at N=1 or were flat; WHY NOT). A `blockIdx` seam runs each body unchanged and bit-identical in one grid, paying the launch floor once. Each merge below: three interleaved pairs at 2K on GPU 1, middle pair reversed, against the one before it.

`gdn_proj_split_pair_*` puts the GDN `qkv` and `gate` projections (1.78 and 0.89 waves alone) in one grid of 2.67; baseline arm flat:

| width | pair 1 | pair 2 | pair 3 |
| :--- | ---: | ---: | ---: |
| N=3 | 210.3 → 213.4 | 210.1 → 213.7 | 210.6 → 212.9 |
| N=1 | 109.3 → 111.1 | 108.9 → 111.3 | 109.3 → 111.7 |

**+1.1–1.7% at N=3 and +1.6–2.2% at N=1.** The second pointer set cost registers (one-token tile 80 → 96) until `__launch_bounds__(128, 6)` restored 80 without spill. The bounds that would help `t8` and `t16` spill, so those stay unbounded; check every tile of a family.

`moe_fused_{ffn,down}_*` run the shared expert in `blockIdx.y` slots past the routed grid; baseline arm drifting −0.5%:

| width | pair 1 | pair 2 | pair 3 |
| :--- | ---: | ---: | ---: |
| N=3 | 213.6 → 216.8 | 212.7 → 214.9 | 212.6 → 216.0 |
| N=1 | 111.9 → 114.7 | 111.1 → 114.2 | 111.2 → 113.9 |

**+1.0–1.6% at N=3 and +2.4–2.8% at N=1**; 80 fewer launches per step at N=1.

Three folds, each bit-identical with its differential: rope and KV append for every sequence in one launch per layer (`attn_decode_rope_append_batch`), the 256-expert dispatch prefix sum as a shuffle scan (`dispatch_token`), and GDN convolution, SiLU and q/k/v split in one launch (`gdn_conv_silu_split_step_batch`):

| width | pair 1 | pair 2 | pair 3 |
| :--- | ---: | ---: | ---: |
| N=3 | 216.5 → 217.4 | 215.0 → 218.0 | 213.5 → 217.0 |
| N=1 | 114.3 → 115.9 | 114.3 → 115.1 | 114.2 → 115.5 |

**+0.7–1.4% at N=1 and +0.4–1.6% at N=3**; the N=3 margin is inside the baseline arm's −1.4% drift and stands on the pairs. On the dense model the applicable merge and folds took `bench_decode_batch 2048 32` (`UD-Q8_K_XL`) from 17.8 → 18.0 tok/s at N=1 and 46.1 → 47.2 / 47.1 at N=3, two clean pairs. The remaining tail under ten microseconds is norms and gates between dependent projections; folding one into a producer's tail lost (WHY NOT, "The next layer's RMSNorm in the MoE combine kernel's tail").

## Parallelism: fill the wave, split the only axis decode has

Decode attention's `(1, q_heads)` grid was sixteen blocks on a 72-SM card. Flash-decoding (split keys across blocks, merge partial `(m, l, acc)`) was worth **114% at 8K depth**. Logical split count sets the numerics: 36 and 48 fail the deep-window 1e-5 gate at `2.28e-5`, 72 and 96 pass. Block count only schedules: at 96 splits, 48 blocks per call take two splits each and N=3's three calls fill all 288 resident block slots in one even wave. Per-sequence work on disjoint state forks onto side streams: 2–4%.

GDN kernels take `STEP_MAX_BATCH` state-pointer slots with `grid.z` picking the sequence, so one capturable launch advances every sequence bit-identically. The recurrent step went from 291 GB/s across three serial launches to **~583 GB/s (87% of streaming peak)**. One warp per state row, reducing with shuffles, replaced a block per row that ran at 44% of peak.

## Prefill: no GDN chunking, and wider passes

The chunked GDN form does **26% more arithmetic** at this geometry and pays only on tensor cores, which Turing lacks for fp32. Prefill runs an unchunked register-resident scan, as llama.cpp's `gated_delta_net_cuda` does: worth 4.5%.

Flattening the N=3 prompts into one pass wins only when it makes the pass wider than any one sequence, not at equal total ubatch: 6,138 rows at 2K, the whole 24,576-row prompt at 8K, 12,288 at 32K and above. This is the shape behind llama.cpp preferring `-ub 4096` over `-ub 2048`.

## A decision prompt is one pass, at its own width

A generated prompt runs as the largest prebuilt passes that fit, so pieces end where the prefix cache restores. A decision prompt (`clef`) is never resumed; decomposed that way, an 877-token prompt was eight passes, and the short Clef set ran 1,742 tokens/s against llama-server's 1,857. `Forward::run_prefix` runs a piece as the unpadded prefix of the narrowest prebuilt pass that holds it, bit-identical at and above the tensor-core tile (`tests/prefix_pass.rs`); narrower pieces are still decomposed.

Two decision-model defaults (Clef long set, three alternating pairs each, all the same sign):

- **The per-session slice is the prefill chunk**, since there are no snapshots to align to. A 2,048 slice → the chunk: 2,361 → 2,451 tokens/s at concurrency 1.
- **The step budget holds every slot's whole prompt**, since there are no decodes to protect. A 4,096 budget → whole prompts: 2,362 → 2,445 tokens/s at concurrency 4.

## Precision is a design constraint, not a tuning knob

Activations are fp32 end to end, every kernel gated against a CPU reference.

- **Decode `Q K^T` splits `Q`.** One fp16 rounding of `Q` for `m16n8k8` breaks the 1e-5 gate at `n_keys = 61`. Rows `g` and `g+8` carry `f16(q[g])` and `f16(q[g] - f32(q_hi[g]))`, so `D[g] + D[g+8]` is `Q K^T` at roughly `2^-22`. It costs 2× the tensor-core issue count, a permanent tax against llama.cpp's shape.
- **The router cannot reassociate.** A last-bit difference changes the top-8 of 256 experts, so the tiled router reproduces the untiled index order exactly, at a cost of 1.1%. A faster wrong version was caught by the golden test's error-growth guard.

`exp2f` in the prefill softmax is an identity (`log2(e)` folded into the score scale), and was still held to the full gate.

## KV cache: q8_0 is capacity, not a default

`-ctk q8_0 -ctv q8_0` stores K2-Horizon's cache in llama.cpp's `block_q8_0` arithmetic, 1,088 bytes per 1,024-element row against 2,048: 192 KiB per token becomes 102, and the Q6_K preflight at `-c 405504` falls from 107.43 to 72.63 GiB. The writers are byte-exact against a port of `quantize_row_q8_0_ref`; the readers turn each code into exactly `f16(q·d)` and meet the 1e-5 gate against CPU attention over the stored values. What remains is the rounding itself. `kv_quality` teacher-forces 2,048 tokens of this repository's docs and compares every position's distribution with the f16 run; the floor is f16 again with 64 decode splits, which only reassociates the softmax:

| K2 | K / V | perplexity | KL vs f16, mean / p99 | top-1 agreement |
| :--- | :--- | ---: | ---: | ---: |
| Q4_K_M | f16 / f16 | 5.7454 | — | — |
| | floor | 5.7376 | 0.0017 / 0.018 | 98.68% |
| | q8_0 / q8_0 | 5.7364 | 0.0072 / 0.054 | 97.02% |
| | q8_0 / f16 | 5.7447 | 0.0063 / 0.050 | 96.88% |
| | f16 / q8_0 | 5.7520 | 0.0051 / 0.046 | 97.22% |
| Q6_K | f16 / f16 | 5.7411 | — | — |
| | floor | 5.7333 | 0.0013 / 0.017 | 98.78% |
| | q8_0 / q8_0 | 5.7364 | 0.0072 / 0.059 | 96.39% |

Perplexity moves less than the floor moves it; the distributions do not. KL is four to six times the floor and one position in thirty picks a different top token instead of one in eighty, with K and V each carrying most of it alone. Expect greedy output to leave f16's path sooner; this measures that it differs, not whether the different text is worse.

Speed is a trade by shape (K2 Q4_K_M, GPU 0, three alternating pairs each, every pair agreeing in sign):

| shape | f16 | q8_0 / q8_0 | pair deltas |
| :--- | ---: | ---: | :--- |
| decode, 2K, N=1 | 81.0 | 81.0 | ±0.1% |
| decode, 2K, N=3 | 141.0 | 149.9 | +5.9 / +6.5 / +6.6% |
| decode, 32K, N=1 | 42.9 | 41.6 | −3.7 / −3.0 / −2.8% |
| decode, 32K, N=3 | 53.2 | 58.4 | +9.6 / +10.0 / +10.2% |
| prefill, 2048 chunk to 2K | 2958.8 | 2922.4 | −2.0 / −1.2 / −0.5% |
| prefill, 2048 chunk to 8K | 2263.1 | 2203.0 | −3.5 / −2.1 / −2.3% |

Three windows' worth of cache traffic is where halving the bytes pays: `bench_k2_attention` puts batched decode over three windows at 0.82× the f16 time at 8K and 32K. One window pays the conversion instead, and so does every prefill tile: nsys puts the 32K N=1 decode kernel at 248.4 µs against 235.9 in the model (the narrow bench had it at 0.96× — it is not a result there), and prefill attention 9% slower in total. The card also held 1,585 MHz against 1,645 at the same ~240 W under q8_0, so part of the conversion's cost is a clock cost. The format stays opt-in and per half: it buys context, and the price is paid in agreement with f16 before it is paid in speed. llama.cpp's own `q8_0` cache on this card is a [measured reject](#rejected-on-measurement) for the baseline and is unaffected by this.

## CUDA graph capture: kept for the host, not for throughput

Measured worth: ~0.4% at prefill shape, bounded above by 5.8% on a decode step. Capture cut idle per decode step from 0.97 ms to 0.405 ms and the wall clock did not move: under replay each of ~1,100 decode kernels ran 0.3–0.5 µs slower. On Turing a graph node costs about the launch gap it replaces; Ampere accelerates graph dispatch, so do not generalize forward.

It stays because it cut a decode step's host cost from 3.1 ms to ~0.05 ms, and because it forces launch shapes to depend on geometry rather than host values (design rule 5), which costs nothing now and would be expensive to reinstate.

## Vision does not touch the text path

Rotary base and cache slot share one position buffer, M-RoPE is bit-exact with the scalar kernel when t == h == w, and the prefix cache hashes a per-slot content lane in place of each `<|image_pad|>` id so different images cannot cross-hit. With vision off, the post-vision binary measured decode N=3 +0.5% (winning 2 of 3 pairs) and prefill at parity once ~0.8% position-in-session drift was controlled for.

## The snapshot arena is all-or-nothing, so size it from the configuration

Publishing yields rather than evicts when slots are scarce, so an undersized arena publishes nothing and the cache never hits. Three growing conversations on one card: 0% reuse against 73–86% once the arena covered the context; 319 s of wall clock against 132. The arena now holds one snapshot per retention interval, per slot, across the slot's share of `--total-context`, capped at a quarter of `MemAvailable` because it is page-locked; `--cache-ram full` lifts the cap.

llama.cpp keeps a slot's KV in place between requests; ours restores from host snapshots. A test-only resident-slot prototype cut follow-up TTFT 8.34–18.31% but whole-conversation time only 0.27–0.83% (six paired file-tool conversations); it is not in production routing. [All pairs, transfer bytes and scope](OPTIMIZATION_CAMPAIGN.md#exact-prefix-resident-continuation).

## An output ceiling is not a reservation

Admission reserves real pages for `prompt + max_output_tokens` for the sequence's life (rule 4). That is right for the prompt, but `max_output_tokens` is the caller's ceiling, not an estimate. Against one card's 1,584 blocks, three 100K-prompt sessions fit at `max_tokens` 4,096 but only two at 65,536. llama.cpp checks only the prompt and treats `n_predict` as a stopping condition. The output reservation is capped at one slot's share of the pool (floor one block); a sequence that reaches it stops with `length`. Refusals carry the scheduler's reason.

## Concurrency is a lock property before it is a scheduler property

Nine sessions across three cards failed to run concurrently for four reasons, none in `llmcuda-sched`:

- **The driver loop yields its lock.** Handlers raise an atomic before blocking and the loop stands aside 1 ms, instead of retaking the lock at once and starving them. Three 64K sessions went from bimodal (1,734 or 826 tok/s) to 1,635–1,708, first tokens spread 2.2%.
- **A step is shared.** Each prefilling session gets at most `--prefill-slice` tokens (the retention interval) per step, from a rotating start: +3.0% at 16K and +3.4% at 64K aggregate prefill, worst-case time to first token 3.5% lower.
- **Workers step independently**, one lock and driver thread each: 4.7–12.1% on aggregate decode and 0.6–16.2% on aggregate prefill across the nine-session table.
- **Routing stays atomic.** Without a lock across score-and-admit, concurrent handlers picked the same idle card: nine 64K sessions placed 4/2/3, the fourth waiting 150 s to first token against 113 s. The routing lock is never held over a GPU step.

## Speculative decode: exact arithmetic before acceptance

Seven drafters (five n-gram policies, MTP, DFlash) feed one batched verify pass, and acceptance requires equality with the target, so verify reproduces decode arithmetic at every width: exact four-row GDN projection slices, FP32 attention projections, and per-query decode attention even past prefill thresholds. Regression tests cover three 49-row windows, mixed acceptance and recurrent rollback.

GPU 1, three interleaved, reversed rounds; N=3 uses fixed scheduler steps and rejects any retired sequence. Synthetic-prompt continuations, not a representative workload. Aggregate tokens/s; changes are the range across pairs ([every observation](OPTIMIZATION_CAMPAIGN.md#speculative-decoding-after-the-kernel-changes)).

| Initial context / width | Policy (draft cap) | Decode mean | Paired change |
| --- | --- | ---: | ---: |
| 256 / N=1 | none | 118.0 | control |
| 256 / N=1 | map-k (48) | 130.3 | +10.37–10.43% |
| 256 / N=1 | MTP (3) | 125.0 | +5.61–6.28% |
| 256 / N=3 | none | 219.6 | control |
| 256 / N=3 | simple (48) | 149.0 | −32.06–32.27% |
| 256 / N=3 | map-k4v (8) | 101.3 | −53.80–53.92% |
| 256 / N=3 | MTP (3) | 168.7 | −23.13–23.25% |
| 32,768 / N=3 | none | 165.8 | control |
| 32,768 / N=3 | simple (48) | 165.3 | −0.24–0.48%; no extra emitted tokens/step |
| 32,768 / N=3 | map-k4v (8) | 117.7 | −28.91–29.07% |
| 32,768 / N=3 | MTP (3) | 127.5 | −22.93–23.25% |
| 122,880 / N=3 | none | 100.6 | control |
| 122,880 / N=3 | map-k4v (8) | 102.0 | +1.29–1.49%; full acceptance |
| 122,880 / N=3 | MTP (3) | 89.3 | −11.14–11.23%; full acceptance |
| 122,880 / N=3 | simple (48) | — | out of memory |

The verify-step cost sets the sign. At 120K map-k4v emits all 27 available tokens per step and wins narrowly, its ~265 ms step costing about 8.9 times a plain 30 ms step; MTP's ~134 ms step outweighs its fourfold output (12 tokens/step). Shallow N=3 lengths diverge over 512 steps (map arm 2,439 / 512 / 3,633 against 512 / 512 / 512). MTP slows 32K prefill to 2,260.1 tokens/s against 2,410.8. GDN snapshot rings grow with the draft cap, so simple's 48-token cap fails allocation at 122,880 tokens. DFlash is not measured (model file absent). The default remains `--spec-type none`.

---

# WHY NOT

Everything below was built or derived far enough to measure, and rejected. It is recorded so it is not re-attempted; several have been proposed twice.

## Rejected on measurement

Each row gives the decisive measurement and why it lost.

| Attempt | Result |
| --- | --- |
| Dense K2 cuBLAS with materialized half operands | Half-rounded dense weights and inputs fail three-sequence batch prefill at **0.31854** logit max-abs (limit **0.02**); high/low half operands with sixteen-bit inputs still fail the 42-token full-width/chunked comparison at **0.30180**. Rejected on correctness before any throughput claim. |
| Exact twelve-bit K2 dots on half tensor cores | Exact in fp32 and bitwise-tested, but the 128-row/64-token prototype reports **276–324 bytes** of spill stores. Split over 512 threads, 512-token query/value still take **1.332/1.428 ms** (Q4) and **1.401/1.793 ms** (Q6), above compact integer calibration. Rejected on timing. |
| Direct register staging for dense Q4 and Q6 | GPU 0, 512 tokens: Q4 query **0.748 ms** vs shared-staged **0.681 ms**; Q6 query/value **1.008/1.412–1.433 ms** vs roughly **0.945/1.05 ms**. Q4 routed values improve, so direct staging is kept for that path only. Rejected for dense and Q6. |
| Smaller Q4 register tiles in both dimensions | GPU 0 routed values at 512 tokens: 128 × 16 **0.683–0.715 ms**, 64 × 32 **0.728–0.774 ms**, 64 × 16 **0.835–0.922 ms**, vs **0.590–0.594 ms** for 128 × 32; also slower at 2K. Fewer live accumulators do not repay repeated weight traffic. Rejected at calibration. |
| Smaller Q6 expert tiles based only on the value benchmark | 32 columns × 64 rows: 512-token routed values **0.919–0.974 ms**, but 2K costs **3.264–3.610 ms**; full-model Q6 at 512 tokens **1,096.91 ± 2.48 tok/s** vs **1,144.49 ± 0.43** for 64 columns. The value-only gain does not survive real routing. Rejected; keep 64 columns. |
| Reusing both Q4 nibble halves inside routed decode | Helps N=1 values (**22.9–23.0 us**) but costs **72.2–72.4 us** at N=3, vs the original ordering's **47–54 us**. Rejected before numerical and model gates. |
| Explicit block-residency bounds on K2 decode | GPU 0, N=3, eight-block bound: Q4 query/value **42.8–43.0/60.8–60.9 us**, Q6 **49.9–50.1/75.2–75.5 us**, vs unrestricted **43/47–54** and **47/62 us**. Four- and sixteen-block bounds also fail to improve the routed path. Rejected. |
| Four lanes per K2 decode scale group | GPU 0, N=3: Q4 query/value **55.4/79.8 us**, Q6 **68.9/106.8 us**, vs control **43/47** and **51/62 us**. Shared dots need extra intra-group reductions before the fixed fp32 tree; the Q4 N=1 query gain (**27.4 us**) does not offset this. Rejected. |
| Precomputed Q4 scale products and byte-expanded Q6 records | 192-byte Q4 and 276-byte Q6 records move unpacking to upload. GPU 0, N=3 query/value: Q4 **43.7/54.9 us**, Q6 **53.0/71.0 us**. Q4 routed prefill improves in calibration, but decode does not repay the extra residency. Rejected before full-model gates. |
| 128-token dense K2 nibble tiles | The 128-column Q4 tile uses **48 KiB shared memory**; at 512 tokens it takes **0.9602/0.9599/0.9618 ms** vs **0.681 ms** for 64 columns. More reuse does not offset the resource cost. Rejected. |
| Unrolling fixed-width K2 decode loops | GPU 0, N=3, unroll factors 2/5/10: Q4 query **44.9/46.0/282.4 us** vs **43.0 us** at factor one; Q6 **51.6/56.3/65.0 us** vs **47.1 us**. Statically specialized routed Q6 also loses (N=1 **41.8 us** vs near **33 us**). Rejected; routed widths stay dynamic. |
| Staging the whole dense K2 activation once | One shared-memory copy per block on the 2,560-wide dense path. GPU 0, N=3 query: Q4 **63.5–63.7 us**, Q6 **50.0 us**, vs unstaged **43/47 us**. Staging and residency cost exceed the savings. Rejected. |
| K2 Q6 low-nibble and high-two-bit planes | Three alternating GPU 0 pairs, N=3: query **54.17 / 54.27 / 54.10 us** vs packed six-bit **50.87 / 50.90 / 51.10 us**; routed values **66.37 / 66.57 / 67.67 us** vs **62.47 / 62.70 / 62.53 us**. Separate planes need two distant word loads per group. Rejected; see WHY. |
| K2 Q4 records in consecutive eight-nibble words | Three alternating GPU 0 pairs, N=3: query **49.37 / 49.50 / 49.40 us** vs **45.77 / 45.80 / 45.80 us**; routed values **58.13 / 52.67 / 52.90 us** vs **48.67 / 48.70 / 50.70 us**. Tensor-core staging gets simpler, but decode loses group-contiguous loads. Rejected; the native layout stays. |
| Double-buffered K2 tensor-core staging | Two 31 KiB Q4 stages (62 KiB shared). GPU 0: `attn_q` **412.6/412.2 us** at 512 tokens and **1526.8/1513.9 us** at 2,048 vs single-stage **362.2** and **1342.3/1342.0 us**; routed `ffn_gate` **437.0/439.0 us** vs **405.6/403.2 us**. Q6's 39 KiB stage cannot be doubled within 64 KiB. Rejected. |
| Sixteen row warps per K2 router block | Sixteen warps per block instead of four; bit-identical. nsys at 2,048 tokens, whole pass: **20.60 ms** (Q4) and **20.25 ms** (Q6) vs **20.86** and **19.69 ms**. The F32 router projection is not bound by re-reading activations. Reverted. |
| 32-token K2 router blocks | Four (or two) token warps per block, cutting the L2 weight stream fourfold; bit-identical. nsys, Q6 2,048-token pass, GPU 1: **12.35 ms** (four) and **12.14 ms** (two) vs **12.16 ms** (one). Weight re-reads do not bind either; L1 load throughput per FMA was not measured. Reverted. |
| FP32-pipe conversion for every K2 Q6 term | Removes `I2F` from the Q6 epilogue; exact. Full model, GPU 0: Q6 **1911.72/1895.33 tok/s** at 512 tokens and **2175.50/2161.89** at 2,048, vs **1980.31/1948.40** and **2255.26/2224.55** with one conversion per pipe. Two adds per conversion overload the FP32 pipe. Rejected; the split stays. |
| No subtile skip in dense K2 tensor-core tiles | Every 8-token subtile computed, padded tail included, to drop per-subtile branches. Same rounds: Q6 **1947.31/1938.97** and **2220.03/2212.80 tok/s** vs the shipped kernel's numbers in the row above; Q4 flat (**2500.02/2501.65** and **2769.18/2767.32** vs **2508.76/2487.25** and **2766.21/2751.43**). Rejected. |
| 64-row K2 tensor-core tiles, two blocks per SM | Full model, GPU 0: Q6 **1658.29/1657.12 tok/s** at 512 tokens and **1893.67/1888.49** at 2,048, vs **1994.39/1986.69** and **2276.96/2269.38** for 128 rows; Q4 **2098.68/2092.06** and **2318.56/2316.96** vs **2557.57/2563.93** and **2834.84/2829.07**. Half the rows doubles activation staging per MMA; the second block does not hide it. Rejected. |
| Magic conversion in only some K2 Q6 scale groups | Two (or one) of four scale groups convert via the FP32 pipe, the rest via `I2F`. Full model, GPU 0: two groups **2003.09/1997.90 tok/s** at 512 and **2301.05/2297.58** at 2,048; one group **1980.11/1979.35** and **2279.68/2278.83**; all four (shipped) **2000.75/1981.36** and **2282.43/2266.92**. Two groups is within spread; one loses. The uniform loop stays. |
| De-phased staging in K2 Q4 tensor-core tiles | Q6's de-phased, double-buffered arrangement applied to Q4 (50,000 bytes shared). Full model, GPU 0: **2590.61/2572.80 tok/s** at 512 tokens and **2872.95/2867.13** at 2,048, vs **2661.70/2656.46** and **2942.16/2945.18** single-buffered. Q4's nibble staging is too light to repay it. Rejected. |
| Q4 sixteen-bit dots using exact half digits | 64 × 64 HMMA prototype on raw Q4 codes and high/low activation bytes as half. GPU 0, 512 tokens: query **1.109–1.208 ms**, routed values **1.955–2.014 ms**, vs integer **0.620–0.647 / 0.898–0.913 ms**. More shared-memory traffic and lower-throughput tensor instructions. Rejected on timing. |
| Interleaved high/low activation words in K2 GEMV | GPU 0, Q4, N=3: `ld.global.v2.u32` loads take **63.3–63.4 us** (query) and **72.3–72.7 us** (routed values) vs separate byte planes **42.9–43.8 / 53.9–54.5 us**. Fewer load instructions, but double the stride between neighbouring scale groups, plus packing work. Rejected. |
| K2 Q6 prefill row tile 128 | GPU 0, 512 tokens: query **0.997–1.083 ms**, routed values **1.458–1.545 ms**, vs 64-row **0.860–0.935 / 1.096–1.130 ms**. **39,168 shared bytes and 177 registers** allow one block per SM; relaxing the launch bound does not recover the loss. Rejected. |
| Fully scaled f16 operands for K2 projections | HMMA on dequantized weights and half activations. N=1 Q4/Q6 query **81.5 / 75.3 us** vs integer **29.1–29.4 / 33.8–33.9 us**; best routed N=3 **166.6 / 218.4 us** vs **49.3–50.0 / 62.4–62.7 us**. At 512 tokens dense query improves (**0.547 / 0.618 ms**); routed values **1.031 / 1.128 ms** vs **0.898–0.925 / 1.135–1.141 ms**. Every decode tile loses. Rejected on timing. |
| Reusing K2 decode rows across matching routes in one CTA | Q4 at three tokens with identical expert ids: **77.9 / 77.8 / 77.8 us** vs the direct path's distinct-route **53.1–53.6 us** (different inputs, not an end-to-end A/B). One larger per-thread accumulator loses more parallelism than it saves in unpacking. Rejected. |
| Asymmetric eight-bit K2 activation projections | Q4_K_M with attention/value and routed FFN at eight bits fails the six-token final-norm gate (**cosine 0.999169**, required **0.9995**). Q6_K all-eight passes, but calibration gives **20.83 / 37.90 ms** (N=1/3, context 512) and **887.15 ± 1.57 tok/s** at 512-token prefill. Sixteen bits stay. |
| Zero-centered Q8_1 arithmetic for K2 projections | Passes six-token captures, fails the 26-token science capture: Q4 logit **max-abs 0.9601** (gate **0.75**), Q6 block 28 **max-abs/RMS 0.4033** (gate **0.3**). Matching MMQ's sum tree still fails (Q4 **0.9264**; Q6 final-norm **cosine 0.9992134**, gate **0.9995**). Half Q4 coefficients fail even six tokens (**0.4350**). Rejected as production precision. |
| K2 routed token tile 128 instead of 64 | GPU 1, three pairs, 64-row/128-token vs 128-row/64-token: Q4 routed values at 512 tokens **2.6729 / 2.6539 / 2.6241 ms** vs **1.0065 / 1.0413 / 1.0066 ms**; at 2K **4.7136 / 4.8219 / 4.7916** vs **2.5676 / 2.6749 / 2.6096 ms**. One resident CTA, and sparse expert runs pad more. Rejected. |
| Predicate Q4 nibble staging on live rows and load coefficients separately | Three alternating GPU 0 pairs, 512 tokens, ordinary → candidate: query **0.6678 / 0.6999 / 0.6978 → 0.6929 / 0.7170 / 0.7253 ms**; routed values **0.9289 / 0.9687 / 0.9732 → 1.0271 / 1.0497 / 1.0670 ms**. Fewer redundant metadata loads lose every pair. Rejected before the full-model gate. |
| Qwen3.6's input embedding in mapped host memory | 0.50 GiB of VRAM. Three alternating GPU 0 pairs, `bench_worker_decode` at 2K: N=1 **−0.5 / −0.2 / −0.1%**, and with each row first staged through shared memory so a gather is one PCIe round trip, N=1 **−0.4 / −0.2 / −0.2%** and N=3 **−0.3%** in all three. The cost is the round trip on the step's critical path, not the gather's loads; it vanishes in K2's longer step, so only K2 does it. |
| K2 Q6_K records at 212 or 216 bytes (from 224) | 1.56 GiB less idle on Q6_K at 212 (1.05 GiB at 216 by the record arithmetic). Three rotated rounds, GPU 0: decode at 2K **+0.9–1.0%** (N=1) and **+1.3–1.9%** (N=3) at 212, but prefill to 8K **−0.5 to −0.9%** at both, and the standing cell's shapes at 212 **−0.0 to −1.4%** (512) and **−0.5 to −1.5%** (2,048) — the thin Q6_K 2K cell. Alignment to 32-byte sectors (216) did not help, 128-byte lines did: the prefill tile stages 384-byte windows of a four-row record group. Kept at 224. |
| Expanding K2 Q6 weights to byte codes | 304-byte records hold **44.57 GiB** at 512-token prefill on a 47.27 GiB card, vs **36.51 GiB** for the 240-byte packed format, leaving little cache capacity. Do not trade the context pool for an unpacking shortcut without an end-to-end measurement. |
| Tiny tensor-core K2 dense decode | GPU 2, Q4 query: eight-token/32-row MMA tiles take **161 / 172 us** at one/three tokens vs vector **31 / 47 us**. Most columns are padding; tensor-core setup cannot be amortized. Rejected. |
| DP2A for K2 sixteen-bit activation digits | Q4 query **33.3 / 65.5 us** at one/three tokens vs DP4A **31.1 / 46.6 us**; correctness gates pass. The wider packed operand does not repay its unpacking and instruction cost. |
| Smaller K2 Q4 prefill row tiles | Q4 query at 512 tokens: 64-row tiles **806.7–832.7 us** vs 128-row **689.5–699.4 us**. A 32-token/64-row routed-value tile (**1004.9–1015.0 us**) does not justify the dense loss. Keep the wider dense tile. |
| Sixteen-bit K2 digits rounded to half coefficient products | 26-token heat-transfer prompt: Q4 logit error **0.8241**, cosine **0.9994492**, vs six-token gates **0.75 / 0.9995**; the sixteen-bit control also misses (**0.8021 / 0.9994252**). Imitating one publisher rounding detail does not establish agreement; six-token captures do not certify longer prompts. |
| Eight-token dense projection tiles at K2 decode width | Three tokens, GPU 0, layer-3 query weights, three rounds: Q4_K four-token tiles **50.2 / 50.1 / 50.1 us** vs eight-token **58.8 / 58.9 / 58.7 us**; Q6_K **36.5 / 36.4 / 36.4** vs **40.8 / 40.8 / 40.8 us**. Keep four-token tiles below eight physical tokens. Rejected. |
| Grouping K2 value experts at decode width | Layer-3 value stacks, top-4 routes, dispatch included, GPU 0. Three-token scalar/grouped: Q4_K **115.7/126.1, 114.6/123.2, 112.0/122.9 us**; Q6_K **98.1/120.2, 98.0/120.3, 98.3/120.5 us**; one token also loses. Padded dispatch and a second launch outweigh row reuse. Grouping starts at eight physical tokens. |
| Cooperative loading before an exact-order GDN gate contraction | Bit-exact at N=1/3/7; registers 64 → 45, but N=3 latency rises **3.857/3.869/3.868 → 4.237/4.245/4.245 us**. Rotating 30 weight pairs: **5.241/5.273/5.272 → 5.008/5.033/5.041 us**, still slower than one head per block. Gates total only 0.181 ms/step. Rejected. [All pairs](OPTIMIZATION_CAMPAIGN.md#exact-order-gdn-gate-experiments). |
| One persistent block per GDN value head for normalization and recurrent update | Exact; **62 registers**. The N=3 grid collapses from 3,072 blocks to 96 (32 on 72 SMs at N=1). Three pairs: N=1 loses **30.6–35.2%** (one state bank) and **21.6–21.7%** (30); N=3 loses **1.20–1.66%** with one bank, gains **2.01–2.35%** rotating 30 (cache-dependent). Rejected. [All pairs](OPTIMIZATION_CAMPAIGN.md#block-persistent-gdn-fragment). |
| Query/key head RMSNorm in one disjoint grid | Exact; isolated N=3 fragment **72.9–73.8%** faster. Three GPU 1 model pairs: 2K favors it by **0.09/0.28/0.14%**, but 32K changes sign (**166.7/166.2/166.6 → 166.5/166.6/166.2 tok/s**, −0.12/+0.24/−0.24%), inside the 0.30% baseline spread. Reverted; no prefill claim. [All pairs](OPTIMIZATION_CAMPAIGN.md#head-normalization-full-model-pairs). |
| One head per block for small target-model F32 GDN gates | 96 one-warp blocks instead of 24 four-warp blocks at N=3; the isolated sign depends on weight reuse. Three GPU 1 model pairs at 2K: N=3 **216.8/216.1/215.7 → 216.6/215.8/215.5 tok/s** (−0.09/−0.14/−0.09%); N=1 gains 0.18–0.26%, below 0.4–0.5% drift. Reverted. [All model pairs](OPTIMIZATION_CAMPAIGN.md#gate-grid-full-model-pairs). |
| Relying on implicit FMA contraction after moving prefill softmax state into registers | Passed all ten attention differentials, then failed the 19-token forward golden at **rank 4** (separation **0.153701**, top-logit error **0.088937**). NVRTC moved a normalizer addition across MMA control flow, splitting an FMA; explicit `__fmaf_rn` restores bit identity. Preserve the compiled arithmetic; isolated tolerance gates cannot replace the model golden. |
| Alternating shared K/V buffers in prefill attention | Removes one barrier per 8-key tile; shared memory 13,952 → 22,272 B; all gates passed. Three GPU 1 N=3 pairs: 2K **−0.13 / +0.23 / −0.14%**, 8K **+0.04 / +0.02 / +0.05%**, inside 8K's 9.80–23.47 tok/s within-process SD. Fewer barriers bought no full-prefill throughput. Reverted; above 8K not measured. |
| Applying two token groups to every Rust tiled RoPE grid | Six alternating GPU 1 pairs: splitting alternate tokens over idle rotary threads gains **51.2–51.6%** at 512 tokens × 2 heads but loses **5.0–5.2%** at 512 × 16, duplicating double-precision frequency work. Use two groups only at most 72 blocks (72 SMs); keep 16-token frequency reuse above that. |
| Keeping the literal production tiled RoPE port's unused rotary threads | Six alternating GPU 1 pairs, 512-token chunk, 2 key heads: NVRTC **19.437–19.511 us**, Rust **20.400–20.426 us** (**4.5–4.7% lower throughput**). 64 blocks on 72 SMs, and half the rotary threads exit. Using that half is a separately gated layout change, not a benefit of translation. |
| Promoting the optimized eight-port Rust bundle to the default backend | Three alternating Qwen3.6 GPU 1 N=3/2K pairs: **−0.05% to +0.05% decode**, **−0.35% to +0.09% prefill**, despite narrow wins for the fused GDN norm and key RoPE. Query RoPE and generic fused hidden-width normalization still lose narrow gates. `rust-kernels` stays opt-in. |
| Removing a Rust RMSNorm barrier by repeating the serial warp-partial sum in every warp | Six alternating GPU 1 pairs vs NVRTC: hidden-width RMSNorm loses **11.4–12.7%** at 3 × 2,048 and **9.7–11.2%** at 1,536 × 2,048. Up to 32 repeated dependent additions per warp cost more than the barrier. Use a parallel second shuffle reduction, gated for its changed addition order. |
| Literal `usize` column loops and a rolled warp sum in the Rust normalization port | Six alternating GPU 1 pairs: 24,576 × 256 **100.606–102.575 us NVRTC** vs **104.244–105.007 us Rust** (2.3–3.5% lower throughput); 49,152 × 128 **100.073–100.855** vs **101.102–101.683 us** (0.8–1.1%). Causes: 64-bit column counters where NVRTC uses 32-bit, and an un-unrolled warp-partial loop. Keep 32-bit counters; lower register counts did not predict throughput. |
| Promoting the literal five-kernel Rust bundle to the default backend | Three alternating Qwen3.6 GPU 1 N=3/2K pairs lose **0.05–0.28% decode** and **0.05–0.38% prefill**; fused width-128 decode normalization loses **3.8–4.4%** kernel throughput. Matching arithmetic and geometry does not make a migration compiler-neutral. The bundle stays behind the disabled-by-default `rust-kernels` switch. |
| Enabling the plain Rust activation ports by default | Six alternating GPU 1 pairs at 8,388,608 elements: SwiGLU **182.156–182.255 us NVRTC** vs **186.574–186.675 us Rust** (2.3–2.4% lower throughput); per-row sigmoid at width 2,048 **132.924–133.251** vs **142.222–142.510 us** (6.4–6.7%). Three N=3/2K model pairs change sign. Kept behind `rust-kernels`, rejected as default. |
| Four-float Rust residual loads/stores with the existing launch grid | Bit-exact. Six GPU 1 pairs vs scalar Rust: 16,384 elements scalar **1.577–1.640 us**, vector **1.798–1.835 us** (8.9–13.9% lower throughput); 8,388,608 elements **180.408–180.593** vs **179.876–180.062 us** (0.20–0.35% higher). The unchanged grid leaves three quarters of its blocks empty at 16K. Rejected. |
| GDN split projection at row tiles 1 and 2 (`uint4` form, N=3) | 78.4 and 46.4 us per qkv/gate call vs RT=4's 33.9. RT=1 has the most warps and in-flight bytes and is worst: memory-level parallelism does not bind; instruction count per byte does, and it falls with RT. |
| The output projection at row tile 2 (2,048 rows, 512 warps at RT = 4 — under half a wave) | Bit-identical; **flat at N=1** (23.0 us), **+25% at N=3** (28.2 → 35.2 us). The grid was 0.44 waves deep, yet doubling warps bought nothing: at three tokens the cost is activation instructions per weight byte, which halving RT doubles. Wave fill does not bind a decode-width GEMV here. |
| The next layer's RMSNorm in the MoE combine kernel's tail (fixed-order last-block reduction, the mixer skips its norm launch; 41 launches a step) | Three pairs, middle reversed, **flat to −0.6%**: N=1 116.3 → 115.8, 115.2 → 114.7, 115.2 → 115.3; N=3 218.4 → 217.5, 218.0 → 216.7, 217.8 → 217.1. Within 2 ulps. Each of eight blocks per row gains a block reduce, fence and atomic, plus a serialized last-block pass, cancelling the ~2.8 us floor removed. Removing norm launches needs the consumer to normalize on the fly. |
| GDN split projection, char4 partition + row tile + prefetch | 40.0/36.7 us vs the shipped `uint4` form's 33.9/29.3, despite coalesced activation loads and ~2.7x fewer L1 wavefronts per byte on paper. The wavefront model predicted the wrong winner. |
| `q8_0` KV cache (llama.cpp side) | −2.5% at depth 0, **−15.5%** at 32K, **−35.8%** at 128K. Turing's in-kernel dequant costs more than the halved traffic saves; the KV path already ran at ~80% of peak. Keep `-ctk f16 -ctv f16`. |
| Requantize experts Q6_K → Q8_0 | 4–8% for **+30% VRAM**. The dequantization format is not what limits that kernel. |
| Marlin-style register prefetch on the MoE MMA kernels | 20% slower at 512 tokens. A register pipeline competes with the deliberate register-for-occupancy trade in `__launch_bounds__`, so ptxas spills instead of exceeding the occupancy target. |
| Widening the prefill softmax-rescale tile (`MMA_KEY_TRIPS` 2/4) | A wash, then **1.96× slower**. The kernel sits at 252 of 255 registers with zero spill; the wider prefetch arrays spill immediately. |
| `MMA_KOCT=4` (32-key prefill staging tile) | **−30%**. The cross-tile prefetch arrays grow with the tile and spill; one octet is the largest tile whose prefetch fits beside `o` and `qa` at head_dim 256. |
| Staging `Q` to shared in the decode MMA kernel | 73 registers freed, zero spill, **37–40% slower** at every depth. Shared grew past the 3-blocks/SM line, and 64 shared loads per warp per trip replaced free registers. llama.cpp's Turing config also keeps `Q` in registers. |
| Per-warp online softmax in decode without the cross-warp round trip | 255 registers and 104–120 B of spill at both occupancy widths; the shared-memory mitigation collapses to 1 block/SM by arithmetic. Not built past `ptxas`. |
| fp16 `P V` accumulation | 54 registers freed, but **209× over `MMA_GATE`** on the constant-`V` test, growing with depth: with coherent `V` the accumulator grows additively until increments round away (an IID-random-`V` simulation tested the wrong regime). fp16- and fp32-accumulate `m16n8k8` measured within 0.4%, so it buys nothing. |
| `__expf` in the prefill softmax | 1.075× on attention, and a **real rank-4 ranking error** against llama.cpp's separation. Greedy decoding would not notice; sampling would. |
| `half2` on the decode `P V` accumulator | Fewer registers and instructions, **11.3% slower**: `pack_h2`'s broadcast of the scalar softmax weight added 40 `PRMT` and 16 `F2F`. Instruction counting is retired as a predictor for this kernel. |
| `dp4a` / Q8_1 activation quantization (llama.cpp's `mmvq`) | Passed every isolated per-kernel gate, then failed `batch_decode` at **5.1e-1** (asymmetric) and **7.1e-1** (symmetric) against a 5e-3 bound: round-to-nearest amplifies upstream residuals into expert-selection flips. Re-attempted with bit-identical paths, it still failed the exact cross-width gate at `4.4e-5`. |
| `ldmatrix.sync.aligned.m8n8.x4` | Both fragment layouts verified on hardware, **0.8% slower**. The kernel is not bound by shared-load instruction count. |
| `mma.m8n8k4` (Turing's "native" shape) | **49.1 TFLOP/s** against `m16n8k8`'s 97.6: half the throughput. Ruled out without writing the kernel. |
| Swapping the flash grid's axes for L2 reuse | **−1.7%**. Sibling blocks drift apart faster than 6 MB of L2 spans; only a staging barrier inside one block makes them share. |
| `ATTN_QT` 16, `GQA_KT` 16, `__maxnreg__(84)` | −10%, −14%, −6%. Each trades the second resident block per SM for something worth less. |
| Copying llama.cpp's 4-query/64-key outer prefill tile | **~8× slower**, `ptxas` clean at 255 registers, zero spill. Four queries launch four times the blocks and the 64-key softmax serialized through four lanes; the tile dimensions are not competitive without llama.cpp's fragment-resident softmax and combine. |
| Prefetching the GDN alpha/beta gate contraction | 21.2 us a call became 28.5 (four blocks staged), 30.8 (two), or 46.3 (one block ahead, the `gdn_proj_split_rows` double-buffer idiom). ptxas already scheduled the tight loop better than the source could. The gates stay as written, and their 1.0 ms stays on the table. |
| Staging the weights instead of the activations at every GEMV width | 0.549 -> 0.585 ms a layer at three tokens and 0.570 -> 0.695 at four. Weight staging is what lets `TT` pass four, but at `TT <= 4` the row tile is 8 and sixty-four live weights cost more than the `TT * 8` activations they replace. Both stagings ship, chosen by `TT`. |
| Hoisting the GDN Gram kernel out of the chunk loop | Bit-identical, 240 launches → 30, and **0.5% slower**, losing every interleaved pair. In the loop the Gram output hits L2 immediately; hoisted, all eight chunks' squares must coexist in a 6 MiB L2. |
| `shared_expert_mma` on the dense FFN at decode width (Qwen3.8-27B) | 109.0 ms per step against the fp32 path's 63.5 (9.17 against 15.74 tok/s). At one token a GEMM discards 63/64 of its 64-token tile; same weight bytes, wrong shape. Replaced by a copy of `GdnBlock`'s split-layout GEMV: 17.26 tok/s, and more accurate (fp32 multiply, no activation quantization). |
| Tiling `gdn_chunk_gram` over target tokens | Nothing. Its keys are 32 KiB per head and sit in L2; arithmetic that says a kernel re-reads its input does not say the re-read costs anything. |
| Shared-memory staging on the GDN decode GEMV and the routed-expert GEMV | −3.6% and −0.8%. The removed re-reads were L1 hits; staging replaced a hit with a copy and a barrier, and cost a resident block. |
| Shared-staged flat decode GEMVs (`int4` staging, as the LM head uses) | Bit-identical, **1.3–2.4% slower**. These kernels are bound by the **integer pipe**, not load issue: Q6_K unpack costs ~9 integer ops per element against a ~10-op budget at the streaming roofline. |
| Shared-memory activation staging in the dense split GEMV | 0.528 ms against 0.512 at one token and 0.862 against 0.684 at four, losing every pair. It halved sector requests as predicted, but the two barriers a 512-element step needs cost more than the sectors save. Third shared-staging loss to registers on a decode-width GEMV. |
| `ld.global.cs` (evict-first) on the dense split GEMV's weight loads | Won every isolated-bench cell (0.508 against 0.512 at one token, 0.601 against 0.690 at four, three interleaved pairs) and **lost in the model**: `dense_proj_split_t1_r4` 166.6 -> 167.6 us, GDN out projection 67.8 -> 71.9, step 52.28 -> 52.65 ms. A narrow bench that wins is a candidate, not a result. |
| `CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 0` on the dense split GEMV | Nothing, at any width. The kernel already requests zero shared memory, so the driver already gave it the large-L1 split. |
| Splitting the GDN gate contraction across eight warps | 20.9 -> 5.4 us on the kernel, 52.28 -> 51.57 ms on the step (**+1.3%**), but `1 - cosine` against llama.cpp's logits grew 2.64e-4 -> 2.94e-4 and `forward_pass` failed. The kernel is a real 1.9% of the decode step; a replacement must be bit-identical to collect it. |
| Unrolling the GDN gate loop for memory-level parallelism | Nothing: 20.9 us before and after. `ptxas` already issued the loads ahead; the kernel is short of *warps*, not of in-flight loads per warp. |
| I2F-free unpack (exact-mantissa trick) | Bit-exact, every gate green, **flat**. With the load-issue and XU-pipe hypotheses dead, the flat decode GEMVs at 473–519 GB/s read as at their practical equilibrium for this quantization on this card. |
| Even/odd MMA accumulator chains | −2% prefill, noise at decode. The schedule was not accumulator-stalled, and eight more registers on kernels at ~230 cost more than the chain relief. |
| Skipping the online-softmax rescale when the running max did not move | Bit-identical, **2.8% slower** at 128K prefill. The skipped multiplies hid under staged-load latency anyway; the vote-and-branch costs more than the work it skips. |
| Speculation during sustained shallow N=3 generation | 256-token prompts, 512 fixed steps, three interleaved reversed pairs: simple at cap 48 lost **32.06–32.27%**, map-k4v at cap 8 **53.80–53.92%**, MTP at cap 3 **23.13–23.25%**. At 32K, map and MTP still lost **28.91–29.07%** and **22.93–23.25%**. Higher emission cannot repay verification. |
| MTP at the N=3 deep-context target, cap 3 | At 122,880 tokens/sequence, **89.3 / 89.3 / 89.4 tok/s** against plain **100.5 / 100.6 / 100.7** (**−11.14–11.23%**, three reversed pairs) despite full acceptance. The ~134 ms draft/verify step cannot repay four times the output of a 30 ms plain step. Map-k4v at cap 8 won **1.29–1.49%** in the same measurement. |
| Widening the trained drafters to the trained maximum (`--spec-draft-n-max 15`) | Lost **55.4%** (MTP) and **65.8%** (DFlash) at N=1, **67.3%** / **54.8%** at N=3. MTP acceptance rose 2.86 → 3.90 tokens/step, but a 3 → 15 block added **57.2 ms/step** (MTP) and **52.6 ms** (DFlash): a trained draft pass scales with the block. Rates are from that configuration, not the current policy table. |
| `ngram-mod` at llama.cpp's default `n_max` 64, N=3 | `CUDA_ERROR_OUT_OF_MEMORY` before the first step. Verify holds one GDN snapshot ring set per decode slot, scaling with `(drafts + 2)`: 4.05 GiB per slot at 64 drafts, 12.15 GiB for three, beside a 29.6 GiB model on 48 GiB. 48 drafts (9.2 GiB) fit shallow and still lost. The draft cap is a VRAM knob. |
| Tensor cores for routed MoE at N=3 (`MMA_MIN_TOKENS` 8 → 3) | 135.2 vs 147.7 tok/s. Padding a three-row dispatch into MMA fragments and quantizing its activations costs more than the arithmetic recovers. |
| The general fp32 expert tiles at N=3 (`MOE_NARROW_DECODE_MAX` 4 → 2) | 129.6 vs 147.7 tok/s. With the row above, both kernel paths neighbouring the current N=3 choice are slower. |
| The scalar warp decode kernel at depth | 133.0 vs 148–151 tok/s at 32K N=3. The depth-aware dispatch onto the tensor-core kernel stands. |
| `WPO=4` decode occupancy width | ~4.5% slower than `WPO=2` at 32K N=3, before and after the register-spill fix; never won at any depth measured. |
| Decode blocks 48 or 60 at 72 logical splits | 150.5 and 139.5 against 36 blocks' 152–153. 48 divides 72 unevenly and the long blocks set the makespan; 60 spills into a second wave. |
| Three independent N=1 decode graphs on one card | 129.9 vs 147.7 tok/s at 2K. Separate streams repeat weight traffic and contend more than their overlap recovers. |
| Shared-expert side-stream overlap | Flat at N=3 (no SM or DRAM slack), **−2.5–3%** at N=1 (launch-latency-bound; two events per MoE layer is pure overhead). The shared expert's blocks now ride the routed expert's grid instead; see WHY, "The same seam carries a whole small kernel". |
| Independent GDN streams at prefill | Within thermal drift, reversing across warmed pairs. Removed rather than kept on a cold-card gain. |
| Cross-sequence prefill flattening at *equal* total ubatch | 0.99×. At fixed physical width, batching does not reduce full-model passes; it wins only when the pass gets wider (see WHY). |
| The two-kernel `bm == 1` split, first attempt | −5.2% before `bucket_live` existed: both kernels walked `sorted_token_ids` and crossed two barriers per bucket to compute `bm`. It landed as a win once `bm` became a table read. |
| GEMV unroll pragmas on the direct-1 helpers | 20% slower on ffn. A register cliff: the unroll depth was tuned for the standalone GEMV's register budget, which the narrow kernel (carrying the tiled fallback in the same function) does not have. |
| A `KU` unroll hint on the GDN tiled projection | Byte-identical SASS at the width N=3 uses; −9% at N=2. `ptxas` already reached that schedule; closing the kernel's 28%-of-roofline ceiling must change what ptxas schedules, not hint at it. |
| Selecting prefill arithmetic for wide speculative verification | Three 49-row windows crossed the GDN projection threshold at 64 total rows and requested a Q8_0 representation production no longer keeps; with residency fixed it accepted **10** tokens where a regression required **24**. Verification now keeps exact split projections and per-query decode attention at any width. |
| Removing the routed-partial clear | Below run-to-run spread, and reversing. Not worth deleting a defensive correctness aid for. |
| Grant alignment as a prefill lever (`ADMISSION_RESERVE_FRACTION` 4 → 2) | Predicted **+27%**, measured **+1.5%**. The width curve is real (2,048-wide passes 2,760 tok/s against 256-wide 1,510; 1,896 vs 1,050 at 64K) and the grant decomposition was verified, but the penalty does not appear end to end. Kept at 2 (never worse); **do not rank work by that width arithmetic**. |
| Snapshot retention as the cause of the concurrent-session penalty | Nothing. At 64K × 3 sessions, `--cache-ram 0` (no snapshots) and `16GiB` (159 per worker) both landed within noise of the 2.41 GiB default's 24. An earlier test at 16K, where the arena cannot bind, proved nothing. Test a capacity hypothesis where capacity is actually exceeded. |
| Community-quant formats in the runtime `dequant_tile` ladder, and `__noinline__` to contain them | The `__forceinline__` runtime-`quant` ladder emits every arm at every call site, so every format pays for each added arm: three formats cost `moe_shared_ffn_gemv` 42 → 62 registers and **−5.05%** on Ornith N=3 2K (195.5 against 205.9). **`__noinline__` is worse** (`moe_expert_ffn` 80 → 124, spilling across the ABI). Reverted; `dequant_tile_ct<Q>` has since landed (see WHY, "Compile-time formats free registers that ptxas then spends"). |
| Forking the Gated DeltaNet input projections onto a side stream | **−1.2%** at N=1 2K, losing both interleaved pairs: the step is launch-latency-bound, and two events per layer is sixty graph nodes with no wave to fill. **Reverted.** The +0.79% at N=3 is not a current claim: it was measured against a stale pinned binary, with the fork leaked into `run_batch_prefill` too. |
| Narrowing the router's expert tile at decode width (`ET` 8 → 4) | **Nothing.** Three interleaved reversed pairs at 2K against a baseline built from the pinned commit read +0.24 / −0.19 / −0.05% at N=3; the sign flipped inside every set at N=3, N=2 and N=1. The earlier +1.31% came from a stale pinned binary; the motivating sequential sweep crossed host drift (**−0.8% at N=3 over ten minutes**) larger than any effect it reported. |
| Reading the Ornith fork's 32K result at all | Three interleaved pairs gave −0.31%, +2.62% and +1.39% while the baseline arm ranged 1.85% across its own runs, so the predicted +0.55–0.6% can be neither confirmed nor refuted. No 32K figure is quoted; the +1.2% mean is not one. |
| Inferring scheduler behaviour from client-side timings | Wrong three times: lock starvation read as a scheduler refusing to share, an admission-pacing artefact as a race, and a 2x throughput collapse as a kernel concurrency limit (the kernel charges 5%, not 50%). Each resolved once a step logged its own grants. Instrument the component first. |
| Single-buffered fp16 weight-only GEMM for dense projections | **0.46–1.07×** the int8 split path on 27B and Clef shapes (27B FFN gate/up at 2,048 tokens **22.6 TFLOP/s** against int8 **43.4 TOP/s**). One Q8_0 tile per k-step behind two barriers, no register stage: every global load stalls its MMAs. The double-buffered rewrite (**1.02–1.33×**, same hour) became `hgemm`. |
| Explicit `prefetch.global.L2` ahead of the `hgemm` stage | Prefetching 128/256/512 bytes ahead was slower than the `L2::128B` control on **6/7, 7/7 and 7/7** 27B/Clef shapes (GDN qkv 10,240×5,120: **52.9 → 50.0 / 48.6 / 45.9 TFLOP/s**); prefetches compete for the issue slots of the loads they hide. Control was the preceding sweep, not interleaved pairs; rejected on the sign's consistency. |
| A second register stage in `hgemm` | **255 registers, 24 bytes of spill stores**, and 50.7 → 50.2 / 41.7 → 40.0 TFLOP/s on the FFN gate and down shapes at 512 tokens. The kernel cannot keep the extra stage resident. |
| Tile-contiguous activation layout for `hgemm` | `[M/128][K/32][128][32]` fp16 was equal or slower on **13 of 14** shape/tile cells against the preceding control. Activations are 1/2 to 1/8 of a stage's bytes and L2-resident; loads wait on the weight stream. Rejected for no gain, not a measured loss. |
| Split-K wherever wave arithmetic said it pays | Splitting any ragged-last-wave grid lost in the harness: 10,240×5,120 split 2/3/4 **1.02 / 1.11 / 1.15 ms** against **0.98** unsplit; 12,288×5,120 split 3 **1.26** against **1.11**. Splitting only under 1.25 waves at an 8% price won all six interleaved pairs on GPU 1: 512 tokens **834.0 → 845.0 tok/s**, 1,536 tokens **774.3 → 821.8**. |

## Rejected on arithmetic, before building

| Idea | Number | Why not |
| --- | --- | --- |
| Widening Qwen3.8-27B's bf16 tensors to fp32 on the host | `output.weight` 2.54 GiB → 5.1 GiB | doubles the head's per-token read; requantizing to Q8_0 changes the model. The LM-head GEMV reads bf16 instead |
| Dense FFN held in both Q8_0 and split int8 | 16.9 GiB twice, beside 11.8 GiB of arena | first attempt hit `CUDA_ERROR_OUT_OF_MEMORY`; see "What the FFN residency is worth" |
| TensorRT | — | not installed; needs an ONNX exporter plus plugins: Q6_K superblock scales and Gated DeltaNet are not builtins |
| vLLM's Marlin MoE | — | int4/int8 weights only, not Q6_K; port its indexing onto llama.cpp's primitives |
| Shared-memory double-buffering in the MoE MMA kernels | `3 × 21,760 = 65,280` of 65,536 B | any double buffer admits one block per SM, the register prefetch's lost trade 3× worse |
| Split-K in the LM head | 248,320 warps vs 2,304 resident (107×) | adds a launch, a `vocab × K` buffer and a split-dependent sum order for no occupancy gain; a unit test asserts it |
| `mma.m16n8k16` | — | NVRTC emits PTX at `compute_75`, `ptxas` rejects it (sm_80+); hand-decompose to `m16n8k8` / `m8n8k16` |

## Diagnoses that were right about the fact and wrong about the cost

- **"The step barrier costs a neighbouring session almost nothing."** The
  median said **1.3x**, the mean **12.1x**: the barrier freezes tokens (21 of
  199) rather than slowing them. Report mean and tail for costs that arrive in
  stalls.
- **"The MoE dispatch kernel is single-block."** True; **0.12%** of runtime.
- **"The LM head computes logits for all positions."** False; it runs one row
  (`get_rows(cur, inp_out_ids)`).
- **"Per-element dequant re-reads the superblock header."** True; hoisting it gave
  **byte-identical SASS**; `ptxas` already did it.
- **"Decode attention is bandwidth-bound."** A calibration ladder with the
  identical grid and loads reaches **90% of streaming roofline**; the cost is
  the softmax.
- **"Sector amplification is costing 4×."** Real in SASS; recovery was
  **1.08×** because L1 absorbed it.
- **Ablations that perturb numerics move expert routing.** Halving an inner
  loop (12% faster) and stripping scale multiplies (10% slower) measured
  nothing; the valid ablation runs the staging loop twice, bit-identical.
- **A result taken during an unrelated regression is not a measurement.** A
  57% "regression" under a live decode bug is a small improvement against a
  clean baseline.
- **Two agents agreeing is one guess twice.** The reasoned `__noinline__` fix
  for a −5% dispatch-ladder regression made it worse: `moe_expert_ffn` 80 → 124
  registers.
- **"The sweep is non-monotonic, so there is nothing there."** Right answer,
  wrong reasoning: non-monotonic single runs mean the method cannot resolve
  the effect. A later +1.31% (three of three pairs) read ±0.2% against a
  baseline built from the pinned commit. One run per point resolves nothing
  below about 1.5%; the box drifts 0.8% in ten minutes.
- **A green test target is not a test that ran.** `forward_pass` reports
  `ok, 8 passed` in 0.36 s without the golden, and **every run from a
  `git worktree` skips it silently** unless `LLMCUDA_GOLDEN` is set.
  `cargo test | grep` reports grep's exit status. Checks: `docs/TESTING.md`.
- **`ptxas` allocates beyond the edit.** One added branch moved an unrelated
  `__global__` function 80 → 126 registers; diff `cuobjdump -sass` for every
  kernel.
- **"It was verified" — at the widths it was benched at.** The narrow expert
  tile, checked at N=1 and N=3, mis-sized its launch at `max_tokens == 2`, a
  width only speculative decode produces; a `replace_all` also put a
  decode-only fork into the prefill path. Gate by entry points and scheduler
  shapes, not by what the change is. Re-measured cleanly, the tile had no win.

---

# Ceilings that are closed by design

Deliberate, verified trades; reopening one means un-making it.

- **Deep decode, against llama.cpp's own kernel.** At head_dim 256, GQA 8 on
  Turing, llama.cpp decodes through its tensor-core prefill kernel. Its edge is
  three things this engine forbids:
  1. **A single fp16 rounding of `Q`** (2× key rate): broke the 1e-5 gate at
     `n_keys = 61`, hence the `q_hi`/`q_lo` split.
  2. **fp16 `VKQ` accumulation**: within 0.4% on throughput, but `half2`
     packing is what fits its barrier-free per-warp design in Turing's
     registers; with fp32 accumulators, measured shut from four directions.
  3. **`mmvq`/dp4a activation quantization**: rejected on the serving
     contract.
- **Aggregate concurrency.** 131.7 MB/token of GDN recurrent and conv state
  **never amortizes at any batch width**; llama.cpp pays it too and holds flat
  at ~41% of the concurrency-aware roofline from one sequence to three. Only
  routed experts batch, and N=3 touches `D(3) = 23.26` distinct experts
  against one sequence's 8.00.

Workstreams that would reopen (1) and (3): [OPTIMIZATION.md](OPTIMIZATION.md).

---

## Reproducing

```sh
# llmcuda-rs, N=3 decode and prefill
CUDA_VISIBLE_DEVICES=1 LLMCUDA_BATCH_N=3 \
  ./target/release/bench_decode_batch 32768 16
CUDA_VISIBLE_DEVICES=2 LLMCUDA_PREFILL_SEQUENCES=3 LLMCUDA_BENCH_CHUNK=6138 \
  LLMCUDA_BENCH_N=2046 cargo run --release -p llmcuda-engine --bin bench_forward

# llama.cpp, same card, alternating with the above
CUDA_VISIBLE_DEVICES=1 llama-batched-bench \
  -m Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf \
  -ngl 99 -sm none -fa on -b 4096 -ub 4096 -ctk f16 -ctv f16 \
  -c 131072 -npp 32768 -ntg 32 -npl 3
```

Dense (`qwen35`) rows, model set by environment variable:

```sh
M=Qwen3.8-27B-UD-Q8_K_XL.gguf
LLMCUDA_MODEL="$M" CUDA_VISIBLE_DEVICES=1 ./target/release/bench_forward
LLMCUDA_MODEL="$M" CUDA_VISIBLE_DEVICES=1 ./target/release/bench_decode

# N=3. bench_decode_batch prints aggregate and per-seq columns itself, and
# carries its own single_stream baseline in the same process.
LLMCUDA_MODEL="$M" LLMCUDA_PREFILL_SEQUENCES=3 LLMCUDA_BENCH_CHUNK=1536 \
  LLMCUDA_BENCH_N=512 CUDA_VISIBLE_DEVICES=1 ./target/release/bench_forward
LLMCUDA_MODEL="$M" LLMCUDA_BATCH_N=1,3 CUDA_VISIBLE_DEVICES=1 \
  ./target/release/bench_decode_batch 512 64

CUDA_VISIBLE_DEVICES=1 llama-batched-bench -m "$M" \
  -ngl 99 -sm none -fa on -b 2048 -ub 2048 -ctk f16 -ctv f16 \
  -npl 1 -npp 512 -ntg 64        # -npl 3 for the N=3 rows

# The dense FFN alone, at decode widths, with the card's measured streaming
# ceiling printed above the table. ~10 s per A/B against `bench_decode`'s
# ~90 s, which is what makes an inner-loop change measurable at all.
LLMCUDA_MODEL="$M" CUDA_VISIBLE_DEVICES=1 ./target/release/bench_dense_ffn
LLMCUDA_DENSE_SPLIT_ROWS=8 ...                  # override the RT table
LLMCUDA_FFN_N=4,6,8,12,16,32 ...                # the GEMV/GEMM width sweep

# A second qwen35moe checkpoint. Requant to the shipped mix first — a
# uniformly-Q6_K file stops at the first projection. Recipe in MODEL.md.
LLMCUDA_MODEL=Ornith-1.5-35B-A3B-UD-Q6_K_XL.gguf CUDA_VISIBLE_DEVICES=2 \
  LLMCUDA_PREFILL_SEQUENCES=3 LLMCUDA_BENCH_CHUNK=6138 LLMCUDA_BENCH_N=130944 \
  ./target/release/bench_forward     # depths are multiples of 6138/3 = 2046

# The speculative table. `LLMCUDA_SPEC` takes any of the seven drafter names.
LLMCUDA_MODEL="$M" LLMCUDA_SPEC=draft-mtp CUDA_VISIBLE_DEVICES=1 \
  ./target/release/bench_worker_spec           # context 256, 512 tok/seq
# Fixed N=3 deep window; use the same reservation for every comparison arm.
LLMCUDA_MODEL="$M" LLMCUDA_SPEC=ngram-map-k4v LLMCUDA_SPEC_DRAFTS=8 \
  LLMCUDA_BATCH_N=3 LLMCUDA_TIMED_STEPS=32 LLMCUDA_MAX_OUTPUT=4096 \
  CUDA_VISIBLE_DEVICES=1 ./target/release/bench_worker_spec 122880 512

# Per-kernel attribution of a decode step. `--cuda-graph-trace=node` is not
# optional: decode replays a captured graph and without it nsys attributes one
# pass and drops the rest.
LLMCUDA_PROFILE_TIMED=1 LLMCUDA_SKIP_SINGLE_STREAM=1 LLMCUDA_BATCH_N=3 \
  CUDA_VISIBLE_DEVICES=1 nsys profile -t cuda --cuda-graph-trace=node \
  -c cudaProfilerApi --capture-range-end=repeat -s none -o d \
  ./target/release/bench_decode_batch 32768 32
nsys export --type sqlite -o d.sqlite d.nsys-rep
```

`LLMCUDA_PROFILE_TIMED` captures only the timed region; `bench_forward` writes
each timed prefill chunk as a numbered report, so export and sum each. Capture
start/stop takes seconds, so profiled rates are not throughput results;
bounded captures also avoid an event-order import failure on long context
setup. Attribution: [OPTIMIZATION_CAMPAIGN.md](OPTIMIZATION_CAMPAIGN.md).

Requantized files (generated, not shipped; the engine cannot read the last
two, see [MODEL.md](MODEL.md#which-formats-the-engine-actually-reads)):

```sh
llama-quantize --allow-requantize "$M" Qwen3.8-27B-req-q8_0.gguf  q8_0   12
llama-quantize --allow-requantize "$M" Qwen3.8-27B-req-q6_k.gguf  q6_k   12
llama-quantize --allow-requantize "$M" Qwen3.8-27B-req-q4_k_m.gguf q4_k_m 12
```

Interleave the files within each repetition; the card drifts. For the
greedy-agreement table use `llama-completion ... -no-cnv < /dev/null`; build
10456's `llama-cli` ignores `-no-cnv` and applies the chat template.

Serving tables come from a running server:

```sh
CUDA_VISIBLE_DEVICES=0 ./target/release/llmcuda \
  --spec-type none -c 405504 -s 3 --token-budget 4096 --prefill-chunk 2048 &
python3 tools/serving/bench_server.py http://127.0.0.1:8000 out.json \
  '{"depths":[2800,5500,11000,22000],"sessions":[1,2,3],"trials":3,"max_tokens":48}'
```

`depths` are words; results are reported against `usage.prompt_tokens`. For
the three-card fleet drop `CUDA_VISIBLE_DEVICES` and set `sessions` to 9.
`--prefill-slice 0` restores whole-step prefill (the sharing A/B).

Narrow harnesses: `bench_attention` (`LLMCUDA_ATTN_CHUNK=1` for decode shape),
`bench_moe`, `bench_moe_mma`, `bench_mma`, `bench_decode`,
`bench_worker_decode`, `profile_forward`, `audit_batch_divergence` (per-layer
batch-vs-single-stream).

## Not measured

- Any hardware counter: efficiency figures are necessary-work ÷ measured-time,
  so they understate distance from peak.
- Multi-GPU under real routed traffic; three-card figures are one session per
  card.
- Long-run output quality under the precision trades; only gates and short
  samples.
- Vision serving throughput and encode latency. Tower VRAM was measured
  (1152 MiB at load, ~1320 MiB after first use, per worker;
  [DEVELOPMENT.md](DEVELOPMENT.md)).
- Sustained thermals; all runs are short.
- The dense model beyond its section: N=1 and N=3 at one depth only.
- **Any accuracy check on the requantized files.** The `all-Q8_0`, Q6_K and
  Q4_K_M files were measured for speed alone. **Do not ship a requant on the
  strength of the throughput table.**
- **Narrowing the activations** (dense GEMV, GDN gates). fp32 activations
  against int8 weights are the remaining cost: the dense split GEMV block
  costs 0.51 ms at one token and 0.68 at four for identical weight bytes.
  **Fixing the GDN-gate Q8_0 alignment is not worth building**: costed at 2x,
  it is 17% of that kernel's sector budget and ~0.3% of the step. fp16 activations would
  cost decode its exactness over llama.cpp (Q8_1 activations); not built.
- **Fusing the dense FFN's SwiGLU and residual add into its GEMVs.** 128
  launches of a 51.8 ms step, about 0.5%; not taken because it moves the
  `valid_tokens` gate (AGENTS.md rule 5) that `forward_pass` checks.
  Recoverable by threading the scalar into the GEMV epilogue.
- **k-quants outside the dense FFN loader.** Q6_K halves the decode budget
  (llama.cpp: 22.13 tok/s), but llama.cpp's Q6_K row streams 463 GB/s against
  Q8_0's 538; a Q6_K GEMV here is unmeasured.
