# Toolchain

**Decision: `cudarc` 0.19 + NVRTC, with CUDA C++ kernel sources compiled at
load time.**

This resolves milestone 00. The evidence below was produced by
`cargo run -p llmcuda-cuda --bin probe` on the target host, and the same checks
run as a test (`spike::tests::milestone_00_gating_spike`), skipping loudly
when no device is present.

## The choice

The design plan offered two options.

| Option | For | Against |
| --- | --- | --- |
| `cudarc` + CUDA C++ via NVRTC | Stable on sm_75; JIT specialization at load; inline PTX always available | Two languages; kernel code outside Rust's type system |
| NVIDIA `cuda-oxide` | Single-source Rust; compile-time kernel policies; monomorphization; PTX interop with existing launchers | Alpha; separate pinned toolchain and a CUDA 13 toolkit to compile; sm_75 needs an explicit target and workload validation |
| NVIDIA `cutile-rs` | Rust tile DSL; compiler-managed layout; stable Rust; graph replay | Explicitly excludes sm_75; cannot target the deployment cards |

The decision rule was explicit: any "no" on the inline-PTX question collapses
the choice to cudarc, because without an escape hatch you cannot reach
`mma.sync.m16n8k8.f16` or `ldmatrix`, and you cannot route around a codegen
bug.

**cuda-oxide is obtainable from NVIDIA** (`NVIDIA/cuda-rust`, which absorbed
`NVlabs/cuda-oxide` and `NVlabs/cutile-rs`). The original availability rejection
is obsolete. Do not confuse it with Protryon's older `cuda-oxide` driver
wrapper. Availability alone does not validate a replacement kernel.

## Rust compiler evaluation

Source revision checked on 2026-11-25: both compilers now live in one
repository, [NVIDIA/cuda-rust `f35d6ccc`](https://github.com/NVIDIA/cuda-rust/tree/f35d6ccc2e03a54e73e3796f415f11313e439d50).

- `cuda-oxide/` declares version 0.4.0 and pins Rust 1.99.1 with `rustc-dev`
  (no longer a nightly). Building its host crates needs a CUDA 13.0+ toolkit;
  the generated PTX has no such requirement. Its installation guide lists
  sm_80+, but `cargo-oxide/src/commands/codegen_env.rs` explicitly handles
  Turing and accepts `--arch sm_75`. Its `ptx_asm!` macro supplies an inline-PTX
  escape hatch. Eight layer kernels run on sm_75; this does not establish
  support or performance for our tensor-core kernels.
- `cutile-rs/` declares version 0.6.0. Its README still states that GPUs below
  sm_80, naming sm_70 and sm_75, are unsupported. Its backend is CUDA Tile IR:
  13.2 for Ampere, Ada and Blackwell, 13.3 for Hopper. Nothing in it can be
  built for or run on the deployment cards, so it has never been measured here.

[`experiments/cuda-oxide`](../experiments/cuda-oxide) holds the Rust source
of the ported kernels, with the raw-pointer ABIs and supported input/output
aliasing their launchers in `layer_ops.rs` and `attention.rs` expect. For most
of them it is the only source: the CUDA C++ was deleted once the port matched
it bit for bit.
It uses a separate Cargo workspace so the kernel compiler does not change the
engine's nightly or dependency graph. The engine embeds the generated PTX and
loads it through its existing cudarc runtime, so it requires neither
cuda-oxide nor its host runtime at build or run time. Source changes require
regeneration.

Which kernels launch from Rust and what a port must show first is in the
[build instructions](DEVELOPMENT.md#rust-cuda-kernels); there is no runtime
switch, and no kernel exists in both languages. A Rust kernel is held by its
CPU-reference differential test and, where a fused CUDA C++ kernel shares its
arithmetic, by the test that compares the two exactly.

[The paired measurements](BENCHMARKS.md#rust-kernel-authoring-can-keep-the-existing-launcher)
show model-level parity. The first normalization ports were faster at narrow
rows and one ulp away from the fused CUDA C++ paths; the ones in the tree
reduce in the original's order and are bit-identical.

The compiler's host crates (`cuda-core`/`cuda-bindings`) require a CUDA 13.0+
toolkit, which the engine does not; the experiment's README records how one
was assembled beside the host's CUDA 12.8. Ordinary engine builds continue to
use NVRTC for every kernel that is not embedded PTX.

## Gating results

Measured on 3× Quadro RTX 8000, driver 595.84, CUDA 12.4, Rust nightly.

| # | Question | Result |
| --- | --- | --- |
| Q1 | Does the toolchain emit sm_75 PTX? | **Pass** — NVRTC accepts `compute_75` and the emitted PTX carries `.target sm_75` |
| Q2 | Is inline PTX available inside kernels? | **Pass** — `asm volatile("fma.rn.f32 …")` compiled, launched, and matched host arithmetic exactly over 1024 elements |
| Q3 | Do warp shuffle/vote intrinsics lower correctly? | **Pass** — `__shfl_xor_sync` butterfly reduction and `__ballot_sync`/`__popc` matched host results across 8 warps (max abs error 1.9e-6) |
| Q4 | Are CUDA Graphs reachable? | **Pass** — a captured launch replayed 5× with the accumulator advancing exactly once per replay |

Each check verifies numerical output. None is satisfied by the API returning
`Ok`, and a check that cannot run reports `Skipped` rather than `Pass` — see
[`CheckOutcome`](../crates/llmcuda-cuda/src/spike.rs).

Q1 was answered more strictly than "it compiled": NVRTC accepting
`--gpu-architecture=compute_75` does not by itself prove the right target came
out, so the `.target` directive in the emitted PTX is inspected directly.

Q4 is the one that mattered most. Graph capture over on-device MoE indirection
is the thesis of this project (milestone 06), and the plan flagged graphs as
"not in the crate list; `cuda-bindings` raw FFI is the fallback". That turned
out to be unnecessary — cudarc 0.19 exposes a safe graph API
(`begin_capture`, `end_capture`, `CudaGraph::launch`) over 90 `cuGraph*`
bindings.

## The event-tracking constraint

Q4 failed on first attempt with:

```
CUDA_ERROR_STREAM_CAPTURE_ISOLATION
  "dependency created on uncaptured work in another stream"
```

This is **not** a hardware or driver limitation, and it is worth understanding
because it constrains milestone 06 permanently.

cudarc tracks a read event and a write event per device allocation. When the
context is in multi-stream mode it injects `cuStreamWaitEvent` before any
launch touching a buffer that another stream last used. For ordinary work that
is exactly right — it is what makes the safe API safe.

During stream capture it is fatal. The injected wait refers to an event
recorded *before* capture began, so the captured graph would depend on
uncaptured work, and CUDA rejects the entire capture.

The resolution is `CudaContext::disable_event_tracking()`, which is `unsafe`
precisely because it hands responsibility for cross-stream ordering back to the
caller.

**Consequence for the engine.** The decode step cannot rely on the wrapper for
ordering. It must:

- own its buffers for the lifetime of the captured graph,
- fully synchronize before capture begins,
- order its own streams explicitly, and
- keep event tracking disabled only across the captured region.

That is consistent with the design anyway — §"No allocation on the hot path"
in [AGENTS.md](../AGENTS.md) already requires pre-allocated arenas, and a
captured graph requires stable device pointers regardless. But it means the
safe wrapper's guarantees do not extend into the hot path, and reviewers should
treat capture-region code as manually ordered.

## What this does not settle

- **Kernel language.** CUDA C++ through NVRTC means kernel code sits outside
  Rust's type system. Shapes are passed as scalars and validated at the call
  boundary in `llmcuda-cuda`, not by the compiler.
- **Tensor-core paths.** Q2 proves inline PTX works; it does not prove any
  particular `mma.sync` shape is correct or profitable on this workload. The
  MoE decode shapes are extremely skinny (batch 3 against 2048×512), so SplitK
  is expected to matter more than MMA. Unmeasured.
- **cuBLASLt.** Enabled as a cudarc feature for the prefill GEMM path but not
  yet exercised by any code here.

## Reproducing

```sh
cargo run -p llmcuda-cuda --bin probe    # full device inventory + gate
cargo test -p llmcuda-cuda               # same checks as tests
```

Exits non-zero if the fleet fails the capability gate or a check that ran did
not pass. On a host with no CUDA driver, the binary reports that and exits
non-zero; the test skips with a printed message.
