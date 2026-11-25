# Rust layer kernels

This standalone experiment compiles `src/main.rs` with cuda-oxide from
[NVIDIA/cuda-rust](https://github.com/NVIDIA/cuda-rust) at
`f35d6ccc2e03a54e73e3796f415f11313e439d50` (0.4.0, Rust 1.99.1 with `rustc-dev`). The
compiler used to live at `NVlabs/cuda-oxide`, which now redirects to that
monorepo; cuda-oxide is its `cuda-oxide/` directory.
The engine's single opt-in `rust-kernels` feature loads residual add,
standalone SwiGLU, sigmoid gating, softplus, both partial RoPE APIs, RMSNorm and
fused RMSNorm/SwiGLU through cudarc. The feature forwards from the server and
engine crates. Normal builds need no cuda-oxide. The
embedded PTX is generated code, not an independently maintained kernel.

Install the matching compiler, with the CUDA 13 toolkit and libclang available:

```sh
cargo +1.99.1 install --git https://github.com/NVIDIA/cuda-rust \
  --rev f35d6ccc2e03a54e73e3796f415f11313e439d50 cargo-oxide
```

For a local compiler checkout, put its `target/debug` directory on `PATH` and
set `CUDA_OXIDE_BACKEND` to the matching `librustc_codegen_cuda.so`. Both must
come from the pinned revision. Regenerate from the repository root:

```sh
python3 experiments/cuda-oxide/regenerate.py
git diff -- crates/llmcuda-cuda/src/kernels/rust/tensor_add.ptx
```

The compiler's host crates need a CUDA 13.0+ toolkit (headers including
`crt/`, `libnvvm`, `libnvJitLink`); the engine itself still builds and runs
against CUDA 12.8. On the target host, which has no system CUDA 13, the
toolkit was assembled from NVIDIA's `cu13` pip wheels into one prefix and the
build used these overrides (libclang from `pip install --target ... libclang`):

```sh
export CUDA_HOME=/path/to/cu13          # include/, lib64/, nvvm/, bin/
export CUDA_TOOLKIT_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LIBCLANG_PATH=/path/to/python/clang/native
export BINDGEN_EXTRA_CLANG_ARGS='-isystem /usr/lib/gcc/x86_64-linux-gnu/13/include'
```

`cargo oxide doctor` still warns about the clang resource directory with this
layout; the build succeeds. A cached build does not re-emit the PTX, so touch
`src/main.rs` if `llmcuda_rust_device.ptx` is missing. The engine benchmark
uses its ordinary CUDA 12.8 NVRTC dependency for the baseline.

Run outside the sandbox, with an idle GPU and no other builds or GPU work
during timing:

```sh
CUDA_VISIBLE_DEVICES=1 cargo run --release -p llmcuda-engine --bin bench_rust_add -- \
  crates/llmcuda-cuda/src/kernels/rust/tensor_add.ptx tensor_add
CUDA_VISIBLE_DEVICES=1 cargo test --release -p llmcuda-engine \
  --features llmcuda-cuda/rust-kernels --test layer_ops_differential -- --nocapture
```

The narrow gate checks empty and ragged lengths, decode and prefill shapes,
multiple grid strides, exact CPU agreement, both in-place aliases, guard
elements, and graph replay. It reports six alternating pairs with reversed
order on odd pairs. Each CUDA event interval contains 100 graph-captured
launches. Small shapes therefore reflect repeated cache-hot execution;
whole-model measurements are required before drawing an inference-speed claim.

An optional third argument selects a baseline PTX containing `tensor_add`,
instead of compiling the NVRTC baseline. This compares a compiler or kernel
change directly with the previous Rust artifact. The gate includes all float
alignments modulo 16, independently offset input/output pointers, and all
four-float tail lengths; wider accesses must preserve the sliced-buffer ABI.

To test inside the model, pass `--features llmcuda-cuda/rust-kernels` to
`bench_forward`, `bench_decode_batch`, and the existing forward tests. Keep
separate target directories for candidate and baseline, build both before
timing, and alternate processes on the same card. The production default
remains NVRTC while this feature is evaluated.

The measured correctness and performance results are recorded in
[BENCHMARKS.md](../../docs/BENCHMARKS.md#rust-kernel-authoring-can-keep-the-existing-launcher).

The activation gate uses the same per-tensor tolerances as
`layer_ops_differential`, checks both sigmoid shapes and the sigmoid waypoint,
empty/ragged/grid-stride lengths, saturation, guarded buffers, supported exact
aliases, and graph replay. Six alternating pairs time 100 captured launches
per CUDA event interval, with reversed order on odd pairs:

```sh
CUDA_VISIBLE_DEVICES=1 cargo run --release -p llmcuda-engine --bin bench_rust_activations
CUDA_VISIBLE_DEVICES=1 cargo test --release -p llmcuda-engine --features rust-kernels \
  --test layer_ops_differential --test forward_pass -- --nocapture --test-threads=1
```

`bench_rust_activations` always compares the embedded Rust artifact against
NVRTC, independently of feature selection. The historical `tensor_add.ptx`
filename now contains all eight generated entries. Standalone SwiGLU is not
used by the model's fused MoE/FFN paths. The current feature changes residual
addition, sigmoid gating, the production tiled RoPE and both normalization
paths in the forward pass. Standalone softplus and layer-ops RoPE are API
ports; production uses other gate and rotary implementations. Combined-switch
results are in [BENCHMARKS.md](../../docs/BENCHMARKS.md#rust-normalization-and-rotary-kernels-need-separate-kernel-and-model-gates).

The normalization ports use two levels of warp shuffles and one block
barrier, preserving exact division/square root and the dynamic shared-memory
launch size. The final warp-partial reduction changes the addition order;
CPU and model gates must pass without relaxed tolerances. Small fused rows
keep the input and gate in registers across the reduction. They use the same
`rust-kernels` switch.
`bench_rust_norm` checks both outputs of fused normalization against the CPU
RMSNorm/SwiGLU composition with the existing layer-op tolerances. It covers
sub-warp and ragged widths, multiple grid-stride iterations, zero/sparse
rows, two epsilon values, guards, in-place standalone RMSNorm, and graph
replay before reporting six alternating CUDA-event pairs:

```sh
CUDA_VISIBLE_DEVICES=1 cargo run --release -p llmcuda-engine --bin bench_rust_norm
```

`bench_rust_rope` checks rotated spans against CPU `apply_rope`, exact tail
bits (including signed zero), independent output sentinels, ragged/full/zero
rotary spans, positions through `u32::MAX`, and graph replay. Its event pairs
use positions starting at 2,048, including the 512-token per-sequence band
in an N=3 1,536-row batch, plus 1,536-token individual calls.
The standalone Rust kernel assigns one thread per rotated pair, sharing its double
precision frequency, sine/cosine and input loads. The tail remains a copy.
`bench_rust_activations` also checks softplus, including the overflow-guard
boundary, and measures the 96-element decode and 49,152-element prefill shapes.

```sh
CUDA_VISIBLE_DEVICES=1 cargo run --release -p llmcuda-engine --bin bench_rust_rope
```

The production attention counterpart is `attn_rope_partial_neox`. It already
shares pair calculations and reuses frequencies over 16 tokens in C++.
Its Rust port preserves that band and reads the base position from device
memory. Grids of at most 72 blocks use both halves of
rotary threads for alternate tokens; larger grids keep 16-token frequency
reuse. This heuristic targets the 72-SM RTX 8000 and preserves the launch ABI.
`bench_rust_rope` measures both APIs separately, at the model's 10,000,000
frequency base. Standalone speedups do not imply the same gain
for the tiled production kernel. Fused batched decode rotary/append and
multimodal IMRoPE remain CUDA C++.
