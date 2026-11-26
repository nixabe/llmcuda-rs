# Rust layer kernels

This standalone experiment compiles `src/main.rs` with cuda-oxide from
[NVIDIA/cuda-rust](https://github.com/NVIDIA/cuda-rust) at
`f35d6ccc2e03a54e73e3796f415f11313e439d50` (0.4.0, Rust 1.99.1 with `rustc-dev`). The
compiler used to live at `NVlabs/cuda-oxide`, which now redirects to that
monorepo; cuda-oxide is its `cuda-oxide/` directory.
The engine loads the result through cudarc. Which kernels launch from it, and
what a port has to show before it does, is in
[DEVELOPMENT.md](../../docs/DEVELOPMENT.md#rust-cuda-kernels). Normal builds
need no cuda-oxide. The embedded PTX is generated code, not an independently
maintained kernel.

`regenerate.py` sets `CUDA_OXIDE_NO_FMA=1`: cuda-oxide otherwise lets the
backend fuse any multiply into a following add, and several of these kernels
are compared bit for bit with CUDA C++ originals that deny that fusion. A
kernel fuses where its source says `mul_add`.

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
`src/main.rs` if `llmcuda_rust_device.ptx` is missing.

## Checking a kernel

Run outside the sandbox, on an idle GPU:

```sh
CUDA_VISIBLE_DEVICES=0 cargo test --release -p llmcuda-cuda
CUDA_VISIBLE_DEVICES=0 cargo test --release -p llmcuda-engine \
  --test layer_ops_differential \
  --test gdn_tail_differential --test gdn_conv_silu_differential \
  --test gdn_verify_differential --test attention_differential
```

`llmcuda-cuda`'s unit tests read this source and the generated PTX: tap
order, the rotary tail copy, the reduction's order, and that the convolution
was compiled without an FMA. `layer_ops_differential` holds each kernel to
its CPU reference. The Gated DeltaNet tests compare these kernels exactly
with the fused CUDA C++ paths that share their arithmetic, which is what
catches a port that is close but not identical.

## Porting another kernel

1. Add the kernel here with the CUDA C++ launch ABI and regenerate.
2. In a test, load both the CUDA C++ entry and the Rust one, run them on the
   same inputs through the same launch geometry, and compare `to_bits()`.
   Two things break this in a port that reads like a transcription: a
   multiply-add NVRTC fuses and the Rust does not spell as `mul_add` (or the
   reverse, where the C++ says `__fmul_rn`), and a reduction that adds in a
   different order.
3. Time both in alternating pairs, then confirm in the model
   (`bench_decode_batch`, `bench_forward`) from two prebuilt target
   directories. The port does not have to win; it has to not cost anything
   measurable.
4. Switch the launch site, delete the CUDA C++, and move the tests that read
   its source to this one.

Things cuda-oxide does differently from NVRTC that have cost throughput
here, each of which made a bit-identical port look like it had to stay C++:

- **Loads are issued where the source reads.** A GPU thread executes in
  order, so a load interleaved with a dependent chain waits on it. Read a
  group of elements into locals, then compute: sixteen for a warp's strided
  contraction, eight for a block-strided loop, four ahead of exponentials.
  Same order of operations, so still bit-identical. Bigger groups are not
  always better; measure.
- **No 16-byte loads.** A bandwidth-bound kernel that NVRTC reads through
  `float4` needs `ld.global.v4.f32` / `st.global.v4.f32` in `ptx_asm!`
  (`load4`/`store4` here).
- **No strength reduction of a 64-bit divide.** Carry a row and a column
  along a strided walk instead of dividing the index per element.
- A fixed-size local array indexed by a loop variable stays in local memory;
  index it with literals (a small macro) and it stays in registers.
- A `usize`/`i64` index product inside a loop is a 64-bit multiply per
  iteration. Walk a pointer instead.

Non-coherent loads (`ld.global.nc`) and unrolling without grouping did not
help. Measured results are in
[BENCHMARKS.md](../../docs/BENCHMARKS.md#rust-kernel-authoring-can-keep-the-existing-launcher).

The tiled attention kernel `attn_rope_partial_neox` reuses frequencies over
16 tokens and reads the base position from device memory. Grids of at most 72
blocks use both halves of the rotary threads for alternate tokens; larger
grids keep 16-token frequency reuse. This heuristic targets the 72-SM RTX
8000. Fused batched decode rotary/append and multimodal IMRoPE remain CUDA
C++.
