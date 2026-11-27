//! The 256-expert mixture-of-experts feed-forward block.
//!
//! This block sits on **every one of the 40 transformer layers** — Gated
//! DeltaNet and Gated Attention alike — plus the MTP head at block 40. It is
//! transcribed from `llama_model_qwen35moe::graph::build_layer_ffn`
//! (`/home/nixabe/llama.cpp/src/models/qwen35moe.cpp`) and the
//! `llm_graph_context::build_moe_ffn` / `build_ffn` helpers it calls
//! (`src/llama-graph.cpp`), not from a description of them.
//!
//! ## The block, step by step, with the graph node each step produces
//!
//! ```text
//!   input   = mixer output + residual                       `attn_residual-N`
//!   normed  = RMSNorm(input, post_attention_norm.weight)     `attn_post_norm-N`
//!   logits  = ffn_gate_inp . normed                          (not captured)
//!   probs   = softmax(logits) over all 256 experts           (not captured)
//!   ids     = top-8 by probability, ties to the lower index  (not captured)
//!   w       = probs[ids] / sum(probs[ids])                   (not captured)
//!   routed  = sum_k w_k * down_k(silu(gate_k . normed) * (up_k . normed))
//!                                                            `ffn_moe_out-N`
//!   shexp   = down_s(silu(gate_s . normed) * (up_s . normed))
//!   g       = sigmoid(ffn_gate_inp_shexp . normed)           one scalar / token
//!   ffn_out = routed + shexp * g                             `ffn_out-N`
//!   l_out   = ffn_out + input                                `l_out-N`
//! ```
//!
//! Three details in there are load-bearing and each one is a place an
//! implementation can be plausibly, silently wrong:
//!
//! 1. **The shared expert has a sigmoid gate.** `ffn_gate_inp_shexp` is a
//!    `[hidden]` vector, so `ffn_gate_inp_shexp . normed` is *one scalar per
//!    token*; `build_layer_ffn` sigmoids it and multiplies the whole shared
//!    expert output by it before the sum. [`llmcuda_cuda::kernels::moe`] has no
//!    reference for this — its `shared_expert` entry point implements
//!    `expert_mlp` only — so the gate lives here, at the block level, and
//!    `tests/moe_block.rs` measures its form out of the golden capture rather
//!    than assuming it.
//! 2. **The renormalization is over the selected 8, not all 256.**
//!    `build_moe_ffn` is called with `norm_w = true`, which divides the
//!    gathered weights by their own sum. `w_scale` is
//!    `hparams.expert_weights_scale`, which `qwen35moe`'s
//!    `load_arch_hparams` never reads, so it keeps its `0.0f` default and the
//!    `w_scale != 0.0f && w_scale != 1.0f` guard skips the scaling entirely.
//!    There is no expert-weight scale on this model.
//! 3. **The residual is taken before the post-mixer norm.** `ffn_residual`
//!    in `build_layer_ffn`'s caller is the `attn_residual` tensor, not the
//!    normalized one. Adding the normalized state back instead produces a
//!    model that still generates text.
//!
//! `build_cvec` between `ffn_out + residual` and the `l_out` callback is the
//! identity without a loaded control vector, so `l_out-N` is exactly the sum.
//!
//! ## Mixed quantization is not optional
//!
//! `Qwen3.6-35B-A3B-UD-Q6_K_XL` stores the expert stacks in **more than one
//! format, and not uniformly per layer**:
//!
//! - `ffn_down_exps` is Q8_0 on all 41 blocks;
//! - `ffn_gate_exps` / `ffn_up_exps` are Q6_K on blocks 0–38 and 40, but
//!   **Q8_0 on block 39**.
//!
//! So the format is read per tensor out of the file's own directory and
//! carried alongside the pointer in [`llmcuda_cuda::kernels::moe::QuantTensor`].
//! Anything that decides the format once per layer — never mind once per
//! model — reads block 39's gate and up projections as garbage.
//!
//! The two router tensors have the same problem in the other direction:
//! `blk.40.ffn_gate_inp.weight` and `blk.40.ffn_gate_inp_shexp.weight` are
//! the **only two `bf16` tensors in the file** (`docs/ORACLE.md` §8.4); every
//! router on blocks 0–39 is `f32`. [`MoeLayerWeights::upload`] widens `bf16`
//! to `f32` on the host — which is exact, `bf16` being a truncated `f32` —
//! and rejects any other dense type by name rather than reading it as
//! something it is not.
//!
//! ## Why the weights are uploaded here rather than aliased out of the arena
//!
//! [`crate::DeviceWeights`] holds the whole model in one
//! [`llmcuda_cuda::DeviceArena`] and hands out [`llmcuda_cuda::Allocation`] byte
//! ranges into a single `CudaSlice<u8>`. `cudarc` 0.19 can produce a borrowed
//! `CudaView` of a sub-range but not an owned `CudaSlice`, and
//! [`llmcuda_cuda::kernels::moe::QuantTensor`] takes `&CudaSlice<u8>`. So
//! [`MoeLayerWeights`] uploads one layer's tensors into allocations of their
//! own.
//!
//! That does **not** double the model on the device: [`crate::forward`] excludes
//! the nine roles this type uploads from the arena entirely
//! ([`crate::DeviceWeights::load_where`]), so every expert tensor is resident
//! exactly once — in 41 sets of allocations rather than in the slab. It also
//! keeps a test able to walk several layers without the model ever being
//! resident, since one layer is ~725 MiB against the model's 29.65 GiB.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};
use llmcuda_cuda::kernels::compile;
use llmcuda_cuda::kernels::layer_ops::{LayerOpsError, LayerOpsKernels};
use llmcuda_cuda::kernels::moe::{
    ExpertQuant, MoeBuffers, MoeError, MoeGeometry, MoeKernels, QuantTensor, SharedExpertInt8,
    to_device_layout,
};
use llmcuda_gguf::{GgmlType, GgufFile};
use llmcuda_model::config::ModelConfig;
use llmcuda_model::weights::{Directory, Role};

/// Threads per block for the three glue kernels below.
const THREADS: u32 = 256;

/// Lanes that contribute to the shared expert's gate dot product.
///
/// Mirrors `MOE_GATE_LANES`, asserted below. Fixed rather than "the block
/// width" because the gate is a floating-point sum: changing how many lanes
/// carry it reassociates 2,048 products and moves the gate.
const GATE_LANES: u32 = 256;

/// Threads in the fused gate-and-combine block.
///
/// Equal to [`GATE_LANES`], which is what makes the gate's reduction the same
/// shape at any grid the combine half wants: threads past the gate's lanes
/// would add literal zeros, and `block_reduce_sum` puts those zeros in the
/// same places whatever the block's width is.
const COMBINE_THREADS: u32 = GATE_LANES;

/// Tokens the wide router instantiation carries. Mirrors its `TT`.
const ROUTER_TT: u32 = 8;
/// Exact token tile for the serving target's three-sequence decode shape.
const ROUTER_TT_N3: u32 = 3;

/// Token count below which the shared expert stays on the fp32 kernel.
///
/// Higher than the routed path's threshold of 8, and measured rather than
/// reasoned. The fp32 shared expert is *two* launches with the SwiGLU fused
/// into the first; the integer path is six — two activation quantizations,
/// three projections and a separate SwiGLU — because `q8_0_proj_split` is a
/// projection and knows nothing about what its output feeds. Six launches and
/// two extra sweeps over `[tokens][intermediate]` need a wide batch to pay
/// for themselves.
///
/// Measured end to end, tok/s at each batch with the threshold at 8 (so the
/// integer path ran) against the fp32 path:
///
/// |  19 | 265.34 -> 240.42 | -9.4%  |
/// | 128 | 893.27 -> 895.87 | +0.3%  |
/// | 512 | 1248.27 -> 1352.61 | +8.4% |
///
/// 128 is where it stops costing anything. Also gates the repack itself, so a
/// decode-shaped pass does not pay 3.5 MB per layer for weights it will never
/// read.
pub(crate) const SHARED_MMA_MIN_TOKENS: usize = 128;
/// Experts one router block covers. Mirrors `ROUTER_ET`.
///
/// The router GEMM is deeply L2-bound: a block moves
/// `(ROUTER_ET + ROUTER_TT) * hidden` floats to do `ROUTER_ET * ROUTER_TT *
/// hidden` multiply-adds, so its arithmetic intensity is the harmonic mean of
/// the two tile widths and widening the narrower one is what pays.
///
/// | experts per block | tok/s |
/// | --- | ---: |
/// | 4 | 1,693.17 |
/// | **8** | **1,721.84** |
/// | 16 | 1,707.63 |
///
/// 16 is past the knee: 128 accumulators per thread. Widening the *token*
/// tile to 16 instead measured 1,703 — same register wall, and it also
/// doubles the staged activation tile.
const ROUTER_ET: u32 = 8;
/// Contraction the router stages per trip. Mirrors `ROUTER_JC`, and **must**
/// equal [`THREADS`] — that equality is what preserves the untiled kernel's
/// per-thread summation order, which the router cannot afford to change.
const ROUTER_JC: u32 = THREADS;

/// Tokens, experts and contraction one wide router block stages. Mirror
/// `ROUTER_WIDE_TT`, `ROUTER_WIDE_ET` and `ROUTER_WIDE_KC`.
const ROUTER_WIDE_TT: u32 = 64;
const ROUTER_WIDE_ET: u32 = 32;
const ROUTER_WIDE_KC: u32 = 16;
/// Warps in the wide router's every-expert block. Mirrors the `EW` of
/// `moe_block_router_logits_wide_w8`.
const ROUTER_WIDE_FULL_WARPS: u32 = 8;
/// Token count from which the router runs as a tiled GEMM
/// (`moe_block_router_logits_wide`) instead of the register-tiled kernel.
///
/// The wide kernel pads its token tile to 64 and walks the whole contraction
/// in one block, which takes about 0.35 ms however few blocks there are, so
/// a narrow pass is faster on the register-tiled kernel. `bench_router`, ms a
/// launch on an idle RTX 8000, three alternating rounds:
///
/// | tokens | register-tiled | one warp | eight warps |
/// | ---: | ---: | ---: | ---: |
/// | 512 | 0.19 | 0.34–0.45 | 0.60 |
/// | 768 | 0.36 | 0.47 | 0.62 |
/// | 1,024 | 0.35–0.37 | 0.34–0.38 | 0.47–0.50 |
/// | 1,536 | 0.54–0.59 | 0.36–0.39 | 0.48–0.53 |
/// | 2,048 | 0.69–0.79 | 0.40–0.44 | 0.49–0.52 |
/// | 3,072 | 1.12–1.20 | 0.58–0.60 | 0.52–0.54 |
/// | 4,096 | 1.46–1.59 | 0.64–0.66 | 0.53–0.55 |
/// | 6,138 | 2.15–2.26 | 1.10–1.14 | 0.89–0.95 |
/// | 8,192 | 3.13–3.21 | 1.29–1.30 | 0.98–1.00 |
///
/// It also has to stay above the routed experts' own tensor-core threshold:
/// below that a pass may be a decode or a verify step, whose router must
/// stay bit-identical across widths.
pub const ROUTER_WIDE_MIN_TOKENS: usize = 1024;
/// Token count from which a wide router block is eight warps covering every
/// expert rather than one warp covering 32. See [`ROUTER_WIDE_MIN_TOKENS`]
/// for the measurement.
pub const ROUTER_WIDE_FULL_MIN_TOKENS: usize = 2560;

/// The width from which a pass routes through the tiled GEMM, as
/// `LLMCUDA_ROUTER_WIDE` leaves it. Read once, at construction.
///
/// - unset, or anything else: [`ROUTER_WIDE_MIN_TOKENS`].
/// - `0`: never. Every width stays on the register-tiled kernels, for
///   `bench_forward`'s `LLMCUDA_BENCH_AB` and as the way back to one router
///   order at every width.
/// - `force`: at every width that runs prefill arithmetic at all, so the
///   19-token golden prompt exercises the wide order through the whole model
///   (`tests/forward_pass.rs`), which the default threshold never lets it.
fn router_wide_min_tokens_from_env() -> usize {
    match std::env::var("LLMCUDA_ROUTER_WIDE").as_deref() {
        Ok("0") => usize::MAX,
        Ok("force") => 0,
        _ => ROUTER_WIDE_MIN_TOKENS,
    }
}

/// Token count from which the combine launches one block per token instead
/// of splitting each token's row over eight.
///
/// Four blocks to an SM on this card: below that the split is what keeps the
/// card busy. The two launch shapes are bit-identical, so unlike the router's
/// this threshold is purely a throughput choice.
pub const COMBINE_ROW_MIN_TOKENS: usize = 288;

/// [`COMBINE_ROW_MIN_TOKENS`], or never with `LLMCUDA_COMBINE_ROW=0` (for
/// `bench_forward`'s `LLMCUDA_BENCH_AB`). Read once, at construction.
fn combine_row_min_tokens_from_env() -> usize {
    match std::env::var("LLMCUDA_COMBINE_ROW").as_deref() {
        Ok("0") => usize::MAX,
        _ => COMBINE_ROW_MIN_TOKENS,
    }
}

/// RMS epsilon when the file does not carry one.
///
/// llama.cpp reads `f_norm_rms_eps` from
/// `qwen35moe.attention.layer_norm_rms_epsilon` and `ModelConfig` has no
/// epsilon field, so [`MoeBlock::eps_from`] prefers the file and falls back to
/// the Qwen3 family's value.
pub const DEFAULT_RMS_EPS: f32 = 1e-6;

/// The architecture-scoped GGUF key suffix llama.cpp reads the RMS epsilon
/// from.
///
/// Both `qwen35moe` and `qwen35` carry it under their own prefix, so the full
/// key comes from [`ModelConfig::hparam_key`] rather than from a literal.
pub(crate) const RMS_EPS_SUFFIX: &str = "attention.layer_norm_rms_epsilon";

/// The three operations the MoE block needs that no landed kernel provides.
///
/// All three are dense fp32 and tiny next to the grouped GEMM — the router
/// projection is 2048x256 per layer against 256 experts of 3x2048x512 — but
/// they must still run on the device, because a host round-trip for the
/// router logits would put a synchronization between every layer and its
/// successor and AGENTS.md rule 5 forbids anything on this path whose size
/// the host has to know.
///
/// Every launch shape here comes from the geometry and the token count lives
/// in the same `valid_tokens` device scalar the MoE kernels already gate on,
/// so adding these does not cost the sequence its capturability.
const GLUE_SRC: &str = r#"
#define MOE_GATE_LANES 256
extern "C" {

extern __shared__ float llmcuda_moe_block_shared[];

// logits[t][e] = dot(ffn_gate_inp row e, normed[t]).
//
// `w` is the GGUF tensor `[hidden, num_experts]` with ne[0] fastest-varying,
// so expert e's row is `hidden` *contiguous* floats at `e * hidden`. Reading
// it the other way round would gather at stride `num_experts` and produce a
// router that is wrong on every token while still summing to one.
//
// # Why this is tiled and the obvious version is not viable
//
// This is the smallest matmul in the layer — 2048x256 against 256 experts of
// 3x2048x512 — and the first version, one block per (expert, token) pair
// reducing a 2048-term dot product, was **7.6% of the whole forward pass**.
// The arithmetic was never the problem. At 512 tokens that grid is 131,072
// blocks, and each one re-read a full 2048-float weight row *and* a full
// 2048-float activation row: 2.1 GB of loads per layer to cover 6 MB of
// distinct data.
//
// A block now covers ROUTER_ET experts and ROUTER_TT tokens at once. The
// weight row is read once per token tile instead of once per token, the
// activation rows once per expert *group* instead of once per expert, and the
// traffic falls about fivefold. Both reads stay fully coalesced.
//
// # Why the summation order is preserved exactly
//
// A first attempt gave each warp its own expert and reduced with a plain warp
// shuffle. It was faster and it was wrong: the per-thread partition of the
// contraction changed, the logits moved in their last bits, and `tests/
// forward_pass.rs` caught block 31's error jumping 5.26x. The router's output
// is not consumed as a number — it is consumed by a top-8 argmax over 256
// experts. A last-bit disagreement between two adjacent logits does not
// perturb an answer slightly; it runs a different expert.
//
// So `ROUTER_JC` equals the block width and the staging loop hands thread
// `tid` the indices `tid, tid + 256, tid + 512, ...` in that order — the
// sequence the untiled `for (j = threadIdx.x; j < hidden; j += blockDim.x)`
// produced — and the reduction repeats `block_reduce_sum`'s shuffle-down and
// ascending warp sum. The logits are bit-identical; only the traffic changed.
//
// That constraint is also why the activation tile is *not* vectorized. A
// `float4` load would hand thread `tid` the indices `4*tid .. 4*tid + 3`, a
// different partition of the contraction, and the note above is what happens
// then: the logits move in their last bits and a different expert runs.
//
// # Why the activation tile is registers and not shared memory
//
// It was shared, and it never needed to be. The staging loop wrote
// `sx[u * ROUTER_JC + tid]` and the inner loop read `sx[u * ROUTER_JC + tid]`
// — the same index, so every thread was the only reader of every slot it
// wrote. `ROUTER_JC == THREADS` is exactly the condition that makes the tile
// private, and it is required for the summation order anyway.
//
// The round trip cost a store, `ROUTER_ET` reloads per token, and two
// barriers, every trip: 88 memory instructions per trip to do 64 multiply-
// adds. As `TT` registers it is 16, the barriers go away, and the block's
// shared footprint drops from 10 KiB to 2 KiB, which is four resident blocks
// per SM instead of one. Bit-identical — the same values in the same order,
// read from a register instead of from shared.
//
// grid: (num_experts / ROUTER_ET, ceil(max_tokens / ROUTER_TT)) — never the
// live token count.

// Tokens one block carries, and experts it covers. The product is the 32
// accumulators each thread holds. `ROUTER_JC` is the contraction staged per
// trip and **must equal the block width**, which is what keeps each thread's
// summation order identical to the untiled kernel's.
#define ROUTER_ET 8
#define ROUTER_JC 256

// Instantiated at two token tile widths. `ROUTER_JC` equals the block width in
// both, which is what keeps each thread's summation order identical to the
// untiled kernel's — and identical *between* the two instantiations, so a
// decode step and a prefill step agree bit for bit on the logits of any token
// they share.
#define ROUTER_LOGITS(NAME, TT)                                                \
__global__ void NAME(                                                          \
    const float* __restrict__ w,                                               \
    const float* __restrict__ x,                                               \
    const int* __restrict__ valid_tokens,                                      \
    int hidden,                                                                \
    int num_experts,                                                           \
    int max_tokens,                                                            \
    float* __restrict__ logits                                                 \
) {                                                                            \
    float* scratch = llmcuda_moe_block_shared;                                    \
                                                                               \
    int tid  = threadIdx.x;                                                    \
    int lane = tid & 31;                                                       \
    int warp = tid >> 5;                                                       \
    int n_warps = (int)(blockDim.x >> 5);                                      \
    int e0 = blockIdx.x * ROUTER_ET;                                           \
    int t0 = blockIdx.y * TT;                                                  \
                                                                               \
    float acc[ROUTER_ET][TT];                                                  \
    _Pragma("unroll")                                                          \
    for (int el = 0; el < ROUTER_ET; ++el)                                     \
        _Pragma("unroll")                                                      \
        for (int u = 0; u < TT; ++u) acc[el][u] = 0.0f;                        \
                                                                               \
    for (int jc = 0; jc < hidden; jc += ROUTER_JC) {                           \
        int j = jc + tid;                                                      \
        float xv[TT];                                                          \
        _Pragma("unroll")                                                      \
        for (int u = 0; u < TT; ++u) {                                         \
            int t = (t0 + u < max_tokens) ? t0 + u : max_tokens - 1;           \
            xv[u] = j < hidden ? x[(long long)t * hidden + j] : 0.0f;          \
        }                                                                      \
        if (j < hidden) {                                                      \
            _Pragma("unroll")                                                  \
            for (int el = 0; el < ROUTER_ET; ++el) {                           \
                int e = e0 + el;                                               \
                float wj = e < num_experts ? w[(long long)e * hidden + j] : 0.0f; \
                _Pragma("unroll")                                              \
                for (int u = 0; u < TT; ++u) {                                 \
                    acc[el][u] += wj * xv[u];                                  \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
                                                                               \
    _Pragma("unroll")                                                          \
    for (int el = 0; el < ROUTER_ET; ++el) {                                   \
        _Pragma("unroll")                                                      \
        for (int u = 0; u < TT; ++u) {                                         \
            float v = acc[el][u];                                              \
            _Pragma("unroll")                                                  \
            for (int offset = 16; offset > 0; offset >>= 1) {                  \
                v += __shfl_down_sync(0xffffffff, v, offset);                  \
            }                                                                  \
            if (lane == 0) scratch[warp * (ROUTER_ET * TT) + el * TT + u] = v; \
        }                                                                      \
    }                                                                          \
    __syncthreads();                                                           \
                                                                               \
    int live = *valid_tokens;                                                  \
    if (tid < ROUTER_ET * TT) {                                                \
        int el = tid / TT;                                                     \
        int u  = tid % TT;                                                     \
        float total = 0.0f;                                                    \
        for (int wv = 0; wv < n_warps; ++wv) {                                 \
            total += scratch[wv * (ROUTER_ET * TT) + el * TT + u];             \
        }                                                                      \
        int e = e0 + el;                                                       \
        int t = t0 + u;                                                        \
        if (e < num_experts && t < live) {                                     \
            logits[(long long)t * num_experts + e] = total;                    \
        }                                                                      \
    }                                                                          \
}

// Eight tokens for prefill. One for decode, where seven of the eight
// accumulators would be a token clamped to the same row — 7/8 of the
// arithmetic and 7/8 of the staged tile spent recomputing one answer.
ROUTER_LOGITS(moe_block_router_logits,    8)
ROUTER_LOGITS(moe_block_router_logits_t3, 3)
ROUTER_LOGITS(moe_block_router_logits_t1, 1)

// The same logits for a prefill-width pass, as a tiled GEMM.
//
// The kernels above keep one accumulator per (expert, token, thread), which
// is what fixes their summation order and also what bounds them: a thread
// has room for about 64 accumulators, so a block can share one pass over the
// contraction between 8 experts and 8 tokens and no more. It moves
// `(8 + 8) * hidden` floats to do `8 * 8 * hidden` multiply-adds, a byte of
// traffic per multiply-add, and at prefill width that is what it is bound
// by: 3.1 GB a layer through L2 for 6,138 tokens.
//
// This one keeps one accumulator per (expert, token) and nothing else, so a
// thread's 64 accumulators are an 8x8 patch of the output and a warp covers
// 64 tokens by 32 experts. The contraction is staged through shared memory
// `ROUTER_WIDE_KC` at a time, each staged value is read by eight threads,
// and the block moves `(64 + 32) * hidden` floats for `64 * 32 * hidden`
// multiply-adds: a fifth of the traffic.
//
// # The summation order, which is not the narrow kernels'
//
// `logits[t][e]` is a single chain over `j` ascending from zero,
// `acc = fma(w[e][j], x[t][j], acc)`. That is a different association from
// the narrow kernels' 256 chains of eight, so the two families do not agree
// to the last bit and are not meant to. What this kernel keeps is that the
// order is the same for every token at every width and every tile position,
// so a token's logits do not depend on what else is in the pass, and the
// host reproduces it exactly with `f32::mul_add`.
//
// It is selected only where the routed experts already run their integer
// tensor-core path, which is not bit-identical with decode either. A
// decode step and a speculative verify window never reach it.
//
// # Layout
//
// One warp per block and no barrier: lane `l` owns tokens `8 * (l / 4)` to
// `+ 7` and experts `8 * (l % 4)` to `+ 7` of the block's tile.
//
// Both staged tiles are row-major with a row stride of `ROUTER_WIDE_KC + 1`
// floats, and both hold their rows in a permuted order. Each choice removes
// a specific cost, and the first one was measured: staging the same tiles
// contraction-major, each lane loading its own row sixteen bytes at a time,
// spent 0.55 ms of a 1.25 ms launch at 6,138 tokens on loads that touched 32
// cache lines an instruction.
//
// - **Row-major**, so a staging load is half a warp reading sixteen
//   consecutive floats of one row and the other half sixteen of another: two
//   coalesced 64-byte reads an instruction.
// - **Stride 17**, so the eight lanes that read eight different rows at the
//   same `kk` in the arithmetic land on eight different banks: `17 * i mod
//   32` is distinct for `i < 32`. At stride 16 they would all share one.
// - **Experts at `q = 4 * (e % 8) + e / 8`** and **tokens at
//   `p = 8 * (t % 8) + t / 8`**, so those eight (or four) rows are
//   consecutive positions `q + (l % 4)` and `p + (l / 4)`.
// - The two rows one staging instruction stores are 16 positions apart,
//   which is 16 banks apart at this stride, so the store is conflict-free
//   as well.
//
// grid: (num_experts / ROUTER_WIDE_ET, ceil(max_tokens / ROUTER_WIDE_TT)) —
// never the live token count. block: one warp.
#define ROUTER_WIDE_TT 64
#define ROUTER_WIDE_ET 32
#define ROUTER_WIDE_KC 16
#define ROUTER_WIDE_STRIDE (ROUTER_WIDE_KC + 1)

} // extern "C"

// `EW` warps, each with its own 32 experts, over one staged 64-token tile.
// The tile and each warp's arithmetic are the same at any `EW`, so the two
// instantiations below agree bit for bit.
template<int EW>
__device__ __forceinline__ void router_logits_wide_impl(
    const float* __restrict__ w,
    const float* __restrict__ x,
    const int* __restrict__ valid_tokens,
    int hidden,
    int num_experts,
    int max_tokens,
    float* __restrict__ logits
) {
    int lane = threadIdx.x & 31;
    int wi   = threadIdx.x >> 5;
    float* sw = llmcuda_moe_block_shared + wi * (ROUTER_WIDE_ET * ROUTER_WIDE_STRIDE);
    float* sx = llmcuda_moe_block_shared + EW * (ROUTER_WIDE_ET * ROUTER_WIDE_STRIDE);

    int live = *valid_tokens;
    int t0 = blockIdx.y * ROUTER_WIDE_TT;
    if (t0 >= live) return;
    int e0 = (blockIdx.x * EW + wi) * ROUTER_WIDE_ET;

    int ti = lane >> 2;
    int ei = lane & 3;
    // Staging: which of an instruction's two rows this lane loads, and which
    // of its sixteen floats.
    int half = lane >> 4;
    int col  = lane & 15;

    // A lane whose eight tokens are all past the live count stages its share
    // (other lanes read it) and skips the arithmetic.
    bool compute = t0 + ti * 8 < live;

    float acc[8][8];
    #pragma unroll
    for (int u = 0; u < 8; ++u)
        #pragma unroll
        for (int v = 0; v < 8; ++v) acc[u][v] = 0.0f;

    for (int k0 = 0; k0 < hidden; k0 += ROUTER_WIDE_KC) {
        // This warp's own expert tile.
        #pragma unroll
        for (int s = 0; s < ROUTER_WIDE_ET / 2; ++s) {
            int q = s + 16 * half;
            int e = e0 + (q & 3) * 8 + (q >> 2);
            sw[q * ROUTER_WIDE_STRIDE + col] = w[(long long)e * hidden + k0 + col];
        }
        // The token tile, its 32 row pairs dealt round the block's warps.
        #pragma unroll
        for (int i = 0; i < ROUTER_WIDE_TT / 2 / EW; ++i) {
            int s = wi + i * EW;
            int p = (s < 16 ? s : s + 16) + 16 * half;
            int t = t0 + (p & 7) * 8 + (p >> 3);
            if (t >= max_tokens) t = max_tokens - 1;
            sx[p * ROUTER_WIDE_STRIDE + col] = x[(long long)t * hidden + k0 + col];
        }
        if (EW == 1) __syncwarp(); else __syncthreads();

        if (compute) {
            #pragma unroll
            for (int kk = 0; kk < ROUTER_WIDE_KC; ++kk) {
                float wv[8];
                #pragma unroll
                for (int v = 0; v < 8; ++v) {
                    wv[v] = sw[(v * 4 + ei) * ROUTER_WIDE_STRIDE + kk];
                }
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    float xv = sx[(u * 8 + ti) * ROUTER_WIDE_STRIDE + kk];
                    #pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        acc[u][v] = fmaf(wv[v], xv, acc[u][v]);
                    }
                }
            }
        }
        if (EW == 1) __syncwarp(); else __syncthreads();
    }

    #pragma unroll
    for (int u = 0; u < 8; ++u) {
        int t = t0 + ti * 8 + u;
        if (t < live) {
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                logits[(long long)t * num_experts + e0 + ei * 8 + v] = acc[u][v];
            }
        }
    }
}

extern "C" {

#define ROUTER_LOGITS_WIDE(NAME, EW)                                           \
__global__ void __launch_bounds__(32 * EW) NAME(                               \
    const float* __restrict__ w,                                               \
    const float* __restrict__ x,                                               \
    const int* __restrict__ valid_tokens,                                      \
    int hidden,                                                                \
    int num_experts,                                                           \
    int max_tokens,                                                            \
    float* __restrict__ logits                                                 \
) {                                                                            \
    router_logits_wide_impl<EW>(w, x, valid_tokens, hidden, num_experts,       \
                                max_tokens, logits);                           \
}

// One warp a block from 1,024 tokens: 32 experts a block, eight blocks a
// token tile, which is what spreads a mid-width pass over the card. Eight
// warps a block from 2,560: every expert in one block, so each activation
// row is read from global memory once a launch instead of eight times, which
// is what the one-warp shape is bound by at depth.
ROUTER_LOGITS_WIDE(moe_block_router_logits_wide,    1)
ROUTER_LOGITS_WIDE(moe_block_router_logits_wide_w8, 8)

}
"#;

/// Something went wrong building or running a feed-forward block.
///
/// Shared with [`crate::block::dense_ffn`], which runs on the same kernels
/// and can fail in the same ways; splitting it would have produced a second
/// enum whose variants were all the same.
#[derive(Debug)]
pub enum MoeBlockError {
    /// A MoE kernel failed.
    Moe(MoeError),
    /// A layer-op kernel failed.
    LayerOps(LayerOpsError),
    /// The driver failed.
    Driver(DriverError),
    /// NVRTC rejected the glue source, or the module failed to load.
    Compile(String),
    /// The weight directory does not carry a tensor this block needs.
    MissingTensor { role: Role, layer: u32 },
    /// The file's data section is shorter than its directory claims.
    UnreadableTensor { name: String },
    /// A dense tensor is stored in a type this block will not guess at.
    ///
    /// Named rather than silently reinterpreted: the two `bf16` routers on
    /// block 40 are exactly the case where reading the bytes as something
    /// else produces finite, plausible, wrong logits.
    UnsupportedDenseType { name: String, found: GgmlType },
    /// An expert stack is stored in a type the grouped GEMM cannot unpack.
    UnsupportedExpertType { name: String, found: GgmlType },
    /// A tensor does not hold the element count the geometry implies.
    ShapeMismatch {
        name: String,
        expected: usize,
        found: usize,
    },
    /// A buffer handed to [`MoeBlock::forward`] is the wrong length.
    BufferLength {
        what: &'static str,
        expected: usize,
        found: usize,
    },
    /// More tokens were presented than the block was sized for.
    TooManyTokens { tokens: usize, max_tokens: usize },
    /// A dense layer is resident as the integer repack but the block's
    /// tensor cores were disabled after it was uploaded.
    ///
    /// Reachable only through `Forward::disable_tensor_cores`, which exists
    /// for `tests/int8_forward.rs` and targets the routed model. Reported
    /// rather than silently falling back, because there is nothing to fall
    /// back *to*: the Q8_0 upload was dropped when the repack replaced it.
    Int8WeightsWithoutTensorCores { layer: u32 },
    /// A feed-forward block was handed the other architecture's weights.
    ///
    /// Structurally unreachable once a pass is built — both come from the
    /// same [`llmcuda_model::config::ModelConfig`] — but checked rather than
    /// asserted, because the alternative failure is a shape error three
    /// launches deep that says nothing about which model was loaded.
    FfnKindMismatch {
        block: &'static str,
        weights: &'static str,
    },
}

impl std::fmt::Display for MoeBlockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Moe(e) => write!(f, "{e}"),
            Self::LayerOps(e) => write!(f, "{e}"),
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::Compile(m) => write!(f, "glue kernel compilation failed: {m}"),
            Self::MissingTensor { role, layer } => {
                write!(f, "layer {layer} has no `{role}`")
            }
            Self::UnreadableTensor { name } => {
                write!(
                    f,
                    "tensor `{name}` is named by the directory but not readable"
                )
            }
            Self::UnsupportedDenseType { name, found } => write!(
                f,
                "tensor `{name}` is {}, and this block only widens f32 and bf16 — \
                 reading it as anything else would produce plausible garbage",
                found.name(),
            ),
            Self::UnsupportedExpertType { name, found } => write!(
                f,
                "expert stack `{name}` is {}, which the grouped GEMM cannot unpack \
                 (it handles Q6_K and Q8_0)",
                found.name(),
            ),
            Self::ShapeMismatch {
                name,
                expected,
                found,
            } => write!(
                f,
                "tensor `{name}` holds {found} elements, the geometry implies {expected}",
            ),
            Self::BufferLength {
                what,
                expected,
                found,
            } => write!(f, "{what} must hold {expected} floats, holds {found}"),
            Self::TooManyTokens { tokens, max_tokens } => write!(
                f,
                "{tokens} tokens exceeds the {max_tokens} this block was sized for",
            ),
            Self::Int8WeightsWithoutTensorCores { layer } => write!(
                f,
                "block {layer}'s dense FFN is resident as the integer repack, which needs \
                 the tensor cores this block was told to stop using",
            ),
            Self::FfnKindMismatch { block, weights } => write!(
                f,
                "a `{block}` feed-forward block was handed `{weights}` weights",
            ),
        }
    }
}

impl std::error::Error for MoeBlockError {}

impl From<MoeError> for MoeBlockError {
    fn from(e: MoeError) -> Self {
        Self::Moe(e)
    }
}

impl From<LayerOpsError> for MoeBlockError {
    fn from(e: LayerOpsError) -> Self {
        Self::LayerOps(e)
    }
}

impl From<DriverError> for MoeBlockError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

/// One layer's MoE weights, resident on the device.
///
/// Uploaded per layer rather than taken out of [`crate::DeviceWeights`]; see
/// the module docs for why. The quantization format of each expert stack is
/// read out of the file's own tensor directory and kept next to the bytes,
/// because this model's is **not uniform** — not even within one layer, and
/// not even across layers for one role.
pub struct MoeLayerWeights {
    layer: u32,
    post_norm: CudaSlice<f32>,
    router: CudaSlice<f32>,
    gate_exps: CudaSlice<u8>,
    up_exps: CudaSlice<u8>,
    down_exps: CudaSlice<u8>,
    gate_quant: ExpertQuant,
    up_quant: ExpertQuant,
    down_quant: ExpertQuant,
    shared_gate_inp: CudaSlice<f32>,
    shared_gate: CudaSlice<u8>,
    shared_up: CudaSlice<u8>,
    shared_down: CudaSlice<u8>,
    shared_gate_quant: ExpertQuant,
    shared_up_quant: ExpertQuant,
    shared_down_quant: ExpertQuant,
    /// The shared expert repacked for the integer tensor cores, when its three
    /// matrices are Q8_0 and the pass is wide enough to want them. About
    /// 3.5 MB per layer.
    shared_int8: Option<SharedExpertInt8>,
    /// How the router and the shared-expert gate were stored in the file.
    router_type: GgmlType,
    shared_gate_inp_type: GgmlType,
}

impl MoeLayerWeights {
    /// Upload every tensor block `layer`'s MoE needs.
    ///
    /// `directory` must have been resolved against `file`.
    pub fn upload(
        stream: &Arc<CudaStream>,
        file: &GgufFile,
        directory: &Directory<'_>,
        layer: u32,
        geometry: &MoeGeometry,
    ) -> Result<Self, MoeBlockError> {
        let hidden = geometry.hidden;
        let one_expert = geometry.intermediate * hidden;
        let stack = geometry.stack_elements();

        let (post_norm, _) = dense_tensor(file, directory, Role::PostMixerNorm, layer, hidden)?;
        let (router, router_type) = dense_tensor(
            file,
            directory,
            Role::MoeRouter,
            layer,
            geometry.num_experts * hidden,
        )?;
        let (shared_gate_inp, shared_gate_inp_type) =
            dense_tensor(file, directory, Role::MoeSharedGateInp, layer, hidden)?;

        let (gate_bytes, gate_quant) =
            quantized_tensor(file, directory, Role::MoeGateExps, layer, stack)?;
        let (up_bytes, up_quant) =
            quantized_tensor(file, directory, Role::MoeUpExps, layer, stack)?;
        let (down_bytes, down_quant) =
            quantized_tensor(file, directory, Role::MoeDownExps, layer, stack)?;

        let (sgate_bytes, shared_gate_quant) =
            quantized_tensor(file, directory, Role::MoeSharedGate, layer, one_expert)?;
        let (sup_bytes, shared_up_quant) =
            quantized_tensor(file, directory, Role::MoeSharedUp, layer, one_expert)?;
        let (sdown_bytes, shared_down_quant) =
            quantized_tensor(file, directory, Role::MoeSharedDown, layer, one_expert)?;

        let shared_gate = stream.clone_htod(&*to_device_layout(shared_gate_quant, sgate_bytes))?;
        let shared_up = stream.clone_htod(&*to_device_layout(shared_up_quant, sup_bytes))?;
        let shared_down = stream.clone_htod(&*to_device_layout(shared_down_quant, sdown_bytes))?;
        // Repacked here, at upload, rather than lazily on the first pass:
        // `AGENTS.md` rule 6 forbids allocating mid-forward, and a lazy repack
        // would also poison the first timed repetition of every benchmark.
        let shared_int8 = if geometry.max_tokens >= SHARED_MMA_MIN_TOKENS
            && [shared_gate_quant, shared_up_quant, shared_down_quant]
                .iter()
                .all(|q| *q == ExpertQuant::Q8_0)
        {
            SharedExpertInt8::repack(
                stream.context(),
                stream,
                QuantTensor {
                    bytes: &shared_gate,
                    quant: shared_gate_quant,
                },
                QuantTensor {
                    bytes: &shared_up,
                    quant: shared_up_quant,
                },
                QuantTensor {
                    bytes: &shared_down,
                    quant: shared_down_quant,
                },
                one_expert,
            )
            .ok()
        } else {
            None
        };

        Ok(Self {
            layer,
            post_norm: stream.clone_htod(&post_norm)?,
            router: stream.clone_htod(&router)?,
            // Re-strided on the way in: the kernels index superblocks by
            // `ExpertQuant::block_bytes`, which is not what the file uses.
            gate_exps: stream.clone_htod(&*to_device_layout(gate_quant, gate_bytes))?,
            up_exps: stream.clone_htod(&*to_device_layout(up_quant, up_bytes))?,
            down_exps: stream.clone_htod(&*to_device_layout(down_quant, down_bytes))?,
            gate_quant,
            up_quant,
            down_quant,
            shared_gate_inp: stream.clone_htod(&shared_gate_inp)?,
            shared_gate,
            shared_up,
            shared_down,
            shared_gate_quant,
            shared_up_quant,
            shared_down_quant,
            shared_int8,
            router_type,
            shared_gate_inp_type,
        })
    }

    /// Which block these weights came from.
    pub fn layer(&self) -> u32 {
        self.layer
    }

    /// Storage format of each routed expert stack, in gate/up/down order.
    ///
    /// Exposed so a caller — or a test — can assert what it actually got
    /// rather than what it expected. On this model the answer is not the same
    /// for every layer.
    pub fn expert_quants(&self) -> [ExpertQuant; 3] {
        [self.gate_quant, self.up_quant, self.down_quant]
    }

    /// Storage format of the shared expert's three matrices.
    pub fn shared_quants(&self) -> [ExpertQuant; 3] {
        [
            self.shared_gate_quant,
            self.shared_up_quant,
            self.shared_down_quant,
        ]
    }

    /// How the file stored this layer's two router tensors.
    ///
    /// `(ffn_gate_inp, ffn_gate_inp_shexp)`. `f32` everywhere except block 40,
    /// where both are `bf16` and nothing else in the file is.
    pub fn router_types(&self) -> (GgmlType, GgmlType) {
        (self.router_type, self.shared_gate_inp_type)
    }

    /// Total device bytes held.
    pub fn bytes(&self) -> usize {
        (self.post_norm.len() + self.router.len() + self.shared_gate_inp.len()) * size_of::<f32>()
            + self.gate_exps.len()
            + self.up_exps.len()
            + self.down_exps.len()
            + self.shared_gate.len()
            + self.shared_up.len()
            + self.shared_down.len()
            + self.shared_int8.as_ref().map_or(0, SharedExpertInt8::bytes)
    }
}

/// Look up one tensor's bytes and stored type.
fn raw<'f>(
    file: &'f GgufFile,
    directory: &Directory<'_>,
    role: Role,
    layer: u32,
) -> Result<(&'f [u8], GgmlType, String), MoeBlockError> {
    let entry = directory
        .find(role, Some(layer))
        .ok_or(MoeBlockError::MissingTensor { role, layer })?;
    let name = entry.spec.name.clone();
    let bytes = file
        .tensor_bytes(&name)
        .ok_or_else(|| MoeBlockError::UnreadableTensor { name: name.clone() })?;
    Ok((bytes, entry.info.ggml_type, name))
}

/// An unquantized tensor, widened to f32 on the host.
///
/// `bf16` is a truncated `f32`, so the widening is exact — a shift, not a
/// conversion. Any other dense type is rejected by name.
pub(crate) fn dense_tensor(
    file: &GgufFile,
    directory: &Directory<'_>,
    role: Role,
    layer: u32,
    expected: usize,
) -> Result<(Vec<f32>, GgmlType), MoeBlockError> {
    let (bytes, ty, name) = raw(file, directory, role, layer)?;
    let values: Vec<f32> = match ty {
        GgmlType::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .map(f32::from_le_bytes)
            .collect(),
        GgmlType::Bf16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| f32::from_bits(u32::from(u16::from_le_bytes(b)) << 16))
            .collect(),
        found => return Err(MoeBlockError::UnsupportedDenseType { name, found }),
    };
    if values.len() != expected {
        return Err(MoeBlockError::ShapeMismatch {
            name,
            expected,
            found: values.len(),
        });
    }
    Ok((values, ty))
}

/// A quantized weight stack, left in its serialized form.
///
/// Nothing is dequantized on the host: one layer's routed experts are 268 M
/// elements per projection, which is 3.1 GiB in fp32 against 725 MiB packed.
/// The GEMM unpacks the element it is about to multiply.
pub(crate) fn quantized_tensor<'f>(
    file: &'f GgufFile,
    directory: &Directory<'_>,
    role: Role,
    layer: u32,
    expected: usize,
) -> Result<(&'f [u8], ExpertQuant), MoeBlockError> {
    let (bytes, ty, name) = raw(file, directory, role, layer)?;
    let quant = match ty {
        GgmlType::Q6K => ExpertQuant::Q6K,
        GgmlType::Q8_0 => ExpertQuant::Q8_0,
        // `Q4_K_M` -- the ordinary community build -- stores `ffn_*_exps` as
        // a mixture of Q4_K, Q5_K and Q6_K by layer. These three read through
        // the `*_community` kernels rather than the tuned families, which are
        // compiled per shipped format; see `ExpertQuant::is_community`.
        GgmlType::Q4_0 => ExpertQuant::Q4_0,
        GgmlType::Q4K => ExpertQuant::Q4K,
        GgmlType::Q5K => ExpertQuant::Q5K,
        found => return Err(MoeBlockError::UnsupportedExpertType { name, found }),
    };
    // The *file's* stride, not the device's: this is validating GGUF bytes.
    let found = bytes.len() / quant.file_block_bytes() * quant.block_elements();
    if !bytes.len().is_multiple_of(quant.file_block_bytes()) || found != expected {
        return Err(MoeBlockError::ShapeMismatch {
            name,
            expected,
            found,
        });
    }
    Ok((bytes, quant))
}

/// The MoE feed-forward block, compiled and sized for one fixed geometry.
pub struct MoeBlock {
    moe: MoeKernels,
    layer_ops: LayerOpsKernels,
    router_logits_fn: CudaFunction,
    router_logits_t3_fn: CudaFunction,
    router_logits_t1_fn: CudaFunction,
    router_logits_wide_fn: [CudaFunction; 2],
    router_wide_min_tokens: usize,
    router_wide_full_min_tokens: usize,
    combine_fn: CudaFunction,
    combine_row_min_tokens: usize,
    buffers: MoeBuffers,
    normed: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    routed: CudaSlice<f32>,
    shexp: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    geometry: MoeGeometry,
    eps: f32,
}

impl MoeBlock {
    /// The RMS epsilon `file` declares, or [`DEFAULT_RMS_EPS`].
    ///
    /// llama.cpp reads this key with the non-optional `get_key`, so a file
    /// without it would not load there at all; the fallback exists for
    /// synthetic files, not for the real one. The key is scoped to
    /// `config`'s architecture: reading `qwen35moe.*` out of a `qwen35` file
    /// would silently take the fallback.
    pub fn eps_from(config: &ModelConfig, file: &GgufFile) -> f32 {
        file.get_f32(&config.hparam_key(RMS_EPS_SUFFIX))
            .unwrap_or(DEFAULT_RMS_EPS)
    }

    /// The real Qwen3.6 MoE geometry, for `max_tokens` per step.
    ///
    /// Derived from [`ModelConfig`] rather than written out, so a config that
    /// disagrees with the file cannot silently produce a block of the wrong
    /// shape. `None` for a model whose layers carry a dense FFN instead — see
    /// [`crate::block::dense_ffn::DenseFfnBlock::geometry_for`].
    pub fn geometry_for(
        config: &ModelConfig,
        block_size: usize,
        max_tokens: usize,
    ) -> Option<MoeGeometry> {
        let m = config.moe()?;
        Some(MoeGeometry {
            num_experts: m.num_experts as usize,
            experts_per_token: m.experts_per_token as usize,
            hidden: config.hidden_size as usize,
            intermediate: m.expert_intermediate as usize,
            block_size,
            max_tokens,
        })
    }

    /// Compile every kernel the block needs and allocate every buffer, once.
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        geometry: MoeGeometry,
        eps: f32,
    ) -> Result<Self, MoeBlockError> {
        let moe = MoeKernels::new(ctx, geometry)?;
        let layer_ops = LayerOpsKernels::new(ctx)?;
        let ptx = compile(GLUE_SRC, "moe_block").map_err(MoeBlockError::Compile)?;
        let module = ctx.load_module(ptx)?;
        let rust = ctx.load_module(cudarc::nvrtc::Ptx::from_src(
            llmcuda_cuda::kernels::RUST_KERNELS_PTX,
        ))?;
        let buffers = moe.buffers(stream)?;

        let n = geometry.max_tokens * geometry.hidden;
        Ok(Self {
            router_logits_fn: module.load_function("moe_block_router_logits")?,
            router_logits_t3_fn: module.load_function("moe_block_router_logits_t3")?,
            router_logits_t1_fn: module.load_function("moe_block_router_logits_t1")?,
            router_logits_wide_fn: [
                module.load_function("moe_block_router_logits_wide")?,
                module.load_function("moe_block_router_logits_wide_w8")?,
            ],
            router_wide_min_tokens: router_wide_min_tokens_from_env(),
            router_wide_full_min_tokens: ROUTER_WIDE_FULL_MIN_TOKENS,
            combine_fn: rust.load_function("moe_block_gate_and_combine")?,
            combine_row_min_tokens: combine_row_min_tokens_from_env(),
            moe,
            layer_ops,
            buffers,
            normed: stream.alloc_zeros::<f32>(n)?,
            logits: stream.alloc_zeros::<f32>(geometry.max_tokens * geometry.num_experts)?,
            routed: stream.alloc_zeros::<f32>(n)?,
            shexp: stream.alloc_zeros::<f32>(n)?,
            gate: stream.alloc_zeros::<f32>(geometry.max_tokens)?,
            geometry,
            eps,
        })
    }

    /// Force every MoE GEMM back onto its fp32 kernel.
    ///
    /// Both the routed experts and the shared one: dropping the `MmaKernels`
    /// handle disables the routed path, and the shared path is gated on the
    /// same flag so it follows. See
    /// [`crate::forward::Forward::disable_tensor_cores`], which calls this so
    /// `tests/int8_forward.rs` covers the MoE at all — without it the test
    /// compared an integer MoE against an integer MoE and said nothing.
    pub fn disable_tensor_cores(&mut self) {
        self.moe.disable_tensor_cores();
    }

    /// Route every expert format through the `*_community` kernels — see
    /// [`MoeKernels::force_community`]. Differential tests set this so their
    /// Q6_K/Q8_0 control exercises the same kernels as the community cases
    /// rather than a different, already-covered path.
    pub fn force_community(&mut self, on: bool) {
        self.moe.force_community(on);
    }

    /// Pin the routed-expert kernels to the flat decode regime regardless of
    /// this pass's token width — see
    /// [`MoeKernels::set_exact_decode_regime`]. Verify passes set this so a
    /// window row's floats land exactly where plain decode's would.
    pub fn set_exact_decode_regime(&mut self, on: bool) {
        self.moe.set_exact_decode_regime(on);
    }

    /// Whether the MoE GEMMs will take the integer tensor-core path.
    pub fn tensor_cores_enabled(&self) -> bool {
        self.moe.tensor_cores_enabled()
    }

    /// Move the widths from which the router runs as a tiled GEMM, and from
    /// which that GEMM's blocks cover every expert.
    ///
    /// For `bench_router`, which measures both crossovers, and for tests that
    /// need a wide kernel on a narrow pass or the narrow one on a wide pass.
    /// `usize::MAX` for the first never selects a wide kernel.
    pub fn set_router_wide_min_tokens(&mut self, wide: usize, full: usize) {
        self.router_wide_min_tokens = wide;
        self.router_wide_full_min_tokens = full;
    }

    /// Move the width from which the combine launches one block per token.
    ///
    /// For the test that checks the two launch shapes agree bit for bit.
    pub fn set_combine_row_min_tokens(&mut self, tokens: usize) {
        self.combine_row_min_tokens = tokens;
    }

    /// Whether a pass at this geometry routes through the tiled GEMM.
    ///
    /// Only where the routed experts are on their prefill arithmetic too —
    /// tensor cores on, not the exact decode regime — and only at a shape the
    /// kernel's fixed tiles divide.
    pub fn router_is_wide(&self) -> bool {
        let g = self.geometry;
        g.max_tokens >= self.router_wide_min_tokens
            && self.moe.prefill_arithmetic()
            && g.num_experts.is_multiple_of(ROUTER_WIDE_ET as usize)
            && g.hidden.is_multiple_of(ROUTER_WIDE_KC as usize)
    }

    /// Warps in a wide router block at this geometry: eight, covering every
    /// expert, when the pass is deep enough and the expert count is eight
    /// 32-expert tiles' worth; otherwise one.
    fn router_wide_warps(&self) -> u32 {
        let g = self.geometry;
        let full = ROUTER_WIDE_ET as usize * ROUTER_WIDE_FULL_WARPS as usize;
        if g.max_tokens >= self.router_wide_full_min_tokens && g.num_experts.is_multiple_of(full) {
            ROUTER_WIDE_FULL_WARPS
        } else {
            1
        }
    }

    /// The router projection alone: logits for the rows of `x`, into
    /// [`Self::router_logits`]. `router` is a layer's `ffn_gate_inp`,
    /// `[num_experts][hidden]`; `x` is `[max_tokens][hidden]`, and the rows
    /// written are the ones [`Self::publish_tokens`] last declared live.
    pub fn route_logits(
        &mut self,
        stream: &Arc<CudaStream>,
        router: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
    ) -> Result<(), MoeBlockError> {
        let g = self.geometry;
        check_len("router", g.num_experts * g.hidden, router.len())?;
        check_len("x", g.max_tokens * g.hidden, x.len())?;
        self.launch_router(stream, router, Some(x))
    }

    /// Launch the router kernel this geometry selects, over `x` or over the
    /// block's own normalized rows.
    fn launch_router(
        &mut self,
        stream: &Arc<CudaStream>,
        router: &CudaSlice<f32>,
        x: Option<&CudaSlice<f32>>,
    ) -> Result<(), MoeBlockError> {
        let g = self.geometry;
        let hidden_i32 = g.hidden as i32;
        let experts_i32 = g.num_experts as i32;
        let max_tokens_i32 = g.max_tokens as i32;
        let wide = self.router_is_wide();
        // Exact N=3 avoids computing the last token five extra times in the
        // wide TT=8 tile.
        let tt = if g.max_tokens == 1 {
            1
        } else if g.max_tokens == ROUTER_TT_N3 as usize {
            ROUTER_TT_N3
        } else {
            ROUTER_TT
        };
        let (f, cfg) = if wide {
            let warps = self.router_wide_warps();
            let cfg = LaunchConfig {
                grid_dim: (
                    g.num_experts as u32 / (ROUTER_WIDE_ET * warps),
                    (g.max_tokens as u32).div_ceil(ROUTER_WIDE_TT),
                    1,
                ),
                block_dim: (32 * warps, 1, 1),
                shared_mem_bytes: ((ROUTER_WIDE_KC + 1) as usize
                    * (ROUTER_WIDE_ET * warps + ROUTER_WIDE_TT) as usize
                    * size_of::<f32>()) as u32,
            };
            let f = &self.router_logits_wide_fn[usize::from(warps != 1)];
            (f, cfg)
        } else {
            let cfg = LaunchConfig {
                grid_dim: (
                    (g.num_experts as u32).div_ceil(ROUTER_ET),
                    (g.max_tokens as u32).div_ceil(tt),
                    1,
                ),
                // The block width *is* the contraction stride: that equality
                // is what preserves the untiled kernel's per-thread summation
                // order, and what makes the activation tile private to each
                // thread.
                block_dim: (ROUTER_JC, 1, 1),
                shared_mem_bytes: ((THREADS.div_ceil(32) * ROUTER_ET * tt) as usize
                    * size_of::<f32>()) as u32,
            };
            let f = match tt {
                1 => &self.router_logits_t1_fn,
                ROUTER_TT_N3 => &self.router_logits_t3_fn,
                _ => &self.router_logits_fn,
            };
            (f, cfg)
        };
        let mut builder = stream.launch_builder(f);
        builder
            .arg(router)
            .arg(x.unwrap_or(&self.normed))
            .arg(self.buffers.valid_tokens())
            .arg(&hidden_i32)
            .arg(&experts_i32)
            .arg(&max_tokens_i32)
            .arg(&mut self.logits);
        // SAFETY: `router` holds `num_experts * hidden` floats and `x`
        // `max_tokens * hidden` — checked at upload and at `new` for the
        // forward path, and in `route_logits` for a caller's buffers. Both
        // kernels clamp their token index to `max_tokens - 1`. The narrow
        // one returns for `e >= num_experts`; the wide ones are selected only
        // when their expert tile (32, or 256 for the eight-warp block)
        // divides `num_experts` and their 16-float stage divides `hidden`,
        // so `e * hidden + j` and `t * hidden + j` are in range for
        // `j < hidden` in all three. `logits` holds
        // `max_tokens * num_experts` and is written only for
        // `t < valid_tokens`. The shared request covers the narrow kernel's
        // per-warp partials and the wide kernel's two staged tiles.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// The geometry this block was compiled for.
    pub fn geometry(&self) -> MoeGeometry {
        self.geometry
    }

    /// The RMS epsilon in use.
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// `attn_post_norm-N`: the post-mixer RMSNorm output, `[max_tokens][hidden]`.
    pub fn normed(&self) -> &CudaSlice<f32> {
        &self.normed
    }

    /// Router logits over all experts, `[max_tokens][num_experts]`.
    ///
    /// Not captured in the golden — `ffn_moe_logits` is named inside
    /// `build_moe_ffn` and the capture's filter list does not cover it — so
    /// this exists for diagnosis, not for a direct comparison.
    pub fn router_logits(&self) -> &CudaSlice<f32> {
        &self.logits
    }

    /// `ffn_moe_out-N`: the routed experts' weighted sum, before the shared
    /// expert.
    pub fn routed(&self) -> &CudaSlice<f32> {
        &self.routed
    }

    /// The shared expert's output **before** its sigmoid gate is applied.
    ///
    /// Not a graph node in llama.cpp's capture; exposed because the gate can
    /// then be recovered from the golden as
    /// `(ffn_out - ffn_moe_out) / shared_ungated`, which is how
    /// `tests/moe_block.rs` measures the gate's form instead of assuming it.
    pub fn shared_ungated(&self) -> &CudaSlice<f32> {
        &self.shexp
    }

    /// `sigmoid(ffn_gate_inp_shexp . normed)`, one scalar per token.
    pub fn shared_gate(&self) -> &CudaSlice<f32> {
        &self.gate
    }

    /// The MoE dispatch buffers, for inspecting the routing decision.
    pub fn buffers(&self) -> &MoeBuffers {
        &self.buffers
    }

    /// Run the whole block over `tokens` tokens.
    ///
    /// `residual` is the mixer output already added back to the block's input
    /// — llama.cpp's `attn_residual-N` — and is both the RMSNorm's input and
    /// the residual the block's output is added to. `ffn_out` receives
    /// `ffn_out-N` and `l_out` receives `l_out-N`; every buffer is
    /// `[max_tokens][hidden]`.
    ///
    /// Rows past `tokens` are never written, which is checkable: a kernel that
    /// ran on a stale slot would leave a plausible value where the caller left
    /// a zero.
    #[allow(clippy::too_many_arguments)]
    /// Publish this pass's token count into the device scalar the MoE kernels
    /// gate on, ahead of the pass.
    ///
    /// Idempotent, and the reason it is public: the write reads host memory,
    /// and a CUDA graph capture may not contain a copy from a pageable host
    /// pointer. `Forward::publish_inputs` calls this before capturing so that
    /// the call inside [`Self::forward`] has nothing left to do.
    pub fn publish_tokens(
        &mut self,
        stream: &Arc<CudaStream>,
        tokens: usize,
    ) -> Result<(), MoeBlockError> {
        self.moe
            .set_valid_tokens(stream, &mut self.buffers, tokens)?;
        Ok(())
    }

    pub fn forward(
        &mut self,
        stream: &Arc<CudaStream>,
        w: &MoeLayerWeights,
        residual: &CudaSlice<f32>,
        tokens: usize,
        ffn_out: &mut CudaSlice<f32>,
        l_out: &mut CudaSlice<f32>,
    ) -> Result<(), MoeBlockError> {
        let g = self.geometry;
        if tokens > g.max_tokens {
            return Err(MoeBlockError::TooManyTokens {
                tokens,
                max_tokens: g.max_tokens,
            });
        }
        let n = g.max_tokens * g.hidden;
        check_len("residual", n, residual.len())?;
        check_len("ffn_out", n, ffn_out.len())?;
        check_len("l_out", n, l_out.len())?;

        // A no-op after the first call at this shape — see
        // `MoeKernels::set_valid_tokens`. The forward path keeps the call so
        // a caller that never reaches `publish_tokens` is still correct.
        self.publish_tokens(stream, tokens)?;

        // 1. post-mixer RMSNorm. Every row of the buffer is normalized, not
        //    just the live ones, so the launch shape is the geometry's: a row
        //    of zeros normalizes to zeros rather than to a NaN, since the
        //    epsilon keeps the divisor positive.
        self.layer_ops.rms_norm(
            stream,
            residual,
            &w.post_norm,
            &mut self.normed,
            g.max_tokens,
            g.hidden,
            self.eps,
        )?;

        // Fork for the shared-expert side stream, recorded HERE — after
        // `normed` lands and before any routed work is enqueued — so the
        // side stream's dependency is exactly `normed` and the shared pair
        // genuinely runs beside the router/dispatch/expert kernels below.
        // Recording at the shared-expert call site instead would capture
        // the whole routed path into the fork and overlap nothing.
        // 2. router logits, then softmax + top-k + renormalize on the device.
        self.launch_router(stream, &w.router, None)?;

        self.moe
            .route_and_dispatch(stream, &mut self.buffers, &self.logits)?;

        let gate = QuantTensor {
            bytes: &w.gate_exps,
            quant: w.gate_quant,
        };
        let up = QuantTensor {
            bytes: &w.up_exps,
            quant: w.up_quant,
        };
        let down = QuantTensor {
            bytes: &w.down_exps,
            quant: w.down_quant,
        };
        let shared_gate = QuantTensor {
            bytes: &w.shared_gate,
            quant: w.shared_gate_quant,
        };
        let shared_up = QuantTensor {
            bytes: &w.shared_up,
            quant: w.shared_up_quant,
        };
        let shared_down = QuantTensor {
            bytes: &w.shared_down,
            quant: w.shared_down_quant,
        };

        // 4 (before 3). The shared expert's tensor-core path, gated on
        //    **this pass's** token count, not merely on the repacked weights
        //    existing. `Forward::reshape` shares one `Vec<MoeLayerWeights>`
        //    between a wide prefill pass and a one-token decode pass, so a
        //    decode step inherits whatever the prefill upload repacked.
        //    Checking only for the repack sent every decode step down the
        //    six-launch integer path to fill one slot of a 64-token tile, and
        //    cost 57% of decode throughput — 15.39 ms per step became 24.13 —
        //    while `bench_forward`'s n = 1 column, which builds its own
        //    weights and so never repacks, showed nothing wrong.
        let wide_enough = g.max_tokens >= SHARED_MMA_MIN_TOKENS;
        let shared_mma = w
            .shared_int8
            .as_ref()
            .filter(|_| wide_enough && self.moe.tensor_cores_enabled());

        // 3 + 4. At decode width the shared expert's blocks ride the routed
        //    expert's two launches instead of taking two of their own; the
        //    bodies are the same, so the outputs are. Stopping at `partial`
        //    either way: step 5 sums the top-k contributions itself, which
        //    saves a launch per layer.
        let fused = if shared_mma.is_none() {
            self.moe.grouped_forward_partial_with_shared(
                stream,
                &mut self.buffers,
                gate,
                up,
                down,
                shared_gate,
                shared_up,
                shared_down,
                &self.normed,
                &mut self.shexp,
            )?
        } else {
            false
        };
        if !fused {
            self.moe.grouped_forward_partial(
                stream,
                &mut self.buffers,
                gate,
                up,
                down,
                &self.normed,
            )?;

            // The shared expert, ungated — the kernel implements `expert_mlp`
            // and nothing else.
            match shared_mma {
                Some(i8w) => self.moe.shared_expert_mma(
                    stream,
                    &mut self.buffers,
                    i8w,
                    &self.normed,
                    &mut self.shexp,
                )?,
                None => {
                    self.moe.shared_expert(
                        stream,
                        &mut self.buffers,
                        shared_gate,
                        shared_up,
                        shared_down,
                        &self.normed,
                        &mut self.shexp,
                    )?;
                }
            }
        }

        // 5. the routed sum, the shared expert's sigmoid gate, and the
        //    combine that consumes both. One launch: the gate a block needs
        //    is the one that block computes, so that dependency is a barrier
        //    rather than a kernel boundary, and the routed sum is a
        //    fused multiply-add per element the combine was going to read
        //    anyway. Still `block_reduce_sum`, so it still needs one float
        //    per warp.
        // At decode width the row is split over `gridDim.y` blocks, because
        // this kernel reads `top_k * hidden` floats per token and one block
        // per token would put 64 KiB of that on one of the card's 72 SMs. At
        // prefill width one block walks the row: there are already more
        // blocks than SMs, and the split only repeats the gate.
        // `COMBINE_THREADS` is `MOE_GATE_LANES`, which is what keeps the
        // gate's reduction the shape `tests/forward_pass.rs` gated on.
        let row_blocks = if g.max_tokens >= self.combine_row_min_tokens {
            1
        } else {
            (g.hidden as u32).div_ceil(COMBINE_THREADS)
        };
        let cfg = LaunchConfig {
            grid_dim: (g.max_tokens as u32, row_blocks, 1),
            block_dim: (COMBINE_THREADS, 1, 1),
            shared_mem_bytes: ((COMBINE_THREADS as usize).div_ceil(32) * size_of::<f32>()) as u32,
        };
        let top_k_i32 = g.experts_per_token as i32;
        let hidden_i32 = g.hidden as i32;
        let mut builder = stream.launch_builder(&self.combine_fn);
        builder
            .arg(&w.shared_gate_inp)
            .arg(&self.normed)
            .arg(self.buffers.partial())
            .arg(&self.shexp)
            .arg(residual)
            .arg(self.buffers.valid_tokens())
            .arg(&top_k_i32)
            .arg(&hidden_i32)
            .arg(&mut self.gate)
            .arg(&mut self.routed)
            .arg(ffn_out)
            .arg(l_out);
        // SAFETY: one block per token slot, gated on the device
        // `valid_tokens`. `w.shared_gate_inp` holds `hidden` floats and
        // `gate` holds `max_tokens`; all five `[max_tokens][hidden]` buffers
        // were length-checked above and both in-row loops are bounded by
        // `hidden`.
        unsafe { builder.launch(cfg) }?;

        Ok(())
    }
}

pub(crate) fn check_len(
    what: &'static str,
    expected: usize,
    found: usize,
) -> Result<(), MoeBlockError> {
    if found == expected {
        Ok(())
    } else {
        Err(MoeBlockError::BufferLength {
            what,
            expected,
            found,
        })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_router_stages_the_contraction_at_the_block_width() {
        // `ROUTER_JC == THREADS` is what makes thread `tid` accumulate the
        // indices `tid, tid + 256, tid + 512, ...` — the untiled kernel's
        // per-thread summation order. The logits feed a top-8 argmax over 256
        // experts, so reassociating that sum does not perturb an answer
        // slightly; it runs a different expert.
        //
        // It is also what makes the activation tile *private* to each thread,
        // which is why the tile is registers and not shared memory.
        assert_eq!(
            ROUTER_JC, THREADS,
            "the router's contraction stride left the block width",
        );
        assert!(
            GLUE_SRC.contains(&format!("#define ROUTER_JC {ROUTER_JC}")),
            "the kernel's ROUTER_JC no longer mirrors the Rust constant",
        );
        assert!(
            GLUE_SRC.contains("acc[el][u] += wj * xv[u];"),
            "the router's activation tile is no longer a register array",
        );
    }

    use super::*;

    #[test]
    fn the_wide_router_is_one_ascending_chain_and_mirrors_its_constants() {
        // The wide kernels promise one order: a single fused multiply-add
        // chain per logit, ascending over the hidden index. The differential
        // test checks the bits; this pins the line that makes them, so a
        // rewrite into `acc += w * x` (which the compiler may or may not
        // contract) or into split accumulators fails here by name.
        assert!(
            GLUE_SRC.contains("acc[u][v] = fmaf(wv[v], xv, acc[u][v]);"),
            "the wide router no longer accumulates with an explicit fma",
        );
        for (name, value) in [
            ("ROUTER_WIDE_TT", ROUTER_WIDE_TT),
            ("ROUTER_WIDE_ET", ROUTER_WIDE_ET),
            ("ROUTER_WIDE_KC", ROUTER_WIDE_KC),
        ] {
            assert!(
                GLUE_SRC.contains(&format!("#define {name} {value}\n")),
                "the kernel's {name} no longer mirrors the Rust constant",
            );
        }
        assert!(GLUE_SRC.contains(&format!(
            "ROUTER_LOGITS_WIDE(moe_block_router_logits_wide_w8, {ROUTER_WIDE_FULL_WARPS})"
        )));
        // The staged row stride has to be odd for the eight rows a warp
        // reads at one `kk` to land on eight banks.
        assert!(GLUE_SRC.contains("#define ROUTER_WIDE_STRIDE (ROUTER_WIDE_KC + 1)"));
        assert_eq!((ROUTER_WIDE_KC + 1) % 2, 1);
        // A lane's 8x8 patch: four expert octets and eight token octets a warp.
        assert_eq!(ROUTER_WIDE_ET, 4 * 8);
        assert_eq!(ROUTER_WIDE_TT, 8 * 8);
    }

    #[test]
    fn the_wide_router_is_never_the_default_at_a_decode_or_verify_width() {
        // Below the routed experts' tensor-core threshold a pass may be a
        // decode step or a speculative verify window, and those need the
        // register-tiled kernels' width-independent bits. The widest such
        // pass is a few sequences times a few draft tokens.
        const { assert!(ROUTER_WIDE_MIN_TOKENS >= 256) };
        const { assert!(ROUTER_WIDE_FULL_MIN_TOKENS >= ROUTER_WIDE_MIN_TOKENS) };
    }

    fn config() -> ModelConfig {
        ModelConfig::qwen3_6_35b_a3b()
    }

    #[test]
    fn the_geometry_is_derived_from_the_model_config_not_written_out() {
        let g = MoeBlock::geometry_for(&config(), 16, 32).expect("qwen35moe is routed");
        assert_eq!(g.num_experts, 256);
        assert_eq!(g.experts_per_token, 8);
        assert_eq!(g.hidden, 2048);
        assert_eq!(g.intermediate, 512);
        // Every buffer and every grid comes from this, so it has to be an
        // upper bound rather than a typical value.
        assert_eq!(g.max_flat_pairs(), 256);
        // grid.y takes the dispatch-*block* capacity, not the slot count.
        assert!(g.expert_block_capacity() <= 65_535, "grid.y limit");
    }

    #[test]
    fn the_shared_expert_gate_is_a_sigmoid_of_a_per_token_scalar() {
        // The step `llmcuda-kernels` has no reference for, and the one an
        // implementation is most likely to leave out entirely: the gate must
        // be `1/(1+exp(-x))` of a dot product against a `[hidden]` vector, and
        // it must multiply the shared expert only — not the routed sum.
        // The kernel is Rust; these are its lines.
        let combine = llmcuda_cuda::kernels::rust_kernel_source("moe_block_gate_and_combine");
        assert!(combine.contains("let g = 1.0 / (1.0 + (-s).exp());"));
        assert!(combine.contains("*gate.add(t as usize) = g;"));
        // The gate multiplies the shared expert alone, and the routed sum is
        // added to the gated result rather than gated with it.
        assert!(combine.contains("let v = (*shexp.add(at)).mul_add(g, r);"));
        assert!(
            combine.contains("while k < top_k {\n                r += *p;"),
            "the routed sum this combine folded in must stay an ascending-k \
             accumulation inside one thread",
        );
        // Mirrored, because `COMBINE_THREADS` derives from it and the gate's
        // summation order depends on the two agreeing.
        assert!(GLUE_SRC.contains(&format!("#define MOE_GATE_LANES {GATE_LANES}")));
        assert!(combine.contains(&format!("if tid < {GATE_LANES} {{")));
        assert!(combine.contains(&format!("j += {GATE_LANES};")));
    }

    #[test]
    fn the_router_reads_an_experts_weights_contiguously() {
        // `ffn_gate_inp.weight` is `[hidden, num_experts]` in ggml order, so
        // one expert's row is `hidden` contiguous floats. Gathering at stride
        // `num_experts` instead produces a router that is wrong on every
        // token and whose weights still sum to one.
        //
        // Asserted as the index expression rather than as one spelling of a
        // hoisted row pointer: the tiled kernel indexes `w` inline, and the
        // invariant is which of the two axes is contiguous, not whether a
        // pointer was named.
        assert!(GLUE_SRC.contains("w[(long long)e * hidden + j]"));
        assert!(
            !GLUE_SRC.contains("num_experts + e]")
                || GLUE_SRC.contains("logits[(long long)t * num_experts + e]"),
            "the only `* num_experts + e` may be the logits store, which is \
             `[max_tokens][num_experts]` and genuinely strided that way",
        );
    }

    #[test]
    fn the_residual_is_added_after_the_ffn_waypoint_not_before() {
        // `ffn_out-N` is captured before the residual add and `l_out-N` after,
        // so the block must produce both. Folding the residual into `ffn_out`
        // would still give the right `l_out` and would fail the golden's
        // intermediate waypoint — which is the point of having it.
        let combine = llmcuda_cuda::kernels::rust_kernel_source("moe_block_gate_and_combine");
        assert!(combine.contains("*ffn_out.add(at) = v;"));
        assert!(combine.contains("*l_out.add(at) = v + *residual.add(at);"));
    }

    #[test]
    fn every_launch_shape_comes_from_the_geometry_and_the_token_count_from_the_device() {
        // AGENTS.md rule 5: nothing on this path may be sized by a host-side
        // value. Every glue kernel must take the device scalar and act on it:
        // the register-tiled router's macro, the wide router's body and the
        // macro that wraps it, and the combine.
        assert_eq!(
            GLUE_SRC
                .matches("const int* __restrict__ valid_tokens")
                .count(),
            3
        );
        // The combine is Rust and takes the same device scalar.
        let combine = llmcuda_cuda::kernels::rust_kernel_source("moe_block_gate_and_combine");
        assert!(combine.contains("valid_tokens: *const i32,"));
        assert!(combine.contains("if t >= *valid_tokens as i64 {"));
        // *Reading* it, not one particular spelling of the guard. The router
        // early-returned on it until it was tiled; now a warp carries eight
        // tokens, so it reads the scalar once and gates the store instead.
        // Asserting the `return` form would have made a correct rewrite look
        // like a rule-5 violation, which is the opposite of what this test is
        // for.
        assert_eq!(GLUE_SRC.matches("*valid_tokens").count(), 2);
        // What rule 5 actually forbids: a launch bound the host had to know.
        // `max_tokens` may size a grid — it is a geometry constant — but no
        // kernel may compare against a *count* passed by value.
        assert!(!GLUE_SRC.contains("int live_tokens"));
    }

    #[test]
    fn bf16_widening_is_a_shift_and_is_therefore_exact() {
        // Block 40's two routers are the file's only bf16 tensors. bf16 is a
        // truncated f32, so widening is `bits << 16` and loses nothing; the
        // point of asserting it is that a *rounding* conversion, or reading
        // the bytes as f16, would both be finite and wrong.
        let widen = |b: [u8; 2]| f32::from_bits(u32::from(u16::from_le_bytes(b)) << 16);
        for v in [1.0f32, -1.0, 0.5, -0.015625, 0.0] {
            let bits = v.to_bits();
            assert_eq!(bits & 0xffff, 0, "{v} is not exactly representable in bf16");
            let truncated = ((bits >> 16) as u16).to_le_bytes();
            assert_eq!(widen(truncated), v);
        }
        // And an f16 reading of the same bytes is genuinely different, so the
        // distinction is not academic.
        let bf16_one = ((1.0f32.to_bits() >> 16) as u16).to_le_bytes();
        assert_eq!(widen(bf16_one), 1.0);
        assert_ne!(u16::from_le_bytes(bf16_one), 0x3c00, "0x3c00 is f16's 1.0");
    }

    #[test]
    fn errors_name_the_tensor_and_the_type_they_refused() {
        let e = MoeBlockError::UnsupportedDenseType {
            name: "blk.40.ffn_gate_inp.weight".into(),
            found: GgmlType::Bf16,
        };
        let text = e.to_string();
        assert!(text.contains("blk.40.ffn_gate_inp.weight"), "{text}");

        let e = MoeBlockError::UnsupportedExpertType {
            name: "blk.39.ffn_gate_exps.weight".into(),
            found: GgmlType::Q4K,
        };
        assert!(e.to_string().contains("blk.39.ffn_gate_exps.weight"));

        let e = MoeBlockError::TooManyTokens {
            tokens: 33,
            max_tokens: 32,
        };
        assert!(e.to_string().contains("33"));

        let e = MoeBlockError::MissingTensor {
            role: Role::MoeSharedGateInp,
            layer: 7,
        };
        assert!(e.to_string().contains("ffn_gate_inp_shexp.weight"));
    }

    #[test]
    fn a_buffer_of_the_wrong_length_is_rejected() {
        assert!(check_len("residual", 4, 4).is_ok());
        let e = check_len("residual", 4, 5).unwrap_err();
        assert!(e.to_string().contains('4'));
    }
}
