//! Gated Attention (flash-attention style) on sm_75.
//!
//! Ten of Qwen3.6's forty layers plus the MTP head. Geometry is 16 query heads
//! against 2 KV heads (GQA ratio 8), head dimension 256, partial rotary over
//! the leading 64 dimensions, binary16 K/V cache. The reference is
//! `llmcuda_kernels::attention::causal_attention_streaming`.
//!
//! # The kernel set, and what selects it
//!
//! | Shape | Kernel | Why this one |
//! | --- | --- | --- |
//! | prefill, `n_query >= 16` | `attn_flash_causal_mma` | `m16n8k8` tensor cores, fp32 accumulate; eight query heads share one staged K/V tile |
//! | prefill, narrower | `attn_flash_causal_gqa` | same GQA-shared staging, scalar fp32 |
//! | prefill, one row | `attn_flash_causal_t1` | no query tile to amortize |
//! | decode, depth >= 16,384 | `attn_flash_decode_mma` | tiled softmax on the tensor cores; wins only where the per-key loop dominates |
//! | decode, shallower | `attn_flash_decode_warp` + `_combine` | per-key online softmax, key axis split 96 ways |
//!
//! [`AttentionKernels::uses_tensor_cores`] and `active_decode_mma` are the
//! dispatch rules, exposed so a differential test picks its tolerance from the
//! kernel that will actually run rather than from a hardcoded copy of the rule.
//!
//! # Invariants a change here must not break
//!
//! **The packed query/gate layout.** `blk.N.attn_q.weight` is `[2048, 8192]`
//! and interleaves the query and its output gate *per head*:
//! `[q_h0, gate_h0, q_h1, gate_h1, ...]`. Confirmed against upstream's
//! `ggml_view_3d(.., stride = head_dim*2, ..)` in `src/models/qwen35moe.cpp`,
//! not inferred from the dimensions. Splitting the tensor into two contiguous
//! halves is arithmetically valid, produces plausible activations, and is a
//! different model — query heads 8..15 would be fed the gates of heads 0..7.
//! [`attn_packed_query_offset`] / [`attn_packed_gate_offset`] state it once so
//! the host and the kernel cannot drift.
//!
//! **The gate is deinterleaved here and applied elsewhere.** There is no CPU
//! reference for the gating nonlinearity in `llmcuda-kernels`, so applying it
//! here would put arithmetic in this module that the differential harness
//! cannot check.
//!
//! **Causal bound.** Query row `i` sits at absolute position `key_offset + i`
//! and attends to `key_offset + i + 1` keys. `key_offset` is a *device*
//! scalar, which is what lets one decode step be recorded as a CUDA graph and
//! replayed at every later position, and what makes chunked prefill and decode
//! the same kernel. The bound appears exactly once, as the loop limit
//! `n_visible`, so there is no separate mask to get off by one against.
//!
//! **`ATTN_MAXD` bounds head_dim at 256.** The query tile lives in registers
//! only while every index folds to a constant, so wider head dimensions are
//! rejected at construction rather than silently spilled to local memory.
//!
//! **Softmax reductions stay serial and ascending within a row.** The
//! normalizer is a floating-point sum; a warp butterfly would reassociate it,
//! and the differential test gates on that exactness.
//!
//! **The decode split count is a numerics parameter, not a scheduling one.**
//! Each split reduces its own key range, so widening the splits lengthens that
//! reduction: 96 passes the deep-window 1e-5 gate and 48 does not. The *block*
//! count is the scheduling knob and is free — blocks grid-stride the splits,
//! so no reduction boundary moves and the partials are bit-identical at any
//! block count.
//!
//! Where the time actually goes, and the levers that were tried and rejected
//! on this kernel family, are in `docs/BENCHMARKS.md`.

use std::sync::Arc;

use cudarc::driver::sys::CUfunction_attribute;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};

use super::compile;

/// Offset of query head `head`'s slice within one token's packed
/// `attn_q.weight` row.
///
/// The row is `[q_h0, gate_h0, q_h1, gate_h1, ...]`, so the per-head stride is
/// `2 * head_dim` and the query is the first slice of each pair. A halves
/// split would compute `head * head_dim` instead, which is the same value only
/// for head 0.
pub const fn attn_packed_query_offset(head: usize, head_dim: usize) -> usize {
    2 * head * head_dim
}

/// Offset of query head `head`'s **output gate** slice within one token's
/// packed `attn_q.weight` row. See [`attn_packed_query_offset`].
pub const fn attn_packed_gate_offset(head: usize, head_dim: usize) -> usize {
    (2 * head + 1) * head_dim
}

/// Keys one warp scores between barriers in the flash kernel.
///
/// Mirrors `ATTN_KT`. The block rescales its running softmax once per
/// `n_warps * KEYS_PER_WARP` keys, so this trades a little shared memory and a
/// longer serial sweep per trip against a quarter of the barriers.
const KEYS_PER_WARP: usize = 4;

/// Query rows one block carries. Mirrors `ATTN_QT`.
///
/// The product `QUERY_TILE * KEYS_PER_WARP` is pinned to 32 by
/// [`the_query_tile_gives_one_thread_per_softmax_slot`], which is what makes
/// `QUERY_TILE * keys_per_tile()` equal the block width at every head
/// dimension: `head_dim/32` warps times `KEYS_PER_WARP` keys times
/// `QUERY_TILE` rows is `head_dim` threads exactly.
const QUERY_TILE: usize = 8;

/// Tokens one rotary block carries. Mirrors `ROPE_TT`.
///
/// The rotary frequency depends on the head dimension and nothing else, so a
/// block per token recomputed a double-precision `pow` for every (token, head,
/// dimension) triple to get one of 32 distinct values. A band hoists it out.
const ROPE_TOKENS: u32 = 16;

/// Head dimensions one lane reduces over, at most. Mirrors `ATTN_MAXD`.
///
/// The query tile lives in registers as `qr[ATTN_QT][ATTN_MAXD]`, and an array
/// indexed by anything the compiler cannot fold to a constant is spilled to
/// local memory — which would cost more than the tiling saves. So the bound is
/// a compile-time constant and `head_dim` is rejected above `32 * ATTN_MAXD`.
const ATTN_MAXD: usize = 8;

/// Query heads per KV head the warp-per-split decode kernel carries. Mirrors
/// `DEC_MAXG`.
///
/// One warp owns every head of its KV head so that a key it has loaded is
/// reused `gqa` times out of registers rather than re-read from shared. The
/// cost is `2 * this * ATTN_MAXD` registers of query and accumulator, which is
/// what bounds it.
const DECODE_MAX_GQA: usize = 8;

/// Query rows one GQA-shared block carries. Mirrors `GQA_QT`.
///
/// This is exactly the factor by which that kernel divides K/V traffic, so
/// bigger is better right up to the register wall — see the kernel's own
/// comment for why 4 is where it stops.
const GQA_QUERY_TILE: usize = 4;

/// Query rows one tensor-core block carries. Mirrors `MMA_QT`.
///
/// Fixed by the `m16n8k8` shape: the A operand is 16 rows and there is nothing
/// to gain from using fewer.
const MMA_QUERY_TILE: usize = 16;

/// Query heads one tensor-core block serves. Mirrors `MMA_HPB`.
///
/// All of them sit under the same KV head, so one staged K/V tile feeds all
/// eight warps. DRAM traffic is one pass over the visible window per block and
/// blocks are `n_query/MMA_QUERY_TILE * q_heads/this`, so this is the direct
/// lever on the traffic the launch issues. The output accumulator's register
/// demand is the opposing constraint.
const MMA_HEADS_PER_BLOCK: usize = 8;

/// Warps serving one query head. Mirrors `MMA_WPH`.
///
/// The block is eight warps and they divide evenly among the heads, so this is
/// `8 / MMA_HEADS_PER_BLOCK` and not an independent choice. Within a group the
/// warps split the output head dimension, so fewer warps per head is a larger
/// output accumulator: `head_dim / (8 * this)` tiles of four floats each.
const MMA_WARPS_PER_HEAD: usize = 8 / MMA_HEADS_PER_BLOCK;

/// Key octets each `Q K^T` warp sweeps per staged tile. Mirrors `MMA_KOCT`.
/// One is the measured optimum at head_dim 256; see the device comment.
const MMA_KEY_OCTETS: usize = 1;

/// Keys the tensor-core kernel stages per trip. Mirrors `MMA_KT`.
///
/// `MMA_KEY_OCTETS` octets per `Q K^T` warp of a head group, so this is
/// pinned to `8 * MMA_WARPS_PER_HEAD * MMA_KEY_OCTETS`: every warp busy,
/// every octet covered once per trip.
const MMA_KEY_TILE: usize = 8 * MMA_WARPS_PER_HEAD * MMA_KEY_OCTETS;

/// Output accumulator tiles a warp can hold. Mirrors `MMA_MAXT`.
///
/// The unroll bound on `o[MMA_MAXT][4]`, so it is four registers apiece whether
/// or not a geometry uses them all. `head_dim / (8 * MMA_WARPS_PER_HEAD)` must
/// not exceed it; at head_dim 256 and one warp per head it is exactly 32.
const MMA_MAX_TILES: usize = 256 / (8 * MMA_WARPS_PER_HEAD);

/// Padding of `v_sh`, in words. Mirrors `MMA_VSTRIDE`.
///
/// The value fragment is read at `dim * this + 4*oct + tig` with `dim`
/// carrying `g`, which lands the 32 lanes of a warp on 32 distinct banks
/// exactly when `gcd(this, 32) == 4`. This is the smallest such stride that
/// still holds `MMA_KEY_TILE / 2` words.
const MMA_VALUE_STRIDE: usize = 4 + 8 * (MMA_KEY_TILE / 2 - 4).div_ceil(8);

/// Dynamic shared memory one tensor-core block needs at a head dimension.
///
/// The staged K and V tiles in fp16. Q, scores, and online softmax state
/// remain in registers. K carries four words of padding per row; the V
/// stride keeps its fragment reads conflict-free.
///
/// Free-standing rather than a method so a unit test can check the budget
/// without a device to build an [`AttentionKernels`] against — the launch does
/// fail loudly past the ceiling, but it fails at the first long prefill rather
/// than at `cargo test`.
const fn mma_shared_bytes(head_dim: usize) -> usize {
    let qstride = head_dim / 2 + 4;
    let vstride = MMA_VALUE_STRIDE;
    // No query tile: Q lives in registers, one head per warp. See the kernel.
    let words = MMA_KEY_TILE * qstride + head_dim * vstride;
    words * size_of::<u32>()
}

/// Dynamic shared memory the tensor-core kernel is allowed to opt in to.
///
/// A block gets 48 KiB without asking; Turing allows up to 64 KiB when the
/// function opts in. This ceiling is shared with the decode variants; the
/// prefill K/V tile fits below the default limit. The opt-in happens once at
/// construction through `cuFuncSetAttribute`, not per launch.
const MMA_SHARED_CEILING: usize = 64 * 1024;

/// Key slices the scalar warp decode uses. Mirrors `DEC_WARP_SPLITS`.
const WARP_DECODE_SPLITS: usize = 96;
const K2_WARP_DECODE_SPLITS: usize = 32;
/// Shared bytes of the staged K2 prefill kernel: 32 keys of K at a 68-word
/// stride and 128 dimensions of V at a 20-word stride. Mirrors `K2S_*`.
const K2_STAGED_SHARED_BYTES: usize = (32 * 68 + 128 * 20) * 4;

/// Key slices the tensor-core decode uses by default.
///
/// NOT pure launch geometry: the slice boundaries are the partial layout
/// and the combine's reduction shape, so the deep-window differential gate
/// must pass at whatever ships here (72 and 96 both have; 48 and 36 both
/// failed it at 2.28e-5, docs/BENCHMARKS.md 2026-08-20).
/// `LLMCUDA_DEC_MMA_SPLITS` overrides at construction. The wave arithmetic,
/// at 228 registers per thread and 64-thread blocks (4 resident blocks/SM,
/// 288 block slots on 72 SMs): the N=3 serving shape runs three concurrent
/// sequence-local calls of `blocks * kv_heads` blocks each. 96 splits at
/// the batch capture's 48 blocks is 288 — exactly one wave, every block
/// grid-striding exactly two splits — and measured 151.2/153.2 tok/s
/// against the prior 72-split/36-block default's 149.5/148.3 at 32K N=3
/// (two interleaved pairs, idle GPU 1, 2026-08-20).
const MMA_DECODE_SPLITS: usize = 96;

/// Largest partial count either decode path can write, used for scratch.
const DECODE_SPLITS: usize = WARP_DECODE_SPLITS;

/// Depth, in cached keys, at and above which [`AttentionKernels::decode`]
/// prefers the tensor-core kernel over `attn_flash_decode_warp` when the
/// tensor-core lever is left at its default (`Auto`).
///
/// From an interleaved `bench_attention` sweep at `LLMCUDA_ATTN_CHUNK=1`
/// (three rounds, CUDA events): the two kernels are within noise of each
/// other at 16,384 keys (0.1917 ms warp vs 0.1910 ms wpo2, ~1.00x) after the
/// tensor-core kernel loses at every shallower depth measured (0.79x at
/// 2,048; 0.92x at 8,192; 0.96x at 12,288) and wins at every deeper one
/// (1.03x at 20,480, climbing to 1.28x at 131,072). 16,384 is the last point
/// before the win becomes consistent, so `decode()` takes it as the
/// threshold rather than a point further out that would strand real
/// win-region depth on the slower kernel. See `docs/BENCHMARKS.md`.
const DECODE_MMA_DEPTH_THRESHOLD: usize = 16_384;

/// Keys the flash-decoding split pass stages per trip. Mirrors `DEC_KT`.
///
/// Capped at 32 because the exponential phase gives one key slot to one lane.
const DECODE_KEY_TILE: usize = 8;

/// Keys the GQA-shared kernel stages in shared memory per trip. Mirrors
/// `GQA_KT`.
///
/// Sets both the shared footprint (`2 * GQA_KEY_TILE * head_dim` floats) and
/// how often the running softmax is rescaled. Capped at 32 because the
/// exponential phase gives one key slot to one lane.
const GQA_KEY_TILE: usize = 8;

/// The q8_0 KV-cache device helpers, verbatim as [`ATTENTION_SRC`] carries
/// them, for other modules that write the same cache (`k2_rope`'s batched
/// decode append). A test pins the two copies together.
pub(crate) const KV_Q8_HELPERS: &str = r#"// q8_0 KV cache (KV_KQ8 / KV_VQ8, set per half by the K2 kernel set).
// One position's row of n = kv_heads * head_dim elements is n int8 codes
// followed by n / 32 binary16 scales; block b covers elements
// [32b, 32b + 32). That is llama.cpp's block_q8_0 arithmetic in a layout
// whose codes stay 16-byte aligned per head. Every reader sees the binary16
// rounding of q * d, so the tensor-core and scalar kernels read the same
// number, and the CPU reference can build it exactly.
#ifndef KV_KQ8
#define KV_KQ8 0
#endif
#ifndef KV_VQ8
#define KV_VQ8 0
#endif
__device__ __forceinline__ long long kvq8_row_bytes(int n) { return (long long)n + (n >> 4); }
// Four codes (one little-endian word) times one scale, as two packed binary16
// pairs. Exact up to one rounding: 0x64XX is 1024 + XX in binary16, so
// (q ^ 0x80) under that exponent is 1152 + q, subtracting 1152 is exact, and
// mul.rn rounds the exact product q * d once -- f16(q * d), no I2F.
__device__ __forceinline__ void kvq8_dequant4(unsigned codes, unsigned short d, unsigned* lo, unsigned* hi) {
    unsigned x = codes ^ 0x80808080u;
    unsigned a = __byte_perm(x, 0x64646464u, 0x4140);
    unsigned b = __byte_perm(x, 0x64646464u, 0x4342);
    unsigned dd = (unsigned)d | ((unsigned)d << 16);
    asm("{ sub.rn.f16x2 %0, %2, %4; mul.rn.f16x2 %0, %0, %5;\n"
        "  sub.rn.f16x2 %1, %3, %4; mul.rn.f16x2 %1, %1, %5; }"
        : "=&r"(*lo), "=&r"(*hi) : "r"(a), "r"(b), "r"(0x64806480u), "r"(dd));
}
// Codes in bytes 0 and 1 of `codes`, each with its own scale: the packed
// binary16 key pair a tensor-core V operand wants.
__device__ __forceinline__ unsigned kvq8_dequant2(unsigned codes, unsigned short d0, unsigned short d1) {
    unsigned a = __byte_perm(codes ^ 0x8080u, 0x64646464u, 0x4140);
    unsigned dd = (unsigned)d0 | ((unsigned)d1 << 16);
    unsigned r;
    asm("{ sub.rn.f16x2 %0, %1, %2; mul.rn.f16x2 %0, %0, %3; }"
        : "=&r"(r) : "r"(a), "r"(0x64806480u), "r"(dd));
    return r;
}
// llama.cpp's quantize_row_q8_0_ref (ggml-quants.c) for the block whose 32
// elements this warp's lanes hold, one each: d = amax / 127, q = round(x / d)
// computed as x * (1 / d), d stored as binary16. fmaxf is exact and
// commutative, so the butterfly order cannot change amax. Every lane must be
// live. Returns this lane's code in the low byte; *d_bits gets the scale.
__device__ __forceinline__ unsigned kvq8_quantize(float x, unsigned short* d_bits) {
    float amax = fabsf(x);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    }
    float d = __fdiv_rn(amax, 127.0f);
    float id = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
    unsigned short h;
    asm("{ .reg .f16 a; cvt.rn.f16.f32 a, %1; mov.b16 %0, a; }" : "=h"(h) : "f"(d));
    *d_bits = h;
    return (unsigned)(int)roundf(__fmul_rn(x, id)) & 0xffu;
}
"#;

/// CUDA C++ baseline exposed for differential CUDA-event benchmarks.
pub const ATTENTION_SRC: &str = r#"
extern "C" {

// NVRTC compiles from a string with no include path, so <math.h>'s INFINITY
// macro is not reachable. `__int_as_float` is a builtin and always is.
__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }

// binary16 <-> fp32, by hand because NVRTC has no cuda_fp16.h here. The KV
// cache stores binary16: it halves the cache (5.06 -> 2.53 GiB at 131,072
// positions), halves the DRAM traffic attention spends most of its time on,
// and lets the tensor-core kernel stage K with no conversion at all, since two
// adjacent dimensions are already the packed operand it wants.
__device__ __forceinline__ float h2f(unsigned short h) {
    float f;
    asm("{ .reg .f16 a; mov.b16 a, %1; cvt.f32.f16 %0, a; }" : "=f"(f) : "h"(h));
    return f;
}
// Both halves of a packed word at once. One 32-bit load feeds two dimensions,
// which is what keeps a binary16 cache from costing twice the load count it
// saves in bytes.
__device__ __forceinline__ void h2f2(unsigned w, float* lo, float* hi) {
    asm("{ .reg .f16 a, b;\n"
        "  mov.b32 {a, b}, %2;\n"
        "  cvt.f32.f16 %0, a;\n"
        "  cvt.f32.f16 %1, b; }\n"
        : "=f"(*lo), "=f"(*hi) : "r"(w));
}

__device__ __forceinline__ unsigned short f2h(float f) {
    unsigned short h;
    asm("{ .reg .f16 a; cvt.rn.f16.f32 a, %1; mov.b16 %0, a; }" : "=h"(h) : "f"(f));
    return h;
}
// q8_0 KV cache (KV_KQ8 / KV_VQ8, set per half by the K2 kernel set).
// One position's row of n = kv_heads * head_dim elements is n int8 codes
// followed by n / 32 binary16 scales; block b covers elements
// [32b, 32b + 32). That is llama.cpp's block_q8_0 arithmetic in a layout
// whose codes stay 16-byte aligned per head. Every reader sees the binary16
// rounding of q * d, so the tensor-core and scalar kernels read the same
// number, and the CPU reference can build it exactly.
#ifndef KV_KQ8
#define KV_KQ8 0
#endif
#ifndef KV_VQ8
#define KV_VQ8 0
#endif
__device__ __forceinline__ long long kvq8_row_bytes(int n) { return (long long)n + (n >> 4); }
// Four codes (one little-endian word) times one scale, as two packed binary16
// pairs. Exact up to one rounding: 0x64XX is 1024 + XX in binary16, so
// (q ^ 0x80) under that exponent is 1152 + q, subtracting 1152 is exact, and
// mul.rn rounds the exact product q * d once -- f16(q * d), no I2F.
__device__ __forceinline__ void kvq8_dequant4(unsigned codes, unsigned short d, unsigned* lo, unsigned* hi) {
    unsigned x = codes ^ 0x80808080u;
    unsigned a = __byte_perm(x, 0x64646464u, 0x4140);
    unsigned b = __byte_perm(x, 0x64646464u, 0x4342);
    unsigned dd = (unsigned)d | ((unsigned)d << 16);
    asm("{ sub.rn.f16x2 %0, %2, %4; mul.rn.f16x2 %0, %0, %5;\n"
        "  sub.rn.f16x2 %1, %3, %4; mul.rn.f16x2 %1, %1, %5; }"
        : "=&r"(*lo), "=&r"(*hi) : "r"(a), "r"(b), "r"(0x64806480u), "r"(dd));
}
// Codes in bytes 0 and 1 of `codes`, each with its own scale: the packed
// binary16 key pair a tensor-core V operand wants.
__device__ __forceinline__ unsigned kvq8_dequant2(unsigned codes, unsigned short d0, unsigned short d1) {
    unsigned a = __byte_perm(codes ^ 0x8080u, 0x64646464u, 0x4140);
    unsigned dd = (unsigned)d0 | ((unsigned)d1 << 16);
    unsigned r;
    asm("{ sub.rn.f16x2 %0, %1, %2; mul.rn.f16x2 %0, %0, %3; }"
        : "=&r"(r) : "r"(a), "r"(0x64806480u), "r"(dd));
    return r;
}
// llama.cpp's quantize_row_q8_0_ref (ggml-quants.c) for the block whose 32
// elements this warp's lanes hold, one each: d = amax / 127, q = round(x / d)
// computed as x * (1 / d), d stored as binary16. fmaxf is exact and
// commutative, so the butterfly order cannot change amax. Every lane must be
// live. Returns this lane's code in the low byte; *d_bits gets the scale.
__device__ __forceinline__ unsigned kvq8_quantize(float x, unsigned short* d_bits) {
    float amax = fabsf(x);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    }
    float d = __fdiv_rn(amax, 127.0f);
    float id = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
    unsigned short h;
    asm("{ .reg .f16 a; cvt.rn.f16.f32 a, %1; mov.b16 %0, a; }" : "=h"(h) : "f"(d));
    *d_bits = h;
    return (unsigned)(int)roundf(__fmul_rn(x, id)) & 0xffu;
}

// Deinterleave the packed query/gate tensor.
//
// grid: (n_tokens, q_heads). block: head_dim threads.
//
// `blk.N.attn_q.weight` emits, per token, [q_h0, gate_h0, q_h1, gate_h1, ...]
// — a per-head stride of 2*head_dim with the query first. This is confirmed
// against llama.cpp's src/models/qwen35moe.cpp, which views it with
// stride = head_dim*2. It is NOT [all queries | all gates]: reading it that
// way feeds query heads 8..15 the gates of heads 0..7, stays finite, and is a
// different model.
__global__ void attn_split_query_gate(
    const float* __restrict__ packed,
    float* __restrict__ q,
    float* __restrict__ gate,
    int q_heads,
    int head_dim
) {
    long long t = blockIdx.x;
    int h = blockIdx.y;
    int d = threadIdx.x;

    long long row = t * (long long)q_heads * 2 * head_dim;
    long long dst = (t * (long long)q_heads + h) * (long long)head_dim + d;

    q[dst]    = packed[row + (long long)(2 * h)     * head_dim + d];
    gate[dst] = packed[row + (long long)(2 * h + 1) * head_dim + d];
}

// Partial rotary position embedding, NEOX (split-half) pairing.
//
// grid: (n_tokens, n_heads). block: head_dim threads.
//
// Rotates dimensions [0, rope_dim) pairing i with i + rope_dim/2, and copies
// [rope_dim, head_dim) through. The copy is a real load/store rather than an
// arithmetic identity so the tail is bit-identical rather than merely close —
// that is what `Tolerance::exact()` in the differential test checks.
//
// The angle is computed in double to match `llmcuda_kernels::rope::apply_rope`,
// which raises theta_base to a f64 power. Doing it in float diverges visibly
// at the positions this model actually reaches: at position 262,143 a float
// angle loses about 5 significant digits.
// Tokens one rotary block carries.
//
// The frequency `theta_base ^ (-2 d / rope_dim)` depends on the head dimension
// and nothing else, yet a block per token made it a **double-precision `pow`
// per (token, head, dimension) triple** -- 262,144 of them per pass for the
// query stream alone, for 32 distinct values. Turing runs fp64 at a
// thirty-second of fp32, and `pow` is a libdevice call on top of that.
//
// A band of tokens computes it once and reuses it, which is bit-identical:
// the same `pow` of the same arguments, hoisted out of a loop. The angle and
// its sine and cosine still have to be per token, and still have to be double
// -- see below.
#define ROPE_TT 16

__global__ void attn_rope_partial_neox(
    const float* __restrict__ in,
    float* __restrict__ out,
    int n_heads,
    int head_dim,
    int rope_dim,
    const int* __restrict__ pos_offset,
    float theta_base,
    int n_tokens
) {
    long long t0 = (long long)blockIdx.x * ROPE_TT;
    int h = blockIdx.y;
    int d = threadIdx.x;

    int half = rope_dim >> 1;

    if (d >= rope_dim) {
        for (int u = 0; u < ROPE_TT; ++u) {
            long long t = t0 + u;
            if (t >= n_tokens) break;
            long long base = (t * (long long)n_heads + h) * (long long)head_dim;
            out[base + d] = in[base + d];
        }
        return;
    }
    // Dimensions [half, rope_dim) are written by their partner thread d-half.
    if (d >= half) return;

    // Hoisted: the one quantity in here that does not depend on the token.
    double freq = pow((double)theta_base, -2.0 * (double)d / (double)rope_dim);

    for (int u = 0; u < ROPE_TT; ++u) {
        long long t = t0 + u;
        if (t >= n_tokens) break;
        long long base = (t * (long long)n_heads + h) * (long long)head_dim;

        double pos = (double)(*pos_offset) + (double)t;
        double angle = pos * freq;
        float sin_a = (float)sin(angle);
        float cos_a = (float)cos(angle);

        float x0 = in[base + d];
        float x1 = in[base + d + half];
        out[base + d]        = x0 * cos_a - x1 * sin_a;
        out[base + d + half] = x0 * sin_a + x1 * cos_a;
    }
}

// Interleaved multimodal rotary (IMROPE), the three-component sibling of
// attn_rope_partial_neox for prefill chunks that overlap an image span.
//
// Each token carries (t, h, w) in `mrope_pos[3*token ..]`. Pair d reads
// component d%3 — h iff d%3==1 and d < h_bound (3*sections[1]), w iff
// d%3==2 and d < w_bound (3*sections[2]), else t — with the frequency on
// the *global* pair index, exactly `ggml_mrope_cache_init`'s
// GGML_ROPE_TYPE_IMROPE branch (reference:
// `llmcuda_kernels::mrope::apply_imrope`). With all three components equal
// this computes bit-for-bit what attn_rope_partial_neox computes — the
// double conversions and the operation order are copied from it — which is
// asserted by differential test rather than assumed.
//
// No ROPE_TT banding: the frequency hoist is per pair as before, but the
// position is per token anyway and this kernel only ever runs on
// image-bearing prefill chunks, off the decode path the banding was
// measured for.
__global__ void attn_rope_partial_imrope(
    const float* __restrict__ in,
    float* __restrict__ out,
    int n_heads,
    int head_dim,
    int rope_dim,
    const int* __restrict__ mrope_pos,
    int h_bound,
    int w_bound,
    float theta_base,
    int n_tokens
) {
    long long t = blockIdx.x;
    int h = blockIdx.y;
    int d = threadIdx.x;
    if (t >= n_tokens) return;

    int half = rope_dim >> 1;
    long long base = (t * (long long)n_heads + h) * (long long)head_dim;

    if (d >= rope_dim) {
        out[base + d] = in[base + d];
        return;
    }
    if (d >= half) return;

    int c = d % 3;
    int component;
    if (c == 1 && d < h_bound) {
        component = mrope_pos[3 * t + 1];
    } else if (c == 2 && d < w_bound) {
        component = mrope_pos[3 * t + 2];
    } else {
        component = mrope_pos[3 * t];
    }

    double freq = pow((double)theta_base, -2.0 * (double)d / (double)rope_dim);
    double angle = (double)component * freq;
    float sin_a = (float)sin(angle);
    float cos_a = (float)cos(angle);

    float x0 = in[base + d];
    float x1 = in[base + d + half];
    out[base + d]        = x0 * cos_a - x1 * sin_a;
    out[base + d + half] = x0 * sin_a + x1 * cos_a;
}

// Causal GQA attention, online-softmax streaming form.
//
// grid: (n_query, q_heads) — one block per (query row, query head).
// block: head_dim threads = head_dim/32 warps.
//
// Per iteration each warp scores one key, so a tile is head_dim/32 keys (8 at
// head_dim 256). Thread `tid` owns output dimension `tid` of the running
// accumulator for the whole kernel, so the value accumulation never leaves
// registers and never needs a reduction.
//
// The online update follows `causal_attention_streaming` in llmcuda-kernels: a
// running max, a correction factor that rescales the accumulator and the
// normalizer into the new max's frame, then the new contributions. The one
// deliberate difference is that the update is applied per *tile* of keys
// rather than per key — mathematically identical, and it cuts the number of
// expf evaluations by the tile width, because otherwise all head_dim threads
// redundantly evaluate the same two exponentials for every key.
// Keys one warp scores per trip, and query rows one block carries.
//
// Their product is pinned to 32, which is what makes the block width exactly
// `ATTN_QT * tile`: the per-tile softmax bookkeeping then has one thread per
// (query row, key slot) and needs no loop at all. See the phases below.
#define ATTN_KT 4
#define ATTN_QT 8
// Head dimensions one lane reduces over, `head_dim / 32`. Bounded because the
// query tile lives in registers and `qr[ATTN_QT][ATTN_MAXD]` must be indexed
// by constants after unrolling — a runtime bound spills it to local memory,
// which is the whole win thrown away. 8 covers head_dim 256.
#define ATTN_MAXD 8

// `exp2f` is one `MUFU.EX2`; `expf` is that plus the range reduction around
// it. Folding `log2(e)` into the score scale makes the swap an identity and
// not an approximation: with `s = x * log2(e)`, `exp2(s_i - max_j s_j)` equals
// `exp(x_i - max_j x_j)` for every i, so weights, normalizers and rescale
// factors are unchanged and only the units of the running maximum differ. The
// multiply it rides on was already being paid.
//
// Applied only where it was measured to help, which is not everywhere: on
// `attn_flash_decode_warp` it measured 3.4% *slower* (1.227 vs 1.188 ms over
// interleaved pairs) despite emitting 80 fewer instructions for the same 16
// `MUFU.EX2` and the same occupancy -- so decode keeps `expf`. A standalone
// cost ladder had predicted a 15% gain there; it omitted the `part_acc`
// writeback and was not a faithful model of the kernel.
//
// It is also not the `__expf` substitution, which *is* an approximation: that
// one measured 1.075x on prefill and was rejected for flipping a rank-4
// ordering against llama.cpp's logits. See docs/BENCHMARKS.md.
#define ATTN_LOG2E 1.4426950408889634f

#define ATTN_FLASH(NAME, QT)                                                  \
__global__ void NAME(                                                           \
    const float* __restrict__ q,                                                \
    const unsigned short* __restrict__ k,                                       \
    const unsigned short* __restrict__ v,                                       \
    float* __restrict__ out,                                                    \
    int q_heads,                                                                \
    int kv_heads,                                                               \
    int head_dim,                                                               \
    const int* __restrict__ key_offset,                                         \
    float scale,                                                                \
    int n_query                                                                 \
) {                                                                             \
    extern __shared__ float smem[];                                             \
    int n_warps = blockDim.x >> 5;                                              \
    int tile = n_warps * ATTN_KT;  /* keys per barrier pair */                  \
    float* score_sh = smem;  /* QT * tile */                                    \
    float* w_sh     = score_sh + QT * tile;                                     \
    float* m_sh     = w_sh + QT * tile;  /* QT, the running maxima */           \
    float* l_sh     = m_sh + QT;  /* QT, the normalizers */                     \
    float* corr_sh  = l_sh + QT;  /* QT, this tile's rescale */                 \
                                                                                \
    long long qi0 = (long long)blockIdx.x * QT;                                 \
    int h = blockIdx.y;                                                         \
    int kvh = h / (q_heads / kv_heads);  /* GQA: never assume 1:1 */            \
    int tid = threadIdx.x;                                                      \
    int lane = tid & 31;                                                        \
    int warp = tid >> 5;                                                        \
    int dpt = head_dim >> 5;                                                    \
                                                                                \
    /* The query tile in registers rather than shared. */                       \
    /* */                                                                       \
    /* Lane `lane` owns dimensions `lane, lane + 32, ...` of *every* row in the */ \
    /* tile — the same partition of the contraction the untiled kernel gave one */ \
    /* block, so the dot product below sums in the same order. Holding it in */ \
    /* registers is what turns the score loop from one shared read per */       \
    /* multiply-add into none: a key element is loaded once and multiplied */   \
    /* QT times. */                                                             \
    float qr[QT][ATTN_MAXD];                                                    \
    _Pragma("unroll")                                                           \
    for (int u = 0; u < QT; ++u) {                                              \
        long long qrow = qi0 + u;                                               \
        const float* qp = q + (qrow * (long long)q_heads + h) * (long long)head_dim; \
        _Pragma("unroll")                                                       \
        for (int i = 0; i < ATTN_MAXD; ++i) {                                   \
            qr[u][i] = (i < dpt && qrow < (long long)n_query) ? qp[lane + 32 * i] : 0.0f; \
        }                                                                       \
    }                                                                           \
                                                                                \
    if (tid < QT) {                                                             \
        m_sh[tid] = neg_inf();                                                  \
        l_sh[tid] = 0.0f;                                                       \
    }                                                                           \
    float acc[QT];                                                              \
    _Pragma("unroll")                                                           \
    for (int u = 0; u < QT; ++u) acc[u] = 0.0f;                                 \
                                                                                \
    /* Rows of the tile that exist, and the deepest key any of them sees. The */ \
    /* causal bound is per row — row `qi0 + u` sees keys `[0, key_offset + qi0 */ \
    /* + u]` inclusive — so the loop runs to the *last* row's bound and the */  \
    /* earlier rows mask the tail off. See the mask in phase 1. */              \
    int qt_live = (int)((long long)n_query - qi0);                              \
    if (qt_live > QT) qt_live = QT;                                             \
    long long n_visible = (long long)(*key_offset) + qi0 + qt_live;             \
                                                                                \
    /* One thread per (query row, key slot) for the softmax phases. */          \
    int su = tid / tile;                                                        \
    int sw = tid - su * tile;                                                   \
                                                                                \
    __syncthreads();                                                            \
                                                                                \
    for (long long j0 = 0; j0 < n_visible; j0 += tile) {                        \
        /* ATTN_KT keys per warp per trip, not one. */                          \
        /* */                                                                   \
        /* `key = j0 + warp * ATTN_KT + r` stored at `score_sh[u * tile + warp */ \
        /* * ATTN_KT + r]` keeps slot `w` holding key `j0 + w`, which is what */ \
        /* lets the value accumulation below stay a single ascending sweep. */  \
        _Pragma("unroll")                                                       \
        for (int r = 0; r < ATTN_KT; ++r) {                                     \
            long long key = j0 + (long long)warp * ATTN_KT + r;                 \
            float part[QT];                                                     \
            _Pragma("unroll")                                                   \
            for (int u = 0; u < QT; ++u) part[u] = 0.0f;                        \
            if (key < n_visible) {                                              \
                const unsigned short* krow =                                    \
                    k + (key * (long long)kv_heads + kvh) * (long long)head_dim; \
                /* Lane l takes dimensions l, l+32, l+64, ...: consecutive lanes */ \
                /* read consecutive floats, so every load is a full 128 B */    \
                /* transaction, and one such load feeds QT multiply-adds. */    \
                _Pragma("unroll")                                               \
                for (int i = 0; i < ATTN_MAXD; ++i) {                           \
                    if (i < dpt) {                                              \
                        float kd = h2f(krow[lane + 32 * i]);                    \
                        _Pragma("unroll")                                       \
                        for (int u = 0; u < QT; ++u) part[u] += qr[u][i] * kd;  \
                    }                                                           \
                }                                                               \
            }                                                                   \
            _Pragma("unroll")                                                   \
            for (int u = 0; u < QT; ++u) {                                      \
                float p = part[u];                                              \
                for (int off = 16; off > 0; off >>= 1) {                        \
                    p += __shfl_xor_sync(0xffffffff, p, off);                   \
                }                                                               \
                if (lane == 0) score_sh[u * tile + warp * ATTN_KT + r] = p * scale; \
            }                                                                   \
        }                                                                       \
        __syncthreads();                                                        \
                                                                                \
        /* Phase 1: the running max, one thread per query row. */               \
        /* */                                                                   \
        /* Serial and ascending over the tile, which is the order the untiled */ \
        /* kernel folded it in. A warp butterfly would be faster and would also */ \
        /* be a different reduction; `fmaxf` happens to be associative in */    \
        /* floating point, but the normalizer in phase 3 is not, so the two are */ \
        /* kept the same shape rather than one being quietly special. */        \
        if (tid < QT) {                                                         \
            long long limit = (long long)(*key_offset) + qi0 + tid;             \
            float tmax = neg_inf();                                             \
            for (int w = 0; w < tile; ++w) {                                    \
                if (j0 + w <= limit) tmax = fmaxf(tmax, score_sh[tid * tile + w]); \
            }                                                                   \
            float m0 = m_sh[tid];                                               \
            float new_m = fmaxf(m0, tmax);                                      \
            /* Matches the reference's guard exactly: on the first tile there is */ \
            /* no accumulator to rescale and expf(-inf - -inf) would be NaN. */ \
            corr_sh[tid] = (m0 == neg_inf()) ? 0.0f : expf(m0 - new_m);         \
            m_sh[tid] = new_m;                                                  \
        }                                                                       \
        __syncthreads();                                                        \
                                                                                \
        /* Phase 2: one exponential per slot, all in parallel. Masked slots */  \
        /* write a literal zero, which is what makes a row whose causal bound */ \
        /* ended earlier contribute exactly nothing to the tiles past it — */   \
        /* `a += 0.0f * v` and `lsum += 0.0f` are both exact. */                \
        if (tid < QT * tile) {                                                  \
            long long limit = (long long)(*key_offset) + qi0 + su;              \
            w_sh[su * tile + sw] = (j0 + sw <= limit)                           \
                ? expf(score_sh[su * tile + sw] - m_sh[su])                     \
                : 0.0f;                                                         \
        }                                                                       \
        __syncthreads();                                                        \
                                                                                \
        /* Phase 3: the normalizer, serial and ascending. Independent of phase */ \
        /* 4, so the two run without a barrier between them. */                 \
        if (tid < QT) {                                                         \
            float lsum = 0.0f;                                                  \
            for (int w = 0; w < tile; ++w) lsum += w_sh[tid * tile + w];        \
            l_sh[tid] = l_sh[tid] * corr_sh[tid] + lsum;                        \
        }                                                                       \
                                                                                \
        /* Phase 4: the values. One global load per key now serves QT */        \
        /* rows instead of one, which is the whole point of the query tile. */  \
        _Pragma("unroll")                                                       \
        for (int u = 0; u < QT; ++u) acc[u] = acc[u] * corr_sh[u];              \
        for (int w = 0; w < tile; ++w) {                                        \
            if (j0 + w >= n_visible) break;                                     \
            float vv =                                                          \
                v[((j0 + w) * (long long)kv_heads + kvh) * (long long)head_dim + tid]; \
            _Pragma("unroll")                                                   \
            for (int u = 0; u < QT; ++u) acc[u] += w_sh[u * tile + w] * vv;     \
        }                                                                       \
                                                                                \
        /* Before the next iteration overwrites score_sh and w_sh. */           \
        __syncthreads();                                                        \
    }                                                                           \
                                                                                \
    _Pragma("unroll")                                                           \
    for (int u = 0; u < QT; ++u) {                                              \
        long long qrow = qi0 + u;                                               \
        if (qrow < (long long)n_query) {                                        \
            out[(qrow * (long long)q_heads + h) * (long long)head_dim + tid] =  \
                acc[u] / l_sh[u];                                               \
        }                                                                       \
    }                                                                           \
}

// Eight query rows for prefill. One for decode, where seven of the eight
// register rows would be a masked-off row past `n_query`: the score loop would
// still run eight multiply-adds and eight shuffle reductions per key to throw
// seven of them away, and attention is 8% of a decode step. Measured 104.1 ->
// 99.7 tok/s before this instantiation existed.
//
// They are separate kernels rather than one macro at two widths because the
// two shapes want different softmax bookkeeping. At eight rows the per-tile
// max and normalizer are one thread per row, because every thread sweeping
// `QT * tile` shared floats would cost more than the value loop it amortizes.
// At one row there is nothing to amortize: the redundant sweep is 32 shared
// broadcasts, and paying it in every thread is cheaper than serializing it
// onto one and adding a barrier. `attn_flash_causal_t1` is therefore the
// pre-tiling kernel, unchanged.
ATTN_FLASH(attn_flash_causal, ATTN_QT)

// ---------------------------------------------------------------------------
// The same attention on the tensor cores.
//
// grid: (n_query / MMA_QT, q_heads). block: 8 warps.
//
// ## Why
//
// The scalar kernels above are at their ceiling and the ceiling is the wrong
// one. Attention is two GEMMs, `Q K^T` and `P V`, and this part does fp32 GEMM
// at 16.3 TFLOP/s and fp16-in/fp32-out GEMM at about 65. llama.cpp reaches ~25
// TFLOP/s on the same prompt, which no fp32 kernel can, and `fattn.cu:461`
// confirms it takes `fattn-mma-f16.cuh` here. Both engines sit at an ordinary
// fraction of their own ceiling; the ceilings differ by 4x.
//
// `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32` takes fp16 operands and
// accumulates in **fp32**, so the rounding is confined to the inputs rather
// than to a 256-term dot product. That is why this and not packed `half2`
// arithmetic, which would have to accumulate in fp16 and would breach the
// tolerance the differential tests are gated at.
//
// The cache stays fp32; K and V are rounded to fp16 on their way into shared.
// This kernel therefore changes the arithmetic without changing the cache
// format, which is a separate (and separately useful) change.
//
// ## Fragment layouts, which are the whole risk
//
// For `m16n8k8` with `groupID = lane >> 2` and `tig = lane & 3`:
//
//   A (16x8, row major, 2 regs): a0 = {A[g][2tig], A[g][2tig+1]},
//                                a1 = {A[g+8][2tig], A[g+8][2tig+1]}
//   B (8x8, col major, 1 reg):   b0 = {B[2tig][g], B[2tig+1][g]}
//   C/D (16x8, f32, 4 regs):     c0,c1 = D[g][2tig], D[g][2tig+1]
//                                c2,c3 = D[g+8][2tig], D[g+8][2tig+1]
//
// D's layout is exactly A's, which is what lets the scores fall straight out of
// `Q K^T` and back into `P V` as the A operand with no transpose: the n index
// of the first becomes the k index of the second.
//
// ## Shape
//
// One block carries 16 query rows for eight query heads sharing a KV head.
// One warp owns each head's entire output dimension and both rows of each
// lane's MMA fragment. At head_dim 256 its Q fragments use 64 registers and
// its output accumulator uses 128. The K/V tile covers eight keys; widening
// it grows the prefetch registers beside those persistent accumulators.
// Fragment-local softmax requires one warp per head and one key octet, which
// is asserted below rather than treating these constants as free tunables.
//
// ## Shared strides
//
// Every fragment read is one 32-bit word, and the padding is chosen so that the
// 32 lanes of a warp hit 32 distinct banks. `k_sh` is indexed
// `row * STRIDE + 4*step + tig` with `row` carrying `g`: stride `hd2 + 4` makes
// the bank `(4g + tig) mod 32`, which is a bijection on `g<8, tig<4`. `v_sh` is
// indexed `dim * VSTRIDE + 4*oct + tig` with `dim` carrying `g`: stride
// `MMA_KT/2 + 4` makes it `(VSTRIDE*g + tig) mod 32`, a bijection whenever
// `gcd(VSTRIDE, 32)` is 4. At `MMA_KT` 16 that stride is 12 and `12g mod 32`
// over `g<8` is {0,12,24,4,16,28,8,20} — eight groups of four consecutive
// banks, as required.
#define MMA_QT 16
#define MMA_HPB 8
#define MMA_WPH (8 / MMA_HPB)
// Key octets each `Q K^T` warp sweeps per staged tile. One octet per warp
// is the measured optimum, not a placeholder: the cross-tile prefetch
// arrays (`kreg`, `vlo`, `vhi`) scale with the tile and are held live
// across the whole compute phase by design, and at head_dim 256 they sit
// against `o` and `qa` at the register ceiling. Widening the tile spills
// them to local memory -- 4 octets measured −30% (32K prefill 2,109 vs
// 2,799), 2 octets −13% (2,424) on 2026-08-20, both rejected. Fewer
// barriers cannot buy back a spilled prefetch.
#define MMA_KOCT 1
#define MMA_KT (8 * MMA_WPH * MMA_KOCT)
#define MMA_MAXT (256 / (8 * MMA_WPH))
// `Q K^T` steps at the largest head dimension the dispatch admits. The Q
// fragments are held in registers across the whole key loop, so every index
// into them must fold at compile time -- a runtime index puts the array in
// local memory and costs more than the shared tile it replaced.
#define MMA_QSTEPS 32
// Padding for `v_sh`, whose fragment read is `dim * VSTRIDE + 4*oct + tig`
// with `dim` carrying `g`. That is a bank bijection exactly when
// `gcd(VSTRIDE, 32) == 4`, so this is the smallest such stride that still
// holds `MMA_KT/2` words: 4, 12, 20 for key tiles of 8, 16, 32. The obvious
// `MMA_KT/2 + 4` gives 8 at a key tile of 8, and `gcd(8, 32)` is 8 -- four
// banks over eight groups, a two-way conflict on every value fragment.
#define MMA_VSTRIDE (4 + 8 * (((MMA_KT / 2) - 4 + 7) / 8))
// Value-staging loads issued before the first store. See the staging loop.
#define MMA_VB 2
// Prefetch registers per thread, sized for the largest geometry the dispatch
// admits (head_dim 256, so `kw4` is at most 32 and a block is 256 threads).
// Both must be compile-time bounds or the arrays spill to local memory.
#define MMA_KREG ((MMA_KT * 32 + 255) / 256)
#define MMA_VREG ((MMA_KT / 2) / MMA_VB)

// Fragment ownership below requires one warp per head and one key octet.
// Widening either changes which lanes own a complete softmax row.
static_assert(MMA_WPH == 1 && MMA_KOCT == 1, "fragment softmax ownership");

__device__ __forceinline__ unsigned pack_h2(float lo, float hi) {
    unsigned r;
    asm("{ .reg .f16 a, b;\n"
        "  cvt.rn.f16.f32 a, %1;\n"
        "  cvt.rn.f16.f32 b, %2;\n"
        "  mov.b32 %0, {a, b}; }\n"
        : "=r"(r) : "f"(lo), "f"(hi));
    return r;
}

__device__ __forceinline__ void mma_m16n8k8(
    float& d0, float& d1, float& d2, float& d3,
    unsigned a0, unsigned a1, unsigned b0
) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(b0));
}

// Each lane holds the even/odd key pair of one row in an MMA fragment.
// Preserve the former eight-lane XOR(4,2,1) sum at its writer (lane zero):
// first reduce the even and odd subsequences, then add them, and broadcast
// the quad leader's result. Other lanes' butterflies have different rounding.
__device__ __forceinline__ float mma_softmax_pair(
    float& even, float& odd, bool live_even, bool live_odd,
    float& maximum, float& normalizer
) {
    even = live_even ? even : neg_inf();
    odd = live_odd ? odd : neg_inf();
    float tile_max = fmaxf(even, odd);
    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 2));
    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 1));
    float next_max = fmaxf(maximum, tile_max);
    float correction = maximum == neg_inf() ? 0.0f : exp2f(maximum - next_max);
    even = live_even ? exp2f(even - next_max) : 0.0f;
    odd = live_odd ? exp2f(odd - next_max) : 0.0f;
    float se = even, so = odd;
    se += __shfl_xor_sync(0xffffffff, se, 2);
    so += __shfl_xor_sync(0xffffffff, so, 2);
    se += __shfl_xor_sync(0xffffffff, se, 1);
    so += __shfl_xor_sync(0xffffffff, so, 1);
    float sum = __shfl_sync(0xffffffff, se + so, 0, 4);
    // Match the old shared-state update even if code motion would separate
    // the multiply and add across the following MMA control flow.
    normalizer = __fmaf_rn(normalizer, correction, sum);
    maximum = next_max;
    return correction;
}

} // extern C
template<bool COMP,int HPB,int MAXT>
__device__ void attn_flash_causal_mma_body(
    const float* __restrict__ q,
    const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v,
    float* __restrict__ out,
    int q_heads,
    int kv_heads,
    int head_dim,
    const int* __restrict__ key_offset,
    float scale,
    int n_query
) {
    // Declared `float` to match every other kernel in this module -- an extern
    // shared array has one type per compilation unit -- and reinterpreted,
    // which is fine because the fp16 tiles are 32-bit-word addressed anyway.
    extern __shared__ float smem_f[];
    unsigned* smem = (unsigned*)smem_f;
    int hd2 = head_dim >> 1;              // 32-bit words per row of a fp16 tile
    int qstride = hd2 + 4;
    int vstride = MMA_VSTRIDE;

    // No `q_sh`. At `MMA_HPB` 8 it would want 67,584 B on its own against a
    // 65,536 B carveout, and it is the one staged tile that is read by exactly
    // one warp -- every warp of a block serves a different head, so nothing is
    // shared by putting Q in shared memory. In registers it costs 64 of them
    // and buys the halved block count, which is what the kernel is bound by.
    unsigned* k_sh = smem;                                      // KT*qstride
    unsigned* v_sh = k_sh + MMA_KT * qstride;                   // head_dim*vstride

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int nthr = blockDim.x;
    int g = lane >> 2;
    int tg = lane & 3;

    // Each warp owns one query head. All eight heads share the K/V tile.
    int hslot = warp / MMA_WPH;
    int sub = warp % MMA_WPH;
    long long qi0 = (long long)blockIdx.x * MMA_QT;
    int h = blockIdx.y * HPB + hslot;
    int kvh = h / (q_heads / kv_heads);

    int dpw = head_dim / MMA_WPH;     // output dims this warp owns
    int dbase = sub * dpw;
    int ntile = dpw >> 3;             // MMA n-tiles of 8 dims, <= MAXT

    float maximum0 = neg_inf(), maximum1 = neg_inf();
    float normalizer0 = 0.0f, normalizer1 = 0.0f;

    // This warp's own Q, in the A fragment layout, held for the whole key
    // loop. Lane `(g, tig)` owns rows `qi0+g` and `qi0+g+8` at the dimension
    // pair `4*s + tig`, which is exactly what `mma.m16n8k8` wants in `a0`,
    // `a1`. Read once per block and never again, so the scatter across eight
    // rows costs nothing measurable against the key stream.
    int steps = head_dim >> 3;
    unsigned qa0[(COMP ? 16 : MMA_QSTEPS)], qa1[(COMP ? 16 : MMA_QSTEPS)];
    unsigned qal0[COMP ? 16 : 1],qal1[COMP ? 16 : 1];
    {
        long long r0 = qi0 + g;
        long long r1 = qi0 + g + 8;
        const float* qp0 = (r0 < (long long)n_query)
            ? q + (r0 * (long long)q_heads + h) * (long long)head_dim : 0;
        const float* qp1 = (r1 < (long long)n_query)
            ? q + (r1 * (long long)q_heads + h) * (long long)head_dim : 0;
        #pragma unroll
        for (int s = 0; s < (COMP ? 16 : MMA_QSTEPS); ++s) {
            qa0[s] = 0u;
            qa1[s] = 0u;
            if (s < steps) {
                int c = 4 * s + tg;
                if (qp0) qa0[s] = pack_h2(qp0[2 * c], qp0[2 * c + 1]);
                if (qp1) qa1[s] = pack_h2(qp1[2 * c], qp1[2 * c + 1]);
                if(COMP){qal0[s]=qp0 ? pack_h2(qp0[2*c]-h2f((unsigned short)qa0[s]),qp0[2*c+1]-h2f((unsigned short)(qa0[s]>>16))) : 0;qal1[s]=qp1 ? pack_h2(qp1[2*c]-h2f((unsigned short)qa1[s]),qp1[2*c+1]-h2f((unsigned short)(qa1[s]>>16))) : 0;}
            }
        }
    }

    float o[MAXT][4];
    #pragma unroll
    for (int t = 0; t < MAXT; ++t) {
        o[t][0] = 0.0f; o[t][1] = 0.0f; o[t][2] = 0.0f; o[t][3] = 0.0f;
    }

    int qt_live = (int)((long long)n_query - qi0);
    if (qt_live > MMA_QT) qt_live = MMA_QT;
    long long n_visible = (long long)(*key_offset) + qi0 + qt_live;

    // Value staging, decomposed once rather than once per tile.
    //
    // A thread owns one output dimension and a slice of the key-pairs. The
    // dimension goes on the fast axis so consecutive lanes still read
    // consecutive dimensions -- the coalescing this kernel already had -- and
    // the key-pair goes on a loop with a compile-time bound, so the staging
    // body contains no integer division at all. Splitting the key-pairs across
    // `head_dim / nthr` slices keeps every thread busy when `head_dim` is
    // smaller than the block, which it is for every geometry but this model's.
    int vcap = (MMA_KT >> 1) / MMA_VB;      // batches of key-pairs to hand out
    int vslices = nthr / head_dim;
    if (vslices < 1) vslices = 1;
    if (vslices > vcap) vslices = vcap;
    // Round down to a divisor of `vcap` so the batches partition exactly; a
    // slice count that does not divide it would leave key-pairs unstaged, and
    // the resulting wrong answer would be finite and plausible.
    while (vcap % vslices != 0) --vslices;
    int vd_col = tid % head_dim;
    int vslice = tid / head_dim;
    bool vactive = vslice < vslices;
    int vk0 = vslice * MMA_VB;
    int vkstep = vslices * MMA_VB;
    const unsigned short* vd =
        v + (long long)kvh * (long long)head_dim + vd_col;
    long long vrow = (long long)kv_heads * (long long)head_dim;

    // The staged tile, in flight.
    //
    // The loop below is software-pipelined: a trip stores the tile that was
    // loaded during the *previous* trip's arithmetic, then immediately issues
    // the loads for the next one, then computes. With one block per SM there is
    // no second block to cover the staging latency, so without this the DRAM
    // round trip sits between two barriers with the tensor cores idle -- which
    // is what held the kernel to a third of peak bandwidth even after each
    // request was widened to 16 bytes.
    //
    // Both arrays must be indexed by a compile-time constant or they land in
    // local memory and the whole point is lost, so the loops are `#pragma
    // unroll` over a fixed bound with the real trip count as a predicate rather
    // than as the bound.
    int kw4 = hd2 >> 2;                              // uint4s per key row
    uint4 kreg[MMA_KREG];
    unsigned short vlo[MMA_VREG][MMA_VB], vhi[MMA_VREG][MMA_VB];

    // K in the B layout of Q K^T, [key][dim/2]: dimensions 2c and 2c+1 of one
    // key are already adjacent binary16 in the cache, which is exactly the B
    // operand's packing -- no conversion, four operands per request.
    //
    // V in the B layout of P V, which contracts over keys: the two halves of a
    // word are two consecutive *keys* at one dimension, so it is the transpose
    // of K's tile and cannot take K's wide-load trick. A `uint4` there is eight
    // consecutive dimensions, and those land eight *rows* apart in `v_sh`; the
    // fragment read needs `gcd(vstride, 32) == 4`, and `8 * vstride` is then a
    // multiple of 32, so all 32 lanes would hit one bank.
#define MMA_PREFETCH(jj)                                                       \
    do {                                                                       \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < MMA_KREG; ++i) {                                   \
            int t = tid + i * nthr;                                            \
            kreg[i] = make_uint4(0u, 0u, 0u, 0u);                              \
            if (t < MMA_KT * kw4) {                                            \
                int r = t / kw4;                                               \
                long long key = (jj) + r;                                      \
                if (key < n_visible) {                                         \
                    const uint4* kp = (const uint4*)(                          \
                        k + (key * (long long)kv_heads + kvh)                  \
                                * (long long)head_dim);                        \
                    kreg[i] = kp[t - r * kw4];                                 \
                }                                                              \
            }                                                                  \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < MMA_VREG; ++i) {                                   \
            int kk0 = vk0 + i * vkstep;                                        \
            bool ok = vactive && kk0 < (MMA_KT >> 1);                          \
            _Pragma("unroll")                                                  \
            for (int u = 0; u < MMA_VB; ++u) {                                 \
                long long k0 = (jj) + 2 * (kk0 + u);                           \
                vlo[i][u] = (ok && k0 < n_visible)                             \
                                ? vd[k0 * vrow] : (unsigned short)0;           \
                vhi[i][u] = (ok && k0 + 1 < n_visible)                         \
                                ? vd[(k0 + 1) * vrow] : (unsigned short)0;     \
            }                                                                  \
        }                                                                      \
    } while (0)

    MMA_PREFETCH(0);

    for (long long j0 = 0; j0 < n_visible; j0 += MMA_KT) {
        __syncthreads();
        // Land the tile the previous trip loaded. `qstride` is `hd2 + 4` and
        // `hd2` is a multiple of 16, so both the row base and `4*c4` are
        // 16-byte aligned; a 128-bit shared store is serviced in quarter-warp
        // phases, so the eight lanes of a phase cover all 32 banks.
        #pragma unroll
        for (int i = 0; i < MMA_KREG; ++i) {
            int t = tid + i * nthr;
            if (t < MMA_KT * kw4) {
                int r = t / kw4;
                *(uint4*)(k_sh + r * qstride + 4 * (t - r * kw4)) = kreg[i];
            }
        }
        #pragma unroll
        for (int i = 0; i < MMA_VREG; ++i) {
            int kk0 = vk0 + i * vkstep;
            if (vactive && kk0 < (MMA_KT >> 1)) {
                #pragma unroll
                for (int u = 0; u < MMA_VB; ++u) {
                    v_sh[vd_col * vstride + kk0 + u] =
                        (unsigned)vlo[i][u] | ((unsigned)vhi[i][u] << 16);
                }
            }
        }
        __syncthreads();

        // Issue the next tile's loads now, so the DRAM round trip overlaps the
        // arithmetic below instead of preceding it. Nothing between here and
        // the next barrier reads `kreg`/`vlo`/`vhi`, so the scheduler is free
        // to leave them in flight for the whole of `Q K^T`, the softmax and
        // `P V`.
        if (j0 + MMA_KT < n_visible) {
            MMA_PREFETCH(j0 + MMA_KT);
        }

        // Q K^T produces the same fragment layout that P V consumes.
        // Keep the scores in that layout instead of staging them through
        // shared memory and redistributing rows for the softmax.
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        #pragma unroll
        for (int s = 0; s < (COMP ? 16 : MMA_QSTEPS); ++s) {
            if (s < steps) {
                unsigned b0 = k_sh[g * qstride + 4 * s + tg];
                mma_m16n8k8(s0, s1, s2, s3, qa0[s], qa1[s], b0);
                if(COMP)mma_m16n8k8(s0,s1,s2,s3,qal0[s],qal1[s],b0);
            }
        }
        float scale2 = scale * ATTN_LOG2E;
        // The old shared store rounded this product before subtraction in
        // exp2. Explicit rounding prevents a new multiply/subtract FMA.
        s0 = __fmul_rn(s0, scale2); s1 = __fmul_rn(s1, scale2);
        s2 = __fmul_rn(s2, scale2); s3 = __fmul_rn(s3, scale2);
        long long key = j0 + 2 * tg;
        long long limit = (long long)(*key_offset) + qi0 + g;
        float cg = mma_softmax_pair(s0, s1,
            key <= limit && key < n_visible,
            key + 1 <= limit && key + 1 < n_visible,
            maximum0, normalizer0);
        float cg8 = mma_softmax_pair(s2, s3,
            key <= limit + 8 && key < n_visible,
            key + 1 <= limit + 8 && key + 1 < n_visible,
            maximum1, normalizer1);
        unsigned a0 = pack_h2(s0, s1);
        unsigned a1 = pack_h2(s2, s3);
        unsigned al0=0,al1=0;if(COMP){al0=pack_h2(s0-h2f((unsigned short)a0),s1-h2f((unsigned short)(a0>>16)));al1=pack_h2(s2-h2f((unsigned short)a1),s3-h2f((unsigned short)(a1>>16)));}

        // Rescale remains unconditional: the old vote to skip an identity
        // correction cost more than these multiplies (see BENCHMARKS.md).
        #pragma unroll
        for (int t = 0; t < MAXT; ++t) {
            o[t][0] *= cg; o[t][1] *= cg; o[t][2] *= cg8; o[t][3] *= cg8;
        }
        #pragma unroll
        for (int t = 0; t < MAXT; ++t) {
            if (t < ntile) {
                unsigned b0 = v_sh[(dbase + 8 * t + g) * vstride + tg];
                mma_m16n8k8(o[t][0], o[t][1], o[t][2], o[t][3], a0, a1, b0);
                if(COMP)mma_m16n8k8(o[t][0],o[t][1],o[t][2],o[t][3],al0,al1,b0);
            }
        }
    }

    #pragma unroll
    for (int t = 0; t < MAXT; ++t) {
        if (t < ntile) {
            int d = dbase + 8 * t + 2 * tg;
            long long r0 = qi0 + g;
            long long r1 = qi0 + g + 8;
            if (r0 < (long long)n_query) {
                float inv = 1.0f / normalizer0;
                float* op = out + (r0 * (long long)q_heads + h) * (long long)head_dim;
                op[d]     = o[t][0] * inv;
                op[d + 1] = o[t][1] * inv;
            }
            if (r1 < (long long)n_query) {
                float inv = 1.0f / normalizer1;
                float* op = out + (r1 * (long long)q_heads + h) * (long long)head_dim;
                op[d]     = o[t][2] * inv;
                op[d + 1] = o[t][3] * inv;
            }
        }
    }
}


extern "C" {
__global__ void attn_flash_causal_mma(const float* q,const unsigned short* k,const unsigned short* v,float* out,int q_heads,int kv_heads,int head_dim,const int* key_offset,float scale,int n_query){attn_flash_causal_mma_body<false,8,32>(q,k,v,out,q_heads,kv_heads,head_dim,key_offset,scale,n_query);}
__global__ void attn_flash_causal_mma_k2(const float* q,const unsigned short* k,const unsigned short* v,float* out,int q_heads,int kv_heads,int head_dim,const int* key_offset,float scale,int n_query){attn_flash_causal_mma_body<true,4,16>(q,k,v,out,q_heads,kv_heads,head_dim,key_offset,scale,n_query);}
// K2 prefill: two 16-query tiles of one KV head's four query heads share each
// staged run of 32 keys, so a barrier pair covers four key octets instead of
// one and each K/V element is staged once per 32 queries. A warp still walks
// its keys in 8-key steps with attn_flash_causal_mma_body<true>'s operations
// and stops at its own tile's visible keys, so its output is unchanged.
#define K2S_QT 16
#define K2S_SK 32
#define K2S_KSTRIDE 68
#define K2S_VSTRIDE 20
__global__ void __launch_bounds__(256,1) attn_flash_causal_mma_k2s(const float* __restrict__ q,const unsigned short* __restrict__ k,const unsigned short* __restrict__ v,float* __restrict__ out,int q_heads,int kv_heads,int head_dim,const int* __restrict__ key_offset,float scale,int n_query){
    extern __shared__ float smem_f[];
    unsigned* k_sh=(unsigned*)smem_f;
    unsigned* v_sh=k_sh+K2S_SK*K2S_KSTRIDE;
    int tid=threadIdx.x,lane=tid&31,warp=tid>>5,g=lane>>2,tg=lane&3;
    // Blocks are handed out longest causal run first: the last query block of
    // every KV head, then the next-to-last, so the deep blocks do not land in
    // the final wave. The grid keeps its (query block, KV head) shape.
    int id=blockIdx.x+gridDim.x*blockIdx.y;
    int kvh=id%gridDim.y,h=kvh*4+(warp&3);
    long long qb=(long long)(gridDim.x-1-id/gridDim.y)*(2*K2S_QT),qi0=qb+(warp>>2)*K2S_QT;
    long long off=*key_offset;
    unsigned qa0[16],qa1[16],qal0[16],qal1[16];
    {
        long long r0=qi0+g,r1=qi0+g+8;
        const float* qp0=(r0<(long long)n_query) ? q+(r0*(long long)q_heads+h)*128LL : 0;
        const float* qp1=(r1<(long long)n_query) ? q+(r1*(long long)q_heads+h)*128LL : 0;
        #pragma unroll
        for(int s=0;s<16;++s) {
            int c=4*s+tg;
            qa0[s]=0u;qa1[s]=0u;
            if(qp0)qa0[s]=pack_h2(qp0[2*c],qp0[2*c+1]);
            if(qp1)qa1[s]=pack_h2(qp1[2*c],qp1[2*c+1]);
            qal0[s]=qp0 ? pack_h2(qp0[2*c]-h2f((unsigned short)qa0[s]),qp0[2*c+1]-h2f((unsigned short)(qa0[s]>>16))) : 0;
            qal1[s]=qp1 ? pack_h2(qp1[2*c]-h2f((unsigned short)qa1[s]),qp1[2*c+1]-h2f((unsigned short)(qa1[s]>>16))) : 0;
        }
    }
    float o[16][4];
    #pragma unroll
    for(int t=0;t<16;++t){o[t][0]=0.0f;o[t][1]=0.0f;o[t][2]=0.0f;o[t][3]=0.0f;}
    float maximum0=neg_inf(),maximum1=neg_inf(),normalizer0=0.0f,normalizer1=0.0f;
    long long qt_live=(long long)n_query-qi0;if(qt_live>K2S_QT)qt_live=K2S_QT;
    long long n_tile=qt_live>0 ? off+qi0+qt_live : 0;
    long long b_live=(long long)n_query-qb;if(b_live>2*K2S_QT)b_live=2*K2S_QT;
    long long n_block=off+qb+b_live;
    // Staging: two 16-byte K chunks and eight packed V key pairs per thread.
    int vdim=tid&127,vslice=tid>>7;
    const unsigned short* vd=v+(long long)kvh*128+vdim;
    long long vrow=(long long)kv_heads*128;
#if KV_KQ8
    // q8_0 K: the same eight dimensions as one uint4 of binary16, as eight
    // codes and their block's scale, held raw across the multiply like kreg
    // and converted at the shared store.
    long long krow8=kvq8_row_bytes(kv_heads*128);
    uint2 kq8[2];unsigned short kd8[2];
#else
    uint4 kreg[2];
#endif
#if KV_VQ8
    // q8_0 V: vreg packs the two keys' codes in bytes 0 and 1; vds their scales.
    long long vrow8=kvq8_row_bytes(kv_heads*128);
    const unsigned char* vc8=(const unsigned char*)v+(long long)kvh*128+vdim;
    const unsigned char* vs8=(const unsigned char*)v+vrow+2LL*(kvh*4+(vdim>>5));
    unsigned vds[8];
#endif
    unsigned vreg[8];
    auto prefetch=[&](long long j) {
        #pragma unroll
        for(int i=0;i<2;++i) {
            int t=tid+256*i,r=t>>4;long long key=j+r;
#if KV_KQ8
            kq8[i]=make_uint2(0u,0u);kd8[i]=0;
            if(key<n_block){const unsigned char* row=(const unsigned char*)k+key*krow8;kq8[i]=((const uint2*)(row+kvh*128))[t&15];kd8[i]=((const unsigned short*)(row+kv_heads*128))[kvh*4+((t&15)>>2)];}
#else
            kreg[i]=make_uint4(0u,0u,0u,0u);
            if(key<n_block)kreg[i]=((const uint4*)(k+(key*(long long)kv_heads+kvh)*128LL))[t&15];
#endif
        }
        #pragma unroll
        for(int u=0;u<8;++u) {
            long long k0=j+2*(vslice*8+u);
#if KV_VQ8
            unsigned lo=k0<n_block ? vc8[k0*vrow8] : 0u,hi=k0+1<n_block ? vc8[(k0+1)*vrow8] : 0u;
            unsigned dlo=k0<n_block ? *(const unsigned short*)(vs8+k0*vrow8) : 0u,dhi=k0+1<n_block ? *(const unsigned short*)(vs8+(k0+1)*vrow8) : 0u;
            vreg[u]=lo|(hi<<8);vds[u]=dlo|(dhi<<16);
#else
            unsigned lo=k0<n_block ? vd[k0*vrow] : 0u,hi=k0+1<n_block ? vd[(k0+1)*vrow] : 0u;
            vreg[u]=lo|(hi<<16);
#endif
        }
    };
    prefetch(0);
    float scale2=scale*ATTN_LOG2E;
    for(long long j0=0;j0<n_block;j0+=K2S_SK) {
        __syncthreads();
        #pragma unroll
#if KV_KQ8
        for(int i=0;i<2;++i){int t=tid+256*i;uint4 h;kvq8_dequant4(kq8[i].x,kd8[i],&h.x,&h.y);kvq8_dequant4(kq8[i].y,kd8[i],&h.z,&h.w);*(uint4*)(k_sh+(t>>4)*K2S_KSTRIDE+4*(t&15))=h;}
#else
        for(int i=0;i<2;++i){int t=tid+256*i;*(uint4*)(k_sh+(t>>4)*K2S_KSTRIDE+4*(t&15))=kreg[i];}
#endif
        #pragma unroll
#if KV_VQ8
        for(int u=0;u<8;++u)v_sh[vdim*K2S_VSTRIDE+vslice*8+u]=kvq8_dequant2(vreg[u],(unsigned short)vds[u],(unsigned short)(vds[u]>>16));
#else
        for(int u=0;u<8;++u)v_sh[vdim*K2S_VSTRIDE+vslice*8+u]=vreg[u];
#endif
        __syncthreads();
        if(j0+K2S_SK<n_block)prefetch(j0+K2S_SK);
        #pragma unroll
        for(int st=0;st<K2S_SK/8;++st) {
            long long j=j0+8*st;
            if(j>=n_tile)break;
            float s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f;
            #pragma unroll
            for(int s=0;s<16;++s) {
                unsigned b0=k_sh[(st*8+g)*K2S_KSTRIDE+4*s+tg];
                mma_m16n8k8(s0,s1,s2,s3,qa0[s],qa1[s],b0);
                mma_m16n8k8(s0,s1,s2,s3,qal0[s],qal1[s],b0);
            }
            s0=__fmul_rn(s0,scale2);s1=__fmul_rn(s1,scale2);
            s2=__fmul_rn(s2,scale2);s3=__fmul_rn(s3,scale2);
            long long key=j+2*tg,limit=off+qi0+g;
            float cg=mma_softmax_pair(s0,s1,key<=limit && key<n_tile,key+1<=limit && key+1<n_tile,maximum0,normalizer0);
            float cg8=mma_softmax_pair(s2,s3,key<=limit+8 && key<n_tile,key+1<=limit+8 && key+1<n_tile,maximum1,normalizer1);
            unsigned a0=pack_h2(s0,s1),a1=pack_h2(s2,s3);
            unsigned al0=pack_h2(s0-h2f((unsigned short)a0),s1-h2f((unsigned short)(a0>>16)));
            unsigned al1=pack_h2(s2-h2f((unsigned short)a1),s3-h2f((unsigned short)(a1>>16)));
            #pragma unroll
            for(int t=0;t<16;++t){o[t][0]*=cg;o[t][1]*=cg;o[t][2]*=cg8;o[t][3]*=cg8;}
            #pragma unroll
            for(int t=0;t<16;++t) {
                unsigned b0=v_sh[(8*t+g)*K2S_VSTRIDE+4*st+tg];
                mma_m16n8k8(o[t][0],o[t][1],o[t][2],o[t][3],a0,a1,b0);
                mma_m16n8k8(o[t][0],o[t][1],o[t][2],o[t][3],al0,al1,b0);
            }
        }
    }
    #pragma unroll
    for(int t=0;t<16;++t) {
        int d=8*t+2*tg;long long r0=qi0+g,r1=qi0+g+8;
        if(r0<(long long)n_query){float inv=1.0f/normalizer0;float* op=out+(r0*(long long)q_heads+h)*128LL;op[d]=o[t][0]*inv;op[d+1]=o[t][1]*inv;}
        if(r1<(long long)n_query){float inv=1.0f/normalizer1;float* op=out+(r1*(long long)q_heads+h)*128LL;op[d]=o[t][2]*inv;op[d+1]=o[t][3]*inv;}
    }
}

// ---------------------------------------------------------------------------
// The same attention, with the KV head — not the query head — on the grid.
//
// grid: (n_query / GQA_QT, kv_heads) — one block per (query tile, KV head).
// block: (q_heads / kv_heads) warps — one warp per query head of that KV head.
//
// ## Why this kernel exists
//
// `attn_flash_causal` above puts the *query* head on grid.y, so at Qwen3.6's
// 16:2 grouping the eight query heads sharing a KV head are eight separate
// blocks, and each one streams the whole K and V window for itself. The work
// needs one pass over K/V per KV head; the kernel paid one per query head.
// Measured at the sixteenth chunk of an 8,192-token chunked prefill, that was
// 16.4 GB moved in 31.65 ms — 77% of this card's 672 GB/s — against a 2.05 GB
// lower bound. Eight times the traffic, and the kernel was bandwidth-bound, so
// it was eight times the time.
//
// Two cheaper fixes were measured first and both lost; see docs/BENCHMARKS.md.
// Widening ATTN_QT to 16 halves the block count but doubles `qr` to 128
// registers and halves resident blocks: 8K prefill 6,004 -> 6,647 ms. Merely
// swapping the grid axes so the eight siblings are *co-resident* and hit in L2
// lost 1.7% across three interleaved pairs — sibling blocks start together but
// drift apart faster than 6 MB of L2 can span, so nothing but an explicit
// staging barrier actually makes them share.
//
// ## The shape, and why GQA_QT is 4 and not 8
//
// DRAM traffic here is `(blocks) * (visible K/V)`, and blocks is
// `n_query/GQA_QT * kv_heads` against the old `n_query/ATTN_QT * q_heads`. The
// gqa ratio and ATTN_QT are both 8, so the reduction is exactly GQA_QT: four
// query rows, four times less traffic. Eight would give eight, and cannot be
// had — a warp now owns a whole head, so a lane carries `head_dim/32`
// accumulator dimensions per row rather than one, and `qr` plus `acc` is
// `2 * GQA_QT * ATTN_MAXD` registers. At GQA_QT 4 that is 64 and ptxas keeps
// two blocks resident; at 8 it is 128, which is the wall the ATTN_QT 16
// experiment already hit from the other side.
//
// ## What moved into shared, and what that costs
//
// K and V for GQA_KT keys are staged once per trip and read by all eight
// warps. The score loop then reads K from shared instead of global, one read
// feeding GQA_QT multiply-adds. Turing does 128 B/clk of shared against 64
// fp32 lanes — two warp-FMAs per warp-load — so a 4:1 ratio is still short of
// the load/store bound, which is why this does not simply move the stall.
//
// Consecutive lanes read consecutive words in both `k_sh` and `v_sh`
// (`lane + 32 * i` with a `head_dim` row stride), so no padding is needed:
// every access is one conflict-free 128 B phase.
//
// ## Softmax
//
// A warp owns a head outright, so the per-tile bookkeeping is warp-local and
// needs `__syncwarp` rather than `__syncthreads` — the only block-wide
// barriers left are the two that fence the staged tile. The running max and
// the normalizer stay serial and ascending over the tile, matching
// `causal_attention_streaming` and the kernel above; only the tile *width*
// differs (GQA_KT rather than `n_warps * ATTN_KT`), which changes where the
// rescales land and so is a floating-point difference, not an algebraic one.
#define GQA_QT 4
#define GQA_KT 8

__global__ void attn_flash_causal_gqa(
    const float* __restrict__ q,
    const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v,
    float* __restrict__ out,
    int q_heads,
    int kv_heads,
    int head_dim,
    const int* __restrict__ key_offset,
    float scale,
    int n_query
) {
    extern __shared__ float smem[];
    int gqa = q_heads / kv_heads;
    int dpt = head_dim >> 5;
    float* k_sh = smem;                            // GQA_KT * head_dim
    float* v_sh = k_sh + GQA_KT * head_dim;        // GQA_KT * head_dim
    float* w_sh = v_sh + GQA_KT * head_dim;        // gqa * GQA_QT * GQA_KT

    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int nthr = blockDim.x;

    long long qi0 = (long long)blockIdx.x * GQA_QT;
    int kvh = blockIdx.y;
    int h   = kvh * gqa + warp;      // this warp's query head
    float* my_w = w_sh + warp * (GQA_QT * GQA_KT);

    // This warp's query tile in registers. Lane `lane` owns dimensions
    // `lane, lane + 32, ...` of every row, so the dot product below reduces
    // over exactly the partition the untiled kernel used.
    float qr[GQA_QT][ATTN_MAXD];
    #pragma unroll
    for (int u = 0; u < GQA_QT; ++u) {
        long long qrow = qi0 + u;
        const float* qp = q + (qrow * (long long)q_heads + h) * (long long)head_dim;
        #pragma unroll
        for (int i = 0; i < ATTN_MAXD; ++i) {
            qr[u][i] = (i < dpt && qrow < (long long)n_query) ? qp[lane + 32 * i] : 0.0f;
        }
    }

    float acc[GQA_QT][ATTN_MAXD];
    float m[GQA_QT], ln[GQA_QT];
    #pragma unroll
    for (int u = 0; u < GQA_QT; ++u) {
        m[u] = neg_inf();
        ln[u] = 0.0f;
        #pragma unroll
        for (int i = 0; i < ATTN_MAXD; ++i) acc[u][i] = 0.0f;
    }

    int qt_live = (int)((long long)n_query - qi0);
    if (qt_live > GQA_QT) qt_live = GQA_QT;
    long long n_visible = (long long)(*key_offset) + qi0 + qt_live;

    for (long long j0 = 0; j0 < n_visible; j0 += GQA_KT) {
        // Stage the tile. `n_visible` is block-uniform, so every warp runs the
        // same trip count and the barriers below are reached by all of them.
        __syncthreads();
        // Every load for the tile is issued before any of it is stored; see
        // the same loop in `attn_flash_decode_split` for why. Costs
        // `2 * GQA_KT` registers, which is the reason it is measured here
        // rather than assumed: this kernel is already at the two-block
        // boundary where the decode one has room to spare.
        for (int d = tid; d < head_dim; d += nthr) {
            float kreg[GQA_KT];
            float vreg[GQA_KT];
            #pragma unroll
            for (int jj = 0; jj < GQA_KT; ++jj) {
                long long key = j0 + jj;
                bool live = key < n_visible;
                long long at = (key * (long long)kv_heads + kvh) * (long long)head_dim + d;
                kreg[jj] = live ? h2f(k[at]) : 0.0f;
                vreg[jj] = live ? h2f(v[at]) : 0.0f;
            }
            #pragma unroll
            for (int jj = 0; jj < GQA_KT; ++jj) {
                k_sh[jj * head_dim + d] = kreg[jj];
                v_sh[jj * head_dim + d] = vreg[jj];
            }
        }
        __syncthreads();

        // Scores. One shared read of a key dimension feeds GQA_QT multiply-adds.
        #pragma unroll
        for (int jj = 0; jj < GQA_KT; ++jj) {
            float part[GQA_QT];
            #pragma unroll
            for (int u = 0; u < GQA_QT; ++u) part[u] = 0.0f;
            #pragma unroll
            for (int i = 0; i < ATTN_MAXD; ++i) {
                if (i < dpt) {
                    float kd = k_sh[jj * head_dim + lane + 32 * i];
                    #pragma unroll
                    for (int u = 0; u < GQA_QT; ++u) part[u] += qr[u][i] * kd;
                }
            }
            #pragma unroll
            for (int u = 0; u < GQA_QT; ++u) {
                float p = part[u];
                for (int off = 16; off > 0; off >>= 1) {
                    p += __shfl_xor_sync(0xffffffff, p, off);
                }
                if (lane == 0) my_w[u * GQA_KT + jj] = p * scale;
            }
        }
        __syncwarp();

        // The running max, serial and ascending over the tile. Every lane
        // sweeps it redundantly: that is GQA_KT conflict-free shared
        // broadcasts, cheaper than serializing onto one lane and broadcasting
        // the result back. `corr` matches the reference's first-tile guard,
        // where there is no accumulator to rescale and expf(-inf - -inf) is NaN.
        float corr[GQA_QT];
        #pragma unroll
        for (int u = 0; u < GQA_QT; ++u) {
            long long limit = (long long)(*key_offset) + qi0 + u;
            float tmax = neg_inf();
            for (int w = 0; w < GQA_KT; ++w) {
                if (j0 + w <= limit) tmax = fmaxf(tmax, my_w[u * GQA_KT + w]);
            }
            float m0 = m[u];
            float nm = fmaxf(m0, tmax);
            corr[u] = (m0 == neg_inf()) ? 0.0f : expf(m0 - nm);
            m[u] = nm;
        }

        // One exponential per slot. Masked slots store a literal zero, which is
        // what lets a row whose causal bound ended earlier contribute exactly
        // nothing to the tiles past it: `a += 0.0f * v` and `l += 0.0f` are
        // both exact. Every read of `my_w` happens before any write to it.
        float wl[GQA_QT];
        #pragma unroll
        for (int u = 0; u < GQA_QT; ++u) {
            long long limit = (long long)(*key_offset) + qi0 + u;
            float s = (lane < GQA_KT) ? my_w[u * GQA_KT + lane] : 0.0f;
            wl[u] = (lane < GQA_KT && j0 + lane <= limit) ? expf(s - m[u]) : 0.0f;
        }
        __syncwarp();
        #pragma unroll
        for (int u = 0; u < GQA_QT; ++u) {
            if (lane < GQA_KT) my_w[u * GQA_KT + lane] = wl[u];
        }
        __syncwarp();

        // The normalizer, serial and ascending like the max above.
        #pragma unroll
        for (int u = 0; u < GQA_QT; ++u) {
            float lsum = 0.0f;
            for (int w = 0; w < GQA_KT; ++w) lsum += my_w[u * GQA_KT + w];
            ln[u] = ln[u] * corr[u] + lsum;
        }

        // The values. One shared read of a value dimension serves GQA_QT rows.
        #pragma unroll
        for (int u = 0; u < GQA_QT; ++u) {
            #pragma unroll
            for (int i = 0; i < ATTN_MAXD; ++i) acc[u][i] *= corr[u];
        }
        for (int w = 0; w < GQA_KT; ++w) {
            if (j0 + w >= n_visible) break;
            float wu[GQA_QT];
            #pragma unroll
            for (int u = 0; u < GQA_QT; ++u) wu[u] = my_w[u * GQA_KT + w];
            #pragma unroll
            for (int i = 0; i < ATTN_MAXD; ++i) {
                if (i < dpt) {
                    float vv = v_sh[w * head_dim + lane + 32 * i];
                    #pragma unroll
                    for (int u = 0; u < GQA_QT; ++u) acc[u][i] += wu[u] * vv;
                }
            }
        }
    }

    #pragma unroll
    for (int u = 0; u < GQA_QT; ++u) {
        long long qrow = qi0 + u;
        if (qrow < (long long)n_query) {
            float* op = out + (qrow * (long long)q_heads + h) * (long long)head_dim;
            #pragma unroll
            for (int i = 0; i < ATTN_MAXD; ++i) {
                if (i < dpt) op[lane + 32 * i] = acc[u][i] / ln[u];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Flash decoding: one query row, the key range split across blocks.
//
// grid: (path-specific splits, kv_heads) for the split pass, (q_heads,) for
// the combine.
// block: (q_heads / kv_heads) warps, one warp per query head, as above.
//
// ## Why decode needs its own shape
//
// `attn_flash_causal_t1` launches `(n_query, q_heads)`, and decode is
// `n_query == 1`. That is **sixteen blocks on a 72-SM card**: 78% of the
// machine is idle before the eightfold K/V redundancy is even counted. Nothing
// about the query tile can fix that, because there is only one query row to
// tile. The parallelism has to come from the key axis instead.
//
// So each block takes a contiguous slice of the key range and runs the same
// online softmax over just that slice, emitting a partial `(m, l, acc)`. A
// second pass merges the slices. The merge is exact in exact arithmetic:
// writing `gm` for `max_s m_s`,
//
//     sum_s exp(m_s - gm) * acc_s = sum_s sum_{j in s} exp(s_j - gm) * v_j
//     sum_s exp(m_s - gm) * l_s   = sum_s sum_{j in s} exp(s_j - gm)
//
// which are the numerator and denominator a single pass would have built, and
// `gm` is the global maximum because the maximum of the slice maxima is it.
//
// ## Rule 5
//
// Both split counts are host constants, but no slice boundary is. The slice
// width is computed on the device from `*key_offset` and rounded up to a whole `DEC_KT`
// tile, so a tile never straddles a boundary and every block's trip count comes
// from device state alone. Splits past the end of a short window run zero trips
// and write the identity partial — `m = -inf`, `l = 0`, `acc = 0` — which the
// combine folds in as `exp(-inf - gm) = 0`, exactly.
#define DEC_KT 8
#define DEC_WARP_SPLITS 96
#define DEC_MMA_SPLITS 72

// ---------------------------------------------------------------------------
// Flash decoding again, with a warp as the whole split and no shared memory.
//
// grid: (DEC_SPLITS, kv_heads). block: one warp.
//
// ## Why the kernel below it was not enough
//
// `attn_flash_decode_split` gives a warp one query head, so a warp needs every
// dimension of every key and the tile has to be staged in shared for the eight
// warps of a KV group to share. Measured at a 131,072-key window it moved
// 184.8 GB/s -- 27.5% of this card -- and the reason is arithmetic rather than
// mystery:
//
//   shared per block-tile = 8 warps * 2 * DEC_KT * head_dim * 4 B = 128 KiB
//   DRAM   per block-tile =           2 * DEC_KT * head_dim * 2 B =   8 KiB
//
// At 128 B/cycle the shared traffic is ~1,152 cycles against ~1,224 for the
// DRAM it is meant to be hiding behind. The staging *is* the bottleneck. Two
// cheaper explanations were measured and rejected first: `DEC_SPLITS` at 144,
// 288, 432 and 576 gives 1.453, 1.453, 1.466, 1.477 ms, so it is not block
// parallelism; and the tile already keeps ~24 KiB in flight per SM against the
// ~4 KiB Little's law asks for, so it is not memory-level parallelism.
//
// That splits sweep says more than it was first credited with. It varies the
// resident warps 3.5x at *fixed* total work and moves the time 1.7%, which
// rules out a latency bound as surely as it rules out a parallelism one: more
// warps would hide more latency. Invariance to parallelism at fixed work is
// the signature of a saturated per-SM resource, and the sector amplification
// documented at the `wpl == 4` branch below is one. The datum sat here for
// some time labelled "no effect" because nothing yet explained it.
//
// ## The shape
//
// A lane holds `head_dim/32` **consecutive** dimensions, so the 32 lanes of one
// warp cover a whole key and can load it as one coalesced run straight from
// global -- though only if the load is written as a single wide access, which
// for a long time it was not; see the `wpl == 4` branch below.
// The warp then owns *every* query head of its KV head, reusing that key
// `gqa` times out of registers. K and V are read exactly once each, by exactly
// one warp, and shared memory disappears along with every barrier.
//
// The partition is the point. "One lane per dimension" would also reuse the
// key, but then each of the `gqa * DEC_KT` dot products reduces across 32 lanes
// separately -- eight times the shuffles. Holding `head_dim/32` dims per lane
// makes the inner sum sequential and free, and leaves the shuffle count per
// warp exactly what the staged kernel already paid.
//
// ## Rescaling is rare, not per key
//
// The online softmax rescales when the running maximum moves, which after the
// first few keys is O(log n) of them. Guarding the rescale on `nm != m` costs a
// warp-uniform branch -- the score is identical in every lane after the
// reduction -- and removes both an `expf` and a `head_dim/32`-wide multiply
// from the common path. Without that guard this shape would pay two `expf` per
// key per head where the staged kernel pays one per *tile*.
//
// A warp is simply a finer split, so `attn_flash_decode_combine` merges these
// with the identity it already implements and needs no change.
#define DEC_MAXG 8
// Keys whose loads are issued before any of them is consumed.
#define DEC_KB 1

// `half2` accumulation for `P V` was tried and measured slower -- 1.334 vs
// 1.198 ms at a 131,072-key window, +11.3%, despite compiling to *fewer*
// registers (155 vs 195) and *fewer* static instructions (1,424 vs 1,536).
// `fma.rn.f16x2`/`mul.f16x2` replace the eight-wide fp32 FMA with a four-wide
// f16x2 one straight against the `uint4` load's packed word, no `h2f2` unpack
// needed. That accounting made it look free; the measurement says the
// packing (`F2F`+`PRMT` per head per key, to broadcast the scalar softmax
// weight into a `half2`) costs more than the FMAs it removes, and where that
// cost lands is still open -- `ncu` cannot run on this host. See
// docs/BENCHMARKS.md for the SASS breakdown. Not applied.
__global__ void attn_flash_decode_warp(
    const float* __restrict__ q,
    const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v,
    float* __restrict__ part_acc,
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    int q_heads,
    int kv_heads,
    int head_dim,
    const int* __restrict__ key_offset,
    float scale,
    int n_splits
) {
    int gqa = q_heads / kv_heads;
    int dpl = head_dim >> 5;          // dimensions this lane owns
    int wpl = dpl >> 1;               // packed 32-bit words behind them
    int lane = threadIdx.x;
    int split = blockIdx.x;
    int kvh = blockIdx.y;

    long long n_visible = (long long)(*key_offset) + 1;
    long long per = (n_visible + n_splits - 1) / n_splits;
    long long begin = (long long)split * per;
    long long end = begin + per;
    if (end > n_visible) end = n_visible;

    float qr[DEC_MAXG][ATTN_MAXD];
    float acc[DEC_MAXG][ATTN_MAXD];
    float m[DEC_MAXG], l[DEC_MAXG];
    #pragma unroll
    for (int hh = 0; hh < DEC_MAXG; ++hh) {
        m[hh] = neg_inf();
        l[hh] = 0.0f;
        #pragma unroll
        for (int i = 0; i < ATTN_MAXD; ++i) { qr[hh][i] = 0.0f; acc[hh][i] = 0.0f; }
        if (hh < gqa) {
            const float* qp =
                q + (long long)(kvh * gqa + hh) * (long long)head_dim;
            #pragma unroll
            for (int i = 0; i < ATTN_MAXD; ++i) {
                if (i < dpl) qr[hh][i] = qp[dpl * lane + i];
            }
        }
    }

    // Keys are taken `DEC_KB` at a time with every load issued before any of
    // the arithmetic. The loop bound is a runtime value, so ptxas cannot
    // unroll it and the next key's loads would otherwise wait on this key's
    // registers -- one key in flight per warp, which is the same
    // memory-parallelism bound that held the prefill staging to a third of
    // peak. A batch of `DEC_KB` puts `8 * DEC_KB` requests in flight instead.
    for (long long j0 = begin; j0 < end; j0 += DEC_KB) {
        unsigned kw[DEC_KB][ATTN_MAXD / 2], vw[DEC_KB][ATTN_MAXD / 2];
        #pragma unroll
        for (int jj = 0; jj < DEC_KB; ++jj) {
            long long key = j0 + jj;
            bool live = key < end;
            long long at = (key * (long long)kv_heads + kvh) * (long long)head_dim;
            const unsigned* kp = (const unsigned*)(k + at);
            const unsigned* vp = (const unsigned*)(v + at);
            // A lane owns `wpl` consecutive words, so `kp[wpl * lane + p]` is a
            // stride-`wpl` access across the warp -- not the single `uint4` the
            // partition above describes. At head_dim 256 that is a 16-byte
            // stride: each of the four loads touches all sixteen 32-byte
            // sectors of the 512-byte key and takes four bytes from each, and
            // the four loads then request the same sixteen sectors again. Four
            // times the sector traffic for the same bytes, which caps the
            // kernel near a quarter of peak -- 27.5% was measured.
            //
            // ptxas cannot rescue it: `wpl` comes from the runtime `head_dim`
            // argument, so neither `wpl == 4` nor 16-byte alignment is provable
            // at compile time, and the per-element `p < wpl` predicate blocks
            // vectorisation on its own. `cuobjdump -sass` confirmed the kernel
            // emitted no `LDG.E.128` at all while `attn_flash_causal_mma` in
            // this same file emits two, so the compiler vectorises here when
            // the pattern permits and this pattern did not permit it.
            //
            // Naming the width restores it. `at` is a multiple of `head_dim`,
            // so at head_dim 256 the address is 512-byte aligned and a `uint4`
            // load is legal; lane `l` takes bytes `[16l, 16l+16)`, which is the
            // same eight binary16 dimensions `[8l, 8l+8)` it read before as
            // four strided words. Identical data, one instruction, and the warp
            // now covers 512 contiguous bytes with every sector fully consumed.
            if (wpl == 4) {
                const uint4* kp4 = (const uint4*)kp;
                const uint4* vp4 = (const uint4*)vp;
                uint4 kq = make_uint4(0u, 0u, 0u, 0u);
                uint4 vq = make_uint4(0u, 0u, 0u, 0u);
                if (live) { kq = kp4[lane]; vq = vp4[lane]; }
                kw[jj][0] = kq.x; kw[jj][1] = kq.y;
                kw[jj][2] = kq.z; kw[jj][3] = kq.w;
                vw[jj][0] = vq.x; vw[jj][1] = vq.y;
                vw[jj][2] = vq.z; vw[jj][3] = vq.w;
            } else {
                #pragma unroll
                for (int p = 0; p < (ATTN_MAXD >> 1); ++p) {
                    kw[jj][p] = (live && p < wpl) ? kp[wpl * lane + p] : 0u;
                    vw[jj][p] = (live && p < wpl) ? vp[wpl * lane + p] : 0u;
                }
            }
        }
        #pragma unroll
        for (int jj = 0; jj < DEC_KB; ++jj) {
        if (j0 + jj >= end) break;
        float kk[ATTN_MAXD], vv[ATTN_MAXD];
        #pragma unroll
        for (int p = 0; p < (ATTN_MAXD >> 1); ++p) {
            if (p < wpl) {
                h2f2(kw[jj][p], &kk[2 * p], &kk[2 * p + 1]);
                h2f2(vw[jj][p], &vv[2 * p], &vv[2 * p + 1]);
            }
        }
        #pragma unroll
        for (int hh = 0; hh < DEC_MAXG; ++hh) {
            if (hh < gqa) {
                float part = 0.0f;
                #pragma unroll
                for (int i = 0; i < ATTN_MAXD; ++i) {
                    if (i < dpl) part += qr[hh][i] * kk[i];
                }
                for (int off = 16; off > 0; off >>= 1) {
                    part += __shfl_xor_sync(0xffffffff, part, off);
                }
                // Identical in every lane from here, so the branch below is
                // warp-uniform and the scalars need no broadcast.
                float s = part * scale;
                float nm = fmaxf(m[hh], s);
                if (nm != m[hh]) {
                    float corr = (m[hh] == neg_inf()) ? 0.0f : expf(m[hh] - nm);
                    l[hh] *= corr;
                    #pragma unroll
                    for (int i = 0; i < ATTN_MAXD; ++i) acc[hh][i] *= corr;
                    m[hh] = nm;
                }
                float e = expf(s - nm);
                l[hh] += e;
                #pragma unroll
                for (int i = 0; i < ATTN_MAXD; ++i) {
                    if (i < dpl) acc[hh][i] += e * vv[i];
                }
            }
        }
        }
    }

    #pragma unroll
    for (int hh = 0; hh < DEC_MAXG; ++hh) {
        if (hh < gqa) {
            int h = kvh * gqa + hh;
            float* pa =
                part_acc + ((long long)split * q_heads + h) * (long long)head_dim;
            #pragma unroll
            for (int i = 0; i < ATTN_MAXD; ++i) {
                if (i < dpl) pa[dpl * lane + i] = acc[hh][i];
            }
            if (lane == 0) {
                part_m[split * q_heads + h] = m[hh];
                part_l[split * q_heads + h] = l[hh];
            }
        }
    }
}

__global__ void attn_flash_decode_split(
    const float* __restrict__ q,
    const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v,
    float* __restrict__ part_acc,
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    int q_heads,
    int kv_heads,
    int head_dim,
    const int* __restrict__ key_offset,
    float scale,
    int n_splits
) {
    extern __shared__ float smem[];
    int gqa = q_heads / kv_heads;
    int dpt = head_dim >> 5;
    float* k_sh = smem;                        // DEC_KT * head_dim
    float* v_sh = k_sh + DEC_KT * head_dim;    // DEC_KT * head_dim
    float* s_sh = v_sh + DEC_KT * head_dim;    // gqa * DEC_KT

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int nthr = blockDim.x;
    int split = blockIdx.x;
    int kvh = blockIdx.y;
    int h = kvh * gqa + warp;
    float* my_s = s_sh + warp * DEC_KT;

    // Decode's single row sits at absolute position *key_offset and sees keys
    // [0, *key_offset]. Every quantity below is block-uniform, so the barriers
    // in the trip loop are reached by every warp the same number of times.
    long long n_visible = (long long)(*key_offset) + 1;
    long long per = (n_visible + n_splits - 1) / n_splits;
    per = ((per + DEC_KT - 1) / DEC_KT) * DEC_KT;
    long long begin = (long long)split * per;
    long long end = begin + per;
    if (end > n_visible) end = n_visible;

    float qr[ATTN_MAXD];
    const float* qp = q + (long long)h * (long long)head_dim;
    #pragma unroll
    for (int i = 0; i < ATTN_MAXD; ++i) qr[i] = (i < dpt) ? qp[lane + 32 * i] : 0.0f;

    float acc[ATTN_MAXD];
    #pragma unroll
    for (int i = 0; i < ATTN_MAXD; ++i) acc[i] = 0.0f;
    float m = neg_inf();
    float l = 0.0f;

    for (long long j0 = begin; j0 < end; j0 += DEC_KT) {
        int n_this = (int)((end - j0) < (long long)DEC_KT ? (end - j0) : (long long)DEC_KT);
        __syncthreads();
        // Every load for the tile is issued before any of it is stored.
        //
        // Writing straight into shared makes each store depend on the load
        // just above it, which leaves the tile with about one outstanding
        // request per thread at a time; the kernel then waits out the full
        // DRAM latency DEC_KT times per trip instead of once. Landing the
        // whole tile in registers first leaves `2 * DEC_KT` requests in
        // flight, which is what turns this loop from latency-bound into
        // bandwidth-bound. Costs `2 * DEC_KT` registers, which this kernel
        // has because one query row makes `qr` and `acc` a quarter of what
        // the prefill kernel carries.
        // Two dimensions per thread, because they are one 32-bit word in a
        // binary16 cache. Loading them singly would halve the bytes and keep
        // the load count, which measured as a net loss on decode even while it
        // won on prefill.
        int hd2 = head_dim >> 1;
        for (int d2 = tid; d2 < hd2; d2 += nthr) {
            unsigned kreg[DEC_KT];
            unsigned vreg[DEC_KT];
            #pragma unroll
            for (int jj = 0; jj < DEC_KT; ++jj) {
                long long key = j0 + jj;
                bool live = jj < n_this;
                long long at = (key * (long long)kv_heads + kvh) * (long long)head_dim;
                const unsigned* kp = (const unsigned*)(k + at);
                const unsigned* vp = (const unsigned*)(v + at);
                kreg[jj] = live ? kp[d2] : 0u;
                vreg[jj] = live ? vp[d2] : 0u;
            }
            #pragma unroll
            for (int jj = 0; jj < DEC_KT; ++jj) {
                float lo, hi;
                h2f2(kreg[jj], &lo, &hi);
                k_sh[jj * head_dim + 2 * d2] = lo;
                k_sh[jj * head_dim + 2 * d2 + 1] = hi;
                h2f2(vreg[jj], &lo, &hi);
                v_sh[jj * head_dim + 2 * d2] = lo;
                v_sh[jj * head_dim + 2 * d2 + 1] = hi;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int jj = 0; jj < DEC_KT; ++jj) {
            float part = 0.0f;
            #pragma unroll
            for (int i = 0; i < ATTN_MAXD; ++i) {
                if (i < dpt) part += qr[i] * k_sh[jj * head_dim + lane + 32 * i];
            }
            for (int off = 16; off > 0; off >>= 1) {
                part += __shfl_xor_sync(0xffffffff, part, off);
            }
            if (lane == 0) my_s[jj] = part * scale;
        }
        __syncwarp();

        // Serial and ascending over the tile, like every other form here.
        float tmax = neg_inf();
        for (int w = 0; w < n_this; ++w) tmax = fmaxf(tmax, my_s[w]);
        float nm = fmaxf(m, tmax);
        float corr = (m == neg_inf()) ? 0.0f : expf(m - nm);

        float wl = (lane < n_this) ? expf(my_s[lane] - nm) : 0.0f;
        __syncwarp();
        if (lane < DEC_KT) my_s[lane] = wl;
        __syncwarp();

        float lsum = 0.0f;
        for (int w = 0; w < n_this; ++w) lsum += my_s[w];
        l = l * corr + lsum;

        #pragma unroll
        for (int i = 0; i < ATTN_MAXD; ++i) acc[i] *= corr;
        for (int w = 0; w < n_this; ++w) {
            float ww = my_s[w];
            #pragma unroll
            for (int i = 0; i < ATTN_MAXD; ++i) {
                if (i < dpt) acc[i] += ww * v_sh[w * head_dim + lane + 32 * i];
            }
        }
        m = nm;
    }

    float* pa = part_acc + ((long long)split * q_heads + h) * (long long)head_dim;
    #pragma unroll
    for (int i = 0; i < ATTN_MAXD; ++i) {
        if (i < dpt) pa[lane + 32 * i] = acc[i];
    }
    if (lane == 0) {
        part_m[split * q_heads + h] = m;
        part_l[split * q_heads + h] = l;
    }
}

// grid: (q_heads,). block: head_dim threads, one per output dimension.
__global__ void attn_flash_decode_combine(
    const float* __restrict__ part_acc,
    const float* __restrict__ part_m,
    const float* __restrict__ part_l,
    float* __restrict__ out,
    int q_heads,
    int head_dim,
    int n_splits
) {
    int h = blockIdx.x;
    int d = threadIdx.x;

    // Split 0 always holds at least key 0, so this is never -inf and the
    // subtraction below never forms inf - inf.
    float gm = neg_inf();
    for (int s = 0; s < n_splits; ++s) gm = fmaxf(gm, part_m[s * q_heads + h]);

    float num = 0.0f;
    float den = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        float ms = part_m[s * q_heads + h];
        float f = (ms == neg_inf()) ? 0.0f : expf(ms - gm);
        num += f * part_acc[((long long)s * q_heads + h) * (long long)head_dim + d];
        den += f * part_l[s * q_heads + h];
    }
    out[(long long)h * (long long)head_dim + d] = num / den;
}

// ---------------------------------------------------------------------------
// Flash decoding on the tensor cores, with Q split into a fp16 high half and
// a fp16 residual instead of the single fp16 rounding attn_flash_causal_mma
// accepts under MMA_GATE.
//
// grid: (DEC_SPLITS, kv_heads), same as attn_flash_decode_warp/_split. block:
// WPO warps.
//
// ## Why Q needs the split and K/V do not
//
// `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32` takes fp16 operands and
// accumulates fp32, so every operand it touches is rounded once. K and V are
// already binary16 in the cache -- `as_cached` rounds the differential test's
// host reference through the identical format before comparing, so that
// budget is spent by every decode kernel equally and is not new. Q is fp32
// end to end in every other kernel in this file; feeding it to `mma` the way
// attn_flash_causal_mma does for its 16-row query tile is what a prior
// attempt at this kernel did, and it broke the differential gate decode
// holds to -- 1.11e-5 against a 1e-5 tolerance, on the `n_keys = 61` case, see
// docs/BENCHMARKS.md. `Q K^T`'s fp32 dot product is a sum of 256 signed
// terms; halving it to 128 terms, each already close to cancelling before the
// rounding is even applied, is what pushed one element over the line.
//
// The fix costs no extra `mma` calls. `m16n8k8`'s A operand is 16 rows and
// this geometry only has `gqa` (8) real query heads to fill it with, so a
// prior attempt zero-padded the dead second half -- computed, then thrown
// away. Filling that second half with `q - f16(q)`'s own fp16 rounding
// instead reuses exactly those otherwise-wasted multiply-adds: row `g` of the
// A fragment carries `q_hi[g] = f16(q[g])`, row `g + 8` carries
// `q_lo[g] = f16(q[g] - f32(q_hi[g]))`, and `D[g] + D[g+8]` (the two halves of
// the same `mma` call's output, for the same real head `g`) is `Q K^T` at
// `q_hi + q_lo` precision -- `q`'s own value to within `q_lo`'s fp16 rounding
// of a residual already two orders of magnitude smaller than `q` itself,
// roughly `2^-22` relative rather than `2^-11`. `K` is unchanged; the residual
// trick only ever touches the operand that was rounding badly.
//
// ## Shape
//
// One block per (split, KV head), same as attn_flash_decode_warp/_split.
// `WPO` warps split `8 * WPO` keys per trip during `Q K^T` -- one key octet
// per warp, mirroring attn_flash_causal_mma's `MMA_WPH` -- and split the
// output head dimension the same way during `P V`. Unlike that kernel there
// is only one logical head-group per block (the 8 real heads all live in one
// `m16n8k8` M-dimension), so every warp of the block cooperates on the same
// score tile and the barriers below are `__syncthreads()` rather than the
// per-group `bar_group`/`__syncwarp` split prefill needs for its several
// independent head-groups.
//
// `P V` pads the dead second half of *its* A operand with zero rather than
// reusing it for anything -- `P` and `V` are already accepted under
// `MMA_GATE` for the fully-tiled prefill kernel, and the calibration ladder
// in docs/BENCHMARKS.md already established `P V` was never the expensive
// half here. The `q_hi`/`q_lo` trick is `Q K^T`-only because `Q K^T`'s
// rounding was the one that broke the gate.
//
// Templated on `WPO` (`ATTN_DECODE_MMA` below) so the occupancy knob can be
// measured without a second hand-written copy of this kernel: `WPO = 4` was
// the value the original attempt built and measured 1.73x slower than
// attn_flash_decode_warp at a 131,072-key window, diagnosed as one resident
// block per SM against Turing's ~64 KiB shared-memory budget with nothing
// else to hide the K/V staging latency behind.
#define ATTN_DECODE_MMA(NAME, WPO, COMP_P, MAX_DIM, KQ8, VQ8, EXTRA_ARGS, SETUP)                                      \
__global__ void NAME(                                                            \
    const float* __restrict__ q,                                                \
    const unsigned short* __restrict__ k,                                       \
    const unsigned short* __restrict__ v,                                       \
    float* __restrict__ part_acc,                                               \
    float* __restrict__ part_m,                                                 \
    float* __restrict__ part_l,                                                 \
    int q_heads,                                                                \
    int kv_heads,                                                               \
    int head_dim,                                                               \
    const int* __restrict__ key_offset,                                        \
    float scale,                                                               \
    int n_splits EXTRA_ARGS                                                    \
) {                                                                            \
    if((MAX_DIM)==128)head_dim=128;                                              \
    SETUP                                                                      \
    extern __shared__ float smem_f[];                                          \
    unsigned* smem = (unsigned*)smem_f;                                        \
    int hd2 = head_dim >> 1;                                                   \
    int qstride = hd2 + 4;                                                     \
    int vstride = 4 + 8 * (((4 * (WPO) - 4) + 7) / 8);                         \
    int k_words = (8 * (WPO)) * qstride;                                      \
    int v_words = head_dim * vstride;                                         \
    int stage_words = k_words > v_words ? k_words : v_words;                  \
    unsigned* k_sh = smem;                                    /* KT*qstride */ \
    unsigned* v_sh = smem;                                    /* head_dim*vstride */ \
    float* s_sh    = (float*)(smem + stage_words);             /* 8*KT */      \
    float* m_sh    = s_sh + 8 * (8 * (WPO));                   /* 8 */         \
    float* l_sh    = m_sh + 8;                                 /* 8 */         \
    float* corr_sh = l_sh + 8;                                 /* 8 */         \
                                                                                \
    int gqa = q_heads / kv_heads;                                              \
    int tid = threadIdx.x;                                                     \
    int lane = tid & 31;                                                       \
    int warp = tid >> 5;                                                       \
    int nthr = (WPO) * 32;                                                     \
    int g = lane >> 2;                                                         \
    int tg = lane & 3;                                                         \
                                                                                \
    int kvh = blockIdx.y;                                                      \
                                                                                \
    /* This warp's own `Q K^T` fragment, held for the whole key loop -- see */ \
    /* the module comment for what a0/a1 carry. */                             \
    int steps = head_dim >> 3;                                                 \
    unsigned qa0[((MAX_DIM) / 8)], qa1[((MAX_DIM) / 8)];                               \
    {                                                                          \
        const float* qp = (g < gqa)                                           \
            ? q + (long long)(kvh * gqa + g) * (long long)head_dim : 0;        \
        _Pragma("unroll")                                                      \
        for (int s = 0; s < ((MAX_DIM) / 8); ++s) {                                \
            qa0[s] = 0u;                                                       \
            qa1[s] = 0u;                                                       \
            if (s < steps && qp) {                                             \
                int c = 4 * s + tg;                                            \
                float v0 = qp[2 * c];                                          \
                float v1 = qp[2 * c + 1];                                      \
                float hi0 = h2f(f2h(v0));                                      \
                float hi1 = h2f(f2h(v1));                                      \
                qa0[s] = pack_h2(v0, v1);                                      \
                qa1[s] = pack_h2(v0 - hi0, v1 - hi1);                          \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    float o[((MAX_DIM) / (8 * (WPO)))][2];                                             \
                                                                                \
    /* `n_splits` is the LOGICAL split count -- the slice boundaries, the */   \
    /* partial layout, and everything attn_flash_decode_combine merges. */    \
    /* `gridDim.x` is pure scheduling: a block grid-strides over its */       \
    /* splits, so launching fewer blocks than splits changes which SM */      \
    /* computes a slice and nothing else -- the partials are bit-identical */ \
    /* at any block count. What that buys: at 228 registers a block, four */  \
    /* blocks per SM are resident (288 slots on 72 SMs), and the N=3 */       \
    /* serving shape's three concurrent calls at 72 splits x 2 KV heads */    \
    /* are 432 blocks -- 1.5 waves, with the third call queueing ~0.22 ms */  \
    /* behind the first two in the trace. 36 blocks per call is 216 */        \
    /* resident together -- one wave -- at identical numerics, which the */   \
    /* 48-LOGICAL-split attempt could not claim (rejected: 2.278388e-5 */     \
    /* against the 1e-5 deep-window gate, docs/BENCHMARKS.md 2026-08-20). */  \
    long long n_visible = (long long)(*key_offset) + 1;                       \
    long long per = (n_visible + n_splits - 1) / n_splits;                     \
                                                                                \
    int dpw = head_dim / (WPO);            /* output dims this warp owns */    \
    int dbase = warp * dpw;                                                    \
    int ntile = dpw >> 3;                                                      \
                                                                                \
    int kw4 = hd2 >> 2;                                                        \
    int dstripes = head_dim / nthr;                                            \
                                                                                \
    /* The loop body below keeps the original single-split indentation so */  \
    /* the per-tile code is diffable against its pre-loop history. The */     \
    /* leading barrier orders the previous split's partial writes (which */   \
    /* read m_sh/l_sh at tid == 0) before this split re-initializes them. */  \
    for (int split = blockIdx.x; split < n_splits; split += gridDim.x) {       \
    __syncthreads();                                                           \
    if (tid < 8) {                                                             \
        m_sh[tid] = neg_inf();                                                 \
        l_sh[tid] = 0.0f;                                                      \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int t = 0; t < ((MAX_DIM) / (8 * (WPO))); ++t) {                              \
        o[t][0] = 0.0f; o[t][1] = 0.0f;                                       \
    }                                                                          \
    long long begin = (long long)split * per;                                  \
    long long end = begin + per;                                               \
    if (end > n_visible) end = n_visible;                                      \
                                                                                \
    /* Same-tile staging only, not the cross-tile register-held double */      \
    /* buffer attn_flash_causal_mma uses. Measured worse: at WPO=2 the */      \
    /* cross-tile prefetch held `kreg`/`vlo`/`vhi` live across the whole */     \
    /* `Q K^T` + softmax + `P V` body simultaneously with `o`/`qa0`/`qa1`, */   \
    /* forcing 255 registers with 16-56 B of spill; llama.cpp's own Turing */   \
    /* path (`fattn-mma-f16.cuh`, `use_cp_async = false` branch of */          \
    /* `flash_attn_ext_f16_load_tile`) does not hold K/V across a tile */       \
    /* boundary either -- it is a same-tile `memcpy_1<16>` batch straight */    \
    /* to shared, relying on the hardware's own outstanding-request */         \
    /* pipelining rather than a programmer-held prefetch register. This */     \
    /* kernel now does the same: `kreg`/`vlo`/`vhi` live only from this */      \
    /* tile's own load to its own store, scoped to a block so their */         \
    /* lifetime cannot leak into the compute phase below. 0 spill at both */    \
    /* WPO=4 (235 registers) and WPO=2 (252), against 255 with spill before.*/  \
    for (long long j0 = begin; j0 < end; j0 += (8 * (WPO))) {                  \
        __syncthreads();                                                       \
        {                                                                       \
            uint4 kreg[8];                                                      \
            uint2 kc8[8];                                                       \
            unsigned short kd8[8];                                              \
            _Pragma("unroll")                                                   \
            for (int i = 0; i < 8; ++i) {                                       \
                int t = tid + i * nthr;                                         \
                kreg[i] = make_uint4(0u, 0u, 0u, 0u);                           \
                kc8[i] = make_uint2(0u, 0u);                                    \
                kd8[i] = 0;                                                     \
                if (t < (8 * (WPO)) * kw4) {                                    \
                    int r = t / kw4;                                            \
                    long long key = j0 + r;                                     \
                    if (key < end) {                                            \
                        if (KQ8) {                                              \
                            /* q8_0: the uint4's eight dims as eight codes */   \
                            /* and their block scale, converted at the store */ \
                            const unsigned char* kr = (const unsigned char*)k   \
                                + key * kvq8_row_bytes(kv_heads * head_dim);    \
                            int c = t - r * kw4;                                \
                            kc8[i] = ((const uint2*)(kr + kvh * head_dim))[c];  \
                            kd8[i] = ((const unsigned short*)(kr + kv_heads * head_dim)) \
                                [kvh * (head_dim >> 5) + (c >> 2)];             \
                        } else {                                                \
                        const uint4* kp = (const uint4*)(                       \
                            k + (key * (long long)kv_heads + kvh)               \
                                    * (long long)head_dim);                     \
                        kreg[i] = kp[t - r * kw4];                              \
                        }                                                       \
                    }                                                           \
                }                                                               \
            }                                                                   \
            _Pragma("unroll")                                                   \
            for (int i = 0; i < 8; ++i) {                                       \
                int t = tid + i * nthr;                                         \
                if (t < (8 * (WPO)) * kw4) {                                    \
                    int r = t / kw4;                                            \
                    if (KQ8) {                                                  \
                        kvq8_dequant4(kc8[i].x, kd8[i], &kreg[i].x, &kreg[i].y); \
                        kvq8_dequant4(kc8[i].y, kd8[i], &kreg[i].z, &kreg[i].w); \
                    }                                                           \
                    *(uint4*)(k_sh + r * qstride + 4 * (t - r * kw4)) = kreg[i]; \
                }                                                               \
            }                                                                   \
        }                                                                       \
        __syncthreads();                                                       \
                                                                                \
        /* Q K^T. Warp `warp` takes key octet `warp` of the tile; the two */   \
        /* fragment halves are the hi/lo score for the SAME real head `g`, */  \
        /* not two different rows, so they are summed before being stored. */  \
        {                                                                       \
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;                  \
            _Pragma("unroll")                                                  \
            for (int s = 0; s < ((MAX_DIM) / 8); ++s) {                            \
                if (s < steps) {                                               \
                    unsigned b0 = k_sh[(8 * warp + g) * qstride + 4 * s + tg]; \
                    mma_m16n8k8(s0, s1, s2, s3, qa0[s], qa1[s], b0);           \
                }                                                              \
            }                                                                  \
            /* Natural-log scale, not the `exp2f`/ATTN_LOG2E fold prefill's */  \
            /* kernel uses: this kernel's (m, l) partials feed */              \
            /* attn_flash_decode_combine's cross-split merge, which reduces */ \
            /* with plain `expf` to match attn_flash_decode_warp's own */      \
            /* partials -- mixing that with a base-2-scaled m/l here would */  \
            /* cancel invisibly for a single real split (`f = exp(m - gm)` */  \
            /* is always 1 when there is only one) and corrupt the merge */    \
            /* the moment two real splits combine. Caught by */                \
            /* `debug_decode_mma_probe`: exact at n_keys=1, 9.8e-2 wrong at */  \
            /* n_keys=2. */                                                    \
            if (g < gqa) {                                                    \
                s_sh[g * (8 * (WPO)) + 8 * warp + 2 * tg]     = (s0 + s2) * scale; \
                s_sh[g * (8 * (WPO)) + 8 * warp + 2 * tg + 1] = (s1 + s3) * scale; \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
                                                                                \
        /* The online softmax: one lane per key, as many of the 8 real rows */ \
        /* at a time as a warp has lanes to spare. Mirrors */                  \
        /* attn_flash_causal_mma's rat/krow/kcol split exactly, with 8 real */ \
        /* rows split over WPO warps instead of MMA_QT over MMA_WPH. */        \
        {                                                                       \
            const int kt = 8 * (WPO);                                         \
            const int rpw = 8 / (WPO);           /* rows this warp owns */    \
            const int rat = 32 / kt;              /* rows covered at once */  \
            int krow = lane / kt;                                             \
            int kcol = lane % kt;                                             \
            int r0 = rpw * warp;                                               \
            _Pragma("unroll")                                                  \
            for (int rr = 0; rr < rpw / rat; ++rr) {                          \
                int row = r0 + rr * rat + krow;                                \
                if (row < gqa) {                                               \
                    bool live = (j0 + kcol < end);                             \
                    float sv = live ? s_sh[row * kt + kcol] : neg_inf();       \
                    float tmax = sv;                                           \
                    for (int off = kt >> 1; off > 0; off >>= 1) {              \
                        tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffff, tmax, off, kt)); \
                    }                                                          \
                    float m0 = m_sh[row];                                      \
                    float nm = fmaxf(m0, tmax);                                \
                    float corr = (m0 == neg_inf()) ? 0.0f : expf(m0 - nm);     \
                    float e = live ? expf(sv - nm) : 0.0f;                     \
                    float lsum = e;                                            \
                    for (int off = kt >> 1; off > 0; off >>= 1) {              \
                        lsum += __shfl_xor_sync(0xffffffff, lsum, off, kt);    \
                    }                                                          \
                    s_sh[row * kt + kcol] = e;                                 \
                    if (kcol == 0) {                                           \
                        m_sh[row] = nm;                                        \
                        l_sh[row] = l_sh[row] * corr + lsum;                   \
                        corr_sh[row] = corr;                                   \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
                                                                                \
        /* K is dead after Q K^T and softmax. Reuse its shared arena for V */   \
        /* so WPO=2 fits four resident blocks per SM instead of three. */       \
        {                                                                       \
            unsigned short vlo[256 / ((WPO) * 32)][4 * (WPO)];                  \
            unsigned short vhi[256 / ((WPO) * 32)][4 * (WPO)];                  \
            unsigned vq8c[256 / ((WPO) * 32)][4 * (WPO)];                     \
            unsigned vq8d[256 / ((WPO) * 32)][4 * (WPO)];                     \
            _Pragma("unroll")                                                   \
            for (int ds = 0; ds < 256 / ((WPO) * 32); ++ds) {                   \
                int vd_col = ds * nthr + tid;                                   \
                bool active = ds < dstripes;                                    \
                _Pragma("unroll")                                               \
                for (int i = 0; i < 4 * (WPO); ++i) {                           \
                    long long k0 = j0 + 2 * i;                                  \
                    if (VQ8) {                                                  \
                        /* q8_0: each key's code byte and block scale */        \
                        const unsigned char* vb = (const unsigned char*)v;      \
                        long long rb = kvq8_row_bytes(kv_heads * head_dim);     \
                        long long cat = (long long)kvh * head_dim + vd_col;     \
                        long long sat = (long long)kv_heads * head_dim          \
                            + 2LL * (kvh * (head_dim >> 5) + (vd_col >> 5));    \
                        bool lo = active && k0 < end, hi = active && k0 + 1 < end; \
                        /* Packed at the load, two keys to a word, so q8 */    \
                        /* staging holds as many registers as binary16's. */     \
                        unsigned c0 = lo ? vb[k0 * rb + cat] : 0u;              \
                        unsigned c1 = hi ? vb[(k0 + 1) * rb + cat] : 0u;        \
                        unsigned d0 = lo ? *(const unsigned short*)(vb + k0 * rb + sat) : 0u; \
                        unsigned d1 = hi ? *(const unsigned short*)(vb + (k0 + 1) * rb + sat) : 0u; \
                        vq8c[ds][i] = c0 | (c1 << 8);                           \
                        vq8d[ds][i] = d0 | (d1 << 16);                          \
                    } else {                                                    \
                    vlo[ds][i] = (active && k0 < end)                           \
                        ? v[(k0 * (long long)kv_heads + kvh) * (long long)head_dim + vd_col] \
                        : (unsigned short)0;                                    \
                    vhi[ds][i] = (active && k0 + 1 < end)                       \
                        ? v[((k0 + 1) * (long long)kv_heads + kvh) * (long long)head_dim + vd_col] \
                        : (unsigned short)0;                                    \
                    }                                                           \
                }                                                               \
            }                                                                   \
            _Pragma("unroll")                                                   \
            for (int ds = 0; ds < 256 / ((WPO) * 32); ++ds) {                   \
                int vd_col = ds * nthr + tid;                                   \
                if (ds < dstripes) {                                            \
                    _Pragma("unroll")                                           \
                    for (int i = 0; i < 4 * (WPO); ++i) {                       \
                        v_sh[vd_col * vstride + i] = (VQ8)                      \
                            ? kvq8_dequant2(vq8c[ds][i], (unsigned short)vq8d[ds][i], \
                                            (unsigned short)(vq8d[ds][i] >> 16)) \
                            : (unsigned)vlo[ds][i] | ((unsigned)vhi[ds][i] << 16); \
                    }                                                           \
                }                                                               \
            }                                                                   \
        }                                                                       \
        __syncthreads();                                                       \
                                                                                \
        /* P V. The dead second half of this operand is zero. */                \
        {                                                                       \
            float cg = (g < gqa) ? corr_sh[g] : 0.0f;                          \
            _Pragma("unroll")                                                  \
            for (int t = 0; t < ((MAX_DIM) / (8 * (WPO))); ++t) {                      \
                o[t][0] *= cg; o[t][1] *= cg;                                  \
            }                                                                  \
            for (int oc = 0; oc < (8 * (WPO)) / 8; ++oc) {                     \
                unsigned a0 = (g < gqa) ? pack_h2(                             \
                    s_sh[g * (8 * (WPO)) + 8 * oc + 2 * tg],                   \
                    s_sh[g * (8 * (WPO)) + 8 * oc + 2 * tg + 1]) : 0u;         \
                unsigned a1 = 0u;                                              \
                if(COMP_P && g<gqa){float p0=s_sh[g*(8*(WPO))+8*oc+2*tg],p1=s_sh[g*(8*(WPO))+8*oc+2*tg+1];a1=pack_h2(p0-h2f((unsigned short)a0),p1-h2f((unsigned short)(a0>>16)));} \
                _Pragma("unroll")                                              \
                for (int t = 0; t < ((MAX_DIM) / (8 * (WPO))); ++t) {                  \
                    if (t < ntile) {                                           \
                        unsigned b0 =                                          \
                            v_sh[(dbase + 8 * t + g) * vstride + 4 * oc + tg]; \
                        float dead0 = 0.0f, dead1 = 0.0f;                       \
                        mma_m16n8k8(o[t][0], o[t][1], dead0, dead1, a0, a1, b0); \
                        if(COMP_P){o[t][0]+=dead0;o[t][1]+=dead1;}              \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
                                                                                \
    /* Write the raw (unnormalized) partial: attn_flash_decode_combine */      \
    /* merges these across DEC_SPLITS blocks with the same log-sum-exp */      \
    /* identity attn_flash_decode_warp's partials already use, unmodified. */  \
    _Pragma("unroll")                                                          \
    for (int t = 0; t < ((MAX_DIM) / (8 * (WPO))); ++t) {                              \
        if (t < ntile && g < gqa) {                                            \
            int d = dbase + 8 * t + 2 * tg;                                    \
            int h = kvh * gqa + g;                                             \
            float* pa =                                                       \
                part_acc + ((long long)split * q_heads + h) * (long long)head_dim; \
            pa[d]     = o[t][0];                                               \
            pa[d + 1] = o[t][1];                                               \
        }                                                                      \
    }                                                                          \
    if (tid == 0) {                                                            \
        _Pragma("unroll")                                                      \
        for (int gg = 0; gg < 8; ++gg) {                                       \
            if (gg < gqa) {                                                    \
                int h = kvh * gqa + gg;                                        \
                part_m[split * q_heads + h] = m_sh[gg];                        \
                part_l[split * q_heads + h] = l_sh[gg];                        \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    }  /* split grid-stride loop */                                            \
}


ATTN_DECODE_MMA(attn_flash_decode_mma_wpo4, 4, false, 256, 0, 0, , )
ATTN_DECODE_MMA(attn_flash_decode_mma_wpo2, 2, false, 256, 0, 0, , )
ATTN_DECODE_MMA(attn_flash_decode_mma_k2_wpo2, 2, true, 256, 0, 0, , )
ATTN_DECODE_MMA(attn_flash_decode_mma_k2_wpo4, 4, true, 256, 0, 0, , )
ATTN_DECODE_MMA(attn_flash_decode_mma_k2_128_wpo2, 2, true, 128, KV_KQ8, KV_VQ8, , )
ATTN_DECODE_MMA(attn_flash_decode_mma_k2_128_wpo4, 4, true, 128, KV_KQ8, KV_VQ8, , )

__global__ void attn_flash_causal_t1(
    const float* __restrict__ q,
    const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v,
    float* __restrict__ out,
    int q_heads,
    int kv_heads,
    int head_dim,
    const int* __restrict__ key_offset,
    float scale,
    int n_query
) {
    (void)n_query;
    extern __shared__ float smem[];
    int n_warps = blockDim.x >> 5;
    int tile = n_warps * ATTN_KT;              // keys per barrier pair
    float* q_sh     = smem;                    // head_dim floats
    float* score_sh = smem + head_dim;         // tile floats
    float* w_sh     = score_sh + tile;         // tile floats

    long long qi = blockIdx.x;
    int h = blockIdx.y;
    int kvh = h / (q_heads / kv_heads);        // GQA: never assume 1:1
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    long long qbase = (qi * (long long)q_heads + h) * (long long)head_dim;
    q_sh[tid] = q[qbase + tid];
    __syncthreads();

    // The causal bound, written once. Query row qi sits at absolute position
    // key_offset + qi and sees keys [0, key_offset + qi] inclusive.
    long long n_visible = (long long)(*key_offset) + qi + 1;

    float m = neg_inf();
    float l = 0.0f;
    float acc = 0.0f;

    for (long long j0 = 0; j0 < n_visible; j0 += tile) {
        // ATTN_KT keys per warp per trip, not one.
        //
        // The barrier pair below is per *trip*, and one key per warp made it
        // one barrier pair per eight keys: a 512-token prefill row crossed 128
        // of them. The scores of several keys are independent, so a warp can
        // compute ATTN_KT of them back to back and the block can rescale its
        // running softmax once for all `tile` of them.
        //
        // `key = j0 + warp * ATTN_KT + r` stored at `score_sh[warp * ATTN_KT
        // + r]` keeps slot `w` holding key `j0 + w`, which is what lets the
        // value accumulation below stay a single ascending sweep.
        #pragma unroll
        for (int r = 0; r < ATTN_KT; ++r) {
            long long key = j0 + (long long)warp * ATTN_KT + r;
            float partial = 0.0f;
            if (key < n_visible) {
                const unsigned short* krow =
                    k + (key * (long long)kv_heads + kvh) * (long long)head_dim;
                // Lane l takes dimensions l, l+32, l+64, ...: consecutive lanes
                // read consecutive floats, so every load is a full 128 B
                // transaction, and q_sh[d] with d = lane + 32*i hits a distinct
                // bank per lane.
                for (int d = lane; d < head_dim; d += 32) partial += q_sh[d] * h2f(krow[d]);
            }
            for (int off = 16; off > 0; off >>= 1) {
                partial += __shfl_xor_sync(0xffffffff, partial, off);
            }
            if (lane == 0) score_sh[warp * ATTN_KT + r] = partial * scale;
        }
        __syncthreads();

        long long remaining = n_visible - j0;
        int n_this = (int)(remaining < (long long)tile ? remaining : (long long)tile);

        float tile_max = neg_inf();
        for (int w = 0; w < n_this; ++w) tile_max = fmaxf(tile_max, score_sh[w]);
        float new_m = fmaxf(m, tile_max);
        // Matches the reference's guard exactly: on the first tile there is
        // no accumulator to rescale and expf(-inf - -inf) would be NaN.
        float corr = (m == neg_inf()) ? 0.0f : expf(m - new_m);

        // w_sh and score_sh are disjoint, so this write races nothing above.
        if (tid < n_this) w_sh[tid] = expf(score_sh[tid] - new_m);
        __syncthreads();

        float lsum = 0.0f;
        for (int w = 0; w < n_this; ++w) lsum += w_sh[w];
        l = l * corr + lsum;

        float a = acc * corr;
        for (int w = 0; w < n_this; ++w) {
            a += w_sh[w]
                * h2f(v[((j0 + w) * (long long)kv_heads + kvh) * (long long)head_dim + tid]);
        }
        acc = a;
        m = new_m;

        // Before the next iteration overwrites score_sh and w_sh.
        __syncthreads();
    }

    out[qbase + tid] = acc / l;
}


// Append this batch's roped keys and raw values to the cache, at the absolute
// position the sequence has reached.
//
// This was two `cuMemcpyDtoDAsync` calls into `cache.k.slice_mut(at..)` and
// `cache.v.slice_mut(at..)`, which is the same traffic and one fewer launch.
// It is a kernel now for one reason: `at` was computed on the host, so the
// destination was a **host-chosen address**. A CUDA graph records addresses,
// so a captured decode step would write position `n` forever. Reading the
// position from device memory is what makes the step replayable, and it is
// `AGENTS.md` rule 5 besides.
//
// Both halves in one launch: they are the same shape and the same stride, and
// the copies are independent, so splitting them would only cost a launch.
//
// grid: (ceil(2 * span / ATTN_APPEND_THREADS),). block: ATTN_APPEND_THREADS.
// out[i] = position[0] + i, for i < n.
//
// A caller that wants to run one query row at a time needs each row's own
// absolute position as a *device* pointer, because that is the only form the
// attention kernels take one in (see the note on `key_offset` above: a host
// position cannot be baked into a captured launch). One thread, `n` of them,
// off the base the caller already holds.
__global__ void attn_row_positions(
    const int* __restrict__ position,
    int* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = position[0] + i;
}

__global__ void attn_kv_append(
    const float* __restrict__ key,
    const float* __restrict__ value,
    unsigned short* __restrict__ k_cache,
    unsigned short* __restrict__ v_cache,
    const int* __restrict__ position,
    int span,
    int row
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * span) return;
    // `span` is `n_tokens * row`; the cache is indexed by *position*, so the
    // destination base is one row per position and not one span.
    long long at = (long long)(*position) * (long long)row;
    if (i < span) {
        k_cache[at + i] = f2h(key[i]);
    } else {
        int j = i - span;
        v_cache[at + j] = f2h(value[j]);
    }
}

// attn_kv_append with either half in q8_0 (`kq8` / `vq8` nonzero). One thread
// per element as above, so with `row` a multiple of 32 and 256-thread blocks
// each warp holds exactly one 32-element block of one half of one position:
// `span` and `2 * span` are multiples of 32, so the early return, the K/V
// choice and the format choice are all warp-uniform and kvq8_quantize sees
// all 32 lanes. A binary16 half is written exactly as attn_kv_append does.
__global__ void attn_kv_append_kv8(
    const float* __restrict__ key,
    const float* __restrict__ value,
    unsigned short* __restrict__ k_cache,
    unsigned short* __restrict__ v_cache,
    const int* __restrict__ position,
    int span,
    int row,
    int kq8,
    int vq8
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * span) return;
    bool is_k = i < span;
    int j = is_k ? i : i - span;
    float x = is_k ? key[j] : value[j];
    unsigned short* cache = is_k ? k_cache : v_cache;
    long long pos = (long long)(*position) + j / row;
    int e = j % row;
    if (!(is_k ? kq8 : vq8)) {
        cache[pos * (long long)row + e] = f2h(x);
        return;
    }
    unsigned short d;
    unsigned code = kvq8_quantize(x, &d);
    unsigned char* dst = (unsigned char*)cache + pos * kvq8_row_bytes(row);
    dst[e] = (unsigned char)code;
    if ((e & 31) == 0) *(unsigned short*)(dst + row + 2 * (e >> 5)) = d;
}

// The decode step's per-sequence rotary and cache append, batched over the
// launch: one grid advances every sequence's query and key through
// `attn_rope_partial_neox`'s arithmetic and appends the roped key and the
// value to that sequence's own cache, each sequence reading its own
// position through a scalar pointer slot -- the idiom `conv1d_step_batch`
// and the GDN step kernels use, so graph capture sees stable pointers.
//
// It replaces three launches per sequence per layer (rope q, rope k,
// append) that were issued on a side stream each. At decode width every
// one of them was under six microseconds, i.e. mostly its own floor.
//
// Bit-identical to the three kernels it replaces, by construction: the
// rotary body is `attn_rope_partial_neox`'s with `t = 0` (the position is
// `(double)*pos + (double)0`, which is exact), the tail is a copy, and the
// cache write is `f2h` of the same float the separate append read back
// from `k_out`. `attention_differential` gates it against the separate
// launches.
//
// grid: (q_heads + kv_heads, n_seq). block: head_dim. Head blocks below
// `q_heads` rope the query; the rest rope one key head and append it with
// its value. Thread `d < rope_dim / 2` owns the pair `(d, d + half)`;
// threads at and past `rope_dim` copy the tail.
#define ATTN_STEP_SLOTS 8

__device__ __forceinline__ unsigned long long attn_slot(
    int z,
    unsigned long long s0, unsigned long long s1,
    unsigned long long s2, unsigned long long s3,
    unsigned long long s4, unsigned long long s5,
    unsigned long long s6, unsigned long long s7
) {
    switch (z) {
        case 0: return s0; case 1: return s1; case 2: return s2; case 3: return s3;
        case 4: return s4; case 5: return s5; case 6: return s6; default: return s7;
    }
}

__global__ void attn_decode_rope_append_batch(
    const float* __restrict__ q_in,
    float* __restrict__ q_out,
    const float* __restrict__ k_in,
    float* __restrict__ k_out,
    const float* __restrict__ v_in,
    unsigned long long rp0, unsigned long long rp1, unsigned long long rp2, unsigned long long rp3,
    unsigned long long rp4, unsigned long long rp5, unsigned long long rp6, unsigned long long rp7,
    unsigned long long ap0, unsigned long long ap1, unsigned long long ap2, unsigned long long ap3,
    unsigned long long ap4, unsigned long long ap5, unsigned long long ap6, unsigned long long ap7,
    unsigned long long kc0, unsigned long long kc1, unsigned long long kc2, unsigned long long kc3,
    unsigned long long kc4, unsigned long long kc5, unsigned long long kc6, unsigned long long kc7,
    unsigned long long vc0, unsigned long long vc1, unsigned long long vc2, unsigned long long vc3,
    unsigned long long vc4, unsigned long long vc5, unsigned long long vc6, unsigned long long vc7,
    int q_heads,
    int kv_heads,
    int head_dim,
    int rope_dim,
    float theta_base
) {
    int seq = blockIdx.y;
    int h = blockIdx.x;
    int d = threadIdx.x;
    const int* rope_pos = (const int*)attn_slot(seq, rp0, rp1, rp2, rp3, rp4, rp5, rp6, rp7);

    bool is_q = h < q_heads;
    const float* in = is_q ? q_in : k_in;
    float* out = is_q ? q_out : k_out;
    int heads = is_q ? q_heads : kv_heads;
    int hh = is_q ? h : h - q_heads;
    long long base = ((long long)seq * heads + hh) * (long long)head_dim;

    // The key head's cache row for this sequence's position.
    unsigned short* k_cache = 0;
    unsigned short* v_cache = 0;
    long long at = 0;
    if (!is_q) {
        const int* pos = (const int*)attn_slot(seq, ap0, ap1, ap2, ap3, ap4, ap5, ap6, ap7);
        k_cache = (unsigned short*)attn_slot(seq, kc0, kc1, kc2, kc3, kc4, kc5, kc6, kc7);
        v_cache = (unsigned short*)attn_slot(seq, vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7);
        at = (long long)(*pos) * (long long)kv_heads * (long long)head_dim
           + (long long)hh * (long long)head_dim;
    }

    int half = rope_dim >> 1;
    if (d >= rope_dim) {
        float x = in[base + d];
        out[base + d] = x;
        if (!is_q) {
            k_cache[at + d] = f2h(x);
            v_cache[at + d] = f2h(v_in[base + d]);
        }
        return;
    }
    if (d >= half) return;

    double freq = pow((double)theta_base, -2.0 * (double)d / (double)rope_dim);
    double pos_d = (double)(*rope_pos) + (double)0;
    double angle = pos_d * freq;
    float sin_a = (float)sin(angle);
    float cos_a = (float)cos(angle);

    float x0 = in[base + d];
    float x1 = in[base + d + half];
    float y0 = x0 * cos_a - x1 * sin_a;
    float y1 = x0 * sin_a + x1 * cos_a;
    out[base + d]        = y0;
    out[base + d + half] = y1;
    if (!is_q) {
        k_cache[at + d]        = f2h(y0);
        k_cache[at + d + half] = f2h(y1);
        v_cache[at + d]        = f2h(v_in[base + d]);
        v_cache[at + d + half] = f2h(v_in[base + d + half]);
    }
}

// Same split arithmetic, with independent sequence pointer slots.
// Pointer slots match the scalar batch API; each z-plane owns one cache.
// MAX_DIM bounds register arrays. The 128 variants also fix address arithmetic;
// Rust dispatch checks head_dim before selecting either specialized entry.
#define K2_MMA_SLOTS \
    , unsigned long long p0, unsigned long long p1, unsigned long long p2, unsigned long long p3, unsigned long long p4, unsigned long long p5, unsigned long long p6, unsigned long long p7 \
    , unsigned long long k0, unsigned long long k1, unsigned long long k2, unsigned long long k3, unsigned long long k4, unsigned long long k5, unsigned long long k6, unsigned long long k7 \
    , unsigned long long v0, unsigned long long v1, unsigned long long v2, unsigned long long v3, unsigned long long v4, unsigned long long v5, unsigned long long v6, unsigned long long v7
#define K2_MMA_SETUP \
    int seq=blockIdx.z; \
    q+=(long long)seq*q_heads*head_dim; \
    part_acc+=(long long)seq*n_splits*q_heads*head_dim; \
    part_m+=(long long)seq*n_splits*q_heads;part_l+=(long long)seq*n_splits*q_heads; \
    key_offset=(const int*)attn_slot(seq,p0,p1,p2,p3,p4,p5,p6,p7); \
    k=(const unsigned short*)attn_slot(seq,k0,k1,k2,k3,k4,k5,k6,k7); \
    v=(const unsigned short*)attn_slot(seq,v0,v1,v2,v3,v4,v5,v6,v7);
ATTN_DECODE_MMA(attn_flash_decode_mma_k2_batch_wpo2, 2, true, 128, KV_KQ8, KV_VQ8, K2_MMA_SLOTS, K2_MMA_SETUP)
ATTN_DECODE_MMA(attn_flash_decode_mma_k2_batch_wpo4, 4, true, 128, KV_KQ8, KV_VQ8, K2_MMA_SLOTS, K2_MMA_SETUP)
#undef K2_MMA_SLOTS
#undef K2_MMA_SETUP
} // extern C
// Keep the scalar reduction tree while bounding K2's four heads and four
// dimensions per lane. Other geometries retain the generic register bounds.
template<int HD,int GQA>
__device__ __forceinline__ void attn_k2_warp_body(const float* __restrict__ q,const unsigned short* __restrict__ k,const unsigned short* __restrict__ v,float* __restrict__ part_acc,float* __restrict__ part_m,float* __restrict__ part_l,int q_heads,int kv_heads,int head_dim,const int* key_offset,float scale,int n_splits) {
    if(HD)head_dim=HD;
    constexpr int MAXD=HD ? HD/32 : ATTN_MAXD, MAXG=GQA ? GQA : DEC_MAXG;
    int gqa = GQA ? GQA : q_heads / kv_heads;
    int dpl = head_dim >> 5;          // dimensions this lane owns
    int wpl = dpl >> 1;               // packed 32-bit words behind them
    int lane = threadIdx.x;
    int split = blockIdx.x;
    int kvh = blockIdx.y;

    long long n_visible = (long long)(*key_offset) + 1;
    long long per = (n_visible + n_splits - 1) / n_splits;
    long long begin = (long long)split * per;
    long long end = begin + per;
    if (end > n_visible) end = n_visible;

    float qr[MAXG][MAXD];
    float acc[MAXG][MAXD];
    float m[MAXG], l[MAXG];
    #pragma unroll
    for (int hh = 0; hh < MAXG; ++hh) {
        m[hh] = neg_inf();
        l[hh] = 0.0f;
        #pragma unroll
        for (int i = 0; i < MAXD; ++i) { qr[hh][i] = 0.0f; acc[hh][i] = 0.0f; }
        if (hh < gqa) {
            const float* qp =
                q + (long long)(kvh * gqa + hh) * (long long)head_dim;
            #pragma unroll
            for (int i = 0; i < MAXD; ++i) {
                if (i < dpl) qr[hh][i] = qp[dpl * lane + i];
            }
        }
    }

    // Keys are taken `DEC_KB` at a time with every load issued before any of
    // the arithmetic. The loop bound is a runtime value, so ptxas cannot
    // unroll it and the next key's loads would otherwise wait on this key's
    // registers -- one key in flight per warp, which is the same
    // memory-parallelism bound that held the prefill staging to a third of
    // peak. A batch of `DEC_KB` puts `8 * DEC_KB` requests in flight instead.
    for (long long j0 = begin; j0 < end; j0 += DEC_KB) {
        unsigned kw[DEC_KB][MAXD / 2], vw[DEC_KB][MAXD / 2];
        #pragma unroll
        for (int jj = 0; jj < DEC_KB; ++jj) {
            long long key = j0 + jj;
            bool live = key < end;
            long long at = (key * (long long)kv_heads + kvh) * (long long)head_dim;
            const unsigned* kp = (const unsigned*)(k + at);
            const unsigned* vp = (const unsigned*)(v + at);
            // A lane owns `wpl` consecutive words, so `kp[wpl * lane + p]` is a
            // stride-`wpl` access across the warp -- not the single `uint4` the
            // partition above describes. At head_dim 256 that is a 16-byte
            // stride: each of the four loads touches all sixteen 32-byte
            // sectors of the 512-byte key and takes four bytes from each, and
            // the four loads then request the same sixteen sectors again. Four
            // times the sector traffic for the same bytes, which caps the
            // kernel near a quarter of peak -- 27.5% was measured.
            //
            // ptxas cannot rescue it: `wpl` comes from the runtime `head_dim`
            // argument, so neither `wpl == 4` nor 16-byte alignment is provable
            // at compile time, and the per-element `p < wpl` predicate blocks
            // vectorisation on its own. `cuobjdump -sass` confirmed the kernel
            // emitted no `LDG.E.128` at all while `attn_flash_causal_mma` in
            // this same file emits two, so the compiler vectorises here when
            // the pattern permits and this pattern did not permit it.
            //
            // Naming the width restores it. `at` is a multiple of `head_dim`,
            // so at head_dim 256 the address is 512-byte aligned and a `uint4`
            // load is legal; lane `l` takes bytes `[16l, 16l+16)`, which is the
            // same eight binary16 dimensions `[8l, 8l+8)` it read before as
            // four strided words. Identical data, one instruction, and the warp
            // now covers 512 contiguous bytes with every sector fully consumed.
#if KV_KQ8 || KV_VQ8
            // A q8_0 half at head_dim 128: lane `l` owns dimensions
            // [4l, 4l + 4), one word of codes and block l / 8's scale, held
            // in the two slots its binary16 words would use and converted
            // with the arithmetic below.
            if (HD == 128) {
                long long rb = kvq8_row_bytes(kv_heads * 128);
                long long cat = (long long)kvh * 128 + 4 * lane;
                long long sat = (long long)kv_heads * 128 + 2 * (kvh * 4 + (lane >> 3));
                const unsigned char* kr = (const unsigned char*)k + key * rb;
                const unsigned char* vr = (const unsigned char*)v + key * rb;
                if (KV_KQ8) {
                    kw[jj][0] = live ? *(const unsigned*)(kr + cat) : 0u;
                    kw[jj][1] = live ? (unsigned)*(const unsigned short*)(kr + sat) : 0u;
                } else {
                    kw[jj][0] = live ? kp[2 * lane] : 0u;
                    kw[jj][1] = live ? kp[2 * lane + 1] : 0u;
                }
                if (KV_VQ8) {
                    vw[jj][0] = live ? *(const unsigned*)(vr + cat) : 0u;
                    vw[jj][1] = live ? (unsigned)*(const unsigned short*)(vr + sat) : 0u;
                } else {
                    vw[jj][0] = live ? vp[2 * lane] : 0u;
                    vw[jj][1] = live ? vp[2 * lane + 1] : 0u;
                }
            } else
#endif
            if (wpl == 4) {
                const uint4* kp4 = (const uint4*)kp;
                const uint4* vp4 = (const uint4*)vp;
                uint4 kq = make_uint4(0u, 0u, 0u, 0u);
                uint4 vq = make_uint4(0u, 0u, 0u, 0u);
                if (live) { kq = kp4[lane]; vq = vp4[lane]; }
                kw[jj][0] = kq.x; kw[jj][1] = kq.y;
                kw[jj][2] = kq.z; kw[jj][3] = kq.w;
                vw[jj][0] = vq.x; vw[jj][1] = vq.y;
                vw[jj][2] = vq.z; vw[jj][3] = vq.w;
            } else {
                #pragma unroll
                for (int p = 0; p < MAXD / 2; ++p) {
                    kw[jj][p] = (live && p < wpl) ? kp[wpl * lane + p] : 0u;
                    vw[jj][p] = (live && p < wpl) ? vp[wpl * lane + p] : 0u;
                }
            }
        }
        #pragma unroll
        for (int jj = 0; jj < DEC_KB; ++jj) {
        if (j0 + jj >= end) break;
        float kk[MAXD], vv[MAXD];
#if KV_KQ8 || KV_VQ8
        if (HD == 128) {
            unsigned kx[2] = {kw[jj][0], kw[jj][1]}, vx[2] = {vw[jj][0], vw[jj][1]};
            if (KV_KQ8) kvq8_dequant4(kw[jj][0], (unsigned short)kw[jj][1], &kx[0], &kx[1]);
            if (KV_VQ8) kvq8_dequant4(vw[jj][0], (unsigned short)vw[jj][1], &vx[0], &vx[1]);
            h2f2(kx[0], &kk[0], &kk[1]);
            h2f2(kx[1], &kk[2], &kk[3]);
            h2f2(vx[0], &vv[0], &vv[1]);
            h2f2(vx[1], &vv[2], &vv[3]);
        } else
#endif
        {
        #pragma unroll
        for (int p = 0; p < MAXD / 2; ++p) {
            if (p < wpl) {
                h2f2(kw[jj][p], &kk[2 * p], &kk[2 * p + 1]);
                h2f2(vw[jj][p], &vv[2 * p], &vv[2 * p + 1]);
            }
        }
        }
        #pragma unroll
        for (int hh = 0; hh < MAXG; ++hh) {
            if (hh < gqa) {
                float part = 0.0f;
                #pragma unroll
                for (int i = 0; i < MAXD; ++i) {
                    if (i < dpl) part += qr[hh][i] * kk[i];
                }
                for (int off = 16; off > 0; off >>= 1) {
                    part += __shfl_xor_sync(0xffffffff, part, off);
                }
                // Identical in every lane from here, so the branch below is
                // warp-uniform and the scalars need no broadcast.
                float s = part * scale;
                float nm = fmaxf(m[hh], s);
                if (nm != m[hh]) {
                    float corr = (m[hh] == neg_inf()) ? 0.0f : expf(m[hh] - nm);
                    l[hh] *= corr;
                    #pragma unroll
                    for (int i = 0; i < MAXD; ++i) acc[hh][i] *= corr;
                    m[hh] = nm;
                }
                float e = expf(s - nm);
                l[hh] += e;
                #pragma unroll
                for (int i = 0; i < MAXD; ++i) {
                    if (i < dpl) acc[hh][i] += e * vv[i];
                }
            }
        }
        }
    }

    #pragma unroll
    for (int hh = 0; hh < MAXG; ++hh) {
        if (hh < gqa) {
            int h = kvh * gqa + hh;
            float* pa =
                part_acc + ((long long)split * q_heads + h) * (long long)head_dim;
            #pragma unroll
            for (int i = 0; i < MAXD; ++i) {
                if (i < dpl) pa[dpl * lane + i] = acc[hh][i];
            }
            if (lane == 0) {
                part_m[split * q_heads + h] = m[hh];
                part_l[split * q_heads + h] = l[hh];
            }
        }
    }
}
extern "C" {
__global__ void attn_flash_decode_warp_k2_batch(
    const float* __restrict__ q,
    float* __restrict__ part_acc,
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    int q_heads,
    int kv_heads,
    int head_dim,
    float scale,
    int n_splits,
    unsigned long long p0, unsigned long long p1, unsigned long long p2, unsigned long long p3, unsigned long long p4, unsigned long long p5, unsigned long long p6, unsigned long long p7,
    unsigned long long k0, unsigned long long k1, unsigned long long k2, unsigned long long k3, unsigned long long k4, unsigned long long k5, unsigned long long k6, unsigned long long k7,
    unsigned long long v0, unsigned long long v1, unsigned long long v2, unsigned long long v3, unsigned long long v4, unsigned long long v5, unsigned long long v6, unsigned long long v7
) {
    int seq=blockIdx.z;
    q+=(long long)seq*q_heads*head_dim;
    part_acc+=(long long)seq*n_splits*q_heads*head_dim;
    part_m+=(long long)seq*n_splits*q_heads;part_l+=(long long)seq*n_splits*q_heads;
    const int* key_offset=(const int*)attn_slot(seq,p0,p1,p2,p3,p4,p5,p6,p7);
    const unsigned short* k=(const unsigned short*)attn_slot(seq,k0,k1,k2,k3,k4,k5,k6,k7);
    const unsigned short* v=(const unsigned short*)attn_slot(seq,v0,v1,v2,v3,v4,v5,v6,v7);
    attn_k2_warp_body<0,0>(q,k,v,part_acc,part_m,part_l,q_heads,kv_heads,head_dim,key_offset,scale,n_splits);
}
__global__ void attn_flash_decode_warp_k2_128_batch(
    const float* __restrict__ q,
    float* __restrict__ part_acc,
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    int q_heads,
    int kv_heads,
    int head_dim,
    float scale,
    int n_splits,
    unsigned long long p0, unsigned long long p1, unsigned long long p2, unsigned long long p3, unsigned long long p4, unsigned long long p5, unsigned long long p6, unsigned long long p7,
    unsigned long long k0, unsigned long long k1, unsigned long long k2, unsigned long long k3, unsigned long long k4, unsigned long long k5, unsigned long long k6, unsigned long long k7,
    unsigned long long v0, unsigned long long v1, unsigned long long v2, unsigned long long v3, unsigned long long v4, unsigned long long v5, unsigned long long v6, unsigned long long v7
) {
    int seq=blockIdx.z;
    q+=(long long)seq*q_heads*head_dim;
    part_acc+=(long long)seq*n_splits*q_heads*head_dim;
    part_m+=(long long)seq*n_splits*q_heads;part_l+=(long long)seq*n_splits*q_heads;
    const int* key_offset=(const int*)attn_slot(seq,p0,p1,p2,p3,p4,p5,p6,p7);
    const unsigned short* k=(const unsigned short*)attn_slot(seq,k0,k1,k2,k3,k4,k5,k6,k7);
    const unsigned short* v=(const unsigned short*)attn_slot(seq,v0,v1,v2,v3,v4,v5,v6,v7);
    attn_k2_warp_body<128,4>(q,k,v,part_acc,part_m,part_l,q_heads,kv_heads,head_dim,key_offset,scale,n_splits);
}
__global__ void attn_flash_decode_warp_k2_128(const float* __restrict__ q,const unsigned short* __restrict__ k,const unsigned short* __restrict__ v,float* __restrict__ part_acc,float* __restrict__ part_m,float* __restrict__ part_l,int q_heads,int kv_heads,int head_dim,const int* key_offset,float scale,int n_splits){attn_k2_warp_body<128,4>(q,k,v,part_acc,part_m,part_l,q_heads,kv_heads,head_dim,key_offset,scale,n_splits);}
__global__ void attn_flash_decode_combine_k2_batch(
    const float* __restrict__ part_acc,
    const float* __restrict__ part_m,
    const float* __restrict__ part_l,
    float* __restrict__ out,
    int q_heads,
    int head_dim,
    int n_splits
) {
    int seq=blockIdx.y;
    part_acc+=(long long)seq*n_splits*q_heads*head_dim;
    part_m+=(long long)seq*n_splits*q_heads;part_l+=(long long)seq*n_splits*q_heads;
    out+=(long long)seq*q_heads*head_dim;
    int h = blockIdx.x;
    int d = threadIdx.x;

    // Split 0 always holds at least key 0, so this is never -inf and the
    // subtraction below never forms inf - inf.
    float gm = neg_inf();
    for (int s = 0; s < n_splits; ++s) gm = fmaxf(gm, part_m[s * q_heads + h]);

    float num = 0.0f;
    float den = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        float ms = part_m[s * q_heads + h];
        float f = (ms == neg_inf()) ? 0.0f : expf(ms - gm);
        num += f * part_acc[((long long)s * q_heads + h) * (long long)head_dim + d];
        den += f * part_l[s * q_heads + h];
    }
    out[(long long)h * (long long)head_dim + d] = num / den;
}

}
"#;

/// Threads per block for [`AttentionKernels::append_kv`].
const APPEND_THREADS: u32 = 256;

/// Sequences one `attn_decode_rope_append_batch` launch carries: the
/// kernel's pointer-slot count (`ATTN_STEP_SLOTS`).
pub const STEP_SLOTS: usize = 8;

/// Element format of one half of the KV cache, as the kernels address it.
///
/// `Q8_0` is the row layout `ATTENTION_SRC` documents beside
/// `kvq8_row_bytes`: a position's `n` codes, then its `n / 32` binary16
/// scales. This crate does not see `llmcuda_model::KvCacheType`; the engine
/// maps one onto the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvFormat {
    #[default]
    F16,
    Q8_0,
}

impl KvFormat {
    /// `u16` words one position's row of `elements` values occupies.
    /// `elements` must be a multiple of 32 for `Q8_0`.
    pub const fn row_words(self, elements: usize) -> usize {
        match self {
            Self::F16 => elements,
            Self::Q8_0 => (elements + elements / 16) / 2,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Q8_0 => "q8_0",
        }
    }
}

/// The K and V halves' formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KvFormats {
    pub k: KvFormat,
    pub v: KvFormat,
}

impl KvFormats {
    pub const F16: Self = Self {
        k: KvFormat::F16,
        v: KvFormat::F16,
    };

    pub const fn is_quantized(self) -> bool {
        !matches!(self.k, KvFormat::F16) || !matches!(self.v, KvFormat::F16)
    }
}

/// Something went wrong compiling or launching an attention kernel.
#[derive(Debug)]
pub enum AttentionError {
    /// NVRTC rejected the source, or the module failed to load.
    Compile(String),
    /// The driver failed.
    Driver(DriverError),
    /// A head dimension the kernel cannot service.
    ///
    /// One thread per head dimension, reduced with warp shuffles, so it must
    /// be a whole number of warps and fit in one block.
    UnsupportedHeadDim { head_dim: usize },
    /// Query heads are not a whole multiple of KV heads.
    UnevenGqaGrouping { q_heads: usize, kv_heads: usize },
    /// `rope_dim` is odd or wider than `head_dim`.
    UnsupportedRopeDim { rope_dim: usize, head_dim: usize },
    /// A buffer is not the size the declared geometry implies.
    ///
    /// Rejected rather than clamped: a short K buffer would be read past its
    /// end for every query deep enough to reach the missing rows, and the
    /// result would still look like attention.
    ///
    /// For `key` and `value`, `expected` is a lower bound rather than an exact
    /// size — see [`AttentionKernels::forward`] on why a cache is allowed to be
    /// longer than its filled window.
    BufferShape {
        what: &'static str,
        expected: usize,
        actual: usize,
    },
    /// The query rows would run past the end of the key window.
    ///
    /// Query row `i` reads keys `[0, key_offset + i]`, so `key_offset +
    /// n_query` must not exceed `n_keys`.
    QueryPastKeys {
        key_offset: usize,
        n_query: usize,
        n_keys: usize,
    },
    /// A KV cache format this kernel set or path cannot read or write.
    UnsupportedKvFormat { kv: KvFormats, what: &'static str },
}

impl std::fmt::Display for AttentionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Compile(m) => write!(f, "kernel compilation failed: {m}"),
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::UnsupportedHeadDim { head_dim } => write!(
                f,
                "head_dim {head_dim} must be a positive multiple of 32 and at most 1024",
            ),
            Self::UnevenGqaGrouping { q_heads, kv_heads } => write!(
                f,
                "{q_heads} query heads do not divide evenly among {kv_heads} kv heads",
            ),
            Self::UnsupportedRopeDim { rope_dim, head_dim } => write!(
                f,
                "rope_dim {rope_dim} must be even and at most head_dim {head_dim}",
            ),
            Self::BufferShape {
                what,
                expected,
                actual,
            } => write!(
                f,
                "{what} holds {actual} floats, but this geometry needs {expected}",
            ),
            Self::QueryPastKeys {
                key_offset,
                n_query,
                n_keys,
            } => write!(
                f,
                "query rows at offset {key_offset}..{} run past the {n_keys}-key window",
                key_offset + n_query,
            ),
            Self::UnsupportedKvFormat { kv, what } => {
                write!(f, "KV cache K={} V={}: {what}", kv.k.name(), kv.v.name(),)
            }
        }
    }
}

impl std::error::Error for AttentionError {}

impl From<DriverError> for AttentionError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

/// Per-slice partials for the flash-decoding path.
///
/// Preallocated because [`AGENTS.md` rule 6] forbids allocating mid-forward,
/// and shared by every attention layer because they run in sequence on one
/// stream and nothing crosses a layer boundary. Roughly 1 MiB at this model's
/// geometry, which is why it is not worth a shape-dependent lifetime.
pub struct AttnDecodeScratch {
    /// `[DECODE_SPLITS][q_heads][head_dim]`, each slice's unnormalized sum.
    acc: CudaSlice<f32>,
    /// `[DECODE_SPLITS][q_heads]`, each slice's running maximum.
    m: CudaSlice<f32>,
    /// `[DECODE_SPLITS][q_heads]`, each slice's softmax normalizer.
    l: CudaSlice<f32>,
}

impl AttnDecodeScratch {
    /// Allocate for one head geometry.
    pub fn new(
        stream: &Arc<CudaStream>,
        q_heads: usize,
        head_dim: usize,
    ) -> Result<Self, AttentionError> {
        Ok(Self {
            acc: stream.alloc_zeros::<f32>(DECODE_SPLITS * q_heads * head_dim)?,
            m: stream.alloc_zeros::<f32>(DECODE_SPLITS * q_heads)?,
            l: stream.alloc_zeros::<f32>(DECODE_SPLITS * q_heads)?,
        })
    }

    /// Allocate independent partials for a fixed decode batch at startup.
    pub fn new_batch(
        stream: &Arc<CudaStream>,
        q_heads: usize,
        head_dim: usize,
        sequences: usize,
    ) -> Result<Self, AttentionError> {
        Ok(Self {
            acc: stream.alloc_zeros::<f32>(sequences * DECODE_SPLITS * q_heads * head_dim)?,
            m: stream.alloc_zeros::<f32>(sequences * DECODE_SPLITS * q_heads)?,
            l: stream.alloc_zeros::<f32>(sequences * DECODE_SPLITS * q_heads)?,
        })
    }

    /// Bytes held, for the VRAM accounting the engine prints.
    pub fn bytes(&self) -> usize {
        (self.acc.len() + self.m.len() + self.l.len()) * size_of::<f32>()
    }
}

/// Compiled Gated Attention kernels for one head geometry.
pub struct AttentionKernels {
    split: CudaFunction,
    rope: CudaFunction,
    rope_imrope: CudaFunction,
    flash: CudaFunction,
    flash_gqa: CudaFunction,
    /// `attn_row_positions`, for the row-at-a-time decode path.
    row_positions: CudaFunction,
    flash_mma: CudaFunction,
    flash_mma_k2: CudaFunction,
    flash_mma_k2s: CudaFunction,
    decode_split: CudaFunction,
    decode_warp: CudaFunction,
    decode_warp_k2_batch: CudaFunction,
    decode_warp_k2_128: CudaFunction,
    decode_warp_k2_128_batch: CudaFunction,
    decode_combine_k2_batch: CudaFunction,
    decode_mma_wpo4: CudaFunction,
    decode_mma_wpo2: CudaFunction,
    decode_mma_k2_batch_wpo2: CudaFunction,
    decode_mma_k2_batch_wpo4: CudaFunction,
    k2_mma_decode_splits: usize,
    decode_mma_k2_128_wpo2: CudaFunction,
    decode_mma_k2_128_wpo4: CudaFunction,
    decode_mma_k2_wpo2: CudaFunction,
    decode_mma_k2_wpo4: CudaFunction,
    decode_combine: CudaFunction,
    flash_t1: CudaFunction,
    append: CudaFunction,
    /// `attn_kv_append_kv8`, the append when either half is q8_0.
    append_kv8: CudaFunction,
    /// `attn_decode_rope_append_batch`: rope q, rope k and the cache append
    /// for every sequence of a decode step under one grid.
    rope_append_batch: CudaFunction,
    /// The cache formats this kernel set was compiled to read and write.
    kv: KvFormats,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    /// Which decode path `decode()` takes: `0` for `Auto` (the default --
    /// depth-aware dispatch against `DECODE_MMA_DEPTH_THRESHOLD`, see
    /// [`Self::active_decode_mma`]), `1` to force `attn_flash_decode_warp`'s
    /// per-key online softmax at every depth, or `2`/`4` to force the
    /// tensor-core kernel at that occupancy width at every depth. For the
    /// forced cases the discriminant doubling as the width is what lets
    /// [`Self::active_decode_mma`] use it directly with no enum to convert;
    /// `Auto` and `ForceWarp` fall out as the two discriminants (`0`, `1`)
    /// no real occupancy width can be. See [`Self::disable_decode_mma`] and
    /// [`Self::set_decode_mma_wpo`].
    ///
    /// An atomic, not a plain field: `GatedAttentionBlock` holds its
    /// `AttentionKernels` behind an `Arc` shared across every layer of the
    /// same geometry, so toggling this lever cannot go through `&mut self`
    /// -- and unlike a `Cell`, this keeps `AttentionKernelSet` `Sync`, which
    /// `Arc<AttentionKernelSet>` already promises callers across threads.
    decode_mma: std::sync::atomic::AtomicU8,
    k2_single_tile: std::sync::atomic::AtomicBool,
    /// Runtime override for the tensor-core decode's launch block count,
    /// `0` meaning "one block per split". Same atomic-behind-`Arc` shape as
    /// `decode_mma`, and set the same way: by the engine right before a
    /// graph capture, so each captured width bakes its own scheduling. The
    /// partials are bit-identical at any value (the kernel grid-strides its
    /// logical splits), so this is a pure wave-shape knob — see
    /// [`MMA_DECODE_SPLITS`] for the arithmetic and docs/BENCHMARKS.md
    /// 2026-08-20 for the measurement (+0.4--1.6% at 32K N=3 with 36; -13%
    /// at N=1, which is why it must be per width and not global).
    decode_mma_blocks: std::sync::atomic::AtomicUsize,
    /// Logical key slices per tensor-core decode call — the slice
    /// boundaries and partial layout the combine merges. Defaults to
    /// [`MMA_DECODE_SPLITS`]; `LLMCUDA_DEC_MMA_SPLITS` overrides it at
    /// construction (numerics-affecting: 48 was measured and rejected at
    /// the deep-window gate). Capped at [`DECODE_SPLITS`], which is what
    /// the partial buffers are sized for.
    mma_decode_splits: usize,
    /// Blocks the tensor-core decode launches over those splits. The kernel
    /// grid-strides `split = blockIdx.x .. n_splits`, so this is pure
    /// scheduling and the partials are **bit-identical at any value**;
    /// fewer blocks than splits exists for the N=3 wave shape (three
    /// concurrent 144-block calls are 1.5 waves of the 288 resident-block
    /// budget; 36 blocks per call is exactly one wave).
    /// `LLMCUDA_DEC_MMA_BLOCKS` overrides at construction; defaults to the
    /// logical split count, i.e. one split per block.
    mma_decode_blocks: usize,
}

impl AttentionKernels {
    /// Compile for a specific head geometry.
    ///
    /// Fixed at construction rather than per launch, matching `GdnKernels`:
    /// the geometry is a property of the model, and validating it once leaves
    /// the launch path with only per-call shapes to reject.
    pub fn new(
        ctx: &Arc<CudaContext>,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, AttentionError> {
        Self::with_kv_formats(ctx, q_heads, kv_heads, head_dim, KvFormats::F16)
    }

    /// [`Self::new`] for a cache whose halves may be q8_0.
    ///
    /// Binary16 on both halves compiles exactly [`ATTENTION_SRC`], so it
    /// shares the PTX cache entry with every other kernel set. A q8_0 half
    /// prepends `KV_KQ8`/`KV_VQ8`, which only K2's compensated kernels read:
    /// `attn_flash_causal_mma_k2s`, the head_dim-128 decode tensor-core and
    /// warp kernels and their batched forms. Every other path either keeps
    /// reading binary16 or is refused at launch, so q8_0 needs that geometry:
    /// head_dim 128 and four query heads per KV head.
    pub fn with_kv_formats(
        ctx: &Arc<CudaContext>,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        kv: KvFormats,
    ) -> Result<Self, AttentionError> {
        if head_dim == 0 || !head_dim.is_multiple_of(32) || head_dim > 32 * ATTN_MAXD {
            return Err(AttentionError::UnsupportedHeadDim { head_dim });
        }
        if kv_heads == 0 || !q_heads.is_multiple_of(kv_heads) {
            return Err(AttentionError::UnevenGqaGrouping { q_heads, kv_heads });
        }
        if kv.is_quantized() && (head_dim != 128 || q_heads != 4 * kv_heads) {
            return Err(AttentionError::UnsupportedKvFormat {
                kv,
                what: "a q8_0 cache needs head_dim 128 and four query heads per KV head",
            });
        }
        let source = if kv.is_quantized() {
            std::borrow::Cow::Owned(format!(
                "#define KV_KQ8 {}\n#define KV_VQ8 {}\n{ATTENTION_SRC}",
                u8::from(kv.k == KvFormat::Q8_0),
                u8::from(kv.v == KvFormat::Q8_0),
            ))
        } else {
            std::borrow::Cow::Borrowed(ATTENTION_SRC)
        };
        let ptx = compile(&source, "attention").map_err(AttentionError::Compile)?;
        let module = ctx.load_module(ptx)?;
        #[cfg(feature = "rust-kernels")]
        let migrated = ctx.load_module(cudarc::nvrtc::Ptx::from_src(include_str!(
            "rust/tensor_add.ptx"
        )))?;
        #[cfg(not(feature = "rust-kernels"))]
        let migrated = &module;
        Ok(Self {
            split: module.load_function("attn_split_query_gate")?,
            rope: migrated.load_function("attn_rope_partial_neox")?,
            rope_imrope: module.load_function("attn_rope_partial_imrope")?,
            flash: module.load_function("attn_flash_causal")?,
            flash_gqa: module.load_function("attn_flash_causal_gqa")?,
            row_positions: module.load_function("attn_row_positions")?,
            flash_mma: {
                let f = module.load_function("attn_flash_causal_mma")?;
                // Ask for the >48 KiB carveout before the first launch. Turing
                // caps a block at 64 KiB and refuses the launch outright
                // without this, so it is done here rather than hopefully.
                f.set_attribute(
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    MMA_SHARED_CEILING as i32,
                )?;
                f
            },
            decode_mma_k2_batch_wpo2: module
                .load_function("attn_flash_decode_mma_k2_batch_wpo2")?,
            decode_mma_k2_batch_wpo4: module
                .load_function("attn_flash_decode_mma_k2_batch_wpo4")?,
            k2_mma_decode_splits: std::env::var("LLMCUDA_DEC_MMA_SPLITS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&s| (1..=DECODE_SPLITS).contains(&s))
                .unwrap_or(32),
            decode_mma_k2_128_wpo2: module.load_function("attn_flash_decode_mma_k2_128_wpo2")?,
            decode_mma_k2_128_wpo4: module.load_function("attn_flash_decode_mma_k2_128_wpo4")?,
            decode_mma_k2_wpo2: module.load_function("attn_flash_decode_mma_k2_wpo2")?,
            decode_mma_k2_wpo4: module.load_function("attn_flash_decode_mma_k2_wpo4")?,
            flash_mma_k2: module.load_function("attn_flash_causal_mma_k2")?,
            flash_mma_k2s: module.load_function("attn_flash_causal_mma_k2s")?,
            decode_split: module.load_function("attn_flash_decode_split")?,
            decode_warp: module.load_function("attn_flash_decode_warp")?,
            decode_warp_k2_128: module.load_function("attn_flash_decode_warp_k2_128")?,
            decode_warp_k2_128_batch: module
                .load_function("attn_flash_decode_warp_k2_128_batch")?,
            decode_warp_k2_batch: module.load_function("attn_flash_decode_warp_k2_batch")?,
            decode_combine_k2_batch: module.load_function("attn_flash_decode_combine_k2_batch")?,
            decode_mma_wpo4: {
                let f = module.load_function("attn_flash_decode_mma_wpo4")?;
                f.set_attribute(
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    MMA_SHARED_CEILING as i32,
                )?;
                f
            },
            decode_mma_wpo2: {
                let f = module.load_function("attn_flash_decode_mma_wpo2")?;
                // WPO=2 fits under the default 48 KiB carveout, but opting in
                // anyway costs nothing and is what buys the third resident
                // block per SM the occupancy experiment is measuring --
                // without it the driver would cap the block scheduler at the
                // default ceiling's math, not the opted-in one.
                f.set_attribute(
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    MMA_SHARED_CEILING as i32,
                )?;
                f
            },
            decode_combine: module.load_function("attn_flash_decode_combine")?,
            flash_t1: module.load_function("attn_flash_causal_t1")?,
            append: module.load_function("attn_kv_append")?,
            append_kv8: module.load_function("attn_kv_append_kv8")?,
            rope_append_batch: module.load_function("attn_decode_rope_append_batch")?,
            kv,
            q_heads,
            kv_heads,
            head_dim,
            decode_mma: std::sync::atomic::AtomicU8::new(0),
            k2_single_tile: std::sync::atomic::AtomicBool::new(false),
            decode_mma_blocks: std::sync::atomic::AtomicUsize::new(0),
            mma_decode_splits: std::env::var("LLMCUDA_DEC_MMA_SPLITS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&s| (1..=DECODE_SPLITS).contains(&s))
                .unwrap_or(MMA_DECODE_SPLITS),
            mma_decode_blocks: std::env::var("LLMCUDA_DEC_MMA_BLOCKS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&b| b >= 1)
                .unwrap_or(0),
        })
    }

    /// Dynamic shared memory `attn_flash_decode_mma_wpo{4,2}` requests: the
    /// aliased K/V tile at `8 * wpo` keys, plus the 8-real-row score, maximum,
    /// normalizer and correction scratch. Q K^T has stopped reading K before
    /// P V stages V into the same arena.
    const fn dmma_shared_bytes(head_dim: usize, wpo: usize) -> usize {
        let qstride = head_dim / 2 + 4;
        let vstride = 4 + 8 * ((4 * wpo + 3) / 8);
        let kt = 8 * wpo;
        let k_words = kt * qstride;
        let v_words = head_dim * vstride;
        let words = if k_words > v_words { k_words } else { v_words };
        let floats = 8 * kt + 24; // s_sh + m_sh/l_sh/corr_sh, 8 rows each
        (words + floats) * size_of::<u32>()
    }

    /// Whether the tensor-core decode kernel can service this geometry at the
    /// given occupancy width.
    ///
    /// The design packs exactly 8 real query heads into one `m16n8k8` M
    /// dimension's first half (the second half carries their fp16 residual,
    /// not a ninth through sixteenth head), so `gqa_ratio()` above 8 has
    /// nowhere to go. `head_dim` must divide into whole key-octets per warp
    /// (`8 * wpo`) for `P V`'s output-dimension split and into whole `nthr`-
    /// wide stripes for `V`'s staging loop -- both hold for this model's 256
    /// at wpo 2 and 4, and are checked rather than assumed for any other.
    fn decode_mma_is_available(&self, wpo: usize) -> bool {
        self.gqa_ratio() >= 1
            && self.gqa_ratio() <= 8
            && self.head_dim.is_multiple_of(8 * wpo)
            && self.head_dim.is_multiple_of(32 * wpo)
            && Self::dmma_shared_bytes(self.head_dim, wpo) <= MMA_SHARED_CEILING
    }

    /// Run K2 prefill attention on the original one-tile kernel, which
    /// stages eight keys per barrier pair. The staged kernel reproduces it bit
    /// for bit; this is the oracle and A/B lever for that claim.
    pub fn use_k2_single_tile_prefill(&self, on: bool) {
        self.k2_single_tile
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// Force decode off the tensor-core kernel and onto
    /// `attn_flash_decode_warp`'s per-key online softmax at every depth,
    /// overriding the default `Auto` dispatch.
    ///
    /// `&self`, not `&mut self`: see the atomic note on the field. Meant for
    /// a benchmark process to pick one configuration and measure it, not to
    /// flip back and forth mid-pass.
    pub fn disable_decode_mma(&self) {
        self.decode_mma
            .store(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Force the tensor-core decode kernel on at a fixed occupancy width at
    /// every depth, overriding the default `Auto` dispatch. `wpo` must be 2
    /// or 4; anything else is a no-op. Exists for `bench_attention`'s
    /// `LLMCUDA_DECODE_MMA_WPO` A/B lever.
    pub fn set_decode_mma_wpo(&self, wpo: usize) {
        if wpo == 2 || wpo == 4 {
            self.decode_mma
                .store(wpo as u8, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Set how many blocks the tensor-core decode launches over its logical
    /// splits — `0` restores one block per split. Pure scheduling: the
    /// kernel grid-strides its splits and the partials are bit-identical at
    /// any value, so this only reshapes the wave the N=3 serving shape's
    /// three concurrent calls form. The engine calls it right before each
    /// graph capture so single-stream and batched captures bake different
    /// block counts; the `LLMCUDA_DEC_MMA_BLOCKS` setup switch, when
    /// present, still wins for whole-process A/B runs.
    pub fn set_decode_mma_blocks(&self, blocks: usize) {
        self.decode_mma_blocks
            .store(blocks, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether `Self::decode` will take the tensor-core kernel this call,
    /// and if so at which occupancy width.
    ///
    /// `depth` is the number of keys the call will actually read (the
    /// caller's `pos_offset + n_query`, host-side and known before launch --
    /// not `max_keys`, which is only the cache's allocated capacity). It
    /// drives the `Auto` lever's choice against
    /// `DECODE_MMA_DEPTH_THRESHOLD`; a forced lever ignores it.
    ///
    /// Exposed so a differential test can pick the tolerance the arithmetic
    /// actually warrants without hardcoding the dispatch rule. Unlike
    /// [`Self::uses_tensor_cores`] (prefill's `n_query >= MMA_QUERY_TILE`
    /// gate) this kernel's `Q K^T` is not a single fp16 rounding of `Q` --
    /// the `q_hi`/`q_lo` split is designed to land back inside the same
    /// tight tolerance the scalar decode kernels hold to, not `MMA_GATE` --
    /// so this returns `None` rather than a wider gate when it is in use;
    /// `attention_differential.rs` gates it identically to
    /// `attn_flash_decode_warp` either way.
    fn active_decode_mma(&self, depth: usize) -> Option<usize> {
        let lever = self.decode_mma.load(std::sync::atomic::Ordering::Relaxed) as usize;
        let wpo = match lever {
            0 if depth >= DECODE_MMA_DEPTH_THRESHOLD => 2,
            0 | 1 => return None,
            forced => forced,
        };
        if self.decode_mma_is_available(wpo) {
            Some(wpo)
        } else {
            None
        }
    }

    /// Query heads sharing each KV head.
    pub fn gqa_ratio(&self) -> usize {
        self.q_heads / self.kv_heads
    }

    /// Keys scored per tile: one per warp, so `head_dim / 32`.
    ///
    /// Exposed so a test can pick a sequence length that is deliberately not a
    /// multiple of it and exercise the ragged final tile.
    pub fn keys_per_tile(&self) -> usize {
        (self.head_dim / 32) * KEYS_PER_WARP
    }

    /// Dynamic shared memory one block requests: the score and weight tiles,
    /// which are now `QUERY_TILE` rows deep, plus the three per-row softmax
    /// scalars. The query tile itself is in registers. See the module docs.
    pub fn shared_bytes(&self) -> usize {
        (2 * QUERY_TILE * self.keys_per_tile() + 3 * QUERY_TILE) * size_of::<f32>()
    }

    /// Dynamic shared memory the GQA-shared kernel requests: the staged key
    /// and value tiles, plus one weight tile per warp. Unlike the kernel above
    /// it stages K and V rather than only the softmax scratch, which is the
    /// whole point — see that kernel's comment.
    pub fn shared_bytes_gqa(&self) -> usize {
        (2 * GQA_KEY_TILE * self.head_dim + self.gqa_ratio() * GQA_QUERY_TILE * GQA_KEY_TILE)
            * size_of::<f32>()
    }

    /// Dynamic shared memory the flash-decoding split pass requests: the staged
    /// key and value tiles plus one score tile per warp.
    pub fn shared_bytes_decode(&self) -> usize {
        (2 * DECODE_KEY_TILE * self.head_dim + self.gqa_ratio() * DECODE_KEY_TILE)
            * size_of::<f32>()
    }

    /// Dynamic shared memory the tensor-core kernel requests.
    ///
    /// The staged Q, K and V tiles in fp16, plus the score tile and the three
    /// per-row softmax scalars in fp32. Q and K carry four words of padding per
    /// row and V four per dimension; see the kernel comment for why those exact
    /// strides are what make every fragment read conflict-free.
    pub fn shared_bytes_mma(&self) -> usize {
        mma_shared_bytes(self.head_dim)
    }

    /// Whether the tensor-core kernel can service this geometry.
    ///
    /// The warps of a head group split the output head dimension, so it must
    /// divide into whole `m16n8k8` n-tiles of 8 per warp, and that share must
    /// fit [`MMA_MAX_TILES`]. The head group must exist — the heads a block
    /// serves have to share a KV head — and the staged tiles must fit the
    /// carveout the function opts in to.
    fn mma_is_available(&self) -> bool {
        let per_warp = 8 * MMA_WARPS_PER_HEAD;
        self.head_dim.is_multiple_of(per_warp)
            && self.head_dim / per_warp <= MMA_MAX_TILES
            && self.gqa_ratio().is_multiple_of(MMA_HEADS_PER_BLOCK)
            && self.q_heads.is_multiple_of(MMA_HEADS_PER_BLOCK)
            && self.shared_bytes_mma() <= MMA_SHARED_CEILING
    }

    /// Whether [`Self::forward`] will take the tensor-core path at this query
    /// count.
    ///
    /// Exposed so a differential test can pick the tolerance the arithmetic
    /// actually warrants — the tensor-core path rounds its operands to fp16 —
    /// instead of hardcoding the dispatch rule and silently drifting from it.
    pub fn uses_tensor_cores(&self, n_query: usize) -> bool {
        n_query >= MMA_QUERY_TILE && self.mma_is_available()
    }

    /// Query rows one block of [`Self::forward`] covers at this query count.
    ///
    /// Together with [`Self::blocks_per_launch`] this pins down the launch's
    /// traffic exactly: the grid is `ceil(n_query / this)` tiles by
    /// `blocks / tiles` heads, and a tile at query offset `o` streams
    /// `key_offset + o + rows` keys rather than the whole window. Exposed for
    /// `bench_attention`, which would otherwise have to approximate the causal
    /// bound and would overstate the shallow rows by a factor of two.
    pub fn query_tile(&self, n_query: usize) -> usize {
        // Decode and the one-row kernel both carry a single query row, by
        // different routes: the first splits the key axis instead of the query
        // axis, the second has fewer rows than a tile to begin with.
        if n_query < GQA_QUERY_TILE {
            1
        } else if self.uses_tensor_cores(n_query) {
            MMA_QUERY_TILE
        } else if self.gqa_shared_is_available() {
            GQA_QUERY_TILE
        } else {
            QUERY_TILE
        }
    }

    /// Whether [`Self::forward`] partitions the key axis across blocks rather
    /// than replicating it.
    ///
    /// The prefill kernels give every block the whole visible window, so their
    /// traffic is `blocks * window`. Flash decoding does the opposite: the
    /// blocks split the key range between them and each reads its slice once,
    /// so the traffic is `kv_heads * window` however many blocks there are.
    /// Exposed because a benchmark that applies the prefill model to the decode
    /// path reports a bandwidth many times the card's, which is how this was
    /// noticed.
    pub fn splits_the_key_axis(&self, n_query: usize) -> bool {
        n_query == 1 && self.decode_split_is_available()
    }

    /// Blocks [`Self::forward`] would launch at this query count.
    ///
    /// Each block streams the whole visible key window for itself, so this is
    /// the multiplier on the launch's DRAM traffic over the one pass per KV
    /// head the problem actually needs. Exposed so `bench_attention` can report
    /// that redundancy from the dispatch rule rather than from a copy of it
    /// that would quietly go stale the next time the block shape moves.
    pub fn blocks_per_launch(&self, n_query: usize) -> usize {
        if n_query == 1 && self.decode_split_is_available() {
            DECODE_SPLITS * self.kv_heads
        } else if n_query < GQA_QUERY_TILE {
            n_query * self.q_heads
        } else if self.uses_tensor_cores(n_query) {
            n_query.div_ceil(MMA_QUERY_TILE) * (self.q_heads / MMA_HEADS_PER_BLOCK)
        } else if self.gqa_shared_is_available() {
            n_query.div_ceil(GQA_QUERY_TILE) * self.kv_heads
        } else {
            n_query.div_ceil(QUERY_TILE) * self.q_heads
        }
    }

    /// Whether flash decoding can service this geometry.
    ///
    /// Same three conditions as the GQA-shared kernel — it is the same block
    /// shape with one query row instead of four.
    /// Whether the warp-per-split decode kernel can service this geometry.
    ///
    /// A lane holds `head_dim / 32` dimensions and they must be a whole number
    /// of packed pairs, and one warp carries every query head of its KV head,
    /// which bounds the register arrays at `DEC_MAXG`.
    fn decode_warp_is_available(&self) -> bool {
        self.gqa_ratio() >= 2
            && self.gqa_ratio() <= DECODE_MAX_GQA
            && self.head_dim.is_multiple_of(64)
            && self.head_dim / 32 <= ATTN_MAXD
    }

    pub fn decode_split_is_available(&self) -> bool {
        self.gqa_ratio() >= 2
            && self.gqa_ratio() * 32 <= 1024
            && self.shared_bytes_decode() <= 48 * 1024
    }

    /// Whether the GQA-shared kernel can service this geometry at all.
    ///
    /// Three things have to hold, and none of them do for every model: there
    /// must be sharing to exploit, one warp per query head has to fit in a
    /// block, and the staged tiles have to fit in the 48 KiB a block gets
    /// without opting in. When any fails the launch falls back to
    /// `attn_flash_causal`, which needs none of them.
    fn gqa_shared_is_available(&self) -> bool {
        self.gqa_ratio() >= 2
            && self.gqa_ratio() * 32 <= 1024
            && self.shared_bytes_gqa() <= 48 * 1024
    }

    /// Dynamic shared memory the one-row kernel requests: `q_sh` plus the two
    /// tile-wide scratch arrays. It stages the query rather than holding it in
    /// registers, so its budget is the pre-tiling one and not a `QUERY_TILE`
    /// of 1 in the formula above.
    fn shared_bytes_t1(&self) -> usize {
        (self.head_dim + 2 * self.keys_per_tile()) * size_of::<f32>()
    }

    /// The `1/sqrt(head_dim)` score scale.
    pub fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    fn expect_len(
        what: &'static str,
        buf_len: usize,
        expected: usize,
    ) -> Result<(), AttentionError> {
        if buf_len != expected {
            return Err(AttentionError::BufferShape {
                what,
                expected,
                actual: buf_len,
            });
        }
        Ok(())
    }

    /// Like [`Self::expect_len`], for the buffers a cache is allowed to
    /// over-allocate. Too small is still fatal — that is the read-past-the-end
    /// case — but too large is the ordinary steady state of a KV cache.
    fn expect_at_least(
        what: &'static str,
        buf_len: usize,
        needed: usize,
    ) -> Result<(), AttentionError> {
        if buf_len < needed {
            return Err(AttentionError::BufferShape {
                what,
                expected: needed,
                actual: buf_len,
            });
        }
        Ok(())
    }

    /// Deinterleave the packed `attn_q` projection output into a query tensor
    /// and an output-gate tensor.
    ///
    /// `packed` is `[n_tokens][q_heads * 2 * head_dim]` in the layout
    /// `[q_h0, gate_h0, q_h1, gate_h1, ...]`; `q` and `gate` are each
    /// `[n_tokens][q_heads][head_dim]`. See the module docs for why this is a
    /// kernel and not a slice.
    pub fn split_query_and_gate(
        &self,
        stream: &Arc<CudaStream>,
        packed: &CudaSlice<f32>,
        q: &mut CudaSlice<f32>,
        gate: &mut CudaSlice<f32>,
        n_tokens: usize,
    ) -> Result<(), AttentionError> {
        let per_head = self.q_heads * self.head_dim;
        Self::expect_len("packed query/gate", packed.len(), n_tokens * per_head * 2)?;
        Self::expect_len("query", q.len(), n_tokens * per_head)?;
        Self::expect_len("gate", gate.len(), n_tokens * per_head)?;

        let cfg = LaunchConfig {
            grid_dim: (n_tokens as u32, self.q_heads as u32, 1),
            block_dim: (self.head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let q_heads = self.q_heads as i32;
        let head_dim = self.head_dim as i32;
        let mut builder = stream.launch_builder(&self.split);
        builder
            .arg(packed)
            .arg(q)
            .arg(gate)
            .arg(&q_heads)
            .arg(&head_dim);
        // SAFETY: the grid is (n_tokens, q_heads) with one thread per head
        // dimension, and the three buffer lengths were just checked against
        // exactly the indices `2*h` and `2*h+1` reach.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// Apply partial rotary embedding to `[n_tokens][n_heads][head_dim]`.
    ///
    /// Token `i` is rotated by position `positions[0] + i`. `n_heads` is
    /// `q_heads` for the query stream and `kv_heads` for the key stream — RoPE
    /// is applied before the GQA broadcast, so the two differ.
    ///
    /// Dimensions `[rope_dim, head_dim)` are copied through unmodified.
    ///
    /// **`positions` is a one-element device scalar, not a host number.** The
    /// position is the only thing that changes between two decode steps, so
    /// keeping it on the device is what lets a whole step be captured once as
    /// a CUDA graph and replayed — a host argument would be baked into the
    /// recorded launch. It is also `AGENTS.md` rule 5: nothing on the forward
    /// path is sized or indexed by a host-side value.
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n_tokens: usize,
        n_heads: usize,
        rope_dim: usize,
        positions: &CudaSlice<i32>,
        theta_base: f32,
    ) -> Result<(), AttentionError> {
        Self::expect_len("rope position", positions.len(), 1)?;
        if !rope_dim.is_multiple_of(2) || rope_dim > self.head_dim {
            return Err(AttentionError::UnsupportedRopeDim {
                rope_dim,
                head_dim: self.head_dim,
            });
        }
        let n = n_tokens * n_heads * self.head_dim;
        Self::expect_len("rope input", input.len(), n)?;
        Self::expect_len("rope output", out.len(), n)?;

        let cfg = LaunchConfig {
            grid_dim: ((n_tokens as u32).div_ceil(ROPE_TOKENS), n_heads as u32, 1),
            block_dim: (self.head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_heads_i = n_heads as i32;
        let head_dim = self.head_dim as i32;
        let rope_dim_i = rope_dim as i32;
        let n_tokens_i = n_tokens as i32;
        let mut builder = stream.launch_builder(&self.rope);
        builder
            .arg(input)
            .arg(out)
            .arg(&n_heads_i)
            .arg(&head_dim)
            .arg(&rope_dim_i)
            .arg(positions)
            .arg(&theta_base)
            .arg(&n_tokens_i);
        // SAFETY: the grid is (ceil(n_tokens / ROPE_TOKENS), n_heads) with one
        // thread per head dimension, and `n_tokens` is passed so the band
        // stops at the last real token; both buffers were checked to hold
        // exactly that many floats, and every thread touches only its own
        // head's slice.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// Interleaved multimodal rotary over `[n_tokens][n_heads][head_dim]` —
    /// the three-component sibling of [`Self::rope`] for prefill chunks
    /// overlapping an image span.
    ///
    /// `mrope_positions` holds `(t, h, w)` i32 triples, one per token, in
    /// token order. Pair `d` reads component `d % 3` bounded by
    /// `3 * sections[1]` / `3 * sections[2]` (`h_bound` / `w_bound`), with
    /// the frequency on the global pair index — `ggml_mrope_cache_init`'s
    /// `GGML_ROPE_TYPE_IMROPE` branch; reference
    /// `llmcuda_kernels::mrope::apply_imrope`. With `t == h == w` per token
    /// this is bit-identical to [`Self::rope`], asserted by differential
    /// test. Dimensions `[rope_dim, head_dim)` are copied through.
    ///
    /// Only ever launched on uncaptured prefill passes; decode keeps the
    /// scalar kernel with a delta-adjusted base.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_imrope(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n_tokens: usize,
        n_heads: usize,
        rope_dim: usize,
        mrope_positions: &CudaSlice<i32>,
        sections: [u32; 3],
        theta_base: f32,
    ) -> Result<(), AttentionError> {
        if mrope_positions.len() < 3 * n_tokens {
            return Err(AttentionError::BufferShape {
                what: "imrope positions",
                expected: 3 * n_tokens,
                actual: mrope_positions.len(),
            });
        }
        if !rope_dim.is_multiple_of(2) || rope_dim > self.head_dim {
            return Err(AttentionError::UnsupportedRopeDim {
                rope_dim,
                head_dim: self.head_dim,
            });
        }
        let n = n_tokens * n_heads * self.head_dim;
        Self::expect_len("imrope input", input.len(), n)?;
        Self::expect_len("imrope output", out.len(), n)?;

        let cfg = LaunchConfig {
            grid_dim: (n_tokens as u32, n_heads as u32, 1),
            block_dim: (self.head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_heads_i = n_heads as i32;
        let head_dim = self.head_dim as i32;
        let rope_dim_i = rope_dim as i32;
        let h_bound = (3 * sections[1]) as i32;
        let w_bound = (3 * sections[2]) as i32;
        let n_tokens_i = n_tokens as i32;
        let mut builder = stream.launch_builder(&self.rope_imrope);
        builder
            .arg(input)
            .arg(out)
            .arg(&n_heads_i)
            .arg(&head_dim)
            .arg(&rope_dim_i)
            .arg(mrope_positions)
            .arg(&h_bound)
            .arg(&w_bound)
            .arg(&theta_base)
            .arg(&n_tokens_i);
        // SAFETY: grid is (n_tokens, n_heads) with one thread per head
        // dimension; both buffers hold exactly n floats and positions holds
        // at least 3*n_tokens ints (checked above); each thread touches only
        // its own head's slice.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// Causal grouped-query attention over a key window.
    ///
    /// - `q`, `out`: `[n_query][q_heads][head_dim]`
    /// - `k`, `v`: `[n_keys][kv_heads][head_dim]`
    ///
    /// Query row `i` sits at absolute position `positions[0] + i` and attends
    /// to keys `[0, positions[0] + i]`. A position of 0 with `n_query` equal
    /// to the filled window is a full prefill; `n_query == 1` at the window's
    /// last position is a decode step against a cached window.
    ///
    /// **`positions` is a one-element device scalar** — see [`Self::rope`] for
    /// why. The consequence here is that the "query rows run past the key
    /// window" check cannot live in this function any more: the position is
    /// not a number this side of the launch. `max_keys` is the caller's
    /// promise about the cache's *capacity*, which is checked against the
    /// buffers, and the caller owns the promise that the position stays inside
    /// it. In this engine that is `GatedAttentionBlock::forward`, which holds
    /// both the host position and `KvCache::max_seq` and returns
    /// [`AttentionError::QueryPastKeys`] itself.
    ///
    /// `k` and `v` may be **longer** than the filled window. A KV cache is
    /// allocated once for the longest sequence the worker admits and then
    /// filled a token at a time. The kernel indexes keys by absolute position
    /// and reads nothing above `positions[0] + n_query - 1`, so the tail is
    /// untouched rather than merely unused. `q` and `out` stay exact: those
    /// are indexed by the launch geometry, so a wrong length there is a wrong
    /// launch.
    ///
    /// `key_depth` is the caller's host-side `positions[0] + n_query` -- the
    /// number of keys this call actually reads, not `max_keys`'s cache
    /// *capacity*. It plays no part in correctness (the launch geometry and
    /// the device-side `positions` already cover that, per the note above on
    /// why the bound check itself cannot live here) and is used only to pick
    /// among decode's kernels by depth; see `Self::decode` and
    /// `DECODE_MMA_DEPTH_THRESHOLD`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        stream: &Arc<CudaStream>,
        dec: &mut AttnDecodeScratch,
        q: &CudaSlice<f32>,
        k: &CudaSlice<u16>,
        v: &CudaSlice<u16>,
        out: &mut CudaSlice<f32>,
        n_query: usize,
        max_keys: usize,
        key_depth: usize,
        positions: &CudaSlice<i32>,
    ) -> Result<(), AttentionError> {
        self.forward_impl(
            stream, dec, q, k, v, out, n_query, max_keys, key_depth, positions, false,
        )
    }

    /// K2 uses one compensated query tile at every prefill width, including
    /// single-token shallow chunks. Deep single-token calls use decode splits.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_k2(
        &self,
        stream: &Arc<CudaStream>,
        dec: &mut AttnDecodeScratch,
        q: &CudaSlice<f32>,
        k: &CudaSlice<u16>,
        v: &CudaSlice<u16>,
        out: &mut CudaSlice<f32>,
        n_query: usize,
        max_keys: usize,
        key_depth: usize,
        positions: &CudaSlice<i32>,
    ) -> Result<(), AttentionError> {
        self.forward_impl(
            stream, dec, q, k, v, out, n_query, max_keys, key_depth, positions, true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_impl(
        &self,
        stream: &Arc<CudaStream>,
        dec: &mut AttnDecodeScratch,
        q: &CudaSlice<f32>,
        k: &CudaSlice<u16>,
        v: &CudaSlice<u16>,
        out: &mut CudaSlice<f32>,
        n_query: usize,
        max_keys: usize,
        key_depth: usize,
        positions: &CudaSlice<i32>,
        canonical_shallow: bool,
    ) -> Result<(), AttentionError> {
        Self::expect_len("attention position", positions.len(), 1)?;
        let q_elems = n_query * self.q_heads * self.head_dim;
        let row = self.kv_heads * self.head_dim;
        Self::expect_len("query", q.len(), q_elems)?;
        Self::expect_at_least("key", k.len(), max_keys * self.kv.k.row_words(row))?;
        Self::expect_at_least("value", v.len(), max_keys * self.kv.v.row_words(row))?;
        Self::expect_len("output", out.len(), q_elems)?;
        // A q8_0 cache is read only by K2's compensated kernels; the
        // single-tile oracle lever and every other shape read binary16.
        if self.kv.is_quantized()
            && (!canonical_shallow
                || self
                    .k2_single_tile
                    .load(std::sync::atomic::Ordering::Relaxed))
        {
            return Err(AttentionError::UnsupportedKvFormat {
                kv: self.kv,
                what: "q8_0 is read only by K2's staged prefill and compensated decode",
            });
        }

        // A partial tile costs its empty rows in full: the score loop runs a
        // multiply-add and a shuffle reduction per row per key whether or not
        // the row exists. Decode is `n_query == 1`, where that is seven
        // eighths of the kernel, so below a whole tile the one-row
        // instantiation is launched instead.
        // Decode gets its own two-pass shape. See the kernel comment: a single
        // query row cannot be tiled, so `attn_flash_causal_t1` could only ever
        // launch `q_heads` blocks, and this splits the key axis instead.
        if n_query == 1
            && self.decode_split_is_available()
            && !(canonical_shallow && key_depth <= 32)
        {
            return self.decode(
                stream,
                dec,
                q,
                k,
                v,
                out,
                key_depth,
                positions,
                canonical_shallow,
            );
        }

        // Three shapes, narrowest first.
        //
        // Below one query tile the tiled kernels cost their empty rows in full
        // — the score loop runs a multiply-add and a shuffle reduction per row
        // per key whether or not the row exists — so decode, which is
        // `n_query == 1`, takes the one-row instantiation.
        //
        // Above it the GQA-shared kernel is the default, because it reads K and
        // V once per KV head instead of once per query head. It cannot service
        // every geometry; `attn_flash_causal` can, and is the fallback.
        let (f, grid, block, shared) = if canonical_shallow
            && self.gqa_ratio() == 4
            && self.head_dim == 128
            && self
                .k2_single_tile
                .load(std::sync::atomic::Ordering::Relaxed)
        {
            (
                &self.flash_mma_k2,
                (
                    (n_query as u32).div_ceil(MMA_QUERY_TILE as u32),
                    (self.q_heads / 4) as u32,
                    1,
                ),
                128,
                self.shared_bytes_mma(),
            )
        } else if canonical_shallow && self.gqa_ratio() == 4 && self.head_dim == 128 {
            // One arithmetic path for every K2 prefill width, including
            // masked short chunks. Changing its precision at sixteen queries
            // perturbs the recurrently reused KV inputs and can change routes.
            (
                &self.flash_mma_k2s,
                (
                    (n_query as u32).div_ceil(2 * MMA_QUERY_TILE as u32),
                    (self.q_heads / 4) as u32,
                    1,
                ),
                256,
                K2_STAGED_SHARED_BYTES,
            )
        } else if n_query < GQA_QUERY_TILE {
            (
                &self.flash_t1,
                (n_query as u32, self.q_heads as u32, 1),
                self.head_dim as u32,
                self.shared_bytes_t1(),
            )
        } else if n_query >= MMA_QUERY_TILE && self.mma_is_available() {
            (
                &self.flash_mma,
                (
                    (n_query as u32).div_ceil(MMA_QUERY_TILE as u32),
                    (self.q_heads / MMA_HEADS_PER_BLOCK) as u32,
                    1,
                ),
                256,
                self.shared_bytes_mma(),
            )
        } else if self.gqa_shared_is_available() {
            (
                &self.flash_gqa,
                (
                    (n_query as u32).div_ceil(GQA_QUERY_TILE as u32),
                    self.kv_heads as u32,
                    1,
                ),
                (self.gqa_ratio() * 32) as u32,
                self.shared_bytes_gqa(),
            )
        } else {
            (
                &self.flash,
                (
                    (n_query as u32).div_ceil(QUERY_TILE as u32),
                    self.q_heads as u32,
                    1,
                ),
                self.head_dim as u32,
                self.shared_bytes(),
            )
        };
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: (block, 1, 1),
            shared_mem_bytes: shared as u32,
        };
        let q_heads = self.q_heads as i32;
        let kv_heads = self.kv_heads as i32;
        let head_dim = self.head_dim as i32;
        let n_query_i32 = n_query as i32;
        let scale = self.scale();
        let mut builder = stream.launch_builder(f);
        builder
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(out)
            .arg(&q_heads)
            .arg(&kv_heads)
            .arg(&head_dim)
            .arg(positions)
            .arg(&scale)
            .arg(&n_query_i32);
        // SAFETY: each of the three shapes covers the whole query range with a
        // grid and block chosen just above, and every one is passed `n_query`
        // so the kernel masks the rows a partial tile does not have — it
        // neither reads `q` nor writes `out` for them. The deepest key index
        // any block reads is `positions[0] + n_query - 1`, which the caller
        // promised is below `max_keys`, and all four buffers were checked
        // against it. Each shape's shared request is computed by the method
        // named beside it and covers everything that shape indexes: the two
        // softmax scratch tiles for `flash`, `q_sh` plus scratch for
        // `flash_t1`, and the staged K/V tiles plus a per-warp weight tile for
        // `flash_gqa`. `gqa_shared_is_available` has already established that
        // the GQA block is at most 1024 threads and its shared request at most
        // the 48 KiB a block gets without opting in.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// The two-pass decode path: slice the key window, then merge the slices.
    ///
    /// Split by [`Self::forward`] rather than inlined there because the two
    /// passes need a launch each and share none of the single-launch shapes'
    /// grid arithmetic. Buffer lengths were checked by the caller.
    #[allow(clippy::too_many_arguments)]
    fn decode(
        &self,
        stream: &Arc<CudaStream>,
        dec: &mut AttnDecodeScratch,
        q: &CudaSlice<f32>,
        k: &CudaSlice<u16>,
        v: &CudaSlice<u16>,
        out: &mut CudaSlice<f32>,
        key_depth: usize,
        positions: &CudaSlice<i32>,
        compensated: bool,
    ) -> Result<(), AttentionError> {
        let partials = DECODE_SPLITS * self.q_heads;
        Self::expect_len(
            "decode partial sums",
            dec.acc.len(),
            partials * self.head_dim,
        )?;
        Self::expect_len("decode partial maxima", dec.m.len(), partials)?;
        Self::expect_len("decode partial normalizers", dec.l.len(), partials)?;

        let q_heads = self.q_heads as i32;
        let kv_heads = self.kv_heads as i32;
        let head_dim = self.head_dim as i32;
        let scale = self.scale();

        // One warp per split when the geometry allows it: it reads K and V
        // once each instead of staging them for a KV group to share, which at
        // a 128K window is the difference between 27% and most of the card's
        // bandwidth. See the kernel comment. The tensor-core kernel, when
        // active, uses the smaller split count its longer tensor-core blocks
        // prefer; the scalar warp path uses more, shorter slices at shallow
        // depth. Both counts are fixed launch geometry and remain capturable.
        let warp_split = self.decode_warp_is_available();
        let mma = if compensated {
            self.k2_decode_mma_wpo(key_depth)
        } else {
            self.active_decode_mma(key_depth)
        };
        let decode_splits = if mma.is_some() && compensated {
            self.k2_mma_decode_splits
        } else if mma.is_some() {
            self.mma_decode_splits
        } else if compensated {
            K2_WARP_DECODE_SPLITS
        } else {
            WARP_DECODE_SPLITS
        };
        let decode_splits_i32 = decode_splits as i32;
        // Blocks may undershoot the logical split count only on the MMA
        // path, whose kernel grid-strides over splits; the scalar kernels
        // still assume one block per split. The setup-time env switch wins
        // over the per-capture runtime setter, so a whole-process A/B stays
        // a whole-process A/B.
        let grid_blocks = if mma.is_some() {
            let runtime = self
                .decode_mma_blocks
                .load(std::sync::atomic::Ordering::Relaxed);
            let chosen = if self.mma_decode_blocks >= 1 {
                self.mma_decode_blocks
            } else if runtime >= 1 {
                runtime
            } else {
                decode_splits
            };
            chosen.min(decode_splits)
        } else {
            decode_splits
        };
        let split_cfg = LaunchConfig {
            grid_dim: (grid_blocks as u32, self.kv_heads as u32, 1),
            block_dim: (
                if let Some(wpo) = mma {
                    (wpo * 32) as u32
                } else if warp_split {
                    32
                } else {
                    (self.gqa_ratio() * 32) as u32
                },
                1,
                1,
            ),
            shared_mem_bytes: if let Some(wpo) = mma {
                Self::dmma_shared_bytes(self.head_dim, wpo) as u32
            } else if warp_split {
                0
            } else {
                self.shared_bytes_decode() as u32
            },
        };
        let f = if let Some(wpo) = mma {
            if compensated && self.head_dim == 128 && wpo == 4 {
                &self.decode_mma_k2_128_wpo4
            } else if compensated && self.head_dim == 128 {
                &self.decode_mma_k2_128_wpo2
            } else if compensated && wpo == 4 {
                &self.decode_mma_k2_wpo4
            } else if compensated {
                &self.decode_mma_k2_wpo2
            } else if wpo == 4 {
                &self.decode_mma_wpo4
            } else {
                &self.decode_mma_wpo2
            }
        } else if warp_split && compensated && self.head_dim == 128 && self.gqa_ratio() == 4 {
            &self.decode_warp_k2_128
        } else if self.kv.is_quantized() {
            return Err(AttentionError::UnsupportedKvFormat {
                kv: self.kv,
                what: "q8_0 decode needs the K2 warp or tensor-core split kernels",
            });
        } else if warp_split {
            &self.decode_warp
        } else {
            &self.decode_split
        };
        let mut builder = stream.launch_builder(f);
        builder
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(&mut dec.acc)
            .arg(&mut dec.m)
            .arg(&mut dec.l)
            .arg(&q_heads)
            .arg(&kv_heads)
            .arg(&head_dim)
            .arg(positions)
            .arg(&scale)
            .arg(&decode_splits_i32);
        // SAFETY: the grid is (DECODE_SPLITS, kv_heads) with one warp per query
        // head under that KV head, so `h` stays below `q_heads` and every
        // partial index below `DECODE_SPLITS * q_heads`, which the three
        // `expect_len` calls above sized. The deepest key read is
        // `positions[0]`, which the caller checked against `max_keys`. Shared
        // covers the staged tiles and the per-warp scores, and
        // `decode_split_is_available` established it fits in 48 KiB and that
        // the block is at most 1024 threads.
        unsafe { builder.launch(split_cfg) }?;

        let combine_cfg = LaunchConfig {
            grid_dim: (self.q_heads as u32, 1, 1),
            block_dim: (self.head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&self.decode_combine);
        builder
            .arg(&dec.acc)
            .arg(&dec.m)
            .arg(&dec.l)
            .arg(out)
            .arg(&q_heads)
            .arg(&head_dim)
            .arg(&decode_splits_i32);
        // SAFETY: one block per query head, one thread per head dimension, so
        // the read of `part_acc` stays inside the buffer sized above and the
        // write covers `out` exactly once — `out` is `q_heads * head_dim` at
        // `n_query == 1`, which the caller checked.
        unsafe { builder.launch(combine_cfg) }?;
        Ok(())
    }

    /// The cache formats this kernel set reads and writes.
    pub fn kv_formats(&self) -> KvFormats {
        self.kv
    }

    /// K2's compensated 128-dimensional tensor-core decode selection.
    /// Small windows keep the scalar path; explicit benchmark overrides apply.
    pub fn k2_decode_mma_wpo(&self, depth: usize) -> Option<usize> {
        let lever = self.decode_mma.load(std::sync::atomic::Ordering::Relaxed);
        if lever == 0 && self.head_dim == 128 {
            return (depth >= 512 && self.decode_mma_is_available(2)).then_some(2);
        }
        self.active_decode_mma(depth)
    }

    /// Whether a K2 call can batch the ordinary warp split path.
    pub fn k2_batch_uses_warp(&self, depth: usize) -> bool {
        depth > 32 && self.decode_warp_is_available() && self.k2_decode_mma_wpo(depth).is_none()
    }

    /// Decode independent K2 caches in one grid, retaining the scalar split order.
    ///
    /// # Safety
    /// The first `n` pointer slots must reference live positions and distinct
    /// caches, each containing keys through its position, as checked by the engine.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn decode_batch_k2_raw(
        &self,
        stream: &Arc<CudaStream>,
        dec: &mut AttnDecodeScratch,
        q: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        positions: &[u64],
        k_caches: &[u64],
        v_caches: &[u64],
    ) -> Result<(), AttentionError> {
        // SAFETY: forwarded unchanged from this method's caller contract.
        unsafe {
            self.launch_batch_k2_raw(stream, dec, q, out, positions, k_caches, v_caches, None)
        }
    }

    /// Compensated tensor-core decode for independent 128-dimensional caches.
    ///
    /// # Safety
    /// The pointer-slot contract is identical to `decode_batch_k2_raw`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn decode_batch_k2_mma_raw(
        &self,
        stream: &Arc<CudaStream>,
        dec: &mut AttnDecodeScratch,
        q: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        positions: &[u64],
        k_caches: &[u64],
        v_caches: &[u64],
        wpo: usize,
    ) -> Result<(), AttentionError> {
        if self.head_dim != 128 || ![2, 4].contains(&wpo) || !self.decode_mma_is_available(wpo) {
            return Err(AttentionError::BufferShape {
                what: "K2 batch MMA geometry",
                expected: 128,
                actual: self.head_dim,
            });
        }
        // SAFETY: forwarded unchanged from this method's caller contract.
        unsafe {
            self.launch_batch_k2_raw(
                stream,
                dec,
                q,
                out,
                positions,
                k_caches,
                v_caches,
                Some(wpo),
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_batch_k2_raw(
        &self,
        stream: &Arc<CudaStream>,
        dec: &mut AttnDecodeScratch,
        q: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        positions: &[u64],
        k_caches: &[u64],
        v_caches: &[u64],
        mma: Option<usize>,
    ) -> Result<(), AttentionError> {
        let n = positions.len();
        if n == 0 || n > STEP_SLOTS || !self.decode_warp_is_available() {
            return Err(AttentionError::BufferShape {
                what: "K2 batch warp geometry",
                expected: STEP_SLOTS,
                actual: n,
            });
        }
        Self::expect_len("K2 batch K slots", k_caches.len(), n)?;
        Self::expect_len("K2 batch V slots", v_caches.len(), n)?;
        Self::expect_len(
            "K2 batch queries",
            q.len(),
            n * self.q_heads * self.head_dim,
        )?;
        Self::expect_len("K2 batch output", out.len(), q.len())?;
        let split_count = if mma.is_some() {
            self.k2_mma_decode_splits
        } else {
            K2_WARP_DECODE_SPLITS
        };
        let partials = n * split_count * self.q_heads;
        Self::expect_at_least("K2 batch partials", dec.acc.len(), partials * self.head_dim)?;
        Self::expect_at_least("K2 batch maxima", dec.m.len(), partials)?;
        Self::expect_at_least("K2 batch normalizers", dec.l.len(), partials)?;
        let fill = |src: &[u64]| {
            let mut slots = [src[0]; STEP_SLOTS];
            slots[..n].copy_from_slice(src);
            slots
        };
        let (p, k, v) = (fill(positions), fill(k_caches), fill(v_caches));
        let (qh, kh, hd, splits) = (
            self.q_heads as i32,
            self.kv_heads as i32,
            self.head_dim as i32,
            split_count as i32,
        );
        let scale = self.scale();
        let kernel = match mma {
            Some(4) => &self.decode_mma_k2_batch_wpo4,
            Some(_) => &self.decode_mma_k2_batch_wpo2,
            None if self.head_dim == 128 && self.gqa_ratio() == 4 => &self.decode_warp_k2_128_batch,
            None => &self.decode_warp_k2_batch,
        };
        let mut b = stream.launch_builder(kernel);
        b.arg(q);
        if mma.is_some() {
            b.arg(&k[0]).arg(&v[0]);
        }
        b.arg(&mut dec.acc)
            .arg(&mut dec.m)
            .arg(&mut dec.l)
            .arg(&qh)
            .arg(&kh)
            .arg(&hd);
        if mma.is_some() {
            b.arg(&p[0]);
        }
        b.arg(&scale).arg(&splits);
        for slot in p.iter().chain(&k).chain(&v) {
            b.arg(slot);
        }
        // SAFETY: fixed grids bound sequence slots and checked partial buffers;
        // the caller guarantees each cache's valid window and position pointer.
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (split_count as u32, self.kv_heads as u32, n as u32),
                block_dim: (mma.unwrap_or(1) as u32 * 32, 1, 1),
                shared_mem_bytes: mma
                    .map_or(0, |wpo| Self::dmma_shared_bytes(self.head_dim, wpo) as u32),
            })
        }?;
        let mut b = stream.launch_builder(&self.decode_combine_k2_batch);
        b.arg(&dec.acc)
            .arg(&dec.m)
            .arg(&dec.l)
            .arg(out)
            .arg(&qh)
            .arg(&hd)
            .arg(&splits);
        // SAFETY: one block per sequence/head, matching the validated output.
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (self.q_heads as u32, n as u32, 1),
                block_dim: (self.head_dim as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;
        Ok(())
    }

    /// Materialize `n` consecutive absolute positions from a base.
    ///
    /// The attention kernels each read one `positions[0]`, so a caller that
    /// wants to run query rows one at a time — which is what the flash-decode
    /// split shape needs, and what a speculative verify pass wants instead of
    /// a two-block prefill launch — has to hand each row its own device
    /// scalar. This writes them once for the whole pass; the caller then
    /// slices `out` a row at a time.
    pub fn row_positions(
        &self,
        stream: &Arc<CudaStream>,
        base: &CudaSlice<i32>,
        out: &mut CudaSlice<i32>,
        n: usize,
    ) -> Result<(), AttentionError> {
        Self::expect_len("row position base", base.len(), 1)?;
        Self::expect_at_least("row positions", out.len(), n)?;
        if n == 0 {
            return Ok(());
        }
        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(APPEND_THREADS), 1, 1),
            block_dim: (APPEND_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i = n as i32;
        let mut builder = stream.launch_builder(&self.row_positions);
        builder.arg(base).arg(&mut *out).arg(&n_i);
        // SAFETY: one thread per element over a grid that covers `n` and
        // returns above it; `base` holds one int and `out` at least `n`,
        // both checked above.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// Write this batch's keys and values into the cache at `positions[0]`.
    ///
    /// `key` and `value` are `[n_tokens][kv_heads][head_dim]`; the caches are
    /// the same layout over `max_keys` positions. This replaced two
    /// device-to-device copies into host-computed slices — see the kernel's
    /// own comment for why a host-computed destination address had to go.
    #[allow(clippy::too_many_arguments)]
    pub fn append_kv(
        &self,
        stream: &Arc<CudaStream>,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        k_cache: &mut CudaSlice<u16>,
        v_cache: &mut CudaSlice<u16>,
        n_tokens: usize,
        max_keys: usize,
        positions: &CudaSlice<i32>,
    ) -> Result<(), AttentionError> {
        Self::expect_len("append position", positions.len(), 1)?;
        let row = self.kv_heads * self.head_dim;
        let span = n_tokens * row;
        Self::expect_len("append key", key.len(), span)?;
        Self::expect_len("append value", value.len(), span)?;
        Self::expect_at_least(
            "append key cache",
            k_cache.len(),
            max_keys * self.kv.k.row_words(row),
        )?;
        Self::expect_at_least(
            "append value cache",
            v_cache.len(),
            max_keys * self.kv.v.row_words(row),
        )?;

        let cfg = LaunchConfig {
            grid_dim: ((2 * span).div_ceil(APPEND_THREADS as usize) as u32, 1, 1),
            block_dim: (APPEND_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let span_i = span as i32;
        let row_i = row as i32;
        let (kq8, vq8) = (
            i32::from(self.kv.k == KvFormat::Q8_0),
            i32::from(self.kv.v == KvFormat::Q8_0),
        );
        let mut builder = stream.launch_builder(if self.kv.is_quantized() {
            &self.append_kv8
        } else {
            &self.append
        });
        builder
            .arg(key)
            .arg(value)
            .arg(&mut *k_cache)
            .arg(&mut *v_cache)
            .arg(positions)
            .arg(&span_i)
            .arg(&row_i);
        if self.kv.is_quantized() {
            // A whole warp per q8_0 block: `row` is a multiple of 32 (the
            // constructor holds head_dim at 128) and APPEND_THREADS of 32.
            builder.arg(&kq8).arg(&vq8);
        }
        // SAFETY: every thread past `2 * span` returns, both sources hold
        // exactly `span` floats, and the deepest destination index is
        // `positions[0] * row + span - 1` — inside `max_keys * row`, which
        // both caches were checked to hold, for any position the caller's own
        // `max_seq` check admits.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// [`Self::rope`] on the query and the key and [`Self::append_kv`] for
    /// every sequence of a decode step, in one launch.
    ///
    /// `q_in`/`q_out` are `[n][q_heads][head_dim]`, `k_in`/`k_out` and
    /// `v_in` `[n][kv_heads][head_dim]`, sequence-major. `rope_positions[i]`
    /// and `positions[i]` are device pointers to sequence `i`'s rotary
    /// position and cache position; `k_caches[i]`/`v_caches[i]` its cache
    /// bases. `n` is at most [`STEP_SLOTS`].
    ///
    /// Bit-identical to the three separate launches; see the kernel.
    ///
    /// # Safety
    ///
    /// Every pointer in the first `n` slots must be live for this launch:
    /// the positions one `i32` each, the caches at least `max_keys *
    /// kv_heads * head_dim` halves each, for a `max_keys` above every
    /// position, and the caches must be distinct sequences' (the kernel
    /// writes them without synchronization between blocks).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rope_append_batch_raw(
        &self,
        stream: &Arc<CudaStream>,
        q_in: &CudaSlice<f32>,
        q_out: &mut CudaSlice<f32>,
        k_in: &CudaSlice<f32>,
        k_out: &mut CudaSlice<f32>,
        v_in: &CudaSlice<f32>,
        rope_positions: &[u64],
        positions: &[u64],
        k_caches: &[u64],
        v_caches: &[u64],
        rope_dim: usize,
        theta_base: f32,
    ) -> Result<(), AttentionError> {
        let n = rope_positions.len();
        if n == 0 || n > STEP_SLOTS {
            return Err(AttentionError::BufferShape {
                what: "rope/append batch (1..=STEP_SLOTS sequences)",
                expected: STEP_SLOTS,
                actual: n,
            });
        }
        Self::expect_len("rope/append batch positions", positions.len(), n)?;
        Self::expect_len("rope/append batch key caches", k_caches.len(), n)?;
        Self::expect_len("rope/append batch value caches", v_caches.len(), n)?;
        if !rope_dim.is_multiple_of(2) || rope_dim > self.head_dim {
            return Err(AttentionError::UnsupportedRopeDim {
                rope_dim,
                head_dim: self.head_dim,
            });
        }
        let q_n = n * self.q_heads * self.head_dim;
        let kv_n = n * self.kv_heads * self.head_dim;
        Self::expect_len("rope/append batch q_in", q_in.len(), q_n)?;
        Self::expect_len("rope/append batch q_out", q_out.len(), q_n)?;
        Self::expect_len("rope/append batch k_in", k_in.len(), kv_n)?;
        Self::expect_len("rope/append batch k_out", k_out.len(), kv_n)?;
        Self::expect_len("rope/append batch v_in", v_in.len(), kv_n)?;

        let fill = |src: &[u64]| {
            let mut slots = [src[0]; STEP_SLOTS];
            slots[..n].copy_from_slice(src);
            slots
        };
        let (rp, ap, kc, vc) = (
            fill(rope_positions),
            fill(positions),
            fill(k_caches),
            fill(v_caches),
        );
        let cfg = LaunchConfig {
            grid_dim: ((self.q_heads + self.kv_heads) as u32, n as u32, 1),
            block_dim: (self.head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let (q_heads, kv_heads, head_dim, rope_dim_i) = (
            self.q_heads as i32,
            self.kv_heads as i32,
            self.head_dim as i32,
            rope_dim as i32,
        );
        let mut builder = stream.launch_builder(&self.rope_append_batch);
        builder
            .arg(q_in)
            .arg(&mut *q_out)
            .arg(k_in)
            .arg(&mut *k_out)
            .arg(v_in);
        for slot in rp.iter().chain(&ap).chain(&kc).chain(&vc) {
            builder.arg(slot);
        }
        builder
            .arg(&q_heads)
            .arg(&kv_heads)
            .arg(&head_dim)
            .arg(&rope_dim_i)
            .arg(&theta_base);
        // SAFETY: `grid.y = n` bounds the pointer slots to the caller's live
        // sequences, whose validity is the caller's contract above; every
        // activation buffer was checked to hold exactly `n` sequences of its
        // width and each thread touches its own head's slice.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_packed_query_layout_is_interleaved_not_halved() {
        // The single most likely way to get this tensor wrong. At head 0 the
        // two layouts agree, which is exactly why a spot check on head 0
        // passes and the model is still broken.
        let head_dim = 256;
        assert_eq!(attn_packed_query_offset(0, head_dim), 0);
        assert_eq!(attn_packed_gate_offset(0, head_dim), head_dim);

        // A halves split would put query head h at h * head_dim. Interleaved
        // puts it at 2 * h * head_dim. They differ for every head but the
        // first.
        for head in 1..16 {
            assert_ne!(
                attn_packed_query_offset(head, head_dim),
                head * head_dim,
                "head {head}: interleaved offset collapsed onto the halves split",
            );
        }

        // And the halves split would read query head 8's slice out of the
        // region that actually holds gates, at the real 16-head geometry.
        let q_dim = 16 * head_dim;
        assert!(attn_packed_query_offset(8, head_dim) >= q_dim);
        // The last head's gate ends exactly at the packed row's width, so the
        // two offset formulas tile the full `2 * q_dim` without a gap.
        assert_eq!(attn_packed_gate_offset(15, head_dim) + head_dim, 2 * q_dim);
    }

    #[test]
    fn the_split_kernel_strides_by_two_head_dims() {
        // Guards the comment above from being "simplified" in the source.
        assert!(
            ATTENTION_SRC.contains("packed[row + (long long)(2 * h)     * head_dim + d]"),
            "query slice is no longer read at the interleaved 2*h stride",
        );
        assert!(
            ATTENTION_SRC.contains("packed[row + (long long)(2 * h + 1) * head_dim + d]"),
            "gate slice is no longer read at the interleaved 2*h+1 stride",
        );
    }

    #[test]
    fn the_causal_bound_is_inclusive_of_the_query_position() {
        // An off-by-one here leaks exactly one future token per row, which
        // moves aggregate error metrics by almost nothing. It is asserted
        // structurally because no tolerance would catch it reliably.
        // The block's loop bound is the *last* row's window; each earlier row
        // masks the tail off itself. Both halves are asserted, because
        // dropping the per-row mask would silently let row `qi0` attend to
        // seven future tokens and every aggregate metric would barely move.
        assert!(
            ATTENTION_SRC
                .contains("long long n_visible = (long long)(*key_offset) + qi0 + qt_live;"),
            "the causal window bound changed shape",
        );
        assert!(
            ATTENTION_SRC.contains("long long limit = (long long)(*key_offset) + qi0 + tid;")
                && ATTENTION_SRC.contains("long long limit = (long long)(*key_offset) + qi0 + su;")
                && ATTENTION_SRC.contains("if (j0 + w <= limit) tmax ="),
            "the per-row causal mask is gone from the softmax phases",
        );
        assert!(
            ATTENTION_SRC.contains("for (long long j0 = 0; j0 < n_visible; j0 += tile)")
                && ATTENTION_SRC.contains("long long key = j0 + (long long)warp * ATTN_KT + r;")
                && ATTENTION_SRC.contains("if (key < n_visible) {"),
            "the key loop no longer stops at the causal bound",
        );
    }

    #[test]
    fn the_rope_tail_is_copied_rather_than_computed() {
        // `out[d] = in[d]` is bit-identical. Anything arithmetic — even
        // multiplying by a cos of angle zero — is not, and would fail the
        // exact-equality gate the differential test puts on the tail.
        assert!(
            ATTENTION_SRC.contains("out[base + d] = in[base + d];"),
            "the untouched rotary tail is no longer a plain copy",
        );
        // And it is still selected by the same bound, now inside the token
        // band rather than outside it.
        assert!(
            ATTENTION_SRC.contains("if (d >= rope_dim) {"),
            "the rotary tail is no longer selected by rope_dim",
        );
    }

    #[test]
    fn the_accumulator_is_rescaled_before_the_new_contributions_land() {
        // Online softmax is only stable if the running accumulator and the
        // normalizer are moved into the new max's frame *first*. Folding the
        // new weights in before rescaling produces a finite, plausible, wrong
        // answer whenever the max increases.
        let rescale_l = ATTENTION_SRC
            .find("l_sh[tid] = l_sh[tid] * corr_sh[tid] + lsum;")
            .expect("normalizer rescale present");
        let rescale_acc = ATTENTION_SRC
            .find("for (int u = 0; u < QT; ++u) acc[u] = acc[u] * corr_sh[u];")
            .expect("accumulator rescale present");
        let fold_in = ATTENTION_SRC
            .find("acc[u] += w_sh[u * tile + w] * vv;")
            .expect("value accumulation present");
        assert!(rescale_acc < fold_in, "values folded in before rescaling");
        assert!(rescale_l < fold_in, "normalizer updated after the values");

        // The one-row kernel is a separate body and needs the same order.
        let t1 = ATTENTION_SRC
            .find("__global__ void attn_flash_causal_t1(")
            .expect("the one-row kernel is still there");
        let tail = &ATTENTION_SRC[t1..];
        let t1_l = tail
            .find("l = l * corr + lsum;")
            .expect("t1 normalizer rescale");
        let t1_acc = tail
            .find("float a = acc * corr;")
            .expect("t1 accumulator rescale");
        // Anchored on the prefix rather than the whole expression: the value
        // load acquired an `h2f(...)` when the cache became binary16, and the
        // property being checked here is the *order* of the fold against the
        // rescale, not how the value is read.
        let t1_fold = tail.find("a += w_sh[w]").expect("t1 value accumulation");
        assert!(t1_acc < t1_fold, "t1 values folded in before rescaling");
        assert!(t1_l < t1_fold, "t1 normalizer updated after the values");
    }

    #[test]
    fn gqa_is_a_grouping_not_an_identity() {
        assert!(
            ATTENTION_SRC.contains("int kvh = h / (q_heads / kv_heads);"),
            "the kv head mapping is no longer a GQA grouping",
        );
        // Mirrors llmcuda_kernels::attention::kv_head_for_query_head at the real
        // 16:2 geometry. Asserted here because llmcuda-cuda must not depend on
        // the oracle crate.
        let kv_of = |h: usize| h / (16 / 2);
        assert_eq!(kv_of(0), 0);
        assert_eq!(kv_of(7), 0);
        assert_eq!(kv_of(8), 1);
        assert_eq!(kv_of(15), 1);
    }

    #[test]
    fn the_shared_memory_budget_fits_turing_at_the_real_geometry() {
        // docs/KERNELS.md: 48 KiB per block, measured. The module docs work
        // through why the textbook tile shapes do not fit at head_dim 256.
        const TURING_SHARED_PER_BLOCK: usize = 48 * 1024;
        let head_dim = 256usize;
        let keys_per_tile = (head_dim / 32) * KEYS_PER_WARP;
        let bytes = (2 * QUERY_TILE * keys_per_tile + 3 * QUERY_TILE) * size_of::<f32>();
        assert_eq!(bytes, 2144);
        assert!(bytes < TURING_SHARED_PER_BLOCK / 20);

        // The tile shapes that do not fit, so the arithmetic in the module
        // docs fails loudly if someone edits it.
        let tiled = |bm: usize, bn: usize| (bm + 2 * bn) * head_dim * 4 + bm * bn * 4;
        assert!(tiled(64, 64) > TURING_SHARED_PER_BLOCK);
        assert!(tiled(32, 32) > TURING_SHARED_PER_BLOCK);
        assert!(tiled(16, 16) > TURING_SHARED_PER_BLOCK);
        assert!(tiled(8, 8) < TURING_SHARED_PER_BLOCK);
    }

    #[test]
    fn head_geometry_is_validated_not_assumed() {
        // The checks that run without a device.
        assert!(
            AttentionError::UnsupportedHeadDim { head_dim: 100 }
                .to_string()
                .contains("100")
        );
        assert!(
            AttentionError::UnevenGqaGrouping {
                q_heads: 16,
                kv_heads: 5,
            }
            .to_string()
            .contains("16")
        );
        assert!(
            AttentionError::QueryPastKeys {
                key_offset: 100,
                n_query: 8,
                n_keys: 104,
            }
            .to_string()
            .contains("108")
        );
        // Qwen3.6's real geometry must be accepted.
        assert_eq!(256 % 32, 0);
        assert_eq!(16 % 2, 0);
        assert_eq!(64 % 2, 0);
    }

    #[test]
    fn a_kv_cache_may_be_longer_than_its_filled_window_but_never_shorter() {
        // The asymmetry is the whole point: a cache is allocated for the
        // longest admissible sequence and filled one token at a time, so
        // "longer than the window" is its steady state from decode step two
        // onward. "Shorter" is the read-past-the-end bug this check exists to
        // catch, and it stays fatal.
        assert!(AttentionKernels::expect_at_least("key", 4096, 4096).is_ok());
        assert!(AttentionKernels::expect_at_least("key", 1 << 20, 4096).is_ok());

        let err = AttentionKernels::expect_at_least("key", 4095, 4096)
            .expect_err("one float short must not be accepted");
        let message = err.to_string();
        assert!(
            message.contains("4096") && message.contains("4095"),
            "{message}"
        );

        // The query and the output are indexed by the launch geometry rather
        // than by absolute position, so they keep the exact check.
        assert!(AttentionKernels::expect_len("query", 4097, 4096).is_err());
    }

    #[test]
    fn the_query_tile_gives_one_thread_per_softmax_slot() {
        // The kernel's softmax phases index `score_sh` and `w_sh` as
        // `su * tile + sw` with `su = tid / tile`, and cover every slot with
        // no loop. That is only true when the block is exactly as wide as the
        // tile is big — which, since the block is `head_dim` threads and the
        // tile is `(head_dim / 32) * KEYS_PER_WARP` keys deep per row, reduces
        // to this product being a warp.
        assert_eq!(
            QUERY_TILE * KEYS_PER_WARP,
            32,
            "QUERY_TILE * KEYS_PER_WARP must be 32, or phase 2 misses slots",
        );
        for head_dim in [32usize, 64, 128, 256] {
            let tile = (head_dim / 32) * KEYS_PER_WARP;
            assert_eq!(
                QUERY_TILE * tile,
                head_dim,
                "head_dim {head_dim}: block width and softmax slot count diverged",
            );
        }
    }

    #[test]
    fn the_register_query_tile_bounds_the_head_dimension() {
        // `qr[ATTN_QT][ATTN_MAXD]` is only in registers while every index
        // folds to a constant, so the head dimension cannot exceed what
        // ATTN_MAXD covers. Asserted because raising head_dim past it would
        // compile, run, and silently spill.
        assert_eq!(32 * ATTN_MAXD, 256);
        assert!(ATTENTION_SRC.contains("#define ATTN_MAXD 8"));
        // k2_rope compiles its q8_0 append against this copy.
        assert!(ATTENTION_SRC.contains(KV_Q8_HELPERS));
        assert!(
            ATTENTION_SRC.contains("float qr[QT][ATTN_MAXD];"),
            "the query tile is no longer a register array",
        );
    }

    #[test]
    fn the_tensor_core_block_shape_is_one_choice_and_not_three() {
        // `MMA_HPB`, `MMA_WPH` and `MMA_KT` look like three tunables and are
        // not. The block is eight warps split evenly among the heads it serves,
        // and `Q K^T` gives each warp of a head exactly one octet of the staged
        // keys. Break either relation and the kernel still compiles: it either
        // leaves warps with no octet to score or leaves octets unscored, and
        // the wrong answer is finite and plausible rather than a crash.
        assert_eq!(
            MMA_HEADS_PER_BLOCK * MMA_WARPS_PER_HEAD,
            8,
            "the head groups no longer partition the block's eight warps",
        );
        assert_eq!(
            MMA_KEY_TILE,
            8 * MMA_WARPS_PER_HEAD * MMA_KEY_OCTETS,
            "the staged key tile is no longer MMA_KEY_OCTETS octets per warp",
        );

        // The device sees these through `#define`s, and a Rust constant that
        // drifted from its mirror would be caught by nothing else: every launch
        // parameter derived here would be self-consistent and wrong.
        assert!(
            ATTENTION_SRC.contains(&format!("#define MMA_HPB {MMA_HEADS_PER_BLOCK}"))
                && ATTENTION_SRC.contains("#define MMA_WPH (8 / MMA_HPB)")
                && ATTENTION_SRC.contains(&format!("#define MMA_KOCT {MMA_KEY_OCTETS}"))
                && ATTENTION_SRC.contains("#define MMA_KT (8 * MMA_WPH * MMA_KOCT)")
                && ATTENTION_SRC.contains("#define MMA_MAXT (256 / (8 * MMA_WPH))"),
            "the device block shape no longer mirrors the host constants",
        );

        assert_eq!(MMA_WARPS_PER_HEAD, 1, "softmax rows must stay warp-owned");
        assert_eq!(
            MMA_KEY_OCTETS, 1,
            "the fragment reduction covers eight keys"
        );

        // The value stride is the one constant with a non-obvious formula on
        // both sides, and a mismatch would be a shared-memory overrun on one
        // side or a bank conflict on the other. Check the closed forms agree
        // at every key tile the shape rule can produce.
        for wph in [1usize, 2, 4] {
            let kt = 8 * wph;
            let host = 4 + 8 * (kt / 2 - 4).div_ceil(8);
            // Transcribed from the `#define` character for character, which is
            // the whole point: rewriting it as `div_ceil` here would compare
            // the host formula against itself and prove nothing.
            #[allow(clippy::manual_div_ceil)]
            let device = 4 + 8 * (((kt / 2) - 4 + 7) / 8);
            assert_eq!(host, device, "value stride disagrees at MMA_KT {kt}");
            let mut a = host;
            let mut b = 32usize;
            while b != 0 {
                let t = b;
                b = a % b;
                a = t;
            }
            assert_eq!(a, 4, "a value stride of {host} is not a bank bijection");
            assert!(
                host >= kt / 2,
                "a value stride of {host} cannot hold {kt} keys"
            );
        }
        assert_eq!(MMA_VALUE_STRIDE, 4 + 8 * (MMA_KEY_TILE / 2 - 4).div_ceil(8));
        assert!(
            ATTENTION_SRC.contains("#define MMA_VSTRIDE (4 + 8 * (((MMA_KT / 2) - 4 + 7) / 8))"),
            "the value stride no longer mirrors MMA_VALUE_STRIDE",
        );

        // The softmax butterfly reduces over the lanes holding one row, so a
        // row's tile must not be wider than a warp and must divide it.
        assert!(
            MMA_KEY_TILE <= 32 && 32_usize.is_multiple_of(MMA_KEY_TILE),
            "a key tile of {MMA_KEY_TILE} does not divide a warp",
        );
        // And a warp's share of the query rows must divide into whole passes.
        let rows_per_warp = MMA_QUERY_TILE / MMA_WARPS_PER_HEAD;
        assert!(
            rows_per_warp.is_multiple_of(32 / MMA_KEY_TILE),
            "{rows_per_warp} rows per warp is not a whole number of softmax passes",
        );
    }

    #[test]
    fn the_staged_tiles_fit_the_shared_memory_that_was_opted_in_to() {
        // Only the K/V tile remains shared; Q and softmax state are registers.
        for head_dim in [32, 64, 128, 256] {
            let bytes = mma_shared_bytes(head_dim);
            assert!(
                bytes <= MMA_SHARED_CEILING,
                "head_dim {head_dim}: {bytes} B of staged tiles exceeds the \
                 {MMA_SHARED_CEILING} B carveout",
            );
        }
        assert_eq!(
            mma_shared_bytes(256),
            8_320,
            "the budget at this model's geometry moved; check it still buys \
             what the block shape was widened for",
        );
        // Well under 32 KiB now that Q is in registers, so shared memory is no
        // longer what holds this kernel to one block per SM -- the accumulator
        // is. Stated because the natural next question is occupancy, and the
        // answer has moved from "shared" to "registers".
        assert!(
            mma_shared_bytes(256) <= 32 * 1024,
            "two blocks per SM are no longer admitted by shared memory",
        );
    }
}
