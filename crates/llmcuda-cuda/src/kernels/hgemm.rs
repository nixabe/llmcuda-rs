//! Weight-only GEMM on Turing's fp16 tensor cores, for prefill-width
//! projections whose weights are Q8_0 or bf16.
//!
//! `out[m][n] (+)= alpha * sum_k half(x[m][k]) * half(w[n][k])`, fp32
//! accumulate, where `half(w)` is the correctly rounded fp16 of the stored
//! weight: `q * d` for the split-int8 Q8_0 layout `MmaKernels::repack_q8_0_half`
//! produces (`wq` `[N][K]` int8, `ws` `[N][K/32]` fp16 bits), or the bf16 value
//! times a per-tensor power of two for a bf16 tensor, undone by `alpha`.
//!
//! # Why not the integer tensor cores
//!
//! `mma_q8_0_proj_split` feeds `m8n8k16` int8 MMAs, but Q8_0 carries a scale
//! per 32 weights and the activation carries one per 32 values, so every 8x8
//! accumulator is converted to float and rescaled after every two
//! instructions. Measured at the 27B's prefill shapes it reaches ~44 TOP/s;
//! cuBLASLt's fp16 GEMM over the same shapes sustains 62-83 TFLOP/s on the
//! same card (`bench_hgemm`). Dequantizing the *weights* to fp16 once per staged
//! tile moves the scale out of the contraction: each staged weight is reused
//! by every token in the block's tile, and the `m16n8k8` fp16 MMA accumulates
//! a whole K in fp32 with no epilogue until the end. It is also the more
//! accurate path — activations are rounded to fp16 (11-bit significand)
//! instead of quantized to int8 against a block absmax.
//!
//! A bf16 tensor has no tensor-core path on Turing at all; converting its
//! values to fp16 is exact once they are scaled into fp16's normal range, and
//! a power-of-two scale chosen from the tensor's own max-abs
//! ([`bf16_half_exponent`]) does that for all but a few subnormal-tiny values
//! of the shipped files.
//!
//! # Tiling
//!
//! 128 tokens by `4 * WN` weight rows per block, eight warps as 2 (tokens) x 4
//! (rows), 64 x `WN` per warp; `WN` is 64 for wide outputs and 32 where a
//! 256-row tile would leave the card's SMs idle. K advances 32 at a time
//! through two shared stages, XOR-swizzled rather than padded, with the next
//! stage prefetched into registers while the current one is consumed — Turing
//! has no `cp.async` — so a k-step costs one barrier. Blocks walk the output
//! in column groups of four so neighbours share weight tiles in L2, and a
//! grid that would leave the card's last wave mostly idle splits the
//! contraction across `blockIdx.z` into fp32 partials summed in a fixed order
//! ([`HgemmKernels::plan`]).

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};

use super::compile;

/// Tokens per block tile.
pub const HGEMM_BM: usize = 128;
/// Contraction elements per stage; also the Q8_0 block, so one scale a row.
pub const HGEMM_BK: usize = 32;

const HGEMM_SRC: &str = r#"
#define HG_BM 128
#define HG_BK 32
#define HG_GROUP 4

extern "C" {

__device__ __forceinline__ unsigned hg_smem_addr(const void* p) {
    unsigned a;
    asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(a) : "l"(p));
    return a;
}

__device__ __forceinline__ void hg_ldsm_x4(unsigned addr, unsigned& r0, unsigned& r1,
                                           unsigned& r2, unsigned& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void hg_mma(float* d, unsigned a0, unsigned a1, unsigned b0) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(b0));
}

// Four int8 weights to two fp16 pairs, times one fp16 scale (both halves of
// `s2`). Biasing to unsigned and splicing each byte under 0x64 gives
// 1024 + (q + 128) exactly; subtracting 1152 leaves q exactly; the multiply
// is the one rounding, the correctly rounded fp16 of q * d.
__device__ __forceinline__ void hg_dequant4(unsigned q, unsigned s2, unsigned& lo, unsigned& hi) {
    q ^= 0x80808080u;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lo) : "r"(q), "r"(0x64646464u), "r"(0x5150u));
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hi) : "r"(q), "r"(0x64646464u), "r"(0x5352u));
    asm("sub.rn.f16x2 %0, %0, %1;" : "+r"(lo) : "r"(0x64806480u));
    asm("sub.rn.f16x2 %0, %0, %1;" : "+r"(hi) : "r"(0x64806480u));
    asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(lo) : "r"(s2));
    asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(hi) : "r"(s2));
}

// Two bf16 weights to one fp16 pair, each times `scale` (a power of two)
// before the one rounding. A bf16 significand is 8 bits, so the result is
// exact whenever the scaled value is an fp16 normal.
__device__ __forceinline__ unsigned hg_bf16x2(unsigned v, float scale) {
    float lo = __uint_as_float(v << 16) * scale, hi = __uint_as_float(v & 0xffff0000u) * scale;
    unsigned short l, h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(l) : "f"(lo));
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(hi));
    return (unsigned)l | ((unsigned)h << 16);
}

// A 16-byte (and a 2-byte) read-only load that asks L2 to fetch the whole
// 128-byte line. A stage reads 32 bytes of a Q8_0 weight row and 64 of an
// activation row, so without the hint each stage opens a fresh DRAM sector
// per row; with it, the next three (one) stages of that row hit L2. Worth
// 4-7% at the 27B's shapes. An explicit `prefetch.global.L2` a line ahead
// on top of it measured slower at every distance tried (128-512 elements).
__device__ __forceinline__ uint4 hg_ldg16(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.L2::128B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ unsigned hg_ldg2(const unsigned short* p) {
    unsigned short v;
    asm volatile("ld.global.nc.L2::128B.u16 %0, [%1];" : "=h"(v) : "l"(p));
    return v;
}

// fp32 -> fp16, round to nearest even, `n` elements.
__global__ void hg_to_half(const float* __restrict__ x, unsigned short* __restrict__ y, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned short h;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(x[i]));
        y[i] = h;
    }
}

// Halves offset of 16-byte chunk `c` (0..3) of row `r` in a [rows][32] fp16
// tile. XOR-swizzling the chunk by bits 1..2 of the row puts the eight rows
// one `ldmatrix` phase reads -- and the eight a store phase writes -- on
// eight distinct 16-byte bank groups, without padding.
__device__ __forceinline__ int hg_sw(int r, int c) { return r * HG_BK + ((c ^ ((r >> 1) & 3)) << 3); }

}  // extern "C"

// out[m][n] (+)= alpha * sum_k x[m][k] * w[n][k]; see the module
// documentation. FMT 0 is the split Q8_0 layout (`w` int8 [N][K], `ws` fp16
// [N][K/32]); FMT 1 is bf16 [N][K], converted as `w * wscale`. WN is the warp
// tile's weight rows (64 or 32); a block is 2 x 4 warps, 128 tokens by 4 * WN
// rows. K must be a multiple of 32; M and N are masked.
//
// Split-K: with gridDim.z > 1, block z contracts k-blocks [z * kps, (z + 1) *
// kps) only and writes alpha times its sum to `part[z][m][n]` (row stride N),
// which `hg_splitk_reduce` adds up in z order. `kps` is a multiple of 8, so a
// split starts on a scale8 boundary.
template <int FMT, int WN>
__device__ __forceinline__ void hg_gemm(
    const unsigned char* __restrict__ w,
    const unsigned short* __restrict__ ws,
    const unsigned short* __restrict__ x,
    float* __restrict__ out,
    float* __restrict__ part,
    int M, int N, int K, int ldo, int accumulate, float alpha, float wscale, int kps
) {
    constexpr int BN = 4 * WN;
    constexpr int NT = WN / 8;            // n8 tiles a warp owns
    __shared__ __align__(16) unsigned short As[2][HG_BM * HG_BK];
    __shared__ __align__(16) unsigned short Ws[2][BN * HG_BK];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    // Grouped rasterization: blocks are issued in launch order, and walking N
    // fastest makes every token tile re-stream the whole weight matrix from
    // DRAM — 32 bytes a row at a time. Walking HG_GROUP token tiles fastest
    // within each column of weight tiles lets those blocks share each weight
    // stage through L2 instead. At 2,048 tokens of the 27B's FFN gate this
    // was 35.6 -> 57.5 TFLOP/s.
    int m0, n0;
    {
        const int nt = gridDim.x, mt = gridDim.y;
        const int lin = blockIdx.y * nt + blockIdx.x;
        const int per = HG_GROUP * nt;
        const int first = (lin / per) * HG_GROUP;
        const int gs = min(HG_GROUP, mt - first);
        const int r = lin % per;
        m0 = (first + r % gs) * HG_BM;
        n0 = (r / gs) * BN;
    }
    const int wm = (warp >> 2) * 64, wn = (warp & 3) * WN;
    const int kblocks = K >> 5;
    const int kb0 = blockIdx.z * kps, kb1 = min(kb0 + kps, kblocks);

    // A: rows tid >> 2 and + 64, chunk tid & 3 (eight halves each).
    const int a_row = tid >> 2, a_ch = tid & 3;
    const bool a_ok0 = m0 + a_row < M, a_ok1 = m0 + a_row + 64 < M;
    const unsigned short* xa0 = x + (long long)(m0 + a_row) * K + a_ch * 8;
    const unsigned short* xa1 = xa0 + 64LL * K;
    const int a_st0 = hg_sw(a_row, a_ch), a_st1 = hg_sw(a_row + 64, a_ch);

    // W. Q8_0: rows tid >> 1 (+ 128 when BN is 256), sixteen int8 at
    // (tid & 1) * 16, plus the row's scale. bf16: rows tid >> 2 (+ 64, ...),
    // chunk tid & 3, eight values a chunk.
    constexpr int WREP = FMT == 0 ? BN / 128 : BN / 64;
    constexpr int WSTEP = FMT == 0 ? 128 : 64;
    const int w_row = FMT == 0 ? tid >> 1 : tid >> 2;
    const int w_sub = FMT == 0 ? tid & 1 : tid & 3;

    uint4 ra0 = make_uint4(0, 0, 0, 0), ra1 = ra0;
    uint4 rw[WREP];
    unsigned rs[WREP];
    // Q8_0 scales, eight k-blocks of them a row: one 16-byte load every
    // eighth stage instead of a 2-byte load every stage, which touched as
    // many cache lines per warp instruction as the weights themselves for a
    // sixteenth of the bytes. Needs K / 32 to be a multiple of 8 (16-byte
    // alignment, and no read past the row); otherwise one scale a stage.
    uint4 rsv[WREP];
    const bool scale8 = (kblocks & 7) == 0;
#pragma unroll
    for (int i = 0; i < WREP; ++i) { rw[i] = ra0; rs[i] = 0; rsv[i] = ra0; }

    auto load = [&](int k0) {
        if (a_ok0) ra0 = hg_ldg16(xa0 + k0);
        if (a_ok1) ra1 = hg_ldg16(xa1 + k0);
#pragma unroll
        for (int i = 0; i < WREP; ++i) {
            const int n = n0 + w_row + i * WSTEP;
            if (n < N) {
                if (FMT == 0) {
                    rw[i] = hg_ldg16(w + (long long)n * K + k0 + w_sub * 16);
                    const int kb = k0 >> 5;
                    if (!scale8) {
                        rs[i] = hg_ldg2(ws + (long long)n * kblocks + kb);
                    } else if ((kb & 7) == 0) {
                        rsv[i] = hg_ldg16(ws + (long long)n * kblocks + kb);
                    }
                } else {
                    rw[i] = hg_ldg16(w + ((long long)n * K + k0 + w_sub * 8) * 2);
                }
            }
        }
    };
    auto store = [&](int s, int kb) {
        *(uint4*)(&As[s][a_st0]) = ra0;
        *(uint4*)(&As[s][a_st1]) = ra1;
#pragma unroll
        for (int i = 0; i < WREP; ++i) {
            const int r = w_row + i * WSTEP;
            if (FMT == 0) {
                unsigned sc = rs[i];
                if (scale8) {
                    const int q = (kb >> 1) & 3;
                    const unsigned wd = q == 0 ? rsv[i].x : q == 1 ? rsv[i].y : q == 2 ? rsv[i].z : rsv[i].w;
                    sc = (wd >> ((kb & 1) * 16)) & 0xffffu;
                }
                const unsigned s2 = sc | (sc << 16);
                uint4 h0, h1;
                hg_dequant4(rw[i].x, s2, h0.x, h0.y);
                hg_dequant4(rw[i].y, s2, h0.z, h0.w);
                hg_dequant4(rw[i].z, s2, h1.x, h1.y);
                hg_dequant4(rw[i].w, s2, h1.z, h1.w);
                *(uint4*)(&Ws[s][hg_sw(r, w_sub * 2)]) = h0;
                *(uint4*)(&Ws[s][hg_sw(r, w_sub * 2 + 1)]) = h1;
            } else {
                uint4 h;
                h.x = hg_bf16x2(rw[i].x, wscale);
                h.y = hg_bf16x2(rw[i].y, wscale);
                h.z = hg_bf16x2(rw[i].z, wscale);
                h.w = hg_bf16x2(rw[i].w, wscale);
                *(uint4*)(&Ws[s][hg_sw(r, w_sub)]) = h;
            }
        }
    };

    float acc[4][NT][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

    // Per-lane ldmatrix addresses. Rows advance in multiples of eight from a
    // lane's base, which leaves bits 1..2 of the row -- the swizzle -- equal
    // to the lane's own `lr`'s, so one chunk offset per k16 serves every tile.
    // A matrix j: rows (j & 1) * 8, chunk j >> 1 (a0/a1 then a2/a3 per k8).
    // B matrix j: rows (j >> 1) * 8 (the second n8 tile), chunk j & 1.
    const int lj = lane >> 3, lr = lane & 7, sw = (lr >> 1) & 3;
    const unsigned a_lane = (unsigned)((wm + (lj & 1) * 8 + lr) * HG_BK) * 2;
    const unsigned b_lane = (unsigned)((wn + (lj >> 1) * 8 + lr) * HG_BK) * 2;
    unsigned a_off[2], b_off[2];
#pragma unroll
    for (int kh = 0; kh < 2; ++kh) {
        a_off[kh] = (unsigned)(((kh * 2 + (lj >> 1)) ^ sw) << 4);
        b_off[kh] = (unsigned)(((kh * 2 + (lj & 1)) ^ sw) << 4);
    }

    load(kb0 * HG_BK);
    store(0, kb0);
    __syncthreads();

    for (int kt = kb0; kt < kb1; ++kt) {
        const int s = (kt - kb0) & 1;
        const bool next = kt + 1 < kb1;
        if (next) load((kt + 1) * HG_BK);

        const unsigned a_base = hg_smem_addr(&As[s][0]) + a_lane;
        const unsigned b_base = hg_smem_addr(&Ws[s][0]) + b_lane;
#pragma unroll
        for (int kh = 0; kh < 2; ++kh) {
            unsigned a[4][4], b[NT / 2][4];
#pragma unroll
            for (int mt = 0; mt < 4; ++mt)
                hg_ldsm_x4(a_base + mt * 16 * HG_BK * 2 + a_off[kh], a[mt][0], a[mt][1], a[mt][2], a[mt][3]);
#pragma unroll
            for (int np = 0; np < NT / 2; ++np)
                hg_ldsm_x4(b_base + np * 16 * HG_BK * 2 + b_off[kh], b[np][0], b[np][1], b[np][2], b[np][3]);
#pragma unroll
            for (int mt = 0; mt < 4; ++mt)
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    hg_mma(acc[mt][nt], a[mt][0], a[mt][1], b[nt >> 1][(nt & 1) * 2]);
                    hg_mma(acc[mt][nt], a[mt][2], a[mt][3], b[nt >> 1][(nt & 1) * 2 + 1]);
                }
        }
        if (next) store(s ^ 1, kt + 1);
        __syncthreads();
    }

    const int g = lane >> 2, t2 = (lane & 3) * 2;
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int m = m0 + wm + mt * 16 + g + half * 8;
            if (m >= M) continue;
            const bool split = gridDim.z > 1;
            float* row = split ? part + ((long long)blockIdx.z * M + m) * N : out + (long long)m * ldo;
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int n = n0 + wn + nt * 8 + t2;
                float v0 = acc[mt][nt][half * 2] * alpha, v1 = acc[mt][nt][half * 2 + 1] * alpha;
                if (split) {
                    // N is even (checked on the host), so a pair never straddles it.
                    if (n < N) *(float2*)(row + n) = make_float2(v0, v1);
                } else if (n + 1 < N) {
                    float2* p = (float2*)(row + n);
                    if (accumulate) { float2 o = *p; v0 += o.x; v1 += o.y; }
                    *p = make_float2(v0, v1);
                } else if (n < N) {
                    row[n] = accumulate ? row[n] + v0 : v0;
                }
            }
        }
    }
}

#define HG_ENTRY(NAME, FMT, WN, MINB)                                                   \
    extern "C" __global__ void __launch_bounds__(256, MINB) NAME(                       \
        const unsigned char* __restrict__ w, const unsigned short* __restrict__ ws,     \
        const unsigned short* __restrict__ x, float* __restrict__ out,                  \
        float* __restrict__ part, int M, int N, int K, int ldo, int accumulate,         \
        float alpha, float wscale, int kps) {                                           \
        hg_gemm<FMT, WN>(w, ws, x, out, part, M, N, K, ldo, accumulate, alpha, wscale, kps); \
    }

// out[m][n] (+)= sum over z = 0, 1, ... of part[z][m][n], in that order, so the
// result does not depend on which split finished first. Two columns a thread.
extern "C" __global__ void hg_splitk_reduce(
    const float* __restrict__ part, float* __restrict__ out,
    int M, int N, int ldo, int splits, int accumulate) {
    const long long i = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const long long mn = (long long)M * N;
    if (i >= mn) return;
    const long long m = i / N, n = i % N;
    float2 v = *(const float2*)(part + i);
    for (int z = 1; z < splits; ++z) {
        const float2 p = *(const float2*)(part + z * mn + i);
        v.x += p.x;
        v.y += p.y;
    }
    float2* o = (float2*)(out + m * ldo + n);
    if (accumulate) {
        const float2 prior = *o;
        v.x = prior.x + v.x;
        v.y = prior.y + v.y;
    }
    *o = v;
}

HG_ENTRY(hg_q8_256, 0, 64, 1)
HG_ENTRY(hg_q8_128, 0, 32, 2)
HG_ENTRY(hg_bf16_256, 1, 64, 1)
HG_ENTRY(hg_bf16_128, 1, 32, 2)
"#;

/// Errors from the weight-only GEMM.
#[derive(Debug)]
pub enum HgemmError {
    /// NVRTC rejected the source.
    Compile(String),
    /// A launch or allocation failed.
    Driver(DriverError),
    /// `k` is not a multiple of [`HGEMM_BK`].
    RaggedContraction { k: usize },
    /// A buffer is shorter than the shape it is launched with.
    Length {
        what: &'static str,
        need: usize,
        got: usize,
    },
    /// `ldo` is narrower than the output rows it must hold, or misaligned.
    Stride { ldo: usize, n: usize },
}

impl core::fmt::Display for HgemmError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Compile(e) => write!(f, "fp16 GEMM compile: {e}"),
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::RaggedContraction { k } => {
                write!(f, "contraction {k} is not a multiple of {HGEMM_BK}")
            }
            Self::Length { what, need, got } => {
                write!(f, "{what}: need {need} elements, buffer has {got}")
            }
            Self::Stride { ldo, n } => write!(f, "output stride {ldo} for {n} columns"),
        }
    }
}

impl core::error::Error for HgemmError {}

impl From<DriverError> for HgemmError {
    fn from(value: DriverError) -> Self {
        Self::Driver(value)
    }
}

fn need(what: &'static str, got: usize, need: usize) -> Result<(), HgemmError> {
    if got < need {
        return Err(HgemmError::Length { what, need, got });
    }
    Ok(())
}

/// The power of two a bf16 tensor is scaled by before its fp16 conversion:
/// the largest that keeps its max-abs `amax` under 2^15, so every value at
/// least 2^-30 of the max stays an fp16 normal and converts exactly. Zero
/// for an all-zero tensor.
pub fn bf16_half_exponent(bf16: &[u8]) -> i32 {
    let amax = bf16
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&b| f32::from_bits(u32::from(u16::from_le_bytes(b)) << 16).abs())
        .filter(|v| v.is_finite())
        .fold(0.0f32, f32::max);
    if amax == 0.0 {
        return 0;
    }
    14 - amax.log2().floor() as i32
}

/// Which stored format a weight operand is in.
#[derive(Debug, Clone, Copy)]
pub enum HalfWeight<'a> {
    /// Split Q8_0: int8 `[N][K]` and fp16 scales `[N][K/32]`.
    Q8_0 {
        q: &'a CudaSlice<i8>,
        scales: &'a CudaSlice<u16>,
    },
    /// bf16 `[N][K]` as raw bytes, converted at `2^exponent` (see
    /// [`bf16_half_exponent`]) and rescaled in the epilogue.
    Bf16 {
        bytes: &'a CudaSlice<u8>,
        exponent: i32,
    },
}

/// How one launch is shaped: the weight-row tile and the split-K factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HgemmPlan {
    /// 256 weight rows a block (one block an SM) rather than 128 (two).
    pub wide: bool,
    /// Contraction splits. Above one, each split writes fp32 partials to a
    /// workspace and a second launch sums them in split order.
    pub splits: usize,
}

/// The compiled GEMM module.
pub struct HgemmKernels {
    q8_256: CudaFunction,
    q8_128: CudaFunction,
    bf16_256: CudaFunction,
    bf16_128: CudaFunction,
    to_half: CudaFunction,
    reduce: CudaFunction,
    sms: usize,
}

/// Most splits a plan uses. Each one costs a `[tokens][n]` fp32 partial
/// written and read back, so past a few the reduction eats the wave gain.
const MAX_SPLITS: usize = 4;

impl HgemmKernels {
    /// Compile for the context's device.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, HgemmError> {
        let ptx = compile(HGEMM_SRC, "hgemm").map_err(HgemmError::Compile)?;
        let module = ctx.load_module(ptx)?;
        let sms = ctx.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )? as usize;
        Ok(Self {
            q8_256: module.load_function("hg_q8_256")?,
            q8_128: module.load_function("hg_q8_128")?,
            bf16_256: module.load_function("hg_bf16_256")?,
            bf16_128: module.load_function("hg_bf16_128")?,
            to_half: module.load_function("hg_to_half")?,
            reduce: module.load_function("hg_splitk_reduce")?,
            sms,
        })
    }

    /// Round `n` fp32 values to fp16 bits.
    pub fn to_half(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<u16>,
        n: usize,
    ) -> Result<(), HgemmError> {
        need("to_half x", x.len(), n)?;
        need("to_half y", y.len(), n)?;
        if n == 0 {
            return Ok(());
        }
        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i = n as i64;
        let mut b = stream.launch_builder(&self.to_half);
        b.arg(x).arg(&mut *y).arg(&n_i);
        // SAFETY: both buffers hold at least `n` elements (checked above) and
        // the grid masks every index past `n`.
        unsafe { b.launch(cfg) }?;
        Ok(())
    }

    /// The launch shape for `tokens x n_rows x k`, from wave arithmetic.
    ///
    /// A 256-row tile halves the activation traffic per weight but runs one
    /// block an SM; the 128-row tile runs two and hides more latency. Measured
    /// at the 27B's shapes (512 tokens, clocks warmed), the narrow tile wins
    /// up to about two and a half waves of the wide one — 10,240 rows at 2.2
    /// waves runs 0.98 ms narrow against 1.06 wide — and loses above it:
    /// 12,288 rows at 2.7 waves is 1.11 ms wide against 1.20 narrow.
    ///
    /// Then the split. A grid of `b` blocks over `slots` resident ones takes
    /// `ceil(b / slots)` rounds, so 160 blocks on 144 slots cost two rounds
    /// for 1.1 rounds of work; splitting the contraction `s` ways makes that
    /// `ceil(s * b / slots) / s` rounds of a full-K block, plus the partials'
    /// round trip, priced at 8% of a round per extra split. Only a grid under
    /// 1.25 waves is considered at all: above it the tail is cheaper than the
    /// model says and every split measured slower (10,240 rows: 0.98 ms whole,
    /// 1.02 / 1.11 / 1.15 split 2 / 3 / 4). Below it the 27B's FFN down
    /// projection goes 1.87 -> 1.61 ms and its 1,024-row k/v 0.20 -> 0.13.
    pub fn plan(&self, tokens: usize, n_rows: usize, k: usize) -> HgemmPlan {
        let m_tiles = tokens.div_ceil(HGEMM_BM);
        let wide = 2 * n_rows.div_ceil(256) * m_tiles >= 5 * self.sms;
        let (blocks, slots) = if wide {
            (n_rows.div_ceil(256) * m_tiles, self.sms)
        } else {
            (n_rows.div_ceil(128) * m_tiles, 2 * self.sms)
        };
        let kblocks = k / HGEMM_BK;
        // Splits start on a scale8 boundary and the partials are stored in
        // pairs; anything else runs whole.
        if blocks == 0
            || 4 * blocks >= 5 * slots
            || !kblocks.is_multiple_of(8)
            || !n_rows.is_multiple_of(2)
        {
            return HgemmPlan { wide, splits: 1 };
        }
        let cost =
            |s: usize| (s * blocks).div_ceil(slots) as f64 / s as f64 + 0.08 * (s - 1) as f64;
        let splits = (1..=MAX_SPLITS)
            .map(|s| Self::effective_splits(kblocks, s))
            .min_by(|&a, &b| cost(a).total_cmp(&cost(b)).then(a.cmp(&b)))
            .unwrap_or(1);
        HgemmPlan { wide, splits }
    }

    /// The split count `s` requested becomes, once each split is rounded up
    /// to a whole number of 8-block scale groups.
    fn effective_splits(kblocks: usize, s: usize) -> usize {
        kblocks.div_ceil(Self::kps(kblocks, s))
    }

    /// K-blocks per split: a multiple of 8, at least 8.
    fn kps(kblocks: usize, splits: usize) -> usize {
        kblocks.div_ceil(splits).next_multiple_of(8).max(8)
    }

    /// The fp32 workspace, in elements, that [`Self::gemm`] needs to split
    /// any `tokens <= max_tokens` launch of this shape: allocate it once at
    /// construction and the forward pass never has to.
    pub fn workspace_elems(&self, max_tokens: usize, n_rows: usize, k: usize) -> usize {
        (1..=max_tokens.div_ceil(HGEMM_BM))
            .map(|m_tiles| {
                let t = (m_tiles * HGEMM_BM).min(max_tokens);
                let p = self.plan(t, n_rows, k);
                if p.splits > 1 {
                    p.splits * t * n_rows
                } else {
                    0
                }
            })
            .max()
            .unwrap_or(0)
    }

    /// `out[m][n] (+)= sum_k x[m][k] * half(w)[n][k]` for `m < tokens`,
    /// `n < n_rows`; `out` rows are `ldo` apart. With `accumulate` the result
    /// is added to what `out` holds (a residual), otherwise it overwrites it.
    ///
    /// Shaped by [`Self::plan`]. A split needs `workspace`; without one, or
    /// with one too short, the launch runs whole — slower, same arithmetic
    /// class, never wrong.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        &self,
        stream: &Arc<CudaStream>,
        w: HalfWeight<'_>,
        x: &CudaSlice<u16>,
        out: &mut CudaSlice<f32>,
        workspace: Option<&mut CudaSlice<f32>>,
        tokens: usize,
        n_rows: usize,
        k: usize,
        ldo: usize,
        accumulate: bool,
    ) -> Result<(), HgemmError> {
        let mut plan = self.plan(tokens, n_rows, k);
        if workspace
            .as_ref()
            .is_none_or(|ws| ws.len() < plan.splits * tokens * n_rows)
        {
            plan.splits = 1;
        }
        self.gemm_with_plan(
            stream, w, x, out, workspace, tokens, n_rows, k, ldo, accumulate, plan,
        )
    }

    /// [`Self::gemm`] with the shape chosen by the caller. Each output's
    /// contraction is the same products in the same per-split order whatever
    /// the tile; a split changes where the fp32 sum is broken, which is why
    /// the differential test covers every plan rather than trusting one.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_with_plan(
        &self,
        stream: &Arc<CudaStream>,
        w: HalfWeight<'_>,
        x: &CudaSlice<u16>,
        out: &mut CudaSlice<f32>,
        workspace: Option<&mut CudaSlice<f32>>,
        tokens: usize,
        n_rows: usize,
        k: usize,
        ldo: usize,
        accumulate: bool,
        plan: HgemmPlan,
    ) -> Result<(), HgemmError> {
        if !k.is_multiple_of(HGEMM_BK) {
            return Err(HgemmError::RaggedContraction { k });
        }
        if ldo < n_rows || !ldo.is_multiple_of(2) {
            return Err(HgemmError::Stride { ldo, n: n_rows });
        }
        need("gemm x", x.len(), tokens * k)?;
        need(
            "gemm out",
            out.len(),
            tokens.saturating_sub(1) * ldo + n_rows,
        )?;
        match w {
            HalfWeight::Q8_0 { q, scales } => {
                need("gemm q8_0 weights", q.len(), n_rows * k)?;
                need("gemm q8_0 scales", scales.len(), n_rows * k / 32)?;
            }
            HalfWeight::Bf16 { bytes, .. } => {
                need("gemm bf16 weights", bytes.len(), n_rows * k * 2)?
            }
        }
        if tokens == 0 || n_rows == 0 {
            return Ok(());
        }
        let kblocks = k / HGEMM_BK;
        let split = plan.splits > 1;
        if split && (!kblocks.is_multiple_of(8) || !n_rows.is_multiple_of(2)) {
            return Err(HgemmError::Stride { ldo, n: n_rows });
        }
        let kps = if split {
            Self::kps(kblocks, plan.splits)
        } else {
            kblocks
        };
        let splits = kblocks.div_ceil(kps);
        let mut workspace = if split {
            let ws = workspace.ok_or(HgemmError::Length {
                what: "gemm split-K workspace",
                need: splits * tokens * n_rows,
                got: 0,
            })?;
            need("gemm split-K workspace", ws.len(), splits * tokens * n_rows)?;
            Some(ws)
        } else {
            None
        };

        let m_tiles = tokens.div_ceil(HGEMM_BM);
        let bn = if plan.wide { 256 } else { 128 };
        let (func, exponent) = match (w, plan.wide) {
            (HalfWeight::Q8_0 { .. }, true) => (&self.q8_256, 0),
            (HalfWeight::Q8_0 { .. }, false) => (&self.q8_128, 0),
            (HalfWeight::Bf16 { exponent, .. }, true) => (&self.bf16_256, exponent),
            (HalfWeight::Bf16 { exponent, .. }, false) => (&self.bf16_128, exponent),
        };
        let cfg = LaunchConfig {
            grid_dim: (n_rows.div_ceil(bn) as u32, m_tiles as u32, splits as u32),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (m_i, n_i, k_i, l_i, a_i, kps_i) = (
            tokens as i32,
            n_rows as i32,
            k as i32,
            ldo as i32,
            i32::from(accumulate),
            kps as i32,
        );
        let wscale = 2f32.powi(exponent);
        let alpha = 2f32.powi(-exponent);
        {
            let mut b = stream.launch_builder(func);
            match w {
                HalfWeight::Q8_0 { q, scales } => b.arg(q).arg(scales),
                // The bf16 entry points never read `ws`; any device pointer
                // will do.
                HalfWeight::Bf16 { bytes, .. } => b.arg(bytes).arg(x),
            };
            b.arg(x).arg(&mut *out);
            // Unsplit launches never touch `part`; any device pointer will do.
            match workspace.as_mut() {
                Some(ws) => b.arg(&mut **ws),
                None => b.arg(x),
            };
            b.arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .arg(&l_i)
                .arg(&a_i)
                .arg(&alpha)
                .arg(&wscale)
                .arg(&kps_i);
            // SAFETY: every buffer was length-checked against the launch shape
            // above — the workspace against `splits * tokens * n_rows` — the
            // kernel masks rows past `tokens` and columns past `n_rows`, and
            // `kps` whole k-blocks per split cover `k` exactly once.
            unsafe { b.launch(cfg) }?;
        }
        if let Some(ws) = workspace {
            let pairs = (tokens * n_rows / 2) as u32;
            let cfg = LaunchConfig {
                grid_dim: (pairs.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let s_i = splits as i32;
            let mut b = stream.launch_builder(&self.reduce);
            b.arg(&*ws)
                .arg(&mut *out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&l_i)
                .arg(&s_i)
                .arg(&a_i);
            // SAFETY: `ws` holds `splits * tokens * n_rows` partials, `n_rows`
            // is even so every pair is in one row, and `out` holds `tokens`
            // rows of stride `ldo >= n_rows`; the grid stops at the last pair.
            unsafe { b.launch(cfg) }?;
        }
        Ok(())
    }
}
