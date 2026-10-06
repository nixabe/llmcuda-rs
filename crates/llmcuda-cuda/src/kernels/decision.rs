//! Device kernels for Clef's joint schema head.
//!
//! The scalar oracle is `llmcuda_kernels::decision::joint_schema_head`; the
//! graph it mirrors is `llama_model_clef::graph::build_head` in llama.cpp's
//! `src/models/clef.cpp` (with `build_head_attn` and `build_head_ffn`), which
//! is itself a port of `JointSchemaHead.forward` in Cloudflare's
//! `joint_schema_model.py`. These kernels are the pieces that graph is built
//! from; `llmcuda-engine`'s `block::decision` strings them together, and its
//! `decision_differential` test checks the whole head against the oracle.
//!
//! # Precision
//!
//! Everything is fp32 — weights (dequantized once at load), activations and
//! accumulation. The head is ~34 MFLOP per prompt token against ~18 GFLOP for
//! the backbone it reads, so a SIMT fp32 GEMM costs little and buys an oracle
//! comparison with no reduced-precision term in its error budget.
//!
//! # LayerNorm is folded into the GEMM that reads it
//!
//! Every `torch.nn.LayerNorm` in the head feeds a linear projection (or a
//! span mean), so [`DecisionKernels::layer_norm`] only produces per-row
//! `(mean, 1/std)` and the GEMM applies `(x - mean) * rstd * g + b` as it
//! loads its X tile. At a 16K-token prompt the normed hidden state would be a
//! 256 MiB buffer that is read once; this way it never exists.
//!
//! # Geometry is per launch
//!
//! Prompt length, question and option counts vary per request, so every
//! launch takes its shape as arguments. The head runs once per request after
//! prefill and is never captured, so host-sized grids are sound here; rule 5
//! of `AGENTS.md` binds captured paths. The launchers take raw device
//! pointers because the engine resolves its workspace once at load and
//! addresses sub-ranges of it (the K and V halves of one projection, the
//! lexical half of the option inputs) on every call.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};

use super::compile;

/// The NVRTC translation unit.
pub const DECISION_SRC: &str = r#"
#define FMT_Q8_0 0
#define FMT_BF16 1
#define FMT_Q6K  2
#define FMT_F16  3
#define FMT_Q4_0 4
#define FMT_Q4K  5
#define FMT_Q5K  6

#define GEMM_GELU    1
#define GEMM_ACC     2
#define GEMM_PARTIAL 4

// NVRTC compiles without the toolkit headers, so no <math.h> INFINITY.
#define DEC_INF __int_as_float(0x7f800000)

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Block-wide sum; blockDim.x a multiple of 32. Every thread gets the total.
// The leading barrier makes back-to-back calls safe on the same `red`.
__device__ float block_sum(float v, float* red) {
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    v = lane < nw ? red[lane] : 0.0f;
    return warp_sum(v);
}

__device__ float block_max(float v, float* red) {
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    v = lane < nw ? red[lane] : -DEC_INF;
    return warp_max(v);
}

// PyTorch's default GELU, the exact erf form (`ggml_gelu_erf`).
__device__ __forceinline__ float gelu_erf(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
}

__device__ __forceinline__ float half_le(const unsigned char* p) {
    unsigned short bits = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(bits));
    return f;
}

// `get_scale_min_k4` from ggml-quants.c, byte for byte.
__device__ __forceinline__ void scale_min_k4(int j, const unsigned char* q, int* sc, int* m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Element `i` of one row in GGUF storage format `fmt`. Operand order follows
// `llmcuda_kernels::quant` exactly and the affine k-quants are spelled with
// `__fmul_rn`/`__fsub_rn` so no FMA contraction separates the two.
__device__ float dequant_element(const unsigned char* row, int fmt, int i) {
    switch (fmt) {
    case FMT_Q8_0: {
        const unsigned char* b = row + (i >> 5) * 34;
        return (float)(signed char)b[2 + (i & 31)] * half_le(b);
    }
    case FMT_BF16: {
        unsigned int bits = (unsigned int)row[2 * i] | ((unsigned int)row[2 * i + 1] << 8);
        return __uint_as_float(bits << 16);
    }
    case FMT_F16:
        return half_le(row + 2 * i);
    case FMT_Q4_0: {
        const unsigned char* b = row + (i >> 5) * 18;
        int e = i & 31;
        int byte = b[2 + (e & 15)];
        int q = e < 16 ? (byte & 0xF) : (byte >> 4);
        return (float)(q - 8) * half_le(b);
    }
    case FMT_Q6K: {
        const unsigned char* b = row + (i >> 8) * 210;
        int e = i & 255, half = e >> 7, r = e & 127, j = r >> 5, l = r & 31;
        const unsigned char* ql = b + half * 64;
        const unsigned char* qh = b + 128 + half * 32;
        const signed char* sc = (const signed char*)(b + 192 + half * 8);
        int lowbyte = ql[l + (j & 1) * 32];
        int low = j < 2 ? (lowbyte & 0xF) : (lowbyte >> 4);
        int high = (qh[l] >> (2 * j)) & 3;
        float d = half_le(b + 208);
        return d * (float)sc[(l >> 4) + 2 * j] * (float)((low | (high << 4)) - 32);
    }
    case FMT_Q4K:
    case FMT_Q5K: {
        const int five = fmt == FMT_Q5K;
        const unsigned char* b = row + (i >> 8) * (five ? 176 : 144);
        int e = i & 255, g = e >> 6, sub = (e >> 5) & 1, l = e & 31;
        float d = half_le(b), dmin = half_le(b + 2);
        int sc, m;
        scale_min_k4(2 * g + sub, b + 4, &sc, &m);
        int byte = b[(five ? 48 : 16) + g * 32 + l];
        int code = sub == 0 ? (byte & 0xF) : (byte >> 4);
        if (five) code += ((b[16 + l] >> (2 * g + sub)) & 1) << 4;
        float d1 = __fmul_rn(d, (float)sc);
        float m1 = __fmul_rn(dmin, (float)m);
        return __fsub_rn(__fmul_rn(d1, (float)code), m1);
    }
    }
    return 0.0f;
}

// Y[m][n] = epilogue(sum_k xin(m, k) * W[n][k]), with xin(m, k) = X[m][k] or,
// with `ln_stats`, the LayerNorm (X[m][k] - mean_m) * rstd_m * g[k] + b[k].
// Epilogue: + bias[n], then exact GELU (GEMM_GELU), then + Y[m][n]
// (GEMM_ACC). With GEMM_PARTIAL the raw sum of this split's K range is
// written to Y + z*M*N (ld N) for `dec_splitk_reduce`.
//
// Requires ldx, ldw, K and the split K length to be multiples of 4 and X, W
// 16-byte aligned: tiles load as float4 along K.
//
// Double-buffered through registers: the next tile's global loads are in
// flight while the current one is multiplied out of shared memory.
template <int BM, int BN, int BK, int TM, int TN>
__device__ __forceinline__ void gemm_nt(
    const float* __restrict__ X, long long ldx,
    const float* __restrict__ W, long long ldw,
    const float* __restrict__ bias,
    float* __restrict__ Y, long long ldy,
    int M, int N, int K, int kchunk,
    const float2* __restrict__ ln_stats, const float* __restrict__ ln_g, const float* __restrict__ ln_b,
    int flags)
{
    constexpr int NT = (BM / TM) * (BN / TN);
    constexpr int KQ = BK / 4;
    constexpr int A_IT = BM * KQ / NT;
    constexpr int B_IT = BN * KQ / NT;
    constexpr int AP = BM + 4;
    constexpr int BP = BN + 4;
    __shared__ __align__(16) float As[2][BK][AP];
    __shared__ __align__(16) float Bs[2][BK][BP];

    const int tid = threadIdx.x;
    const int tx = tid % (BN / TN), ty = tid / (BN / TN);
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int k_begin = blockIdx.z * kchunk;
    const int k_end = min(K, k_begin + kchunk);
    const int ktiles = (k_end - k_begin + BK - 1) / BK;

    float2 stat[A_IT];
    #pragma unroll
    for (int it = 0; it < A_IT; ++it) {
        int row = (tid + it * NT) / KQ;
        stat[it] = make_float2(0.0f, 1.0f);
        if (ln_stats && m0 + row < M) stat[it] = ln_stats[m0 + row];
    }

    float4 ra[A_IT], rb[B_IT];
    auto load = [&](int k0) {
        #pragma unroll
        for (int it = 0; it < A_IT; ++it) {
            int idx = tid + it * NT, row = idx / KQ, kq = (idx % KQ) * 4;
            int gm = m0 + row, gk = k0 + kq;
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (gm < M && gk < k_end) {
                v = *(const float4*)(X + (long long)gm * ldx + gk);
                if (ln_stats) {
                    float mu = stat[it].x, rs = stat[it].y;
                    v.x = (v.x - mu) * rs * ln_g[gk + 0] + ln_b[gk + 0];
                    v.y = (v.y - mu) * rs * ln_g[gk + 1] + ln_b[gk + 1];
                    v.z = (v.z - mu) * rs * ln_g[gk + 2] + ln_b[gk + 2];
                    v.w = (v.w - mu) * rs * ln_g[gk + 3] + ln_b[gk + 3];
                }
            }
            ra[it] = v;
        }
        #pragma unroll
        for (int it = 0; it < B_IT; ++it) {
            int idx = tid + it * NT, row = idx / KQ, kq = (idx % KQ) * 4;
            int gn = n0 + row, gk = k0 + kq;
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (gn < N && gk < k_end) v = *(const float4*)(W + (long long)gn * ldw + gk);
            rb[it] = v;
        }
    };
    auto store = [&](int buf) {
        #pragma unroll
        for (int it = 0; it < A_IT; ++it) {
            int idx = tid + it * NT, row = idx / KQ, kq = (idx % KQ) * 4;
            As[buf][kq + 0][row] = ra[it].x;
            As[buf][kq + 1][row] = ra[it].y;
            As[buf][kq + 2][row] = ra[it].z;
            As[buf][kq + 3][row] = ra[it].w;
        }
        #pragma unroll
        for (int it = 0; it < B_IT; ++it) {
            int idx = tid + it * NT, row = idx / KQ, kq = (idx % KQ) * 4;
            Bs[buf][kq + 0][row] = rb[it].x;
            Bs[buf][kq + 1][row] = rb[it].y;
            Bs[buf][kq + 2][row] = rb[it].z;
            Bs[buf][kq + 3][row] = rb[it].w;
        }
    };

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    if (ktiles > 0) {
        load(k_begin);
        store(0);
    }
    __syncthreads();
    for (int t = 0; t < ktiles; ++t) {
        const int cur = t & 1;
        if (t + 1 < ktiles) load(k_begin + (t + 1) * BK);
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a[TM], b[TN];
            #pragma unroll
            for (int g = 0; g < TM / 4; ++g) {
                float4 v = *(const float4*)&As[cur][kk][g * (BM / (TM / 4)) + ty * 4];
                a[g * 4 + 0] = v.x; a[g * 4 + 1] = v.y; a[g * 4 + 2] = v.z; a[g * 4 + 3] = v.w;
            }
            #pragma unroll
            for (int g = 0; g < TN / 4; ++g) {
                float4 v = *(const float4*)&Bs[cur][kk][g * (BN / (TN / 4)) + tx * 4];
                b[g * 4 + 0] = v.x; b[g * 4 + 1] = v.y; b[g * 4 + 2] = v.z; b[g * 4 + 3] = v.w;
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
        }
        if (t + 1 < ktiles) store(cur ^ 1);
        __syncthreads();
    }

    float* Yp = (flags & GEMM_PARTIAL) ? Y + (long long)blockIdx.z * M * N : Y;
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int m = m0 + (i / 4) * (BM / (TM / 4)) + ty * 4 + (i % 4);
        if (m >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int n = n0 + (j / 4) * (BN / (TN / 4)) + tx * 4 + (j % 4);
            if (n >= N) continue;
            float v = acc[i][j];
            if (flags & GEMM_PARTIAL) {
                Yp[(long long)m * N + n] = v;
            } else {
                if (bias) v += bias[n];
                if (flags & GEMM_GELU) v = gelu_erf(v);
                long long at = (long long)m * ldy + n;
                if (flags & GEMM_ACC) v += Y[at];
                Y[at] = v;
            }
        }
    }
}

// One query tile (16 rows) of one head against a contiguous key range, with
// an online softmax: no [queries][keys] buffer exists at any width. Eight
// lanes own a query; each scores keys `kl + 8j` of a 32-key tile and then
// accumulates dims `32g + 4kl ..+4` of that query's output.
//
// With `part == 0` the block owns every key and writes the normalized output.
// Otherwise it writes its unnormalized accumulator and (max, sum) to `part`
// for `dec_attention_combine`.
template <int DH>
__device__ __forceinline__ void attention_tile(
    const float* __restrict__ Q, long long ldq,
    const float* __restrict__ Kp, long long ldk,
    const float* __restrict__ Vp, long long ldv,
    float* __restrict__ O, long long ldo,
    float* __restrict__ part,
    int nq, int nk, int heads, int chunk, float scale)
{
    constexpr int QT = 16, KT = 32, NT = 128, DQ = DH / 4, DP = DH + 4, KJ = KT / 8, DG = DH / 32;
    __shared__ __align__(16) float Qs[QT][DP];
    __shared__ __align__(16) float Ks[KT][DP];
    __shared__ __align__(16) float Vs[KT][DH];
    __shared__ float Ps[QT][KT];

    const int tid = threadIdx.x, qi = tid >> 3, kl = tid & 7;
    const int q0 = blockIdx.x * QT, h = blockIdx.y, z = blockIdx.z;
    const int kbeg = z * chunk, kend = min(nk, kbeg + chunk);

    for (int idx = tid; idx < QT * DQ; idx += NT) {
        int r = idx / DQ, c = (idx % DQ) * 4;
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (q0 + r < nq) v = *(const float4*)(Q + (long long)(q0 + r) * ldq + h * DH + c);
        Qs[r][c + 0] = v.x * scale; Qs[r][c + 1] = v.y * scale;
        Qs[r][c + 2] = v.z * scale; Qs[r][c + 3] = v.w * scale;
    }

    float m = -DEC_INF, l = 0.0f;
    float o[DG * 4];
    #pragma unroll
    for (int t = 0; t < DG * 4; ++t) o[t] = 0.0f;

    for (int k0 = kbeg; k0 < kend; k0 += KT) {
        __syncthreads();
        for (int idx = tid; idx < KT * DQ; idx += NT) {
            int r = idx / DQ, c = (idx % DQ) * 4;
            float4 kv = make_float4(0.0f, 0.0f, 0.0f, 0.0f), vv = kv;
            if (k0 + r < kend) {
                kv = *(const float4*)(Kp + (long long)(k0 + r) * ldk + h * DH + c);
                vv = *(const float4*)(Vp + (long long)(k0 + r) * ldv + h * DH + c);
            }
            *(float4*)&Ks[r][c] = kv;
            *(float4*)&Vs[r][c] = vv;
        }
        __syncthreads();

        float s[KJ];
        #pragma unroll
        for (int j = 0; j < KJ; ++j) s[j] = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < DH; d += 4) {
            float4 qv = *(const float4*)&Qs[qi][d];
            #pragma unroll
            for (int j = 0; j < KJ; ++j) {
                float4 kv = *(const float4*)&Ks[kl + 8 * j][d];
                s[j] += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
            }
        }
        float mx = -DEC_INF;
        #pragma unroll
        for (int j = 0; j < KJ; ++j) {
            if (k0 + kl + 8 * j >= kend) s[j] = -DEC_INF;
            mx = fmaxf(mx, s[j]);
        }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
        const float mnew = fmaxf(m, mx);
        const float alpha = expf(m - mnew);
        float ps = 0.0f;
        #pragma unroll
        for (int j = 0; j < KJ; ++j) {
            float p = expf(s[j] - mnew);
            Ps[qi][kl + 8 * j] = p;
            ps += p;
        }
        ps += __shfl_xor_sync(0xffffffffu, ps, 1);
        ps += __shfl_xor_sync(0xffffffffu, ps, 2);
        ps += __shfl_xor_sync(0xffffffffu, ps, 4);
        l = l * alpha + ps;
        m = mnew;
        #pragma unroll
        for (int t = 0; t < DG * 4; ++t) o[t] *= alpha;
        // Row `qi` of Ps is written and read by the same eight lanes.
        __syncwarp();
        #pragma unroll 4
        for (int j = 0; j < KT; ++j) {
            float p = Ps[qi][j];
            #pragma unroll
            for (int g = 0; g < DG; ++g) {
                float4 v = *(const float4*)&Vs[j][g * 32 + kl * 4];
                o[g * 4 + 0] += p * v.x; o[g * 4 + 1] += p * v.y;
                o[g * 4 + 2] += p * v.z; o[g * 4 + 3] += p * v.w;
            }
        }
    }

    const int q = q0 + qi;
    if (q >= nq) return;
    if (part == 0) {
        float inv = 1.0f / l;
        float* out = O + (long long)q * ldo + h * DH;
        #pragma unroll
        for (int g = 0; g < DG; ++g)
            #pragma unroll
            for (int t = 0; t < 4; ++t) out[g * 32 + kl * 4 + t] = o[g * 4 + t] * inv;
    } else {
        long long slot = ((long long)z * nq + q) * heads + h;
        float* acc = part + slot * DH;
        #pragma unroll
        for (int g = 0; g < DG; ++g)
            #pragma unroll
            for (int t = 0; t < 4; ++t) acc[g * 32 + kl * 4 + t] = o[g * 4 + t];
        if (kl == 0) {
            float* ml = part + (long long)gridDim.z * nq * heads * DH + slot * 2;
            ml[0] = m;
            ml[1] = l;
        }
    }
}

extern "C" {

// One block per row: two-pass fp32 mean and variance. Writes (mean, rstd) to
// `stats` when non-null and the normed row to `y` when non-null.
__global__ void dec_layer_norm(
    const float* __restrict__ x, long long ldx, int rows, int cols, float eps,
    const float* __restrict__ g, const float* __restrict__ b,
    float2* __restrict__ stats, float* __restrict__ y, long long ldy)
{
    __shared__ float red[32];
    int r = blockIdx.x;
    if (r >= rows) return;
    const float* xr = x + (long long)r * ldx;
    float s = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) s += xr[c];
    const float mean = block_sum(s, red) / (float)cols;
    float v = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float d = xr[c] - mean;
        v += d * d;
    }
    const float var = block_sum(v, red) / (float)cols;
    const float rstd = 1.0f / sqrtf(var + eps);
    if (stats && threadIdx.x == 0) stats[r] = make_float2(mean, rstd);
    if (y) {
        float* yr = y + (long long)r * ldy;
        for (int c = threadIdx.x; c < cols; c += blockDim.x) yr[c] = (xr[c] - mean) * rstd * g[c] + b[c];
    }
}

__global__ void __launch_bounds__(256, 2) dec_gemm_128(
    const float* X, long long ldx, const float* W, long long ldw, const float* bias,
    float* Y, long long ldy, int M, int N, int K, int kchunk,
    const float2* ln_stats, const float* ln_g, const float* ln_b, int flags)
{
    gemm_nt<128, 128, 8, 8, 8>(X, ldx, W, ldw, bias, Y, ldy, M, N, K, kchunk, ln_stats, ln_g, ln_b, flags);
}

__global__ void __launch_bounds__(256) dec_gemm_64(
    const float* X, long long ldx, const float* W, long long ldw, const float* bias,
    float* Y, long long ldy, int M, int N, int K, int kchunk,
    const float2* ln_stats, const float* ln_g, const float* ln_b, int flags)
{
    gemm_nt<64, 64, 16, 4, 4>(X, ldx, W, ldw, bias, Y, ldy, M, N, K, kchunk, ln_stats, ln_g, ln_b, flags);
}

// Sum the split-K partials in split order, then the GEMM epilogue.
__global__ void dec_splitk_reduce(
    const float* __restrict__ part, int splits, int M, int N,
    const float* __restrict__ bias, float* __restrict__ Y, long long ldy, int flags)
{
    long long total = (long long)M * N;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
         i += (long long)gridDim.x * blockDim.x) {
        float v = 0.0f;
        for (int z = 0; z < splits; ++z) v += part[z * total + i];
        int m = (int)(i / N), n = (int)(i % N);
        if (bias) v += bias[n];
        if (flags & GEMM_GELU) v = gelu_erf(v);
        long long at = (long long)m * ldy + n;
        if (flags & GEMM_ACC) v += Y[at];
        Y[at] = v;
    }
}

__global__ void __launch_bounds__(128) dec_attention_32(
    const float* Q, long long ldq, const float* K, long long ldk, const float* V, long long ldv,
    float* O, long long ldo, float* part, int nq, int nk, int heads, int chunk, float scale)
{
    attention_tile<32>(Q, ldq, K, ldk, V, ldv, O, ldo, part, nq, nk, heads, chunk, scale);
}

__global__ void __launch_bounds__(128) dec_attention_64(
    const float* Q, long long ldq, const float* K, long long ldk, const float* V, long long ldv,
    float* O, long long ldo, float* part, int nq, int nk, int heads, int chunk, float scale)
{
    attention_tile<64>(Q, ldq, K, ldk, V, ldv, O, ldo, part, nq, nk, heads, chunk, scale);
}

__global__ void __launch_bounds__(128) dec_attention_128(
    const float* Q, long long ldq, const float* K, long long ldk, const float* V, long long ldv,
    float* O, long long ldo, float* part, int nq, int nk, int heads, int chunk, float scale)
{
    attention_tile<128>(Q, ldq, K, ldk, V, ldv, O, ldo, part, nq, nk, heads, chunk, scale);
}

// Merge key-split partials: grid (nq, heads), one thread per head dim.
__global__ void dec_attention_combine(
    const float* __restrict__ part, int splits, int nq, int heads, int dh,
    float* __restrict__ O, long long ldo)
{
    int q = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    if (d >= dh) return;
    const float* ml = part + (long long)splits * nq * heads * dh;
    float mx = -DEC_INF;
    for (int z = 0; z < splits; ++z) mx = fmaxf(mx, ml[(((long long)z * nq + q) * heads + h) * 2]);
    float num = 0.0f, den = 0.0f;
    for (int z = 0; z < splits; ++z) {
        long long slot = ((long long)z * nq + q) * heads + h;
        float w = expf(ml[slot * 2] - mx);
        den += ml[slot * 2 + 1] * w;
        num += part[slot * dh + d] * w;
    }
    O[(long long)q * ldo + h * dh + d] = num / den;
}

// out[s][c] = mean over rows t of span s of LayerNorm(x[t])[c], with the
// rows' (mean, rstd) from `dec_layer_norm`. Spans are (start, end) pairs at
// `spans + s * stride`. Grid (ceil(cols / blockDim), nspans).
__global__ void dec_span_mean(
    const float* __restrict__ x, long long ldx, int cols,
    const float2* __restrict__ stats, const float* __restrict__ g, const float* __restrict__ b,
    const int* __restrict__ spans, int stride, int nspans,
    float* __restrict__ out, long long ldo)
{
    int s = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nspans || c >= cols) return;
    int start = spans[s * stride], end = spans[s * stride + 1];
    float acc = 0.0f;
    for (int t = start; t < end; ++t) {
        float2 st = stats[t];
        acc += (x[(long long)t * ldx + c] - st.x) * st.y;
    }
    out[(long long)s * ldo + c] = acc / (float)(end - start) * g[c] + b[c];
}

// out[s][c] = mean over the span's token ids of row `id` of an embedding
// table stored as raw GGUF rows of `row_bytes` in format `fmt`.
__global__ void dec_lexical(
    const unsigned char* __restrict__ table, int fmt, long long row_bytes, int cols,
    const int* __restrict__ tokens, const int* __restrict__ spans, int stride, int nspans,
    float* __restrict__ out, long long ldo)
{
    int s = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nspans || c >= cols) return;
    int start = spans[s * stride], end = spans[s * stride + 1];
    float acc = 0.0f;
    for (int t = start; t < end; ++t) acc += dequant_element(table + (long long)tokens[t] * row_bytes, fmt, c);
    out[(long long)s * ldo + c] = acc / (float)(end - start);
}

// y[o][c] += src[owner(o)][c], owner at `spans[o * stride + 2]`.
__global__ void dec_add_owner_rows(
    float* __restrict__ y, long long ldy, const float* __restrict__ src, long long lds,
    const int* __restrict__ spans, int stride, int n, int cols)
{
    int o = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n || c >= cols) return;
    int q = spans[o * stride + 2];
    y[(long long)o * ldy + c] += src[(long long)q * lds + c];
}

// One block per question: softmax over its own options of
// options[o] . base[q] * scale, the weighted option summary, and
// fields[q] = base[q] + LN(summary) + gproj + type_embd[type(q)].
// `weights` is [n_opts] scratch; each option's entry is touched only by its
// owner's block. Dynamic shared memory: `width` floats.
__global__ void dec_question_fields(
    const float* __restrict__ opts, long long ldo, int n_opts,
    const int* __restrict__ ospans, int ostride,
    const float* __restrict__ base, long long ldb,
    const float* __restrict__ gproj, const float* __restrict__ type_embd,
    const int* __restrict__ qspans, int qstride,
    const float* __restrict__ sg, const float* __restrict__ sb, float eps,
    float* __restrict__ weights, float* __restrict__ fields, long long ldf,
    int width, float scale)
{
    extern __shared__ float summary[];
    __shared__ float red[32];
    const int q = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    const float* bq = base + (long long)q * ldb;

    for (int o = warp; o < n_opts; o += nw) {
        if (ospans[o * ostride + 2] != q) continue;
        const float* oo = opts + (long long)o * ldo;
        float d = 0.0f;
        for (int c = lane; c < width; c += 32) d += oo[c] * bq[c];
        d = warp_sum(d);
        if (lane == 0) weights[o] = d * scale;
    }
    __syncthreads();
    float mx = -DEC_INF;
    for (int o = tid; o < n_opts; o += blockDim.x)
        if (ospans[o * ostride + 2] == q) mx = fmaxf(mx, weights[o]);
    mx = block_max(mx, red);
    float se = 0.0f;
    for (int o = tid; o < n_opts; o += blockDim.x)
        if (ospans[o * ostride + 2] == q) se += expf(weights[o] - mx);
    se = block_sum(se, red);
    for (int o = tid; o < n_opts; o += blockDim.x)
        if (ospans[o * ostride + 2] == q) weights[o] = expf(weights[o] - mx) / se;
    __syncthreads();

    float s = 0.0f;
    for (int c = tid; c < width; c += blockDim.x) {
        float acc = 0.0f;
        for (int o = 0; o < n_opts; ++o)
            if (ospans[o * ostride + 2] == q) acc += weights[o] * opts[(long long)o * ldo + c];
        summary[c] = acc;
        s += acc;
    }
    const float mean = block_sum(s, red) / (float)width;
    float v = 0.0f;
    for (int c = tid; c < width; c += blockDim.x) {
        float d = summary[c] - mean;
        v += d * d;
    }
    const float rstd = 1.0f / sqrtf(block_sum(v, red) / (float)width + eps);
    const int type = qspans[q * qstride + 2];
    float* fq = fields + (long long)q * ldf;
    for (int c = tid; c < width; c += blockDim.x) {
        float ln = (summary[c] - mean) * rstd * sg[c] + sb[c];
        fq[c] = bq[c] + ln + gproj[c] + type_embd[(long long)type * width + c];
    }
}

// One block per option: the scorer's input features
// [field, option, field * option, |field - option|], the cosine of field and
// option, and the lexical prior cos(lexical, question mean + global).
// aux[o] = (prior, cosine).
__global__ void dec_features(
    const float* __restrict__ fieldn, long long ldf,
    const float* __restrict__ optn, long long ldo,
    const int* __restrict__ ospans, int ostride, int n_opts,
    const float* __restrict__ qmean, long long ldq, const float* __restrict__ global,
    const float* __restrict__ lexical, long long ldl,
    int hidden, int width, float* __restrict__ feat, long long ldfeat, float* __restrict__ aux)
{
    __shared__ float red[32];
    const int o = blockIdx.x, tid = threadIdx.x;
    if (o >= n_opts) return;
    const int q = ospans[o * ostride + 2];
    const float* f = fieldn + (long long)q * ldf;
    const float* x = optn + (long long)o * ldo;
    float* out = feat + (long long)o * ldfeat;
    float ff = 0.0f, xx = 0.0f, fx = 0.0f;
    for (int c = tid; c < width; c += blockDim.x) {
        float a = f[c], b = x[c];
        ff += a * a; xx += b * b; fx += a * b;
        out[c] = a;
        out[width + c] = b;
        out[2 * width + c] = a * b;
        out[3 * width + c] = fabsf(a - b);
    }
    const float* qm = qmean + (long long)q * ldq;
    const float* lx = lexical + (long long)o * ldl;
    float aa = 0.0f, ll = 0.0f, al = 0.0f;
    for (int c = tid; c < hidden; c += blockDim.x) {
        float a = qm[c] + global[c], b = lx[c];
        aa += a * a; ll += b * b; al += a * b;
    }
    ff = block_sum(ff, red); xx = block_sum(xx, red); fx = block_sum(fx, red);
    aa = block_sum(aa, red); ll = block_sum(ll, red); al = block_sum(al, red);
    if (tid == 0) {
        float cosine = fx / (fmaxf(sqrtf(ff), 1e-12f) * fmaxf(sqrtf(xx), 1e-12f));
        float prior = al / (fmaxf(sqrtf(ll), 1e-12f) * fmaxf(sqrtf(aa), 1e-12f));
        aux[o * 2] = prior;
        aux[o * 2 + 1] = cosine;
    }
}

// One warp per option: residual = scorer_out(h) and the final combination
// prior_scale * prior + gate * (joint_scale * cosine + residual).
__global__ void dec_final_scores(
    const float* __restrict__ h, long long ldh, int n_opts, int width,
    const float* __restrict__ w_out, float b_out, const float* __restrict__ aux,
    float prior_scale, float joint_scale, float gate, float* __restrict__ scores)
{
    int o = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (o >= n_opts) return;
    const float* hr = h + (long long)o * ldh;
    float d = 0.0f;
    for (int c = lane; c < width; c += 32) d += hr[c] * w_out[c];
    d = warp_sum(d);
    if (lane == 0) {
        float joint = joint_scale * aux[o * 2 + 1] + (d + b_out);
        scores[o] = prior_scale * aux[o * 2] + gate * joint;
    }
}

}
"#;

/// Storage format of the embedding table the lexical gather reads, as the
/// kernel's `fmt` argument. Matches `lm_head::HeadFormat` one to one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum RowFormat {
    /// 34-byte blocks of 32 int8 codes and an fp16 delta.
    Q8_0 = 0,
    /// Dense bf16.
    Bf16 = 1,
    /// 210-byte k-quant superblocks of 256 6-bit codes.
    Q6K = 2,
    /// Dense IEEE half.
    F16 = 3,
    /// 18-byte blocks of 32 4-bit codes and an fp16 delta.
    Q4_0 = 4,
    /// 144-byte affine k-quant superblocks of 256 4-bit codes.
    Q4K = 5,
    /// 176-byte affine k-quant superblocks of 256 5-bit codes.
    Q5K = 6,
}

impl RowFormat {
    /// `(elements, bytes)` of one storage block.
    pub const fn block(self) -> (usize, usize) {
        match self {
            Self::Q8_0 => (32, 34),
            Self::Bf16 | Self::F16 => (1, 2),
            Self::Q6K => (256, 210),
            Self::Q4_0 => (32, 18),
            Self::Q4K => (256, 144),
            Self::Q5K => (256, 176),
        }
    }

    /// Bytes of one row of `cols` elements, or `None` if `cols` is not a
    /// whole number of blocks.
    pub const fn row_bytes(self, cols: usize) -> Option<usize> {
        let (elements, bytes) = self.block();
        if cols.is_multiple_of(elements) {
            Some(cols / elements * bytes)
        } else {
            None
        }
    }

    /// The format of an LM-head tensor.
    pub const fn from_head(format: super::lm_head::HeadFormat) -> Self {
        use super::lm_head::HeadFormat as H;
        match format {
            H::Q8_0 => Self::Q8_0,
            H::Bf16 => Self::Bf16,
            H::Q6K => Self::Q6K,
            H::F16 => Self::F16,
            H::Q4_0 => Self::Q4_0,
            H::Q4K => Self::Q4K,
            H::Q5K => Self::Q5K,
        }
    }
}

/// Loading or launching a decision-head kernel failed.
#[derive(Debug)]
pub enum DecisionKernelError {
    /// NVRTC rejected the source, or the module failed to load.
    Compile(String),
    /// The driver failed.
    Driver(DriverError),
    /// A launch's geometry is outside what the kernels support.
    Shape(String),
}

impl std::fmt::Display for DecisionKernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Compile(e) => write!(f, "decision kernel compilation failed: {e}"),
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::Shape(e) => write!(f, "decision kernel geometry: {e}"),
        }
    }
}

impl std::error::Error for DecisionKernelError {}

impl From<DriverError> for DecisionKernelError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

fn shape(msg: String) -> DecisionKernelError {
    DecisionKernelError::Shape(msg)
}

/// A device pointer, as the raw `CUdeviceptr` the launchers take.
pub type DevicePtr = u64;

/// Per-row LayerNorm folded into a GEMM's X load: `stats` is `[M]` `(mean,
/// rstd)` pairs from [`DecisionKernels::layer_norm`], `weight` and `bias` are
/// `[K]`.
#[derive(Debug, Clone, Copy)]
pub struct RowNorm {
    /// `[M]` float2.
    pub stats: DevicePtr,
    /// `[K]` scale.
    pub weight: DevicePtr,
    /// `[K]` shift.
    pub bias: DevicePtr,
}

/// `Y[m][n] (+)= act(sum_k xin(m, k) * W[n][k] + bias[n])`: both operands
/// K-contiguous, `W` row-major `[N][K]` like a `torch.nn.Linear` weight.
#[derive(Debug, Clone, Copy)]
pub struct Gemm {
    /// `[M][ldx]`, 16-byte aligned.
    pub x: DevicePtr,
    /// Row stride of `x` in elements, a multiple of 4.
    pub ldx: usize,
    /// `[N][ldw]`, 16-byte aligned.
    pub w: DevicePtr,
    /// Row stride of `w` in elements, a multiple of 4.
    pub ldw: usize,
    /// `[N]`, or 0 for none.
    pub bias: DevicePtr,
    /// `[M][ldy]`.
    pub y: DevicePtr,
    /// Row stride of `y` in elements.
    pub ldy: usize,
    /// Rows of X and Y.
    pub m: usize,
    /// Rows of W, columns of Y.
    pub n: usize,
    /// Reduction length, a multiple of 4.
    pub k: usize,
    /// LayerNorm applied to X rows on load.
    pub norm: Option<RowNorm>,
    /// Apply exact GELU after the bias.
    pub gelu: bool,
    /// Add into `y` instead of overwriting it (a residual connection).
    pub accumulate: bool,
}

/// Unmasked multi-head attention: `o[q] = softmax(Q[q] K^T / sqrt(dh)) V`
/// per head, heads laid out contiguously across each row.
#[derive(Debug, Clone, Copy)]
pub struct Attention {
    /// `[nq][ldq]`.
    pub q: DevicePtr,
    /// Row stride of `q`.
    pub ldq: usize,
    /// `[nk][ldk]`.
    pub k: DevicePtr,
    /// Row stride of `k`.
    pub ldk: usize,
    /// `[nk][ldv]`.
    pub v: DevicePtr,
    /// Row stride of `v`.
    pub ldv: usize,
    /// `[nq][ldo]`, written.
    pub o: DevicePtr,
    /// Row stride of `o`.
    pub ldo: usize,
    /// Query rows.
    pub nq: usize,
    /// Key/value rows.
    pub nk: usize,
    /// Heads; the head dim is `width / heads`.
    pub heads: usize,
    /// Per-head width: 32, 64 or 128.
    pub head_dim: usize,
}

/// Device scratch a launch may use, `[capacity]` f32.
#[derive(Debug, Clone, Copy)]
pub struct Scratch {
    /// Base pointer.
    pub ptr: DevicePtr,
    /// Elements available.
    pub capacity: usize,
}

/// The compiled head kernels.
pub struct DecisionKernels {
    layer_norm: CudaFunction,
    gemm_128: CudaFunction,
    gemm_64: CudaFunction,
    splitk_reduce: CudaFunction,
    attention_32: CudaFunction,
    attention_64: CudaFunction,
    attention_128: CudaFunction,
    attention_combine: CudaFunction,
    span_mean: CudaFunction,
    lexical: CudaFunction,
    add_owner_rows: CudaFunction,
    question_fields: CudaFunction,
    features: CudaFunction,
    final_scores: CudaFunction,
    sm_count: usize,
}

const ROW_THREADS: u32 = 256;
const ATTN_QUERIES: usize = 16;
const ATTN_KEYS: usize = 32;

fn grid_1d(n: usize, threads: u32) -> u32 {
    n.div_ceil(threads as usize) as u32
}

fn i32_of(v: usize, what: &str) -> Result<i32, DecisionKernelError> {
    i32::try_from(v).map_err(|_| shape(format!("{what} = {v} exceeds i32")))
}

impl DecisionKernels {
    /// Compile the head's kernels for `ctx`.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, DecisionKernelError> {
        let ptx = compile(DECISION_SRC, "decision_head").map_err(DecisionKernelError::Compile)?;
        let module = ctx.load_module(ptx)?;
        let f = |name: &str| module.load_function(name);
        let sm_count = ctx
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )?
            .max(1) as usize;
        Ok(Self {
            layer_norm: f("dec_layer_norm")?,
            gemm_128: f("dec_gemm_128")?,
            gemm_64: f("dec_gemm_64")?,
            splitk_reduce: f("dec_splitk_reduce")?,
            attention_32: f("dec_attention_32")?,
            attention_64: f("dec_attention_64")?,
            attention_128: f("dec_attention_128")?,
            attention_combine: f("dec_attention_combine")?,
            span_mean: f("dec_span_mean")?,
            lexical: f("dec_lexical")?,
            add_owner_rows: f("dec_add_owner_rows")?,
            question_fields: f("dec_question_fields")?,
            features: f("dec_features")?,
            final_scores: f("dec_final_scores")?,
            sm_count,
        })
    }

    /// Streaming multiprocessors on this device; sizes the key and K splits.
    pub fn sm_count(&self) -> usize {
        self.sm_count
    }

    /// Scratch elements [`Self::attention`] needs to split `nq` queries over
    /// `splits` key ranges.
    pub const fn attention_scratch(
        nq: usize,
        heads: usize,
        head_dim: usize,
        splits: usize,
    ) -> usize {
        splits * nq * heads * (head_dim + 2)
    }

    /// Row LayerNorm statistics, and optionally the normed rows.
    ///
    /// Writes `(mean, rstd)` per row to `stats` if non-zero and
    /// `(x - mean) * rstd * weight + bias` to `y` if non-zero (`weight` and
    /// `bias` are read only then).
    ///
    /// # Safety
    /// Every non-zero pointer must address a live device allocation covering
    /// the rows and columns named.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn layer_norm(
        &self,
        stream: &Arc<CudaStream>,
        x: DevicePtr,
        ldx: usize,
        rows: usize,
        cols: usize,
        eps: f32,
        weight: DevicePtr,
        bias: DevicePtr,
        stats: DevicePtr,
        y: DevicePtr,
        ldy: usize,
    ) -> Result<(), DecisionKernelError> {
        if rows == 0 {
            return Ok(());
        }
        let (rows_i, cols_i) = (i32_of(rows, "rows")?, i32_of(cols, "cols")?);
        let (ldx, ldy) = (ldx as i64, ldy as i64);
        // SAFETY: the caller guarantees the pointers cover the shape.
        unsafe {
            stream
                .launch_builder(&self.layer_norm)
                .arg(&x)
                .arg(&ldx)
                .arg(&rows_i)
                .arg(&cols_i)
                .arg(&eps)
                .arg(&weight)
                .arg(&bias)
                .arg(&stats)
                .arg(&y)
                .arg(&ldy)
                .launch(LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (ROW_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// Run one [`Gemm`], splitting K across blocks into `scratch` when the
    /// output alone would leave most of the device idle.
    ///
    /// # Safety
    /// Every pointer in `g` and `scratch` must address live device memory
    /// covering the shape; `y` must not alias `x`, `w` or `scratch`.
    pub unsafe fn gemm(
        &self,
        stream: &Arc<CudaStream>,
        g: &Gemm,
        scratch: Scratch,
    ) -> Result<(), DecisionKernelError> {
        if g.m == 0 || g.n == 0 {
            return Ok(());
        }
        if !g.k.is_multiple_of(4) || !g.ldx.is_multiple_of(4) || !g.ldw.is_multiple_of(4) {
            return Err(shape(format!(
                "gemm K {} / ldx {} / ldw {} must be multiples of 4",
                g.k, g.ldx, g.ldw
            )));
        }
        if !g.x.is_multiple_of(16) || !g.w.is_multiple_of(16) {
            return Err(shape("gemm operands must be 16-byte aligned".into()));
        }
        let large = g.m >= 512;
        let (bm, bn, bk) = if large { (128, 128, 8) } else { (64, 64, 16) };
        let tiles = g.m.div_ceil(bm) * g.n.div_ceil(bn);
        // Split K only when the output tiles cannot fill the device and the
        // reduction is long enough for each split to stay a real loop.
        let mut splits = 1;
        if tiles < self.sm_count && g.k >= 512 {
            let want = (2 * self.sm_count).div_ceil(tiles).min(g.k / 256).min(16);
            let fit = scratch.capacity / (g.m * g.n);
            splits = want.min(fit).max(1);
        }
        let kchunk = g.k.div_ceil(splits).next_multiple_of(bk);
        let splits = g.k.div_ceil(kchunk).max(1);
        let flags = i32::from(g.gelu) | (i32::from(g.accumulate) << 1);
        let (m, n, k) = (
            i32_of(g.m, "gemm M")?,
            i32_of(g.n, "gemm N")?,
            i32_of(g.k, "gemm K")?,
        );
        let kchunk_i = i32_of(kchunk, "gemm K chunk")?;
        let (ln_stats, ln_g, ln_b) = g.norm.map_or((0, 0, 0), |r| (r.stats, r.weight, r.bias));
        let (ldx, ldw) = (g.ldx as i64, g.ldw as i64);
        let partial = splits > 1;
        let (y, ldy, bias, launch_flags) = if partial {
            (scratch.ptr, g.n as i64, 0u64, 4i32)
        } else {
            (g.y, g.ldy as i64, g.bias, flags)
        };
        let func = if large { &self.gemm_128 } else { &self.gemm_64 };
        let cfg = LaunchConfig {
            grid_dim: (
                g.n.div_ceil(bn) as u32,
                g.m.div_ceil(bm) as u32,
                splits as u32,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the caller guarantees the operands cover the shape; the
        // split partials need `splits * M * N <= scratch.capacity`, which
        // `fit` enforces.
        unsafe {
            stream
                .launch_builder(func)
                .arg(&g.x)
                .arg(&ldx)
                .arg(&g.w)
                .arg(&ldw)
                .arg(&bias)
                .arg(&y)
                .arg(&ldy)
                .arg(&m)
                .arg(&n)
                .arg(&k)
                .arg(&kchunk_i)
                .arg(&ln_stats)
                .arg(&ln_g)
                .arg(&ln_b)
                .arg(&launch_flags)
                .launch(cfg)?;
        }
        if partial {
            let total = g.m * g.n;
            let splits_i = splits as i32;
            let ldy = g.ldy as i64;
            // SAFETY: as above; the partials were just written.
            unsafe {
                stream
                    .launch_builder(&self.splitk_reduce)
                    .arg(&scratch.ptr)
                    .arg(&splits_i)
                    .arg(&m)
                    .arg(&n)
                    .arg(&g.bias)
                    .arg(&g.y)
                    .arg(&ldy)
                    .arg(&flags)
                    .launch(LaunchConfig {
                        grid_dim: (grid_1d(total, 256).min(4096), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
        }
        Ok(())
    }

    /// Unmasked multi-head attention, split over keys into `scratch` when the
    /// query tiles alone would leave the device idle.
    ///
    /// # Safety
    /// Every pointer in `a` and `scratch` must address live device memory
    /// covering the shape; `o` must not alias the inputs.
    pub unsafe fn attention(
        &self,
        stream: &Arc<CudaStream>,
        a: &Attention,
        scratch: Scratch,
    ) -> Result<(), DecisionKernelError> {
        if a.nq == 0 {
            return Ok(());
        }
        if a.nk == 0 {
            return Err(shape("attention over zero keys".into()));
        }
        let func = match a.head_dim {
            32 => &self.attention_32,
            64 => &self.attention_64,
            128 => &self.attention_128,
            d => {
                return Err(shape(format!(
                    "attention head dim {d} (want 32, 64 or 128)"
                )));
            }
        };
        if [a.ldq, a.ldk, a.ldv].iter().any(|ld| !ld.is_multiple_of(4))
            || [a.q, a.k, a.v].iter().any(|p| !p.is_multiple_of(16))
        {
            return Err(shape("attention operands must be float4-aligned".into()));
        }
        let qtiles = a.nq.div_ceil(ATTN_QUERIES);
        let blocks = qtiles * a.heads;
        let fit = scratch.capacity / Self::attention_scratch(a.nq, a.heads, a.head_dim, 1);
        let want = (4 * self.sm_count)
            .div_ceil(blocks)
            .min(a.nk.div_ceil(ATTN_KEYS))
            .min(fit)
            .max(1);
        let chunk = a.nk.div_ceil(want).next_multiple_of(ATTN_KEYS);
        let splits = a.nk.div_ceil(chunk);
        let part = if splits > 1 { scratch.ptr } else { 0u64 };
        let (nq, nk, heads, chunk_i) = (
            i32_of(a.nq, "attention queries")?,
            i32_of(a.nk, "attention keys")?,
            i32_of(a.heads, "attention heads")?,
            i32_of(chunk, "attention chunk")?,
        );
        let scale = 1.0 / (a.head_dim as f32).sqrt();
        let (ldq, ldk, ldv, ldo) = (a.ldq as i64, a.ldk as i64, a.ldv as i64, a.ldo as i64);
        // SAFETY: the caller guarantees the operands; `fit` bounds the
        // partials by the scratch capacity.
        unsafe {
            stream
                .launch_builder(func)
                .arg(&a.q)
                .arg(&ldq)
                .arg(&a.k)
                .arg(&ldk)
                .arg(&a.v)
                .arg(&ldv)
                .arg(&a.o)
                .arg(&ldo)
                .arg(&part)
                .arg(&nq)
                .arg(&nk)
                .arg(&heads)
                .arg(&chunk_i)
                .arg(&scale)
                .launch(LaunchConfig {
                    grid_dim: (qtiles as u32, a.heads as u32, splits as u32),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        if splits > 1 {
            let (splits_i, dh) = (splits as i32, a.head_dim as i32);
            // SAFETY: the partials were just written for this shape.
            unsafe {
                stream
                    .launch_builder(&self.attention_combine)
                    .arg(&scratch.ptr)
                    .arg(&splits_i)
                    .arg(&nq)
                    .arg(&heads)
                    .arg(&dh)
                    .arg(&a.o)
                    .arg(&ldo)
                    .launch(LaunchConfig {
                        grid_dim: (a.nq as u32, a.heads as u32, 1),
                        block_dim: (a.head_dim as u32, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
        }
        Ok(())
    }

    /// `out[s] = mean over span s of LayerNorm(x[t])`, spans as `(start,
    /// end)` i32 pairs every `stride` ints.
    ///
    /// # Safety
    /// Every pointer must address live device memory covering the shape, and
    /// every span must be non-empty and inside `x` and `stats`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn span_mean(
        &self,
        stream: &Arc<CudaStream>,
        x: DevicePtr,
        ldx: usize,
        cols: usize,
        norm: RowNorm,
        spans: DevicePtr,
        stride: usize,
        nspans: usize,
        out: DevicePtr,
        ldo: usize,
    ) -> Result<(), DecisionKernelError> {
        if nspans == 0 {
            return Ok(());
        }
        let (cols_i, stride_i, n_i) = (
            i32_of(cols, "span cols")?,
            i32_of(stride, "span stride")?,
            i32_of(nspans, "spans")?,
        );
        let (ldx, ldo) = (ldx as i64, ldo as i64);
        // SAFETY: the caller guarantees the operands.
        unsafe {
            stream
                .launch_builder(&self.span_mean)
                .arg(&x)
                .arg(&ldx)
                .arg(&cols_i)
                .arg(&norm.stats)
                .arg(&norm.weight)
                .arg(&norm.bias)
                .arg(&spans)
                .arg(&stride_i)
                .arg(&n_i)
                .arg(&out)
                .arg(&ldo)
                .launch(LaunchConfig {
                    grid_dim: (grid_1d(cols, ROW_THREADS), nspans as u32, 1),
                    block_dim: (ROW_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// `out[s] = mean over span s of table row tokens[t]`, rows stored in
    /// GGUF format `fmt`.
    ///
    /// # Safety
    /// Every pointer must address live device memory covering the shape, and
    /// every token id inside a span must name a row of `table`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn lexical(
        &self,
        stream: &Arc<CudaStream>,
        table: DevicePtr,
        fmt: RowFormat,
        cols: usize,
        tokens: DevicePtr,
        spans: DevicePtr,
        stride: usize,
        nspans: usize,
        out: DevicePtr,
        ldo: usize,
    ) -> Result<(), DecisionKernelError> {
        if nspans == 0 {
            return Ok(());
        }
        let row_bytes = fmt
            .row_bytes(cols)
            .ok_or_else(|| shape(format!("{cols} columns are not whole {fmt:?} blocks")))?
            as i64;
        let (fmt_i, cols_i, stride_i, n_i) = (
            fmt as i32,
            i32_of(cols, "lexical cols")?,
            i32_of(stride, "span stride")?,
            i32_of(nspans, "spans")?,
        );
        let ldo = ldo as i64;
        // SAFETY: the caller guarantees the operands.
        unsafe {
            stream
                .launch_builder(&self.lexical)
                .arg(&table)
                .arg(&fmt_i)
                .arg(&row_bytes)
                .arg(&cols_i)
                .arg(&tokens)
                .arg(&spans)
                .arg(&stride_i)
                .arg(&n_i)
                .arg(&out)
                .arg(&ldo)
                .launch(LaunchConfig {
                    grid_dim: (grid_1d(cols, ROW_THREADS), nspans as u32, 1),
                    block_dim: (ROW_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// `y[o] += src[owner(o)]`, the owner index at `spans[o * stride + 2]`.
    ///
    /// # Safety
    /// Every pointer must address live device memory covering the shape, and
    /// every owner must name a row of `src`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn add_owner_rows(
        &self,
        stream: &Arc<CudaStream>,
        y: DevicePtr,
        ldy: usize,
        src: DevicePtr,
        lds: usize,
        spans: DevicePtr,
        stride: usize,
        n: usize,
        cols: usize,
    ) -> Result<(), DecisionKernelError> {
        if n == 0 {
            return Ok(());
        }
        let (stride_i, n_i, cols_i) = (
            i32_of(stride, "span stride")?,
            i32_of(n, "rows")?,
            i32_of(cols, "cols")?,
        );
        let (ldy, lds) = (ldy as i64, lds as i64);
        // SAFETY: the caller guarantees the operands.
        unsafe {
            stream
                .launch_builder(&self.add_owner_rows)
                .arg(&y)
                .arg(&ldy)
                .arg(&src)
                .arg(&lds)
                .arg(&spans)
                .arg(&stride_i)
                .arg(&n_i)
                .arg(&cols_i)
                .launch(LaunchConfig {
                    grid_dim: (grid_1d(cols, ROW_THREADS), n as u32, 1),
                    block_dim: (ROW_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// Each question's option summary and its field vector (see the kernel).
    ///
    /// # Safety
    /// Every pointer must address live device memory covering the shape;
    /// every option's owner must be `< nq` and every question type `< 3`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn question_fields(
        &self,
        stream: &Arc<CudaStream>,
        opts: DevicePtr,
        ldo: usize,
        n_opts: usize,
        ospans: DevicePtr,
        ostride: usize,
        base: DevicePtr,
        ldb: usize,
        gproj: DevicePtr,
        type_embd: DevicePtr,
        qspans: DevicePtr,
        qstride: usize,
        nq: usize,
        summary_norm: (DevicePtr, DevicePtr),
        eps: f32,
        weights: DevicePtr,
        fields: DevicePtr,
        ldf: usize,
        width: usize,
    ) -> Result<(), DecisionKernelError> {
        if nq == 0 {
            return Ok(());
        }
        let (n_i, os_i, qs_i, w_i) = (
            i32_of(n_opts, "options")?,
            i32_of(ostride, "option stride")?,
            i32_of(qstride, "question stride")?,
            i32_of(width, "width")?,
        );
        let (ldo, ldb, ldf) = (ldo as i64, ldb as i64, ldf as i64);
        let scale = 1.0 / (width as f32).sqrt();
        let smem = u32::try_from(width * 4).map_err(|_| shape("width".into()))?;
        // SAFETY: the caller guarantees the operands.
        unsafe {
            stream
                .launch_builder(&self.question_fields)
                .arg(&opts)
                .arg(&ldo)
                .arg(&n_i)
                .arg(&ospans)
                .arg(&os_i)
                .arg(&base)
                .arg(&ldb)
                .arg(&gproj)
                .arg(&type_embd)
                .arg(&qspans)
                .arg(&qs_i)
                .arg(&summary_norm.0)
                .arg(&summary_norm.1)
                .arg(&eps)
                .arg(&weights)
                .arg(&fields)
                .arg(&ldf)
                .arg(&w_i)
                .arg(&scale)
                .launch(LaunchConfig {
                    grid_dim: (nq as u32, 1, 1),
                    block_dim: (ROW_THREADS, 1, 1),
                    shared_mem_bytes: smem,
                })?;
        }
        Ok(())
    }

    /// The scorer's features and each option's (prior, cosine).
    ///
    /// # Safety
    /// Every pointer must address live device memory covering the shape,
    /// and every option's owner must name a row of `fieldn` and `qmean`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn features(
        &self,
        stream: &Arc<CudaStream>,
        fieldn: DevicePtr,
        ldf: usize,
        optn: DevicePtr,
        ldo: usize,
        ospans: DevicePtr,
        ostride: usize,
        n_opts: usize,
        qmean: DevicePtr,
        ldq: usize,
        global: DevicePtr,
        lexical: DevicePtr,
        ldl: usize,
        hidden: usize,
        width: usize,
        feat: DevicePtr,
        ldfeat: usize,
        aux: DevicePtr,
    ) -> Result<(), DecisionKernelError> {
        if n_opts == 0 {
            return Ok(());
        }
        let (os_i, n_i, h_i, w_i) = (
            i32_of(ostride, "option stride")?,
            i32_of(n_opts, "options")?,
            i32_of(hidden, "hidden")?,
            i32_of(width, "width")?,
        );
        let (ldf, ldo, ldq, ldl, ldfeat) = (
            ldf as i64,
            ldo as i64,
            ldq as i64,
            ldl as i64,
            ldfeat as i64,
        );
        // SAFETY: the caller guarantees the operands.
        unsafe {
            stream
                .launch_builder(&self.features)
                .arg(&fieldn)
                .arg(&ldf)
                .arg(&optn)
                .arg(&ldo)
                .arg(&ospans)
                .arg(&os_i)
                .arg(&n_i)
                .arg(&qmean)
                .arg(&ldq)
                .arg(&global)
                .arg(&lexical)
                .arg(&ldl)
                .arg(&h_i)
                .arg(&w_i)
                .arg(&feat)
                .arg(&ldfeat)
                .arg(&aux)
                .launch(LaunchConfig {
                    grid_dim: (n_opts as u32, 1, 1),
                    block_dim: (ROW_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// `scores[o] = prior_scale * prior + gate * (joint_scale * cosine +
    /// h[o] . w_out + b_out)`, `scales = [prior_scale, joint_scale, gate]`.
    ///
    /// # Safety
    /// Every pointer must address live device memory covering the shape.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn final_scores(
        &self,
        stream: &Arc<CudaStream>,
        h: DevicePtr,
        ldh: usize,
        n_opts: usize,
        width: usize,
        w_out: DevicePtr,
        b_out: f32,
        aux: DevicePtr,
        scales: [f32; 3],
        scores: DevicePtr,
    ) -> Result<(), DecisionKernelError> {
        if n_opts == 0 {
            return Ok(());
        }
        let (n_i, w_i) = (i32_of(n_opts, "options")?, i32_of(width, "width")?);
        let ldh = ldh as i64;
        let [prior, joint, gate] = scales;
        // SAFETY: the caller guarantees the operands.
        unsafe {
            stream
                .launch_builder(&self.final_scores)
                .arg(&h)
                .arg(&ldh)
                .arg(&n_i)
                .arg(&w_i)
                .arg(&w_out)
                .arg(&b_out)
                .arg(&aux)
                .arg(&prior)
                .arg(&joint)
                .arg(&gate)
                .arg(&scores)
                .launch(LaunchConfig {
                    grid_dim: (grid_1d(n_opts * 32, ROW_THREADS), 1, 1),
                    block_dim: (ROW_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }
}
