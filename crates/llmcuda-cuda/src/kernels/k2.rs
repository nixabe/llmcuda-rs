//! K2-Horizon operations, following IFM's llama.cpp `src/models/k2-horizon.cpp`.
//! Routing bias affects selection only; original sigmoid scores are normalized
//! with ggml's 2^-14 floor, then scaled. Grids and buffers are fixed per shape.
use super::{
    compile,
    moe::{ExpertQuant, MoeError},
};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use std::sync::Arc;

const SRC: &str = r#"
extern "C" {
__global__ void k2_norm(const float* x, const float* w, float* y, int width, int groups, float eps) {
    int g = blockIdx.x % groups, token = blockIdx.x / groups, part = width / groups;
    long long base = (long long)token * width + g * part;
    float acc = 0.0f;
    for (int j = threadIdx.x; j < part; j += blockDim.x) acc += x[base+j] * x[base+j];
    for (int d = 16; d > 0; d >>= 1) acc += __shfl_down_sync(0xffffffff, acc, d);
    __shared__ float sums[8];
    if ((threadIdx.x & 31) == 0) sums[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f; for (int i = 0; i < 8; ++i) sum += sums[i];
        sums[0] = 1.0f / sqrtf(sum / part + eps);
    }
    __syncthreads();
    for (int j = threadIdx.x; j < part; j += blockDim.x) y[base+j] = (x[base+j] * sums[0]) * w[g*part+j];
}
// One thread selects the small fixed expert population; no atomics or host reads.
__global__ void k2_route(const float* logits, const float* bias, int* ids, float* weights,
                         int experts, int topk, float scale) {
    int token = blockIdx.x;
    if (threadIdx.x != 0) return;
    float p[512], selection[512];
    for (int e = 0; e < experts; ++e) {
        p[e] = 1.0f / (1.0f + expf(-logits[(long long)token*experts+e]));
        selection[e] = p[e] + bias[e];
    }
    float sum = 0.0f;
    for (int k = 0; k < topk; ++k) {
        int best = 0;
        for (int e = 1; e < experts; ++e) if (selection[e] > selection[best]) best = e;
        ids[token*topk+k] = best; weights[token*topk+k] = p[best]; sum += p[best];
        selection[best] = -__int_as_float(0x7f800000);
    }
    sum = fmaxf(sum, 6.103515625e-5f);
    for (int k = 0; k < topk; ++k) weights[token*topk+k] = (weights[token*topk+k] / sum) * scale;
}
// One warp per output row, four consecutive contraction elements per lane.
// `quant=5` is a dense fp32 matrix, used by the FFN router.
// Routed output is [token][topk][rows]; ids are consumed on-device.
__global__ void k2_project(const unsigned char* w, int quant, const float* x, const int* ids,
                           float* out, int inner, int rows, int topk, int routed) {
    int row = blockIdx.x*4 + threadIdx.y, token = blockIdx.y / topk, slot = blockIdx.y % topk;
    if (row >= rows) return;
    int expert = routed ? ids[token*topk+slot] : 0;
    long long start = ((long long)expert*rows + row)*inner;
    float acc = 0.0f;
    for (int j = 0; j < inner; j += 128) {
        float v[4];
        if (quant == 5) { for (int z=0; z<4; ++z) v[z] = ((const float*)w)[start+j+threadIdx.x*4+z]; }
        else dequant_tile_community(w, quant, start+j, threadIdx.x, v);
        for (int z=0; z<4; ++z) acc = fmaf(v[z], x[(long long)token*inner+j+threadIdx.x*4+z], acc);
    }
    for (int d=16; d>0; d>>=1) acc += __shfl_down_sync(0xffffffff, acc, d);
    if (threadIdx.x == 0) out[((long long)token*topk+slot)*rows+row] = acc;
}
__global__ void k2_values(const float* projections, const float* weights, float* out, int rows, int topk) {
    int j=blockIdx.x*blockDim.x+threadIdx.x, token=blockIdx.y;
    if (j>=rows) return;
    float acc=0.0f;
    for (int k=0;k<topk;++k) {
        float v=projections[((long long)token*topk+k)*rows+j];
        acc += (v/(1.0f+expf(-v))) * weights[token*topk+k];
    }
    out[(long long)token*rows+j]=acc;
}
__global__ void k2_gate(const float* x, const float* gate, float* out, long long n) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    float v=gate[i]*0.6931471805599453f;
    float sp=fmaxf(v,0.0f)+log1pf(expf(-fabsf(v)));
    out[i]=x[i]*(sp*1.4426950408889634f);
}
}
"#;

/// The quantized projection format code, shared with the MoE unpackers.
pub fn quant_code(q: ExpertQuant) -> i32 {
    match q {
        ExpertQuant::Q6K => 0,
        ExpertQuant::Q8_0 => 1,
        ExpertQuant::Q4_0 => 2,
        ExpertQuant::Q4K => 3,
        ExpertQuant::Q5K => 4,
    }
}

/// Compiled operations, shared across layers and shapes.
pub struct K2Kernels {
    norm: CudaFunction,
    route: CudaFunction,
    project: CudaFunction,
    values: CudaFunction,
    gate: CudaFunction,
}
impl K2Kernels {
    /// Reuse the existing, tested quant unpackers; compile only their helpers.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, MoeError> {
        let prologue = super::moe::MOE_SRC
            .split("// The GEMM prologue, with the format")
            .next()
            .unwrap();
        let src = format!("{prologue}\n{SRC}");
        let module = ctx.load_module(compile(&src, "k2_horizon").map_err(MoeError::Compile)?)?;
        Ok(Self {
            norm: module.load_function("k2_norm")?,
            route: module.load_function("k2_route")?,
            project: module.load_function("k2_project")?,
            values: module.load_function("k2_values")?,
            gate: module.load_function("k2_gate")?,
        })
    }
    /// Normalize contiguous groups with a full-width learned weight vector.
    #[allow(clippy::too_many_arguments)]
    pub fn norm(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        width: usize,
        groups: usize,
        eps: f32,
    ) -> Result<(), MoeError> {
        check("group norm input", y.len(), x.len())?;
        check("group norm weight", width, w.len())?;
        if groups == 0
            || width == 0
            || !width.is_multiple_of(groups)
            || !x.len().is_multiple_of(width)
        {
            return Err(MoeError::WrongElementCount {
                which: "group norm geometry",
                expected: width,
                found: groups,
            });
        }
        let cfg = LaunchConfig {
            grid_dim: ((x.len() / width * groups) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (width, groups) = (width as i32, groups as i32);
        unsafe {
            stream
                .launch_builder(&self.norm)
                .arg(x)
                .arg(w)
                .arg(y)
                .arg(&width)
                .arg(&groups)
                .arg(&eps)
                .launch(cfg)
        }?;
        Ok(())
    }
    /// Bias selects routes; unbiased sigmoid probabilities supply their weights.
    #[allow(clippy::too_many_arguments)]
    pub fn route(
        &self,
        stream: &Arc<CudaStream>,
        logits: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        ids: &mut CudaSlice<i32>,
        weights: &mut CudaSlice<f32>,
        topk: usize,
        scale: f32,
    ) -> Result<(), MoeError> {
        let experts = bias.len();
        if experts == 0
            || experts > 512
            || topk == 0
            || topk > experts
            || !logits.len().is_multiple_of(experts)
        {
            return Err(MoeError::WrongElementCount {
                which: "K2 route geometry",
                expected: 512,
                found: experts,
            });
        }
        let tokens = logits.len() / experts;
        check("K2 route ids", tokens * topk, ids.len())?;
        check("K2 route weights", tokens * topk, weights.len())?;
        let cfg = LaunchConfig {
            grid_dim: (tokens as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let (experts, topk) = (experts as i32, topk as i32);
        unsafe {
            stream
                .launch_builder(&self.route)
                .arg(logits)
                .arg(bias)
                .arg(ids)
                .arg(weights)
                .arg(&experts)
                .arg(&topk)
                .arg(&scale)
                .launch(cfg)
        }?;
        Ok(())
    }
    /// Dense or selected stacked projections; all launch dimensions are fixed.
    #[allow(clippy::too_many_arguments)]
    pub fn project(
        &self,
        stream: &Arc<CudaStream>,
        w: &CudaSlice<u8>,
        quant: Option<ExpertQuant>,
        x: &CudaSlice<f32>,
        ids: &CudaSlice<i32>,
        out: &mut CudaSlice<f32>,
        inner: usize,
        rows: usize,
        experts: usize,
        topk: usize,
        routed: bool,
    ) -> Result<(), MoeError> {
        if inner == 0
            || !inner.is_multiple_of(128)
            || topk == 0
            || rows == 0
            || experts == 0
            || (!routed && experts != 1)
            || quant.is_some_and(|q| !inner.is_multiple_of(q.block_elements()))
        {
            return Err(MoeError::WrongElementCount {
                which: "K2 projection geometry",
                expected: 128,
                found: inner,
            });
        }
        let tokens = x.len() / inner;
        check("K2 projection input", tokens * inner, x.len())?;
        check("K2 projection output", tokens * topk * rows, out.len())?;
        if routed {
            check("K2 projection ids", tokens * topk, ids.len())?;
        }
        let elems = inner * rows * experts;
        let bytes = match quant {
            Some(q) => elems / q.block_elements() * q.block_bytes(),
            None => elems * 4,
        };
        check("K2 projection weights", bytes, w.len())?;
        let code = quant.map(quant_code).unwrap_or(5);
        let (inner, rows, topk, routed) =
            (inner as i32, rows as i32, topk as i32, i32::from(routed));
        let cfg = LaunchConfig {
            grid_dim: ((rows as u32).div_ceil(4), tokens as u32 * topk as u32, 1),
            block_dim: (32, 4, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.project)
                .arg(w)
                .arg(&code)
                .arg(x)
                .arg(ids)
                .arg(out)
                .arg(&inner)
                .arg(&rows)
                .arg(&topk)
                .arg(&routed)
                .launch(cfg)
        }?;
        Ok(())
    }
    /// Apply SiLU to each value projection before the weighted sum.
    pub fn values(
        &self,
        stream: &Arc<CudaStream>,
        p: &CudaSlice<f32>,
        weights: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        topk: usize,
    ) -> Result<(), MoeError> {
        if rows == 0 || topk == 0 {
            return Err(MoeError::WrongElementCount {
                which: "K2 values geometry",
                expected: 1,
                found: 0,
            });
        }
        let tokens = out.len() / rows;
        check("K2 values output", tokens * rows, out.len())?;
        check("K2 values projections", tokens * topk * rows, p.len())?;
        check("K2 values weights", tokens * topk, weights.len())?;
        let cfg = LaunchConfig {
            grid_dim: ((rows as u32).div_ceil(256), tokens as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (rows, topk) = (rows as i32, topk as i32);
        unsafe {
            stream
                .launch_builder(&self.values)
                .arg(p)
                .arg(weights)
                .arg(out)
                .arg(&rows)
                .arg(&topk)
                .launch(cfg)
        }?;
        Ok(())
    }
    /// Attention output gate: softplus(gate * ln(2)) / ln(2).
    pub fn gate(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), MoeError> {
        check("K2 gate", x.len(), gate.len())?;
        check("K2 gate output", x.len(), out.len())?;
        let n = x.len() as i64;
        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.gate)
                .arg(x)
                .arg(gate)
                .arg(out)
                .arg(&n)
                .launch(cfg)
        }?;
        Ok(())
    }
}
fn check(which: &'static str, expected: usize, found: usize) -> Result<(), MoeError> {
    if expected == found {
        Ok(())
    } else {
        Err(MoeError::WrongElementCount {
            which,
            expected,
            found,
        })
    }
}
