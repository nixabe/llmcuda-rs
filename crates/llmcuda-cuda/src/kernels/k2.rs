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

pub(crate) const K2_ROUTED_TILE: usize = 64;

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
// Parallel score evaluation and stable argmax. Equal scores select the lowest
// expert index, as in the scalar scan. Lane zero retains the scalar top-k sum
// order, so changing the reduction does not change the combine weights.
__global__ void k2_route(const float* logits, const float* bias, int* ids, float* weights,
                         int experts, int topk, float scale) {
    int token = blockIdx.x;
    __shared__ float p[512], selection[512], best_scores[4], sum;
    __shared__ int best_ids[4], chosen;
    for (int e = threadIdx.x; e < experts; e += blockDim.x) {
        p[e] = 1.0f / (1.0f + expf(-logits[(long long)token*experts+e]));
        selection[e] = p[e] + bias[e];
    }
    if (threadIdx.x == 0) sum = 0.0f;
    __syncthreads();
    for (int k = 0; k < topk; ++k) {
        int best = 512;
        float score = -__int_as_float(0x7f800000);
        for (int e = threadIdx.x; e < experts; e += blockDim.x) {
            float s = selection[e];
            if (s > score || (s == score && e < best)) { score = s; best = e; }
        }
        for (int d = 16; d > 0; d >>= 1) {
            float s = __shfl_down_sync(0xffffffff, score, d);
            int e = __shfl_down_sync(0xffffffff, best, d);
            if (s > score || (s == score && e < best)) { score = s; best = e; }
        }
        if ((threadIdx.x & 31) == 0) {
            best_scores[threadIdx.x >> 5] = score;
            best_ids[threadIdx.x >> 5] = best;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            chosen = best_ids[0]; score = best_scores[0];
            for (int w = 1; w < 4; ++w) {
                float s = best_scores[w]; int e = best_ids[w];
                if (s > score || (s == score && e < chosen)) { score = s; chosen = e; }
            }
            ids[token*topk+k] = chosen;
            weights[token*topk+k] = p[chosen]; sum += p[chosen];
            selection[chosen] = -__int_as_float(0x7f800000);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        sum = fmaxf(sum, 6.103515625e-5f);
        for (int k = 0; k < topk; ++k) weights[token*topk+k] = (weights[token*topk+k] / sum) * scale;
    }
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
// Batched dense projections reuse the unpacked row across BT tokens. The
// contraction and shuffle order of each row matches k2_project exactly.
}
template<int BT>
__device__ void project_dense_tile(const unsigned char* w, int quant, const float* x,
                                  float* out, int inner, int rows, int tokens) {
    int row = blockIdx.x*4 + threadIdx.y, t0 = blockIdx.y*BT;
    if (row >= rows) return;
    long long start = (long long)row*inner;
    float acc[BT];
    #pragma unroll
    for (int b=0; b<BT; ++b) acc[b]=0.0f;
    for (int j=0; j<inner; j+=128) {
        float v[4];
        if (quant == 5) { for (int z=0; z<4; ++z) v[z]=((const float*)w)[start+j+threadIdx.x*4+z]; }
        else dequant_tile_community(w, quant, start+j, threadIdx.x, v);
        #pragma unroll
        for (int b=0; b<BT; ++b) {
            if (t0+b >= tokens) continue;
            float4 xv=*(const float4*)(x+(long long)(t0+b)*inner+j+threadIdx.x*4);
            acc[b]=fmaf(v[0],xv.x,acc[b]); acc[b]=fmaf(v[1],xv.y,acc[b]);
            acc[b]=fmaf(v[2],xv.z,acc[b]); acc[b]=fmaf(v[3],xv.w,acc[b]);
        }
    }
    #pragma unroll
    for (int b=0; b<BT; ++b) {
        for (int d=16; d>0; d>>=1) acc[b]+=__shfl_down_sync(0xffffffff,acc[b],d);
        if (threadIdx.x==0 && t0+b<tokens) out[(long long)(t0+b)*rows+row]=acc[b];
    }
}
extern "C" {
__global__ void k2_project_b4(const unsigned char* w, int quant, const float* x,
                             float* out, int inner, int rows, int tokens) {
    project_dense_tile<4>(w,quant,x,out,inner,rows,tokens);
}
__global__ void k2_project_b8(const unsigned char* w, int quant, const float* x,
                             float* out, int inner, int rows, int tokens) {
    project_dense_tile<8>(w,quant,x,out,inner,rows,tokens);
}
// One thread per expert gathers flat (token, slot) ids in their original
// order. Prefix offsets pack padded runs; unused runs carry expert=-1.
// This follows moe_align_block_size's device-side counting and padding.
__global__ void k2_dispatch(const int* ids, int* sorted, int* owners,
                            int tokens, int topk, int experts, int capacity, int tile) {
    __shared__ int counts[128], starts[128];
    int e=threadIdx.x, pairs=tokens*topk;
    for(int b=threadIdx.x;b<capacity;b+=blockDim.x) owners[b]=-1;
    int count=0;
    if(e<experts) for(int f=0;f<pairs;++f) count += ids[f]==e;
    counts[e]=count;
    __syncthreads();
    if(e==0) {
        int cursor=0;
        for(int j=0;j<experts;++j) { starts[j]=cursor; cursor+=(counts[j]+tile-1)/tile*tile; }
    }
    __syncthreads();
    if(e>=experts) return;
    int cursor=starts[e];
    for(int f=0;f<pairs;++f) if(ids[f]==e) sorted[cursor++]=f;
    int end=starts[e]+(counts[e]+tile-1)/tile*tile;
    for(int f=cursor;f<end;++f) sorted[f]=pairs;
    for(int f=starts[e];f<end;f+=tile) owners[f/tile]=e;
}
// Each lane owns a contiguous slice of flat ids. The prefix of lane counts
// keeps the original route order without host counts or atomic scatter.
__global__ void k2_dispatch_count(const int* ids,int* counts,int pairs) {
    int e=blockIdx.x,t=threadIdx.x,chunk=(pairs+127)/128;
    int n=0;for(int f=t*chunk;f<min((t+1)*chunk,pairs);++f)n+=ids[f]==e;
    counts[e*128+t]=n;
}
__global__ void k2_dispatch_prefix(const int* counts,int* starts,int* owners,int experts,int capacity,int tile) {
    __shared__ int totals[128];int t=threadIdx.x,n=0;
    if(t<experts)for(int j=0;j<128;++j)n+=counts[t*128+j];
    totals[t]=(n+tile-1)/tile*tile;
    for(int b=t;b<capacity;b+=128)owners[b]=-1;
    __syncthreads();int start=0;for(int e=0;e<t;++e)start+=totals[e];
    if(t<experts){starts[t]=start;for(int f=start;f<start+totals[t];f+=tile)owners[f/tile]=t;}
}
__global__ void k2_dispatch_gather(const int* ids,const int* counts,const int* starts,int* sorted,int pairs,int tile) {
    int e=blockIdx.x,t=threadIdx.x,chunk=(pairs+127)/128;
    __shared__ int prefix[128];prefix[t]=counts[e*128+t];__syncthreads();
    for(int d=1;d<128;d*=2){int v=t>=d ? prefix[t-d] : 0;__syncthreads();prefix[t]+=v;__syncthreads();}
    int total=prefix[127],cursor=starts[e]+prefix[t]-counts[e*128+t];
    for(int f=t*chunk;f<min((t+1)*chunk,pairs);++f)if(ids[f]==e)sorted[cursor++]=f;
    for(int f=total+t;f<(total+tile-1)/tile*tile;f+=128)sorted[starts[e]+f]=pairs;
}
__global__ void k2_project_grouped(const unsigned char* w, int quant, const float* x,
                                  const int* sorted, const int* owners, float* out,
                                  int inner, int rows, int tokens, int topk) {
    int e=owners[blockIdx.y], row=blockIdx.x*4+threadIdx.y;
    if(e<0 || row>=rows) return;
    int flats[8]; float acc[8];
    #pragma unroll
    for(int b=0;b<8;++b) { flats[b]=sorted[blockIdx.y*8+b]; acc[b]=0.0f; }
    long long start=((long long)e*rows+row)*inner;
    for(int j=0;j<inner;j+=128) {
        float v[4]; dequant_tile_community(w,quant,start+j,threadIdx.x,v);
        #pragma unroll
        for(int b=0;b<8;++b) {
            if(flats[b]>=tokens*topk) continue;
            float4 xv=*(const float4*)(x+(long long)(flats[b]/topk)*inner+j+threadIdx.x*4);
            acc[b]=fmaf(v[0],xv.x,acc[b]); acc[b]=fmaf(v[1],xv.y,acc[b]);
            acc[b]=fmaf(v[2],xv.z,acc[b]); acc[b]=fmaf(v[3],xv.w,acc[b]);
        }
    }
    #pragma unroll
    for(int b=0;b<8;++b) {
        for(int d=16;d>0;d>>=1) acc[b]+=__shfl_down_sync(0xffffffff,acc[b],d);
        if(threadIdx.x==0 && flats[b]<tokens*topk) out[(long long)flats[b]*rows+row]=acc[b];
    }
}
__global__ void k2_combine(const float* p,const float* w,float* out,int rows,int topk) {
    int j=blockIdx.x*blockDim.x+threadIdx.x,t=blockIdx.y;if(j>=rows)return;
    float a=0;for(int k=0;k<topk;++k) a+=p[((long long)t*topk+k)*rows+j]*w[t*topk+k];
    out[(long long)t*rows+j]=a;
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

/// Fixed device dispatch workspace for selected projections. Natural token
/// capacity plus at most one padded run per expert bounds every write.
pub struct K2Dispatch {
    counts: CudaSlice<i32>,
    starts: CudaSlice<i32>,
    pub(crate) direct: CudaSlice<i32>,
    pub(crate) sorted: CudaSlice<i32>,
    pub(crate) owners: CudaSlice<i32>,
    pub(crate) tokens: usize,
    pub(crate) topk: usize,
    pub(crate) experts: usize,
    pub(crate) tile: usize,
}
impl K2Dispatch {
    /// Allocate once for a fixed projection shape, never during inference.
    pub fn new(
        stream: &Arc<CudaStream>,
        tokens: usize,
        topk: usize,
        experts: usize,
    ) -> Result<Self, MoeError> {
        Self::with_tile(stream, tokens, topk, experts, 8)
    }
    /// Fixed 64-slot runs for tensor-core batching.
    pub fn new_gemm(
        stream: &Arc<CudaStream>,
        tokens: usize,
        topk: usize,
        experts: usize,
    ) -> Result<Self, MoeError> {
        Self::with_tile(stream, tokens, topk, experts, K2_ROUTED_TILE)
    }
    fn with_tile(
        stream: &Arc<CudaStream>,
        tokens: usize,
        topk: usize,
        experts: usize,
        tile: usize,
    ) -> Result<Self, MoeError> {
        if tokens == 0 || experts == 0 || experts > 128 || topk == 0 || topk > experts {
            return Err(MoeError::WrongElementCount {
                which: "K2 dispatch geometry",
                expected: 128,
                found: experts,
            });
        }
        let capacity = (tokens * topk).div_ceil(tile) + experts;
        Ok(Self {
            counts: stream.alloc_zeros::<i32>(experts * 128)?,
            starts: stream.alloc_zeros::<i32>(experts)?,
            direct: stream.alloc_zeros::<i32>(tokens * topk)?,
            sorted: stream.alloc_zeros::<i32>(capacity * tile)?,
            owners: stream.alloc_zeros::<i32>(capacity)?,
            tokens,
            topk,
            experts,
            tile,
        })
    }
}

/// Compiled operations, shared across layers and shapes.
pub struct K2Kernels {
    norm: CudaFunction,
    route: CudaFunction,
    project: CudaFunction,
    project_tiles: [CudaFunction; 2],
    dispatch: CudaFunction,
    dispatch_count: CudaFunction,
    dispatch_prefix: CudaFunction,
    dispatch_gather: CudaFunction,
    project_grouped: CudaFunction,
    values: CudaFunction,
    combine: CudaFunction,
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
            project_tiles: [
                module.load_function("k2_project_b4")?,
                module.load_function("k2_project_b8")?,
            ],
            dispatch: module.load_function("k2_dispatch")?,
            dispatch_count: module.load_function("k2_dispatch_count")?,
            dispatch_prefix: module.load_function("k2_dispatch_prefix")?,
            dispatch_gather: module.load_function("k2_dispatch_gather")?,
            project_grouped: module.load_function("k2_project_grouped")?,
            values: module.load_function("k2_values")?,
            combine: module.load_function("k2_combine")?,
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
            block_dim: (128, 1, 1),
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
        self.project_with_tile(
            stream,
            w,
            quant,
            x,
            ids,
            out,
            inner,
            rows,
            experts,
            topk,
            routed,
            if routed || topk != 1 {
                1
            } else if x.len() / inner.max(1) >= 8 {
                8
            } else {
                4
            },
        )
    }
    /// Projection tile control for differential tests and kernel benchmarks.
    /// Routed projections currently use the one-token kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn project_with_tile(
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
        tile: usize,
    ) -> Result<(), MoeError> {
        if ![1, 4, 8].contains(&tile) || (tile != 1 && (routed || topk != 1)) {
            return Err(MoeError::WrongElementCount {
                which: "K2 projection tile",
                expected: 8,
                found: tile,
            });
        }
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
        if tile != 1 {
            let cfg = LaunchConfig {
                grid_dim: ((rows as u32).div_ceil(4), tokens.div_ceil(tile) as u32, 1),
                block_dim: (32, 4, 1),
                shared_mem_bytes: 0,
            };
            let (inner, rows, tokens) = (inner as i32, rows as i32, tokens as i32);
            let f = &self.project_tiles[usize::from(tile == 8)];
            unsafe {
                stream
                    .launch_builder(f)
                    .arg(w)
                    .arg(&code)
                    .arg(x)
                    .arg(out)
                    .arg(&inner)
                    .arg(&rows)
                    .arg(&tokens)
                    .launch(cfg)
            }?;
            return Ok(());
        }
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
    /// Group selected value projections by expert, preserving each dot's
    /// fp32 contraction order. Dispatch and launch sizes stay device-resident.
    #[allow(clippy::too_many_arguments)]
    pub fn project_grouped(
        &self,
        stream: &Arc<CudaStream>,
        w: &CudaSlice<u8>,
        quant: ExpertQuant,
        x: &CudaSlice<f32>,
        ids: &CudaSlice<i32>,
        out: &mut CudaSlice<f32>,
        inner: usize,
        rows: usize,
        dispatch: &mut K2Dispatch,
    ) -> Result<(), MoeError> {
        if inner == 0
            || !inner.is_multiple_of(128)
            || !inner.is_multiple_of(quant.block_elements())
            || rows == 0
        {
            return Err(MoeError::WrongElementCount {
                which: "K2 grouped projection geometry",
                expected: quant.block_elements(),
                found: inner,
            });
        }
        let (tokens, topk, experts) = (dispatch.tokens, dispatch.topk, dispatch.experts);
        check("K2 grouped input", tokens * inner, x.len())?;
        check("K2 grouped ids", tokens * topk, ids.len())?;
        check("K2 grouped output", tokens * topk * rows, out.len())?;
        check(
            "K2 grouped weights",
            inner * rows * experts / quant.block_elements() * quant.block_bytes(),
            w.len(),
        )?;
        check("K2 scalar dispatch tile", 8, dispatch.tile)?;
        let tile = dispatch.tile as i32;
        let capacity = dispatch.owners.len() as i32;
        let (tokens, topk, experts, inner, rows, code) = (
            tokens as i32,
            topk as i32,
            experts as i32,
            inner as i32,
            rows as i32,
            quant_code(quant),
        );
        unsafe {
            stream
                .launch_builder(&self.dispatch)
                .arg(ids)
                .arg(&mut dispatch.sorted)
                .arg(&mut dispatch.owners)
                .arg(&tokens)
                .arg(&topk)
                .arg(&experts)
                .arg(&capacity)
                .arg(&tile)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })?;
            stream
                .launch_builder(&self.project_grouped)
                .arg(w)
                .arg(&code)
                .arg(x)
                .arg(&dispatch.sorted)
                .arg(&dispatch.owners)
                .arg(out)
                .arg(&inner)
                .arg(&rows)
                .arg(&tokens)
                .arg(&topk)
                .launch(LaunchConfig {
                    grid_dim: ((rows as u32).div_ceil(4), capacity as u32, 1),
                    block_dim: (32, 4, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }
    /// Build fixed expert runs on-device for the batched contraction.
    pub fn dispatch_gemm(
        &self,
        stream: &Arc<CudaStream>,
        ids: &CudaSlice<i32>,
        d: &mut K2Dispatch,
    ) -> Result<(), MoeError> {
        if ![8, 16, 32, 64].contains(&d.tile) {
            return Err(MoeError::WrongElementCount {
                which: "K2 dispatch tile",
                expected: 64,
                found: d.tile,
            });
        };
        check("K2 BLAS dispatch ids", d.tokens * d.topk, ids.len())?;
        stream.memcpy_dtod(ids, &mut d.direct)?;
        // Small shapes read direct ids and never consume padded runs.
        if d.tokens < 8 {
            return Ok(());
        }
        let pairs = (d.tokens * d.topk) as i32;
        let experts = d.experts as i32;
        let capacity = d.owners.len() as i32;
        let tile = d.tile as i32;
        // SAFETY: one fixed block per expert, disjoint count and padded runs;
        // the prefix is device-resident and capacity includes expert padding.
        unsafe {
            let cfg = LaunchConfig {
                grid_dim: (d.experts as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            stream
                .launch_builder(&self.dispatch_count)
                .arg(ids)
                .arg(&mut d.counts)
                .arg(&pairs)
                .launch(cfg)?;
            stream
                .launch_builder(&self.dispatch_prefix)
                .arg(&d.counts)
                .arg(&mut d.starts)
                .arg(&mut d.owners)
                .arg(&experts)
                .arg(&capacity)
                .arg(&tile)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    ..cfg
                })?;
            stream
                .launch_builder(&self.dispatch_gather)
                .arg(ids)
                .arg(&d.counts)
                .arg(&d.starts)
                .arg(&mut d.sorted)
                .arg(&pairs)
                .arg(&tile)
                .launch(cfg)?;
        }
        Ok(())
    }
    /// Sum routed FFN projections in route order without an activation.
    pub fn combine(
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
                which: "K2 combine geometry",
                expected: 1,
                found: 0,
            });
        }
        let tokens = out.len() / rows;
        check("K2 combine output", tokens * rows, out.len())?;
        check("K2 combine partial", tokens * topk * rows, p.len())?;
        check("K2 combine weights", tokens * topk, weights.len())?;
        let (rows, topk) = (rows as i32, topk as i32);
        // SAFETY: output, partial and route-weight checks bound every access.
        unsafe {
            stream
                .launch_builder(&self.combine)
                .arg(p)
                .arg(weights)
                .arg(out)
                .arg(&rows)
                .arg(&topk)
                .launch(LaunchConfig {
                    grid_dim: ((rows as u32).div_ceil(256), tokens as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
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
