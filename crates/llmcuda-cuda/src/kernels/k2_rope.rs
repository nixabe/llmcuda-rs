//! K2 rotary coefficients shared by the heads and layers of a fixed pass.
//! The double-precision frequency and angle evaluation matches `attention`
//! exactly; only the resulting fp32 sine/cosine values are cached.
use super::{compile, moe::MoeError};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use std::sync::Arc;
const SRC: &str = r#"
__device__ __forceinline__ unsigned long long slot(int n,unsigned long long p0,unsigned long long p1,unsigned long long p2,unsigned long long p3,unsigned long long p4,unsigned long long p5,unsigned long long p6,unsigned long long p7){switch(n){case 0:return p0;case 1:return p1;case 2:return p2;case 3:return p3;case 4:return p4;case 5:return p5;case 6:return p6;default:return p7;}}
__device__ __forceinline__ unsigned short half_bits(float v){unsigned short h;asm("cvt.rn.f16.f32 %0,%1;":"=h"(h):"f"(v));return h;}
extern "C" {
__global__ void k2_rotary_prepare(float* coefficients,int chunk,int rope_dim,float theta,int offset,
 unsigned long long p0,unsigned long long p1,unsigned long long p2,unsigned long long p3,unsigned long long p4,unsigned long long p5,unsigned long long p6,unsigned long long p7){
 int t=blockIdx.x*8+threadIdx.y,d=threadIdx.x,seq=blockIdx.y;if(t>=chunk || d>=rope_dim/2)return;
 const int* position=(const int*)slot(seq,p0,p1,p2,p3,p4,p5,p6,p7);
 double freq=pow((double)theta,-2.0*(double)d/(double)rope_dim);
 double pos=(double)(*position)+(double)t,angle=pos*freq;
 long long at=((long long)offset+seq*chunk+t)*rope_dim+2*d;
 coefficients[at]=(float)sin(angle);coefficients[at+1]=(float)cos(angle);
}
__global__ void k2_rotary_apply(const float* input,float* out,const float* coefficients,int heads,int dim,int rope_dim,int offset,int tokens){
 int t0=blockIdx.x*8,h=blockIdx.y,d=threadIdx.x,half=rope_dim/2;
 if(d>=half && d<rope_dim)return;
 for(int u=0;u<8;++u){int t=t0+u;if(t>=tokens)break;long long base=((long long)t*heads+h)*dim;
  if(d>=rope_dim){out[base+d]=input[base+d];continue;}
  float sin_a=coefficients[((long long)offset+t)*rope_dim+2*d],cos_a=coefficients[((long long)offset+t)*rope_dim+2*d+1];
  float x0=input[base+d],x1=input[base+d+half];
  out[base+d]=x0*cos_a-x1*sin_a;out[base+d+half]=x0*sin_a+x1*cos_a;
 }
}
__global__ void k2_rotary_append(const float* q,float* qr,const float* k,float* kr,const float* v,const float* coefficients,
 unsigned long long ap0,unsigned long long ap1,unsigned long long ap2,unsigned long long ap3,unsigned long long ap4,unsigned long long ap5,unsigned long long ap6,unsigned long long ap7,
 unsigned long long kc0,unsigned long long kc1,unsigned long long kc2,unsigned long long kc3,unsigned long long kc4,unsigned long long kc5,unsigned long long kc6,unsigned long long kc7,
 unsigned long long vc0,unsigned long long vc1,unsigned long long vc2,unsigned long long vc3,unsigned long long vc4,unsigned long long vc5,unsigned long long vc6,unsigned long long vc7,
 int qh,int kh,int dim,int rope_dim){
 int seq=blockIdx.y,h=blockIdx.x,d=threadIdx.x;bool is_q=h<qh;int hh=is_q ? h : h-qh,heads=is_q ? qh : kh;
 const float* in=is_q ? q : k;float* out=is_q ? qr : kr;long long base=((long long)seq*heads+hh)*dim;
 unsigned short* kc=0;unsigned short* vc=0;long long at=0;
 if(!is_q){const int* pos=(const int*)slot(seq,ap0,ap1,ap2,ap3,ap4,ap5,ap6,ap7);kc=(unsigned short*)slot(seq,kc0,kc1,kc2,kc3,kc4,kc5,kc6,kc7);vc=(unsigned short*)slot(seq,vc0,vc1,vc2,vc3,vc4,vc5,vc6,vc7);at=(long long)*pos*kh*dim+(long long)hh*dim;}
 int half=rope_dim/2;if(d>=rope_dim){float x=in[base+d];out[base+d]=x;if(!is_q){kc[at+d]=half_bits(x);vc[at+d]=half_bits(v[base+d]);}return;}if(d>=half)return;
 float sin_a=coefficients[(long long)seq*rope_dim+2*d],cos_a=coefficients[(long long)seq*rope_dim+2*d+1];
 float x0=in[base+d],x1=in[base+d+half],y0=x0*cos_a-x1*sin_a,y1=x0*sin_a+x1*cos_a;
 out[base+d]=y0;out[base+d+half]=y1;
 if(!is_q){kc[at+d]=half_bits(y0);kc[at+d+half]=half_bits(y1);vc[at+d]=half_bits(v[base+d]);vc[at+d+half]=half_bits(v[base+d+half]);}
}
}
"#;
/// Fixed coefficients for one physical token shape, refreshed on the device
/// at layer zero, including on each graph replay.
pub struct K2Rope {
    prepare: CudaFunction,
    apply: CudaFunction,
    append: CudaFunction,
    coefficients: CudaSlice<f32>,
    tokens: usize,
    dim: usize,
    rope_dim: usize,
}
impl K2Rope {
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        tokens: usize,
        dim: usize,
        rope_dim: usize,
    ) -> Result<Self, MoeError> {
        if tokens == 0
            || dim == 0
            || dim > 1024
            || !rope_dim.is_multiple_of(2)
            || rope_dim == 0
            || rope_dim > dim
            || rope_dim > 256
        {
            return Err(MoeError::WrongElementCount {
                which: "K2 rotary geometry",
                expected: dim,
                found: rope_dim,
            });
        }
        let module = ctx.load_module(compile(SRC, "k2_rotary").map_err(MoeError::Compile)?)?;
        Ok(Self {
            prepare: module.load_function("k2_rotary_prepare")?,
            apply: module.load_function("k2_rotary_apply")?,
            append: module.load_function("k2_rotary_append")?,
            coefficients: stream.alloc_zeros(tokens * rope_dim)?,
            tokens,
            dim,
            rope_dim,
        })
    }
    /// Prepare one contiguous batch of at most eight sequence chunks.
    ///
    /// # Safety
    /// Each position slot must point to a live device i32 on this stream.
    pub unsafe fn prepare_raw(
        &mut self,
        stream: &Arc<CudaStream>,
        positions: &[u64],
        chunk: usize,
        offset: usize,
        theta: f32,
    ) -> Result<(), MoeError> {
        if !theta.is_finite()
            || theta <= 0.0
            || positions.is_empty()
            || positions.len() > 8
            || chunk == 0
            || offset + positions.len() * chunk > self.tokens
        {
            return Err(MoeError::WrongElementCount {
                which: "K2 rotary positions",
                expected: self.tokens,
                found: offset + positions.len() * chunk,
            });
        }
        let mut p = [positions[0]; 8];
        p[..positions.len()].copy_from_slice(positions);
        let (chunk_i, rd, off) = (chunk as i32, self.rope_dim as i32, offset as i32);
        let mut launch = stream.launch_builder(&self.prepare);
        launch
            .arg(&mut self.coefficients)
            .arg(&chunk_i)
            .arg(&rd)
            .arg(&theta)
            .arg(&off);
        for slot in &p {
            launch.arg(slot);
        }
        // SAFETY: caller owns live position slots; checked output token range.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: ((chunk as u32).div_ceil(8), positions.len() as u32, 1),
                block_dim: ((self.rope_dim / 2) as u32, 8, 1),
                shared_mem_bytes: 0,
            })?;
        }
        Ok(())
    }
    pub fn apply(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        heads: usize,
        offset: usize,
    ) -> Result<(), MoeError> {
        if input.is_empty() || heads == 0 || !input.len().is_multiple_of(heads * self.dim) {
            return Err(MoeError::WrongElementCount {
                which: "K2 rotary heads",
                expected: self.dim,
                found: heads,
            });
        }
        let tokens = input.len() / (heads * self.dim);
        check("K2 rotary output", input.len(), out.len())?;
        if offset + tokens > self.tokens {
            return Err(MoeError::WrongElementCount {
                which: "K2 rotary range",
                expected: self.tokens,
                found: offset + tokens,
            });
        }
        let (h, d, rd, off, t) = (
            heads as i32,
            self.dim as i32,
            self.rope_dim as i32,
            offset as i32,
            tokens as i32,
        );
        // SAFETY: input/output and coefficient ranges have been checked.
        unsafe {
            stream
                .launch_builder(&self.apply)
                .arg(input)
                .arg(out)
                .arg(&self.coefficients)
                .arg(&h)
                .arg(&d)
                .arg(&rd)
                .arg(&off)
                .arg(&t)
                .launch(LaunchConfig {
                    grid_dim: ((tokens as u32).div_ceil(8), heads as u32, 1),
                    block_dim: (self.dim as u32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }
    /// Rotate and append one decode row for every sequence.
    ///
    /// # Safety
    /// Position slots must address live i32 values. Cache slots must be
    /// distinct live f16 arrays covering those positions and these heads.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn append_raw(
        &self,
        stream: &Arc<CudaStream>,
        q: &CudaSlice<f32>,
        qr: &mut CudaSlice<f32>,
        k: &CudaSlice<f32>,
        kr: &mut CudaSlice<f32>,
        v: &CudaSlice<f32>,
        positions: &[u64],
        keys: &[u64],
        values: &[u64],
        qh: usize,
        kh: usize,
    ) -> Result<(), MoeError> {
        let n = positions.len();
        if n == 0 || n > 8 || n != self.tokens || qh == 0 || kh == 0 {
            return Err(MoeError::WrongElementCount {
                which: "K2 rotary decode batch",
                expected: self.tokens,
                found: n,
            });
        }
        check("K2 key slots", n, keys.len())?;
        check("K2 value slots", n, values.len())?;
        check("K2 query", n * qh * self.dim, q.len())?;
        check("K2 query output", q.len(), qr.len())?;
        check("K2 key", n * kh * self.dim, k.len())?;
        check("K2 key output", k.len(), kr.len())?;
        check("K2 value", k.len(), v.len())?;
        let fill = |s: &[u64]| {
            let mut a = [s[0]; 8];
            a[..n].copy_from_slice(s);
            a
        };
        let (ap, kc, vc) = (fill(positions), fill(keys), fill(values));
        let (qh_i, kh_i, d, rd) = (qh as i32, kh as i32, self.dim as i32, self.rope_dim as i32);
        let mut launch = stream.launch_builder(&self.append);
        launch
            .arg(q)
            .arg(qr)
            .arg(k)
            .arg(kr)
            .arg(v)
            .arg(&self.coefficients);
        for slot in ap.iter().chain(&kc).chain(&vc) {
            launch.arg(slot);
        }
        launch.arg(&qh_i).arg(&kh_i).arg(&d).arg(&rd);
        // SAFETY: checked input shapes; caller guarantees live independent caches.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: ((qh + kh) as u32, n as u32, 1),
                block_dim: (self.dim as u32, 1, 1),
                shared_mem_bytes: 0,
            })?;
        }
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
