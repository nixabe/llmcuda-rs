//! Fixed-buffer K2 integer projections for Turing. Decode uses DP4A; prefill
//! uses integer tensor cores. Per-group scales, affine correction, and the
//! reduction tree are shared across dense, routed, and flattened shapes.
//!
//! Q4 records retain 128 bytes of nibble codes, half d/dmin, eight scale/min
//! bytes and packed code sums (160 bytes). Q6 records keep 192 bytes of six-bit
//! codes, half d and sixteen signed subscales (224 bytes, padded).
//! Expert weights interleave each word across four rows; dense rows stay
//! contiguous. Repacking and activation arenas are allocated at construction.
//!
//! Activation codes use 32-value scales. The production path retains 16 bits,
//! split into signed high/low bytes for DP4A or four nibble planes for IMMA.
//! Q4's affine offset uses the original activation sum. Every launch adds four
//! 32-value terms as `(t0+t2)+(t1+t3)` before advancing the 128-value window,
//! including Q6's paired 16-value subscales. Fixed dispatch runs preserve flat
//! route indices, so grouping never changes the reduction order.
use super::{
    compile,
    k2::K2Dispatch,
    moe::{ExpertQuant, MoeError},
};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use std::sync::Arc;

const REPACK_SRC: &str = r#"
template<int Q>
__device__ __forceinline__ unsigned raw_pack4(const unsigned char* w,long long i,float* scale,float* minimum) {
    int r=i&255;unsigned p=0;
    if(Q==3) {
        const unsigned char* b=w+(i>>8)*144;int sc,m;moe_scale_min_k4(r/32,b+4,&sc,&m);
        *scale=load_half_le(b)*(float)sc;*minimum=load_half_le(b+2)*(float)m;
        unsigned z=*(const unsigned*)(b+16+(r/64)*32+r%32);
        p=((r/32)&1) ? (z>>4)&0x0f0f0f0f : z&0x0f0f0f0f;
    } else {
        const unsigned char* b=w+(i>>8)*224;int half=r/128,g=(r%128)/32,l=r%32;
        *scale=load_half_le(b+208)*(float)((const signed char*)b)[192+r/16];*minimum=0;
        unsigned low=*(const unsigned*)(b+half*64+(g%2)*32+l),high=*(const unsigned*)(b+128+half*32+l);
        unsigned z=(((g/2) ? low>>4 : low)&0x0f0f0f0f)|(((high>>(2*g))&0x03030303)<<4);
        // Signed byte subtraction cannot borrow into a neighbour: each
        // byte has a spare high bit before subtracting 32.
        p=((z|0x80808080)-0x20202020)^0x80808080;
    }
    return p;
}
extern "C" {
__global__ void k2_repack_q4(const unsigned char* src,unsigned char* dst,long long elems,int inner,int rows,int IR) {
    long long i=((long long)blockIdx.x*blockDim.x+threadIdx.x)*4;if(i>=elems)return;
    int r=i%256,g=r/32,z=(r%32)/4;if(z>=4)return;float scale,minimum;
    unsigned lo=raw_pack4<3>(src,i,&scale,&minimum),hi=raw_pack4<3>(src,i+16,&scale,&minimum);
    long long row=i/inner;int expert=row/rows,rw=row%rows;unsigned char* out=dst+(((long long)expert*((rows+IR-1)/IR)+rw/IR)*(inner/256)+((i%inner)/256))*160*IR+(rw%IR)*4;*(unsigned*)(out+(z*32+g*4)*IR)=lo|(hi<<4);
    if(r==0){const unsigned char* in=src+(i/256)*144;*(unsigned*)(out+128*IR)=*(const unsigned*)in;}
    if(r%32==0){const unsigned char* in=src+(i/256)*144;int sc,m;moe_scale_min_k4(g,in+4,&sc,&m);out[((132+g)>>2)*(4*IR)+((132+g)&3)]=sc;out[((140+g)>>2)*(4*IR)+((140+g)&3)]=m;}
    if(r==0){unsigned words[3]={};for(int g=0;g<8;++g){int sum=0;for(int z=0;z<8;++z){float s,m;unsigned p=raw_pack4<3>(src,(i/256)*256+g*32+z*4,&s,&m);int v;asm("dp4a.s32.s32 %0,%1,%2,%3;":"=r"(v):"r"(p),"r"(0x01010101),"r"(sum));sum=v;}unsigned v=sum+0,bit=g*9,shift=bit&31;words[bit/32]|=v<<shift;if(shift>23)words[bit/32+1]|=v>>(32-shift);}for(int n=0;n<3;++n)*(unsigned*)(out+148*IR+n*(4*IR))=words[n];}
}
__global__ void k2_repack_q6(const unsigned char* src,unsigned char* dst,long long elems,int inner,int rows,int IR) {
    long long i=((long long)blockIdx.x*blockDim.x+threadIdx.x)*4;if(i>=elems || i%16)return;
    int r=i%256,g=r/16;long long row=i/inner;int e=row/rows,rw=row%rows;
    unsigned char* out=dst+(((long long)e*((rows+IR-1)/IR)+rw/IR)*(inner/256)+((i%inner)/256))*224*IR+(rw%IR)*4;
    unsigned words[3]={};int sum=0;
    for(int z=0;z<4;++z){float sc,m;unsigned p=raw_pack4<0>(src,i+z*4,&sc,&m);
        for(int n=0;n<4;++n){int code=(int)(signed char)(p>>(n*8));sum+=code;unsigned c=code+32,bit=(z*4+n)*6,pos=bit/32,shift=bit%32;words[pos]|=c<<shift;if(shift>26)words[pos+1]|=c>>(32-shift);}
    }
    for(int n=0;n<3;++n)*(unsigned*)(out+(g*12+n*4)*IR)=words[n];
    const unsigned char* in=src+(i/256)*224;
    if(r==0){out[192*IR]=in[208];out[192*IR+1]=in[209];}
    int off=194+g;out[(off/4)*(4*IR)+(off%4)]=in[192+g];

}
}
"#;

const SRC: &str = r#"#ifndef K2_ROUTED_GEMV_ROWS
#define K2_ROUTED_GEMV_ROWS 4
#endif
#ifndef K2_ROUTED_TILE
#define K2_ROUTED_TILE 64
#endif
#ifndef K2_ACTIVATION_BITS
#define K2_ACTIVATION_BITS 16
#endif

extern "C" __global__ void k2_quantize(const float* __restrict__ x,signed char* q,signed char* ql,float* __restrict__ scales,float* __restrict__ sl,float* __restrict__ sums,int inner,int count,unsigned* nib,int pack_nibbles) {
    int b=blockIdx.x*4+(threadIdx.x>>5),lane=threadIdx.x&31;if(b>=count)return;
    float v=x[(long long)b*32+lane],a=fabsf(v);
    for(int d=16;d>0;d>>=1)a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,d));
    constexpr float LIMIT=K2_ACTIVATION_BITS==8 ? 127.0f : (K2_ACTIVATION_BITS==12 ? 2039.0f : 32639.0f);
    float lower=v,upper=v;
    if(K2_ACTIVATION_BITS==8)for(int d=16;d>0;d>>=1){lower=fminf(lower,__shfl_xor_sync(0xffffffff,lower,d));upper=fmaxf(upper,__shfl_xor_sync(0xffffffff,upper,d));}
    float scale=K2_ACTIVATION_BITS==8 ? (a>0 ? fmaxf(fmaxf((upper*0.5f-lower*0.5f)/127.f,a/32512.f),1.17549435e-38f) : 1.f) : (a>0 ? a/LIMIT : 1.0f);
    float center=K2_ACTIVATION_BITS==8 ? fmaxf(-32512.f,fminf(32512.f,rintf((lower*0.5f+upper*0.5f)/scale))) : 0;
    int code=K2_ACTIVATION_BITS==8 ? (int)rintf(v/scale)-(int)center : (a>0 ? (int)rintf(v*(LIMIT/a)) : 0);code=max(-(int)LIMIT,min((int)LIMIT,code));
    constexpr int RADIX=K2_ACTIVATION_BITS==12 ? 16 : 256;
    int hi=K2_ACTIVATION_BITS==8 ? code : (code+RADIX/2)/(RADIX);
    if(K2_ACTIVATION_BITS!=8)hi=(code+RADIX/2)>>(K2_ACTIVATION_BITS==12 ? 4 : 8);
    int lo=K2_ACTIVATION_BITS==8 ? 0 : code-hi*RADIX;
    q[(long long)b*32+lane]=(signed char)hi;ql[(long long)b*32+lane]=(signed char)lo;
    if(pack_nibbles && K2_ACTIVATION_BITS>=12){
        #pragma unroll
        for(int n=0;n<K2_ACTIVATION_BITS/4;++n){unsigned word=(((unsigned)code>>(4*n))&15)<<(4*(lane&7));
            #pragma unroll
            for(int d=4;d>0;d>>=1)word|=__shfl_xor_sync(0xffffffff,word,d);
            if((lane&7)==0)nib[(long long)b*16+(lane/8)*4+n]=word;
        }
    }

    float sum=v;for(int d=16;d>0;d>>=1)sum+=__shfl_xor_sync(0xffffffff,sum,d);
    if(lane==0){scales[b]=scale;sl[b]=K2_ACTIVATION_BITS==8 ? center : scale;sums[b]=sum;}
}

template<int Q,int IR=4>
__device__ __forceinline__ const unsigned char* weight_row(const unsigned char* __restrict__ w,int e,int r,int inner,int rows){return w+((long long)e*((rows+IR-1)/IR)+r/IR)*IR*(inner/256)*(Q==3 ? 160 : 224)+(r%IR)*4;}
// Native records interleave aligned words across rows. Load d/dmin as one
// word and Q6 d as one halfword without changing the float conversion.
__device__ __forceinline__ float k2_half_bits(unsigned short bits){float f;asm("cvt.f32.f16 %0,%1;":"=f"(f):"h"(bits));return f;}
template<int Q,int IR=4>
__device__ __forceinline__ unsigned raw_pack4(const unsigned char* __restrict__ w,unsigned i,float* __restrict__ scale,float* __restrict__ minimum) {
    int r=i&255,g=r>>5,z=(r&31)>>2;const unsigned char* b=w+(i>>8)*(Q==3 ? 160 : 224)*IR;
    if(Q==3){unsigned dm=*(const unsigned*)(b+128*IR);*scale=k2_half_bits((unsigned short)dm)*(float)b[((132+g)>>2)*(4*IR)+((132+g)&3)];*minimum=k2_half_bits((unsigned short)(dm>>16))*(float)b[((140+g)>>2)*(4*IR)+((140+g)&3)];unsigned code=*(const unsigned*)(b+((z%4)*32+g*4)*IR);return (z<4 ? code : code>>4)&0x0f0f0f0f;}
    *scale=k2_half_bits(*(const unsigned short*)(b+192*IR))*(float)((const signed char*)b)[((194+r/16)>>2)*(4*IR)+((194+r/16)&3)];*minimum=0;
    int bit=(r%16)*6,word=bit/32,shift=bit%32,base=(r/16)*12;
    unsigned v=*(const unsigned*)(b+(base+word*4)*IR)>>shift;
    if(shift>8)v|=*(const unsigned*)(b+(base+(word+1)*4)*IR)<<(32-shift);
    unsigned code=(v&63)|(((v>>6)&63)<<8)|(((v>>12)&63)<<16)|(((v>>18)&63)<<24);
    return ((code|0x80808080)-0x20202020)^0x80808080;
}
template<int Q>
__device__ __forceinline__ float raw_term(float scale,int dot,int dotl,float minimum,float sum,float xs,float xsl) {
    float coefficient,result;
    if(K2_ACTIVATION_BITS==8) {
        int corrected=dot+(int)xsl*dotl;
        coefficient=__fmul_rn(scale,xs);result=__fmul_rn(coefficient,(float)corrected);
        return Q==0 ? result : __fmaf_rn(-minimum,sum,result);
    }
    asm("mul.rn.f32 %0,%1,%2;":"=f"(coefficient):"f"(scale),"f"(xs));
    asm("mul.rn.f32 %0,%1,%2;":"=f"(result):"f"(coefficient),"f"((float)(dot*(K2_ACTIVATION_BITS==12 ? 16 : 256)+dotl)));
    if(Q==3)asm("fma.rn.f32 %0,%1,%2,%3;":"=f"(result):"f"(-minimum),"f"(sum),"f"(result));return result;
}
__device__ __forceinline__ int raw_dp4a(unsigned a,unsigned b,int c) {int v;asm("dp4a.s32.s32 %0,%1,%2,%3;":"=r"(v):"r"(a),"r"(b),"r"(c));return v;}
__device__ __forceinline__ void raw_imma(int* d,unsigned a,unsigned b) {
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1},{%2},{%3},{%0,%1};":"+r"(d[0]),"+r"(d[1]):"r"(a),"r"(b));
}
__device__ __forceinline__ unsigned raw_nibbles(unsigned a,unsigned b) {
    // Pair even and odd bytes, then select their low nibbles. The masks
    // also preserve signed low activation digits in the twelve-bit path.
    unsigned even=__byte_perm(a,b,0x6420),odd=__byte_perm(a,b,0x7531);
    return (even&0x0f0f0f0f)|((odd<<4)&0xf0f0f0f0);
}
__device__ __forceinline__ void raw_imma4(int* d,unsigned a,unsigned b) {
    asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.s4.u4.s32 {%0,%1},{%2},{%3},{%0,%1};":"+r"(d[0]),"+r"(d[1]):"r"(a),"r"(b));
}

template<int Q,int IR=4>
__device__ __forceinline__ int weight_sum(const unsigned char* __restrict__ row,unsigned i){
    if(Q==0){int sum=0;float sc,m;for(int z=0;z<4;++z)sum=raw_dp4a(raw_pack4<Q,IR>(row,i+z*4,&sc,&m),0x01010101,sum);return sum;}
    const unsigned char* b=row+(i/256)*160*IR;unsigned bit=((i&255)/32)*9,pos=bit>>5,shift=bit&31;
    const unsigned* p=(const unsigned*)(b+148*IR);unsigned code=p[pos*IR]>>shift;if(shift>23)code|=p[(pos+1)*IR]<<(32-shift);return (int)(code&511);
}
template<int IR> __device__ __forceinline__ float q6_subscale(const unsigned char* __restrict__ row,int i){const unsigned char* b=row+(i/256)*224*IR;int off=194+(i%256)/16;return (float)((const signed char*)b)[(off/4)*(4*IR)+(off%4)];}
template<int IR> __device__ __forceinline__ float q6_delta(const unsigned char* __restrict__ row,int i){return k2_half_bits(*(const unsigned short*)(row+(i/256)*224*IR+192*IR));}
// Fixed hidden width removes dynamic address and tail arithmetic only.
// Every scale group and the fp32 accumulation tree stay in the same order.
template<int Q,int NT,bool ROUTED,int INNER=0>
__device__ __forceinline__ void raw_gemv(const unsigned char* __restrict__ w,const signed char* __restrict__ x,const signed char* __restrict__ xl,const float* __restrict__ sx,const float* __restrict__ sl,const float* __restrict__ sums,const int* __restrict__ ids,float* __restrict__ out,int inner,int rows,int pairs,int topk,int mode) {
    if(INNER)inner=INNER;
    constexpr int IR=ROUTED ? 4 : 1;
    constexpr int RW=ROUTED ? K2_ROUTED_GEMV_ROWS : 4,LANES=32/RW;
    if(blockIdx.x*4*RW>=rows)return;
    int lane=threadIdx.x&31,within=lane%LANES;
    int r=(blockIdx.x*4+(threadIdx.x>>5))*RW+lane/LANES,f0=blockIdx.y*NT;bool live=r<rows;
    int expert=mode ? ids[f0] : 0,group=32;
    int groups=inner>>5,blocks=inner>>5;float acc[NT]={};
    for(int g0=0;g0<groups;g0+=LANES) {
        int g=g0+within;float value[NT]={};
        if(g<groups && live) {

            float scale=0,minimum=0;int dot[NT]={},dotl[NT]={};const unsigned char* row=weight_row<Q,IR>(w,expert,r,inner,rows);
            if(Q==0){
                int dh0[NT]={},dl0[NT]={},dh1[NT]={},dl1[NT]={};
                #pragma unroll
                for(int z=0;z<8;++z){unsigned weight=raw_pack4<Q,IR>(row,g*32+z*4,&scale,&minimum);
                    #pragma unroll
                    for(int n=0;n<NT;++n)if(f0+n<pairs){int t=mode==1 ? (f0+n)/topk : f0+n;unsigned xh=*(const unsigned*)(x+(long long)t*inner+g*32+z*4),lo=*(const unsigned*)(xl+(long long)t*inner+g*32+z*4);
                        if(z<4){dh0[n]=raw_dp4a(weight,xh,dh0[n]);if(K2_ACTIVATION_BITS!=8)dl0[n]=raw_dp4a(weight,lo,dl0[n]);}
                        else {dh1[n]=raw_dp4a(weight,xh,dh1[n]);if(K2_ACTIVATION_BITS!=8)dl1[n]=raw_dp4a(weight,lo,dl1[n]);}
                    }
                }
                float d=q6_delta<IR>(row,g*32),s0=q6_subscale<IR>(row,g*32),s1=q6_subscale<IR>(row,g*32+16);
                #pragma unroll
                for(int n=0;n<NT;++n)if(f0+n<pairs){int t=mode==1 ? (f0+n)/topk : f0+n,b=t*blocks+g;
                    int v0=K2_ACTIVATION_BITS==8 ? dh0[n]+(int)sl[b]*weight_sum<Q,IR>(row,g*32) : dh0[n]*(K2_ACTIVATION_BITS==12 ? 16 : 256)+dl0[n];
                    int v1=K2_ACTIVATION_BITS==8 ? dh1[n]+(int)sl[b]*weight_sum<Q,IR>(row,g*32+16) : dh1[n]*(K2_ACTIVATION_BITS==12 ? 16 : 256)+dl1[n];
                    value[n]=__fmul_rn(__fmul_rn(d,sx[b]),__fmaf_rn((float)v1,s1,__fmul_rn((float)v0,s0)));
                }
            }else{
            if(ROUTED){
                #pragma unroll
                for(int z=0;z<(Q==3 ? 8 : 4);++z){unsigned weight=raw_pack4<Q,IR>(row,g*group+z*4,&scale,&minimum);
                    #pragma unroll
                    for(int n=0;n<NT;++n)if(f0+n<pairs){int t=mode==1 ? (f0+n)/topk : f0+n;dot[n]=raw_dp4a(weight,*(const unsigned*)(x+(long long)t*inner+g*group+z*4),dot[n]);if(K2_ACTIVATION_BITS!=8)dotl[n]=raw_dp4a(weight,*(const unsigned*)(xl+(long long)t*inner+g*group+z*4),dotl[n]);}
                }
            }else{
            raw_pack4<Q,IR>(row,g*group,&scale,&minimum);
            const unsigned char* block=row+(g/(Q==3 ? 8 : 16))*(Q==3 ? 160 : 224)*IR;
            #pragma unroll
            for(int z=0;z<4;++z){unsigned packed=*(const unsigned*)(block+((Q==3 ? z*32+(g&7)*4 : (z+((g&1)*4))*32+((g>>1)&7)*4)*IR));
                #pragma unroll
                for(int half=0;half<(Q==3 ? 2 : 1);++half){unsigned weight=Q==3 ? (packed>>(half*4))&0x0f0f0f0f : packed;
                    #pragma unroll
                    for(int n=0;n<NT;++n)if(f0+n<pairs){int t=mode==1 ? (f0+n)/topk : f0+n,idx=t*inner+g*group+z*4+half*16;dot[n]=raw_dp4a(weight,*(const unsigned*)(x+idx),dot[n]);if(K2_ACTIVATION_BITS!=8)dotl[n]=raw_dp4a(weight,*(const unsigned*)(xl+idx),dotl[n]);}
                }
            }
            }
            }
            int b=g;
            #pragma unroll
            for(int n=0;n<NT;++n)if(Q==3 && f0+n<pairs) {
                int t=mode==1 ? (f0+n)/topk : f0+n;
                value[n]=raw_term<Q>(scale,dot[n],K2_ACTIVATION_BITS==8 ? weight_sum<Q,IR>(weight_row<Q,IR>(w,expert,r,inner,rows),g*group) : dotl[n],minimum,sums[t*blocks+b],sx[t*blocks+b],sl[t*blocks+b]);
            }
        }
        #pragma unroll
        for(int n=0;n<NT;++n) {
            #pragma unroll
            for(int d=2;d>0;d>>=1)value[n]+=__shfl_xor_sync(0xffffffff,value[n],d);
            for(int l=0;l<min(LANES,groups-g0);l+=4)acc[n]+=__shfl_sync(0xffffffff,value[n],l+lane/LANES*LANES);
        }
    }
    if(within==0 && live) {
        #pragma unroll
        for(int n=0;n<NT;++n)if(f0+n<pairs)out[(long long)(f0+n)*rows+r]=acc[n];
    }
}
template<int Q,int BT,int MF,int IR>
__device__ void raw_gemm_fallback(const unsigned char* __restrict__ w,const signed char* __restrict__ x,const signed char* __restrict__ xl,const float* __restrict__ sx,const float* __restrict__ sl,const float* __restrict__ xsum,const int* __restrict__ sorted,const int* __restrict__ owners,float* __restrict__ out,int inner,int rows,int pairs,int topk,int mode) {
    constexpr int RN=64,NF=RN/8,NG=Q==3 ? 4 : 8,CS=NG+1;
    constexpr bool S4=Q==3 && K2_ACTIVATION_BITS==12;
    int expert=mode ? owners[blockIdx.y] : 0;if(expert<0)return;
    __shared__ unsigned ws[RN][32],xs[BT][32],xlow[BT][S4 ? 16 : 32],wlow[S4 ? RN : 1][S4 ? 16 : 1];
    __shared__ float scales[RN][CS],mins[Q==3 ? RN : 1][Q==3 ? CS : 1],xscale[BT][4],xscalel[BT][4];
    __shared__ float sums[BT][4];__shared__ int flats[BT];__shared__ float delta[Q==0 ? RN : 1];__shared__ int wsum[RN][CS];
    int tid=threadIdx.x,lane=tid&31,warp=tid>>5,warps=BT/(8*MF),r0=blockIdx.x*RN,blocks=inner/32;
    for(int t=tid;t<BT;t+=blockDim.x)flats[t]=mode ? sorted[blockIdx.y*BT+t] : blockIdx.y*BT+t;
    __syncthreads();float acc[MF][NF][2];
    #pragma unroll
    for(int a=0;a<MF;++a)for(int n=0;n<NF;++n)acc[a][n][0]=acc[a][n][1]=0;
    for(int j=0;j<inner;j+=128) {

        for(int rb=warp*4;rb<RN;rb+=warps*4){int r=rb+lane/8;
            #pragma unroll
            for(int z=0;z<4;++z){int pos=(lane%4)*8+(lane/4%2)*4+z;float sc=0,mn=0;unsigned v=0;const unsigned char* row=weight_row<Q,IR>(w,expert,r0+r,inner,rows);if(r0+r<rows)v=raw_pack4<Q,IR>(row,j+pos*4,&sc,&mn);
                ws[r][pos^((r&7)*4)]=v;
                if(S4 && (pos&1)==0){float s1,m1;unsigned next=0;if(r0+r<rows)next=raw_pack4<Q,IR>(row,j+pos*4+4,&s1,&m1);wlow[r][(pos/2)^(((r>>1)&3)*4)]=raw_nibbles(v,next);}
                if(pos%(Q==3 ? 8 : 4)==0){int g=pos/(Q==3 ? 8 : 4);scales[r][g]=Q==0 && r0+r<rows ? q6_subscale<IR>(row,j+pos*4) : sc;if(Q==0 && pos==0)delta[r]=r0+r<rows ? q6_delta<IR>(row,j) : 0;if(Q==3)mins[r][g]=mn;if(K2_ACTIVATION_BITS==8)wsum[r][g]=r0+r<rows ? weight_sum<Q,IR>(row,j+pos*4) : 0;}
            }
        }
        for(int t=warp;t<BT;t+=warps) {
            int f=flats[t],input=mode==1 ? f/topk : f;
            xs[t][lane^((t&7)*4)]=f<pairs ? *(const unsigned*)(x+(long long)input*inner+j+lane*4) : 0;
            if(S4){if(lane<16){unsigned lo=0,hi=0;if(f<pairs){const unsigned* p=(const unsigned*)(xl+(long long)input*inner+j+lane*8);lo=p[0];hi=p[1];}xlow[t][lane^(((t>>1)&3)*4)]=raw_nibbles(lo,hi);}}
            else xlow[t][lane^((t&7)*4)]=f<pairs ? *(const unsigned*)(xl+(long long)input*inner+j+lane*4) : 0;
            if(lane<4){xscalel[t][lane]=f<pairs ? sl[(long long)input*blocks+j/32+lane] : 0;xscale[t][lane]=f<pairs ? sx[(long long)input*blocks+j/32+lane] : 0;sums[t][lane]=f<pairs ? xsum[(long long)input*blocks+j/32+lane] : 0;}
        }
        __syncthreads();
        #pragma unroll
        for(int a=0;a<MF;++a)if(flats[(warp*MF+a)*8]<pairs) {
            #pragma unroll
            for(int nb=0;nb<NF;nb+=4) {
                constexpr int NP=2;
                float partial[NP][4][2]={},first[4][2]={};
                #pragma unroll
                for(int g=0;g<NG;++g) {
                    int tmp[4][2]={},tmpl[4][2]={};
                    #pragma unroll
                    for(int k=0;k<(Q==3 ? 2 : 1);++k) {
                        int pos=g*(Q==3 ? 8 : 4)+k*4+(lane&3);
                        unsigned act=xs[(warp*MF+a)*8+(lane>>2)][pos^((lane>>2)*4)],actl=S4 ? 0 : xlow[(warp*MF+a)*8+(lane>>2)][pos^((lane>>2)*4)];
                        #pragma unroll
                        for(int n=0;n<4;++n){unsigned weight=ws[(nb+n)*8+(lane>>2)][pos^((lane>>2)*4)];raw_imma(tmp[n],act,weight);if(K2_ACTIVATION_BITS!=8 && !S4)raw_imma(tmpl[n],actl,weight);}
                    }
                    if(S4) {
                        int pos=g*4+(lane&3);
                        unsigned low=xlow[(warp*MF+a)*8+(lane>>2)][pos^(((lane>>3)&3)*4)];
                        #pragma unroll
                        for(int n=0;n<4;++n){unsigned weight=wlow[(nb+n)*8+(lane>>2)][pos^(((lane>>3)&3)*4)];raw_imma4(tmpl[n],low,weight);}
                    }
                    #pragma unroll
                    for(int n=0;n<4;++n) {
                        int t=(warp*MF+a)*8+(lane>>2),r=(nb+n)*8+(lane&3)*2,h=Q==3 ? g : g/2;
                        float scale=xscale[t][h],scalel=xscalel[t][h],sum=Q==3 ? sums[t][h] : 0;
                        if(Q==3){
                        partial[g%NP][n][0]+=raw_term<Q>(scales[r][g],tmp[n][0],K2_ACTIVATION_BITS==8 ? wsum[r][g] : tmpl[n][0],Q==3 ? mins[r][g] : 0,sum,scale,scalel);
                        partial[g%NP][n][1]+=raw_term<Q>(scales[r+1][g],tmp[n][1],K2_ACTIVATION_BITS==8 ? wsum[r+1][g] : tmpl[n][1],Q==3 ? mins[r+1][g] : 0,sum,scale,scalel);
                        }else{
                            int v0=K2_ACTIVATION_BITS==8 ? tmp[n][0]+(int)scalel*wsum[r][g] : tmp[n][0]*(K2_ACTIVATION_BITS==12 ? 16 : 256)+tmpl[n][0];
                            int v1=K2_ACTIVATION_BITS==8 ? tmp[n][1]+(int)scalel*wsum[r+1][g] : tmp[n][1]*(K2_ACTIVATION_BITS==12 ? 16 : 256)+tmpl[n][1];
                            if((g&1)==0){first[n][0]=__fmul_rn((float)v0,scales[r][g]);first[n][1]=__fmul_rn((float)v1,scales[r+1][g]);}
                            else {int p=(g/2)%NP;partial[p][n][0]+=__fmul_rn(__fmul_rn(delta[r],scale),__fmaf_rn((float)v0,scales[r][g],first[n][0]));partial[p][n][1]+=__fmul_rn(__fmul_rn(delta[r+1],scale),__fmaf_rn((float)v1,scales[r+1][g],first[n][1]));}
                        }

                    }
                }
                #pragma unroll
                for(int n=0;n<4;++n)for(int z=0;z<2;++z) {
                    float total=partial[0][n][z]+partial[1][n][z];
                    acc[a][nb+n][z]+=total;
                }
            }
        }
        __syncthreads();
    }
    for(int a=0;a<MF;++a)for(int n=0;n<NF;++n)for(int z=0;z<2;++z){int r=r0+n*8+(lane&3)*2+z,t=(warp*MF+a)*8+(lane>>2),f=flats[t];if(r<rows && f<pairs)out[(long long)f*rows+r]=acc[a][n][z];}
}

// Operand ownership follows llama.cpp's Turing MMQ register tiles:
// ggml-cuda/mmq-vec-dot.cuh, ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma.
// A warp retains one weight row fragment and its scales while visiting token
// columns. The sixteen-bit dot and 128-value fp32 tree are unchanged.
template<int Q,int BT,int MF,int IR>
__device__ void raw_gemm_reuse(const unsigned char* __restrict__ w,const signed char* __restrict__ x,const signed char* __restrict__ xl,const float* __restrict__ sx,const float* __restrict__ sl,const float* __restrict__ xsum,const int* __restrict__ sorted,const int* __restrict__ owners,float* __restrict__ out,int inner,int rows,int pairs,int topk,int mode) {
    constexpr int RN=64,NG=Q==3 ? 4 : 8,CS=NG+1;
    constexpr bool S4=Q==3 && K2_ACTIVATION_BITS==12;
    int expert=mode ? owners[blockIdx.y] : 0;if(expert<0)return;
    __shared__ unsigned ws[RN][32],xs[BT][32],xlow[BT][S4 ? 16 : 32],wlow[S4 ? RN : 1][S4 ? 16 : 1];
    __shared__ float scales[RN][CS],mins[Q==3 ? RN : 1][Q==3 ? CS : 1],xscale[BT][4],xscalel[BT][4];
    __shared__ float sums[BT][4];__shared__ int flats[BT];__shared__ float delta[Q==0 ? RN : 1];__shared__ int wsum[RN][CS];
    int tid=threadIdx.x,lane=tid&31,warp=tid>>5,warps=BT/(8*MF),r0=blockIdx.x*RN,blocks=inner/32;
    for(int t=tid;t<BT;t+=blockDim.x)flats[t]=mode ? sorted[blockIdx.y*BT+t] : blockIdx.y*BT+t;
    constexpr int WR=RN/(BT/(8*MF))/8,TF=BT/8;
    __syncthreads();float acc[TF][WR][2];
    #pragma unroll
    for(int a=0;a<TF;++a)for(int n=0;n<WR;++n)acc[a][n][0]=acc[a][n][1]=0;
    for(int j=0;j<inner;j+=128) {

        for(int rb=warp*4;rb<RN;rb+=warps*4){int r=rb+lane/8;
            #pragma unroll
            for(int z=0;z<4;++z){int pos=(lane%4)*8+(lane/4%2)*4+z;float sc=0,mn=0;unsigned v=0;const unsigned char* row=weight_row<Q,IR>(w,expert,r0+r,inner,rows);if(r0+r<rows)v=raw_pack4<Q,IR>(row,j+pos*4,&sc,&mn);
                ws[r][pos^((r&7)*4)]=v;
                if(S4 && (pos&1)==0){float s1,m1;unsigned next=0;if(r0+r<rows)next=raw_pack4<Q,IR>(row,j+pos*4+4,&s1,&m1);wlow[r][(pos/2)^(((r>>1)&3)*4)]=raw_nibbles(v,next);}
                if(pos%(Q==3 ? 8 : 4)==0){int g=pos/(Q==3 ? 8 : 4);scales[r][g]=Q==0 && r0+r<rows ? q6_subscale<IR>(row,j+pos*4) : sc;if(Q==0 && pos==0)delta[r]=r0+r<rows ? q6_delta<IR>(row,j) : 0;if(Q==3)mins[r][g]=mn;if(K2_ACTIVATION_BITS==8)wsum[r][g]=r0+r<rows ? weight_sum<Q,IR>(row,j+pos*4) : 0;}
            }
        }
        for(int t=warp;t<BT;t+=warps) {
            int f=flats[t],input=mode==1 ? f/topk : f;
            xs[t][lane^((t&7)*4)]=f<pairs ? *(const unsigned*)(x+(long long)input*inner+j+lane*4) : 0;
            if(S4){if(lane<16){unsigned lo=0,hi=0;if(f<pairs){const unsigned* p=(const unsigned*)(xl+(long long)input*inner+j+lane*8);lo=p[0];hi=p[1];}xlow[t][lane^(((t>>1)&3)*4)]=raw_nibbles(lo,hi);}}
            else xlow[t][lane^((t&7)*4)]=f<pairs ? *(const unsigned*)(xl+(long long)input*inner+j+lane*4) : 0;
            if(lane<4){xscalel[t][lane]=f<pairs ? sl[(long long)input*blocks+j/32+lane] : 0;xscale[t][lane]=f<pairs ? sx[(long long)input*blocks+j/32+lane] : 0;sums[t][lane]=f<pairs ? xsum[(long long)input*blocks+j/32+lane] : 0;}
        }
        __syncthreads();
        unsigned weights[WR][8];float sc[WR][8],ds[WR];
        #pragma unroll
        for(int n=0;n<WR;++n) {
            int r=(warp*WR+n)*8+(lane>>2);ds[n]=delta[r];
            #pragma unroll
            for(int g=0;g<8;++g) {
                int pos=(g*4+(lane&3))^((lane>>2)*4);
                weights[n][g]=ws[r][pos];sc[n][g]=scales[r][g];
            }
        }
        #pragma unroll
        for(int a=0;a<TF;++a)if(flats[a*8]<pairs) {
            float partial[2][WR][2]={},first[WR][2]={};
            #pragma unroll
            for(int g=0;g<8;++g) {
                int total[WR][2]={},low[WR][2]={};int pos=(g*4+(lane&3))^((lane>>2)*4);
                unsigned act=xs[a*8+(lane>>2)][pos],actl=xlow[a*8+(lane>>2)][pos];
                #pragma unroll
                for(int n=0;n<WR;++n) {
                    raw_imma(total[n],weights[n][g],act);raw_imma(low[n],weights[n][g],actl);
                    int t=a*8+(lane&3)*2;
                    #pragma unroll
                    for(int z=0;z<2;++z) {
                        int v=total[n][z]*(K2_ACTIVATION_BITS==12 ? 16 : 256)+low[n][z];
                        if((g&1)==0)first[n][z]=__fmul_rn((float)v,sc[n][g]);
                        else partial[(g/2)%2][n][z]+=__fmul_rn(__fmul_rn(ds[n],xscale[t+z][g/2]),__fmaf_rn((float)v,sc[n][g],first[n][z]));
                    }
                }
            }
            #pragma unroll
            for(int n=0;n<WR;++n)for(int z=0;z<2;++z)acc[a][n][z]+=partial[0][n][z]+partial[1][n][z];
        }
        __syncthreads();
    }
    for(int a=0;a<TF;++a)for(int n=0;n<WR;++n)for(int z=0;z<2;++z){int r=r0+(warp*WR+n)*8+(lane>>2),t=a*8+(lane&3)*2+z,f=flats[t];if(r<rows && f<pairs)out[(long long)f*rows+r]=acc[a][n][z];}
}

__device__ __forceinline__ void raw_imma4u(int* d,unsigned a,unsigned b) {
    asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.u4.u4.s32 {%0,%1},{%2},{%3},{%0,%1};":"+r"(d[0]),"+r"(d[1]):"r"(a),"r"(b));
}
__device__ __forceinline__ float raw_full_term(float scale,int dot,float minimum,float sum,float xs) {
    float coefficient,result;
    asm("mul.rn.f32 %0,%1,%2;":"=f"(coefficient):"f"(scale),"f"(xs));
    asm("mul.rn.f32 %0,%1,%2;":"=f"(result):"f"(coefficient),"f"((float)dot));
    asm("fma.rn.f32 %0,%1,%2,%3;":"=f"(result):"f"(-minimum),"f"(sum),"f"(result));return result;
}
__device__ __forceinline__ void raw_imma4us(int* d,unsigned a,unsigned b) {
    asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.u4.s4.s32 {%0,%1},{%2},{%3},{%0,%1};":"+r"(d[0]),"+r"(d[1]):"r"(a),"r"(b));
}
template<int BT,int MF,int IR,int INNER=0>
__device__ __forceinline__ void raw_gemm_q4_nibbles(const unsigned char* __restrict__ w,const signed char* __restrict__ x,const signed char* __restrict__ xl,const float* __restrict__ sx,const float* __restrict__ sl,const float* __restrict__ xsum,const int* __restrict__ sorted,const int* __restrict__ owners,float* __restrict__ out,int inner,int rows,int pairs,int topk,int mode,const unsigned* __restrict__ nib) {
    if(INNER)inner=INNER;
    constexpr int RN=BT==64 ? 128 : 64,NP=K2_ACTIVATION_BITS/4;
    if(blockIdx.x*RN>=rows)return;
    int expert=mode ? owners[blockIdx.y] : 0;if(expert<0)return;
    __shared__ unsigned wn[RN][16],an[NP][BT][16];
    __shared__ float scales[RN][5],mins[RN][5],xscale[BT][4],sums[BT][4];
    __shared__ int flats[BT];
    int tid=threadIdx.x,lane=tid&31,warp=tid>>5,warps=BT/(8*MF),r0=blockIdx.x*RN,blocks=inner/32;
    for(int t=tid;t<BT;t+=blockDim.x)flats[t]=mode ? sorted[blockIdx.y*BT+t] : blockIdx.y*BT+t;
    constexpr int WR=RN/(BT/(8*MF))/8, TF=BT/8;
    __syncthreads();float acc[TF][WR][2]={};
    for(int j=0;j<inner;j+=128) {

        for(int rb=warp*4;rb<RN;rb+=warps*4){int r=rb+lane/8;
            #pragma unroll
            for(int u=0;u<2;++u){int pos=lane%8+u*8;float scale=0,minimum=0,s1,m1;unsigned a=0,b=0;if(r0+r<rows){const unsigned char* row=weight_row<3,IR>(w,expert,r0+r,inner,rows);a=raw_pack4<3,IR>(row,j+pos*8,&scale,&minimum);b=raw_pack4<3,IR>(row,j+pos*8+4,&s1,&m1);}
                wn[r][pos^(((r>>1)&3)*4)]=raw_nibbles(a,b);if((pos&3)==0){scales[r][pos/4]=scale;mins[r][pos/4]=minimum;}
            }
        }
        for(int t=warp;t<BT;t+=warps)if(lane<16) {
            int f=flats[t],input=mode==1 ? f/topk : f;
            int pos=lane^(((t>>1)&3)*4);
            uint4 packed=make_uint4(0u,0u,0u,0u);if(f<pairs)packed=((const uint4*)nib)[(long long)input*blocks*4+j/8+lane];
            an[0][t][pos]=packed.x;an[1][t][pos]=packed.y;an[2][t][pos]=packed.z;if(NP==4)an[3][t][pos]=packed.w;
            if(lane<4){xscale[t][lane]=f<pairs ? sx[(long long)input*blocks+j/32+lane] : 0;sums[t][lane]=f<pairs ? xsum[(long long)input*blocks+j/32+lane] : 0;}
        }
        __syncthreads();
        // Two weight row fragments per warp, reused over all token columns.
        unsigned weights[WR][4];float sc[WR][4],mn[WR][4];
        #pragma unroll
        for(int n=0;n<WR;++n) {
            int r=(warp*WR+n)*8+(lane>>2);
            #pragma unroll
            for(int g=0;g<4;++g) {
                int pos=(g*4+(lane&3))^(((lane>>3)&3)*4);
                weights[n][g]=wn[r][pos];sc[n][g]=scales[r][g];mn[n][g]=mins[r][g];
            }
        }
        #pragma unroll
        for(int a=0;a<TF;++a)if(flats[a*8]<pairs) {
            float partial[2][WR][2]={};
            #pragma unroll
            for(int g=0;g<4;++g) {
                int total[WR][2]={};int pos=(g*4+(lane&3))^(((lane>>3)&3)*4);
                #pragma unroll
                for(int ni=NP-1;ni>=0;--ni) {
                    unsigned act=an[ni][a*8+(lane>>2)][pos];
                    #pragma unroll
                    for(int n=0;n<WR;++n) {
                        if(ni!=NP-1){total[n][0]*=16;total[n][1]*=16;}
                        if(ni==NP-1)raw_imma4us(total[n],weights[n][g],act);else raw_imma4u(total[n],weights[n][g],act);
                    }
                }
                int t=a*8+(lane&3)*2;
                #pragma unroll
                for(int n=0;n<WR;++n) {
                    partial[g%2][n][0]+=raw_full_term(sc[n][g],total[n][0],mn[n][g],sums[t][g],xscale[t][g]);
                    partial[g%2][n][1]+=raw_full_term(sc[n][g],total[n][1],mn[n][g],sums[t+1][g],xscale[t+1][g]);
                }
            }
            #pragma unroll
            for(int n=0;n<WR;++n)for(int z=0;z<2;++z)acc[a][n][z]+=partial[0][n][z]+partial[1][n][z];
        }
        __syncthreads();
    }
    for(int a=0;a<TF;++a)for(int n=0;n<WR;++n)for(int z=0;z<2;++z){int r=r0+(warp*WR+n)*8+(lane>>2),t=a*8+(lane&3)*2+z,f=flats[t];if(r<rows && f<pairs)out[(long long)f*rows+r]=acc[a][n][z];}
}

// Each fixed 64-slot dispatch run becomes two 32-token CTAs. Put the
// subtile in grid.x so dispatch capacity does not consume more grid.y.
// Unlike the dense tile, every warp reads its own weight fragments directly.
template<int INNER=0>
__device__ __forceinline__ void raw_gemm_q4_registers(const unsigned char* __restrict__ w,const signed char* __restrict__ x,const signed char* __restrict__ xl,const float* __restrict__ sx,const float* __restrict__ sl,const float* __restrict__ xsum,const int* __restrict__ sorted,const int* __restrict__ owners,float* __restrict__ out,int inner,int rows,int pairs,int topk,int mode,const unsigned* __restrict__ nib) {
    if(INNER)inner=INNER;
    constexpr int RN=128,BT=32,IR=4,NP=K2_ACTIVATION_BITS/4;
    int row_tile=blockIdx.x/2,token_base=blockIdx.y*64+(blockIdx.x%2)*BT;
    if(row_tile*RN>=rows)return;
    int expert=owners[blockIdx.y];if(expert<0)return;
    __shared__ unsigned an[NP][BT][16];
    __shared__ float xscale[BT][4],sums[BT][4];
    __shared__ int flats[BT];
    int tid=threadIdx.x,lane=tid&31,warp=tid>>5,warps=8,r0=row_tile*RN,blocks=inner/32;
    for(int t=tid;t<BT;t+=blockDim.x)flats[t]=sorted[token_base+t];
    constexpr int WR=RN/8/8, TF=BT/8;
    __syncthreads();if(flats[0]>=pairs)return;float acc[TF][WR][2]={};
    for(int j=0;j<inner;j+=128) {

        unsigned weights[WR][4];float sc[WR][4],mn[WR][4];
        #pragma unroll
        for(int n=0;n<WR;++n) {
            int r=r0+(warp*WR+n)*8+(lane>>2);
            const unsigned char* row=weight_row<3,IR>(w,expert,r,inner,rows);
            #pragma unroll
            for(int g=0;g<4;++g) {
                int pos=j+g*32+(lane&3)*8;float scale=0,minimum=0,s1,m1;unsigned a=0,b=0;
                if(r<rows){a=raw_pack4<3,IR>(row,pos,&scale,&minimum);b=raw_pack4<3,IR>(row,pos+4,&s1,&m1);}
                weights[n][g]=raw_nibbles(a,b);sc[n][g]=scale;mn[n][g]=minimum;
            }
        }
        for(int t=warp;t<BT;t+=warps)if(lane<16) {
            int f=flats[t],input=mode==1 ? f/topk : f;
            int pos=lane^(((t>>1)&3)*4);
            uint4 packed=make_uint4(0u,0u,0u,0u);if(f<pairs)packed=((const uint4*)nib)[(long long)input*blocks*4+j/8+lane];
            an[0][t][pos]=packed.x;an[1][t][pos]=packed.y;an[2][t][pos]=packed.z;if(NP==4)an[3][t][pos]=packed.w;
            if(lane<4){xscale[t][lane]=f<pairs ? sx[(long long)input*blocks+j/32+lane] : 0;sums[t][lane]=f<pairs ? xsum[(long long)input*blocks+j/32+lane] : 0;}
        }
        __syncthreads();
        #pragma unroll
        for(int a=0;a<TF;++a)if(flats[a*8]<pairs) {
            float partial[2][WR][2]={};
            #pragma unroll
            for(int g=0;g<4;++g) {
                int total[WR][2]={};int pos=(g*4+(lane&3))^(((lane>>3)&3)*4);
                #pragma unroll
                for(int ni=NP-1;ni>=0;--ni) {
                    unsigned act=an[ni][a*8+(lane>>2)][pos];
                    #pragma unroll
                    for(int n=0;n<WR;++n) {
                        if(ni!=NP-1){total[n][0]*=16;total[n][1]*=16;}
                        if(ni==NP-1)raw_imma4us(total[n],weights[n][g],act);else raw_imma4u(total[n],weights[n][g],act);
                    }
                }
                int t=a*8+(lane&3)*2;
                #pragma unroll
                for(int n=0;n<WR;++n) {
                    partial[g%2][n][0]+=raw_full_term(sc[n][g],total[n][0],mn[n][g],sums[t][g],xscale[t][g]);
                    partial[g%2][n][1]+=raw_full_term(sc[n][g],total[n][1],mn[n][g],sums[t+1][g],xscale[t+1][g]);
                }
            }
            #pragma unroll
            for(int n=0;n<WR;++n)for(int z=0;z<2;++z)acc[a][n][z]+=partial[0][n][z]+partial[1][n][z];
        }
        __syncthreads();
    }
    for(int a=0;a<TF;++a)for(int n=0;n<WR;++n)for(int z=0;z<2;++z){int r=r0+(warp*WR+n)*8+(lane>>2),t=a*8+(lane&3)*2+z,f=flats[t];if(r<rows && f<pairs)out[(long long)f*rows+r]=acc[a][n][z];}
}

extern "C" {
#define GEMV_ENTRY(NAME,Q,NT,ROUTED) __global__ void NAME(const unsigned char* __restrict__ w,const signed char* __restrict__ x,const signed char* __restrict__ xl,const float* __restrict__ sx,const float* __restrict__ sl,const float* __restrict__ sums,const int* __restrict__ ids,float* __restrict__ out,int inner,int rows,int pairs,int topk,int mode,const unsigned* __restrict__ nib) {if(!ROUTED && inner==2560)raw_gemv<Q,NT,ROUTED,2560>(w,x,xl,sx,sl,sums,ids,out,inner,rows,pairs,topk,mode);else raw_gemv<Q,NT,ROUTED>(w,x,xl,sx,sl,sums,ids,out,inner,rows,pairs,topk,mode);}
#define GEMM_ENTRY(NAME,Q,BT,MF,IR,INNER) __global__ __launch_bounds__(256,2) void NAME(const unsigned char* __restrict__ w,const signed char* __restrict__ x,const signed char* __restrict__ xl,const float* __restrict__ sx,const float* __restrict__ sl,const float* __restrict__ sums,const int* __restrict__ sorted,const int* __restrict__ owners,float* __restrict__ out,int inner,int rows,int pairs,int topk,int mode,const unsigned* __restrict__ nib) {if(Q==3 && K2_ACTIVATION_BITS>=12)raw_gemm_q4_nibbles<BT,MF,IR,INNER>(w,x,xl,sx,sl,sums,sorted,owners,out,inner,rows,pairs,topk,mode,nib);else if(Q==0 && K2_ACTIVATION_BITS>=12)raw_gemm_reuse<Q,BT,MF,IR>(w,x,xl,sx,sl,sums,sorted,owners,out,inner,rows,pairs,topk,mode);else raw_gemm_fallback<Q,BT,MF,IR>(w,x,xl,sx,sl,sums,sorted,owners,out,inner,rows,pairs,topk,mode);}
#define GEMM_REG_ENTRY(NAME,INNER) __global__ __launch_bounds__(256,(K2_ACTIVATION_BITS>=12 ? 3 : 2)) void NAME(const unsigned char* w,const signed char* x,const signed char* xl,const float* sx,const float* sl,const float* sums,const int* sorted,const int* owners,float* out,int inner,int rows,int pairs,int topk,int mode,const unsigned* nib) {if(K2_ACTIVATION_BITS>=12)raw_gemm_q4_registers<INNER>(w,x,xl,sx,sl,sums,sorted,owners,out,inner,rows,pairs,topk,mode,nib);else raw_gemm_fallback<3,64,1,4>(w,x,xl,sx,sl,sums,sorted,owners,out,inner,rows,pairs,topk,mode);}
GEMV_ENTRY(k2_gemv_q4,3,1,true)
GEMV_ENTRY(k2_gemv_dense_q4,3,K2_DENSE_TOKENS,false)
GEMV_ENTRY(k2_gemv_q6,0,1,true)
GEMV_ENTRY(k2_gemv_dense_q6,0,K2_DENSE_TOKENS,false)
GEMM_REG_ENTRY(k2_gemm_q4,0)
GEMM_ENTRY(k2_gemm_dense_q4,3,64,1,1,0)
GEMM_ENTRY(k2_gemm_q6,0,K2_ROUTED_TILE,1,4,0)
GEMM_ENTRY(k2_gemm_dense_q6,0,64,1,1,0)
GEMM_REG_ENTRY(k2_gemm_hidden_q4,2560)
GEMM_ENTRY(k2_gemm_hidden_dense_q4,3,64,1,1,2560)
}
"#;

/// A generation of quantized input in one fixed projection workspace.
/// Reuse explicitly while the input is unchanged; preparing again invalidates it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct K2Prepared {
    workspace: u64,
    generation: u64,
    inner: usize,
    len: usize,
    nibbles: bool,
}
static NEXT_WORKSPACE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Compiled quantized tensor-core projections, with fixed shape bounds.
pub struct K2Gemm {
    workspace: u64,
    prepared: Option<K2Prepared>,
    activation_bits: usize,
    routed_gemv_rows: usize,
    hidden_q4: CudaFunction,
    hidden_dense_q4: CudaFunction,
    functions: [CudaFunction; 2],
    gemv: [CudaFunction; 2],
    gemv_dense: [CudaFunction; 2],
    dense: [CudaFunction; 2],
    quantize: CudaFunction,
    q: CudaSlice<i8>,
    ql: CudaSlice<i8>,
    nibbles: CudaSlice<u32>,
    sl: CudaSlice<f32>,
    scales: CudaSlice<f32>,
    sums: CudaSlice<f32>,
    tokens: usize,
    capacity: usize,
    max_inner: usize,
    max_rows: usize,
    max_weight_elements: usize,
}
impl K2Gemm {
    /// Resident bytes after packing, including four-row padding for experts.
    pub fn weight_bytes(quant: ExpertQuant, inner: usize, rows: usize, experts: usize) -> usize {
        let rows = if experts > 1 {
            rows.div_ceil(4) * 4
        } else {
            rows
        };
        inner * rows * experts / 256 * if quant == ExpertQuant::Q4K { 160 } else { 224 }
    }
    /// Repack once at upload. Both formats retain their original scales;
    /// Q6 codes stay at six bits with their half delta and signed subscales.
    pub fn repack(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        src: &CudaSlice<u8>,
        quant: ExpertQuant,
        inner: usize,
        rows: usize,
        experts: usize,
    ) -> Result<CudaSlice<u8>, MoeError> {
        let elems = inner * rows * experts;
        let interleaved = experts > 1;
        if !inner.is_multiple_of(256)
            || inner == 0
            || rows == 0
            || experts == 0
            || !elems.is_multiple_of(256)
            || !matches!(quant, ExpertQuant::Q4K | ExpertQuant::Q6K)
        {
            return Err(MoeError::WrongElementCount {
                which: "K2 repack format/geometry",
                expected: 256,
                found: elems,
            });
        }
        check(
            "repack source",
            elems / quant.block_elements() * quant.block_bytes(),
            src.len(),
        )?;
        let prologue = super::moe::MOE_SRC
            .split("// The GEMM prologue, with the format")
            .next()
            .unwrap();
        let module = ctx.load_module(
            compile(&format!("{prologue}\n{REPACK_SRC}"), "k2_repack")
                .map_err(MoeError::Compile)?,
        )?;
        let f = module.load_function(if quant == ExpertQuant::Q4K {
            "k2_repack_q4"
        } else {
            "k2_repack_q6"
        })?;
        let mut out = stream.alloc_zeros::<u8>(Self::weight_bytes(quant, inner, rows, experts))?;
        let ir = if interleaved { 4i32 } else { 1 };
        let (elems, inner, rows) = (elems as i64, inner as i32, rows as i32);
        // SAFETY: four source elements per thread, masked at elems; every
        // destination record is 160/224 bytes and header writers are disjoint.
        unsafe {
            stream
                .launch_builder(&f)
                .arg(src)
                .arg(&mut out)
                .arg(&elems)
                .arg(&inner)
                .arg(&rows)
                .arg(&ir)
                .launch(LaunchConfig {
                    grid_dim: ((elems as u64).div_ceil(4 * 256) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(out)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        tokens: usize,
        topk: usize,
        experts: usize,
        max_weight_elements: usize,
        max_inner: usize,
        max_rows: usize,
    ) -> Result<Self, MoeError> {
        Self::new_with_activation_bits(
            ctx,
            stream,
            tokens,
            topk,
            experts,
            max_weight_elements,
            max_inner,
            max_rows,
            16,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_activation_bits(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        tokens: usize,
        topk: usize,
        experts: usize,
        max_weight_elements: usize,
        max_inner: usize,
        max_rows: usize,
        activation_bits: usize,
    ) -> Result<Self, MoeError> {
        if tokens == 0
            || topk == 0
            || experts == 0
            || topk > experts
            || max_inner == 0
            || !max_inner.is_multiple_of(256)
            || max_rows == 0
        {
            return Err(MoeError::WrongElementCount {
                which: "K2 workspace geometry",
                expected: 256,
                found: max_inner,
            });
        }
        if ![8, 12, 16].contains(&activation_bits) {
            return Err(MoeError::WrongElementCount {
                which: "activation bits",
                expected: 16,
                found: activation_bits,
            });
        }
        let routed_gemv_rows = if tokens == 1 { 2 } else { 4 };
        let route_tile = super::k2::K2_ROUTED_TILE;
        let prologue = super::moe::MOE_SRC
            .split("// The GEMM prologue, with the format")
            .next()
            .unwrap();
        let dense_tokens = tokens.min(4);
        let ptx = compile(&format!("#define K2_DENSE_TOKENS {dense_tokens}\n#define K2_ROUTED_GEMV_ROWS {routed_gemv_rows}\n#define K2_ACTIVATION_BITS {activation_bits}\n#define K2_ROUTED_TILE {route_tile}\n{prologue}\n{SRC}"), "k2_gemm").map_err(MoeError::Compile)?;
        let module = ctx.load_module(ptx)?;
        Ok(Self {
            workspace: NEXT_WORKSPACE.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            prepared: None,
            activation_bits,
            routed_gemv_rows,
            hidden_q4: module.load_function("k2_gemm_hidden_q4")?,
            hidden_dense_q4: module.load_function("k2_gemm_hidden_dense_q4")?,
            functions: [
                module.load_function("k2_gemm_q4")?,
                module.load_function("k2_gemm_q6")?,
            ],
            dense: [
                module.load_function("k2_gemm_dense_q4")?,
                module.load_function("k2_gemm_dense_q6")?,
            ],
            gemv: [
                module.load_function("k2_gemv_q4")?,
                module.load_function("k2_gemv_q6")?,
            ],
            gemv_dense: [
                module.load_function("k2_gemv_dense_q4")?,
                module.load_function("k2_gemv_dense_q6")?,
            ],
            quantize: module.load_function("k2_quantize")?,
            q: stream.alloc_zeros::<i8>(tokens * topk * max_inner)?,
            ql: stream.alloc_zeros::<i8>(tokens * topk * max_inner)?,
            nibbles: stream.alloc_zeros::<u32>(tokens * topk * max_inner / 32 * 16)?,
            sl: stream.alloc_zeros::<f32>(tokens * topk * max_inner / 32)?,
            scales: stream.alloc_zeros::<f32>(tokens * topk * max_inner / 32)?,
            sums: stream.alloc_zeros::<f32>(tokens * topk * max_inner / 32)?,
            tokens,
            capacity: (tokens * topk).div_ceil(route_tile) + experts,
            max_inner,
            max_rows,
            max_weight_elements,
        })
    }
    /// Fixed activation workspaces, for numerical diagnostics.
    pub fn activation_parts(
        &self,
    ) -> (
        &CudaSlice<i8>,
        &CudaSlice<i8>,
        &CudaSlice<f32>,
        &CudaSlice<f32>,
    ) {
        (&self.q, &self.ql, &self.scales, &self.sl)
    }
    /// Quantize once for projections that share an unchanged input. Set
    /// `q4_nibbles` if any consumer is Q4_K; Q6 ignores the packed planes.
    pub fn prepare(
        &mut self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        inner: usize,
        q4_nibbles: bool,
    ) -> Result<K2Prepared, MoeError> {
        if inner == 0
            || !inner.is_multiple_of(256)
            || inner > self.max_inner
            || x.is_empty()
            || !x.len().is_multiple_of(inner)
            || x.len() / inner > self.q.len() / self.max_inner
        {
            return Err(MoeError::WrongElementCount {
                which: "K2 prepared input",
                expected: self.max_inner,
                found: inner,
            });
        }
        let prepared = K2Prepared {
            workspace: self.workspace,
            generation: self.prepared.map_or(1, |p| p.generation.wrapping_add(1)),
            inner,
            len: x.len(),
            nibbles: q4_nibbles,
        };
        let count = (x.len() / 32) as i32;
        let inner_i32 = inner as i32;
        let pack_nibbles = i32::from(q4_nibbles && self.tokens >= 8);
        // SAFETY: all arenas cover the checked token and inner dimensions.
        unsafe {
            stream
                .launch_builder(&self.quantize)
                .arg(x)
                .arg(&mut self.q)
                .arg(&mut self.ql)
                .arg(&mut self.scales)
                .arg(&mut self.sl)
                .arg(&mut self.sums)
                .arg(&inner_i32)
                .arg(&count)
                .arg(&mut self.nibbles)
                .arg(&pack_nibbles)
                .launch(LaunchConfig {
                    grid_dim: ((count as u32).div_ceil(4), 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        self.prepared = Some(prepared);
        Ok(prepared)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn project(
        &mut self,
        stream: &Arc<CudaStream>,
        w: &CudaSlice<u8>,
        quant: ExpertQuant,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        inner: usize,
        rows: usize,
        experts: usize,
        d: Option<&K2Dispatch>,
        flat_input: bool,
    ) -> Result<(), MoeError> {
        let prepared = self.prepare(stream, x, inner, quant == ExpertQuant::Q4K)?;
        self.project_prepared(
            stream, w, quant, prepared, out, inner, rows, experts, d, flat_input,
        )
    }
    /// Contract a previously prepared generation without requantizing.
    #[allow(clippy::too_many_arguments)]
    pub fn project_prepared(
        &mut self,
        stream: &Arc<CudaStream>,
        w: &CudaSlice<u8>,
        quant: ExpertQuant,
        prepared: K2Prepared,
        out: &mut CudaSlice<f32>,
        inner: usize,
        rows: usize,
        experts: usize,
        d: Option<&K2Dispatch>,
        flat_input: bool,
    ) -> Result<(), MoeError> {
        if self.prepared != Some(prepared)
            || prepared.workspace != self.workspace
            || prepared.inner != inner
            || (quant == ExpertQuant::Q4K && self.tokens >= 8 && !prepared.nibbles)
        {
            return Err(MoeError::WrongElementCount {
                which: "K2 prepared generation",
                expected: inner,
                found: prepared.inner,
            });
        }
        let (topk, batches, mode) = if let Some(d) = d {
            check("tokens", self.tokens, d.tokens)?;
            check("experts", experts, d.experts)?;
            check("tile", super::k2::K2_ROUTED_TILE, d.tile)?;
            (d.topk, d.owners.len(), if flat_input { 2i32 } else { 1 })
        } else {
            check("dense experts", 1, experts)?;
            (1, self.tokens.div_ceil(super::k2::K2_ROUTED_TILE), 0i32)
        };
        if inner == 0
            || rows == 0
            || !inner.is_multiple_of(256)
            || inner > self.max_inner
            || rows > self.max_rows
            || batches > self.capacity
            || inner * rows * experts > self.max_weight_elements
            || !matches!(quant, ExpertQuant::Q4K | ExpertQuant::Q6K)
        {
            return Err(MoeError::WrongElementCount {
                which: "K2 tensor geometry",
                expected: self.max_weight_elements,
                found: inner * rows * experts,
            });
        }
        check(
            "weights",
            Self::weight_bytes(quant, inner, rows, experts),
            w.len(),
        )?;
        check(
            "input",
            self.tokens * inner * if flat_input { topk } else { 1 },
            prepared.len,
        )?;
        check("output", self.tokens * topk * rows, out.len())?;
        let (inner, rows, pairs, topk) = (
            inner as i32,
            rows as i32,
            (self.tokens * topk) as i32,
            topk as i32,
        );
        // SAFETY: fixed workspace and geometry checks cover all indices;
        // rows and dispatch padding are masked, and dense ignores route pointers.
        unsafe {
            let small = self.tokens < 8;
            let wide = d.is_none() && self.tokens >= 8;
            let f = if small && d.is_none() {
                &self.gemv_dense[usize::from(quant == ExpertQuant::Q6K)]
            } else if small {
                &self.gemv[usize::from(quant == ExpertQuant::Q6K)]
            } else if quant == ExpertQuant::Q4K && self.activation_bits >= 12 && inner == 2560 {
                if wide {
                    &self.hidden_dense_q4
                } else {
                    &self.hidden_q4
                }
            } else if wide {
                &self.dense[usize::from(quant == ExpertQuant::Q6K)]
            } else {
                &self.functions[usize::from(quant == ExpertQuant::Q6K)]
            };
            let mut b = stream.launch_builder(f);
            b.arg(w)
                .arg(&self.q)
                .arg(&self.ql)
                .arg(&self.scales)
                .arg(&self.sl)
                .arg(&self.sums);
            if small {
                if let Some(d) = d {
                    b.arg(&d.direct);
                } else {
                    b.arg(w);
                }
            } else if let Some(d) = d {
                b.arg(&d.sorted).arg(&d.owners);
            } else {
                b.arg(w).arg(w);
            }
            b.arg(out)
                .arg(&inner)
                .arg(&rows)
                .arg(&pairs)
                .arg(&topk)
                .arg(&mode)
                .arg(&self.nibbles)
                .launch(LaunchConfig {
                    grid_dim: (
                        (rows as u32).div_ceil(if small {
                            if d.is_none() {
                                16
                            } else {
                                (4 * self.routed_gemv_rows) as u32
                            }
                        } else if quant == ExpertQuant::Q4K && self.activation_bits >= 12 {
                            128
                        } else {
                            64
                        }) * if !small
                            && d.is_some()
                            && quant == ExpertQuant::Q4K
                            && self.activation_bits >= 12
                        {
                            2
                        } else {
                            1
                        },
                        if small {
                            if d.is_none() {
                                (pairs as u32).div_ceil(4)
                            } else {
                                pairs as u32
                            }
                        } else if wide {
                            (self.tokens as u32).div_ceil(64)
                        } else {
                            batches as u32
                        },
                        1,
                    ),
                    block_dim: (
                        if small {
                            128
                        } else if wide {
                            256
                        } else {
                            super::k2::K2_ROUTED_TILE as u32 * 4
                        },
                        1,
                        1,
                    ),
                    shared_mem_bytes: 0,
                })?;
        }

        Ok(())
    }
}
fn check(which: &'static str, expected: usize, found: usize) -> Result<(), MoeError> {
    if expected != found {
        Err(MoeError::WrongElementCount {
            which,
            expected,
            found,
        })
    } else {
        Ok(())
    }
}
