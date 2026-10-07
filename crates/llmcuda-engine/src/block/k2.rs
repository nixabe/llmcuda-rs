//! K2-Horizon mixer and FFN integration. IFM llama.cpp `src/models/k2-horizon.cpp`
//! is the source for grouped norms, separate softplus gate, routed SiLU values,
//! sigmoid FFN routing and the ungated shared expert.
use super::{
    attention::{AttentionBlockError, AttnScratch, KvCache, RopeSource},
    moe::{MoeBlockError, dense_tensor, quantized_tensor},
};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr};
use llmcuda_cuda::kernels::{
    attention::{AttentionKernels, AttnDecodeScratch, STEP_SLOTS},
    k2::{K2Dispatch, K2Kernels},
    k2_gemm::{K2Gemm, K2Prepared},
    k2_rope::K2Rope,
    layer_ops::LayerOpsKernels,
    moe::{ExpertQuant, MoeGeometry, to_device_layout},
};
use llmcuda_gguf::{GgmlType, GgufFile};
use llmcuda_model::{Directory, ModelConfig, Role};
use std::sync::Arc;

/// A projection with its own format; Q6_K uses the MoE padded device layout.
pub struct Projection {
    bytes: CudaSlice<u8>,
    quant: Option<ExpertQuant>,
    inner: usize,
    rows: usize,
    experts: usize,
}
impl Projection {
    fn upload(
        stream: &Arc<CudaStream>,
        file: &GgufFile,
        directory: &Directory<'_>,
        role: Role,
        layer: u32,
    ) -> Result<Self, MoeBlockError> {
        let entry = directory
            .find(role, Some(layer))
            .ok_or(MoeBlockError::MissingTensor { role, layer })?;
        let dims = &entry.info.dims;
        let inner = dims[0] as usize;
        let rows = dims[1] as usize;
        let experts = dims.get(2).copied().unwrap_or(1) as usize;
        let elems = inner * rows * experts;
        let (bytes, quant) = if entry.info.ggml_type == GgmlType::F32 {
            let src = file.tensor_bytes(&entry.spec.name).ok_or_else(|| {
                MoeBlockError::UnreadableTensor {
                    name: entry.spec.name.clone(),
                }
            })?;
            (stream.clone_htod(src)?, None)
        } else {
            let (src, q) = quantized_tensor(file, directory, role, layer, elems)?;
            {
                let source = stream.clone_htod(&*to_device_layout(q, src))?;
                (
                    K2Gemm::repack(stream.context(), stream, &source, q, inner, rows, experts)?,
                    Some(q),
                )
            }
        };
        Ok(Self {
            bytes,
            quant,
            inner,
            rows,
            experts,
        })
    }
    fn prefill(
        &self,
        ops: &K2Kernels,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        ids: &CudaSlice<i32>,
        out: &mut CudaSlice<f32>,
        gemm: &mut K2Gemm,
    ) -> Result<(), MoeBlockError> {
        if let Some(q) = self.quant {
            gemm.project(
                stream,
                &self.bytes,
                q,
                x,
                out,
                self.inner,
                self.rows,
                self.experts,
                None,
                false,
            )?;
        } else {
            self.forward(ops, stream, x, ids, out, 1, false)?;
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn prefill_prepared(
        &self,
        ops: &K2Kernels,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        ids: &CudaSlice<i32>,
        out: &mut CudaSlice<f32>,
        gemm: &mut K2Gemm,
        prepared: K2Prepared,
    ) -> Result<(), MoeBlockError> {
        if let Some(q) = self.quant {
            gemm.project_prepared(
                stream,
                &self.bytes,
                q,
                prepared,
                out,
                self.inner,
                self.rows,
                self.experts,
                None,
                false,
            )?;
        } else {
            self.forward(ops, stream, x, ids, out, 1, false)?;
        }
        Ok(())
    }
    /// `prefill` of `silu(gate) * up`. Quantized weights quantize the product
    /// straight from `gate` and `up`, bit for bit as after a separate SwiGLU;
    /// F32 weights materialize it in `inter`.
    #[allow(clippy::too_many_arguments)]
    fn prefill_swiglu(
        &self,
        ops: &K2Kernels,
        layer_ops: &LayerOpsKernels,
        stream: &Arc<CudaStream>,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        inter: &mut CudaSlice<f32>,
        ids: &CudaSlice<i32>,
        out: &mut CudaSlice<f32>,
        gemm: &mut K2Gemm,
    ) -> Result<(), MoeBlockError> {
        if let Some(q) = self.quant {
            let prepared =
                gemm.prepare_swiglu(stream, gate, up, self.inner, q == ExpertQuant::Q4K)?;
            gemm.project_prepared(
                stream,
                &self.bytes,
                q,
                prepared,
                out,
                self.inner,
                self.rows,
                self.experts,
                None,
                false,
            )?;
        } else {
            layer_ops.swiglu(stream, gate, up, inter, gate.len())?;
            self.forward(ops, stream, inter, ids, out, 1, false)?;
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        ops: &K2Kernels,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        ids: &CudaSlice<i32>,
        out: &mut CudaSlice<f32>,
        topk: usize,
        routed: bool,
    ) -> Result<(), MoeBlockError> {
        ops.project(
            stream,
            &self.bytes,
            self.quant,
            x,
            ids,
            out,
            self.inner,
            self.rows,
            self.experts,
            topk,
            routed,
        )?;
        Ok(())
    }
}
fn vector(
    stream: &Arc<CudaStream>,
    file: &GgufFile,
    d: &Directory<'_>,
    role: Role,
    layer: u32,
    n: usize,
) -> Result<CudaSlice<f32>, MoeBlockError> {
    let (v, _) = dense_tensor(file, d, role, layer, n)?;
    Ok(stream.clone_htod(&v)?)
}

pub(crate) struct K2ValueScratch {
    rotary: K2Rope,
    dispatch: K2Dispatch,
    batch_decode: Option<AttnDecodeScratch>,
    gemm: K2Gemm,
    logits: CudaSlice<f32>,
    ids: CudaSlice<i32>,
    weights: CudaSlice<f32>,
    projections: CudaSlice<f32>,
}
impl K2ValueScratch {
    pub(crate) fn new(
        stream: &Arc<CudaStream>,
        c: &ModelConfig,
        t: usize,
    ) -> Result<Self, AttentionBlockError> {
        let k = c.k2.expect("K2 config");
        let top = k.values_per_token as usize;
        let rows = (c.attention.kv_heads * c.attention.head_dim) as usize;
        Ok(Self {
            rotary: K2Rope::new(
                stream.context(),
                stream,
                t,
                c.attention.head_dim as usize,
                c.attention.rope_dim as usize,
            )?,
            dispatch: K2Dispatch::new_gemm(stream, t, top, k.value_experts as usize)?,
            batch_decode: if t <= STEP_SLOTS {
                Some(AttnDecodeScratch::new_batch(
                    stream,
                    c.attention.q_heads as usize,
                    c.attention.head_dim as usize,
                    t,
                )?)
            } else {
                None
            },
            gemm: K2Gemm::new(
                stream.context(),
                stream,
                t,
                top,
                k.value_experts as usize,
                (k.value_experts as usize * c.hidden_size as usize * rows).max(
                    c.hidden_size as usize
                        * c.attention.q_heads as usize
                        * c.attention.head_dim as usize,
                ),
                (c.hidden_size as usize)
                    .max(c.attention.q_heads as usize * c.attention.head_dim as usize),
                (c.hidden_size as usize)
                    .max(c.attention.q_heads as usize * c.attention.head_dim as usize),
            )?,
            logits: stream.alloc_zeros::<f32>(t * k.value_experts as usize)?,
            ids: stream.alloc_zeros::<i32>(t * top)?,
            weights: stream.alloc_zeros::<f32>(t * top)?,
            projections: stream.alloc_zeros::<f32>(t * top * rows)?,
        })
    }
}
pub(crate) struct K2AttentionWeights {
    layer: u32,
    norm: CudaSlice<f32>,
    q: Projection,
    k: Projection,
    v: Projection,
    gate: Projection,
    out: Projection,
    router: Option<(Projection, CudaSlice<f32>)>,
}
impl K2AttentionWeights {
    pub(crate) fn upload(
        stream: &Arc<CudaStream>,
        file: &GgufFile,
        d: &Directory<'_>,
        c: &ModelConfig,
        layer: u32,
    ) -> Result<Self, MoeBlockError> {
        let k = c.k2.expect("K2 config");
        let p = |r| Projection::upload(stream, file, d, r, layer);
        let router = if layer >= k.leading_dense_layers {
            Some((
                p(Role::AttnValueRouter)?,
                vector(
                    stream,
                    file,
                    d,
                    Role::AttnValueBias,
                    layer,
                    k.value_experts as usize,
                )?,
            ))
        } else {
            None
        };
        Ok(Self {
            layer,
            norm: vector(
                stream,
                file,
                d,
                Role::InputNorm,
                layer,
                c.hidden_size as usize,
            )?,
            q: p(Role::AttnQ)?,
            k: p(Role::AttnK)?,
            v: p(if router.is_some() {
                Role::AttnValueExps
            } else {
                Role::AttnV
            })?,
            gate: p(Role::AttnGate)?,
            out: p(Role::AttnOut)?,
            router,
        })
    }
    pub(crate) fn bytes(&self) -> usize {
        self.norm.len() * 4
            + [&self.q, &self.k, &self.v, &self.gate, &self.out]
                .iter()
                .map(|p| p.bytes.len())
                .sum::<usize>()
            + self
                .router
                .as_ref()
                .map_or(0, |(p, b)| p.bytes.len() + b.len() * 4)
    }
}
pub(crate) struct K2AttentionKernels {
    pub ops: Arc<K2Kernels>,
    pub attention: AttentionKernels,
    add: LayerOpsKernels,
}
impl K2AttentionKernels {
    pub(crate) fn new(
        ctx: &Arc<CudaContext>,
        c: &ModelConfig,
    ) -> Result<Self, AttentionBlockError> {
        Ok(Self {
            ops: Arc::new(K2Kernels::new(ctx)?),
            attention: AttentionKernels::new(
                ctx,
                c.attention.q_heads as usize,
                c.attention.kv_heads as usize,
                c.attention.head_dim as usize,
            )?,
            add: LayerOpsKernels::new(ctx)?,
        })
    }
}
pub(crate) struct K2AttentionBlock {
    pub kernels: Arc<K2AttentionKernels>,
    pub weights: Arc<K2AttentionWeights>,
    config: ModelConfig,
    tokens: usize,
    eps: f32,
    theta: f32,
}
impl K2AttentionBlock {
    pub(crate) fn new(
        kernels: Arc<K2AttentionKernels>,
        weights: Arc<K2AttentionWeights>,
        config: &ModelConfig,
        tokens: usize,
        eps: f32,
        theta: f32,
    ) -> Self {
        Self {
            kernels,
            weights,
            config: config.clone(),
            tokens,
            eps,
            theta,
        }
    }
    fn projections(
        &self,
        stream: &Arc<CudaStream>,
        sc: &mut AttnScratch,
        x: &CudaSlice<f32>,
    ) -> Result<(), AttentionBlockError> {
        let c = &self.config;
        let k = c.k2.unwrap();
        let ops = &self.kernels.ops;
        let w = &self.weights;
        ops.norm(
            stream,
            x,
            &w.norm,
            &mut sc.normed,
            c.hidden_size as usize,
            k.norm_groups as usize,
            self.eps,
        )?;
        let vs = sc.k2.as_mut().expect("K2 scratch allocated at startup");
        let prepared = vs.gemm.prepare(
            stream,
            &sc.normed,
            c.hidden_size as usize,
            [&w.q, &w.k, &w.gate, &w.v]
                .iter()
                .any(|p| p.quant == Some(ExpertQuant::Q4K))
                || w.router
                    .as_ref()
                    .is_some_and(|(p, _)| p.quant == Some(ExpertQuant::Q4K)),
        )?;
        w.q.prefill_prepared(
            ops,
            stream,
            &sc.normed,
            &vs.ids,
            &mut sc.query,
            &mut vs.gemm,
            prepared,
        )?;
        w.k.prefill_prepared(
            ops,
            stream,
            &sc.normed,
            &vs.ids,
            &mut sc.key,
            &mut vs.gemm,
            prepared,
        )?;
        w.gate.prefill_prepared(
            ops,
            stream,
            &sc.normed,
            &vs.ids,
            &mut sc.gate,
            &mut vs.gemm,
            prepared,
        )?;
        if let Some((router, bias)) = &w.router {
            router.prefill_prepared(
                ops,
                stream,
                &sc.normed,
                &vs.ids,
                &mut vs.logits,
                &mut vs.gemm,
                prepared,
            )?;
            ops.route(
                stream,
                &vs.logits,
                bias,
                &mut vs.ids,
                &mut vs.weights,
                k.values_per_token as usize,
                k.route_scale(),
            )?;
            {
                ops.dispatch_gemm(stream, &vs.ids, &mut vs.dispatch)?;
                vs.gemm.project_prepared(
                    stream,
                    &w.v.bytes,
                    w.v.quant.unwrap(),
                    prepared,
                    &mut vs.projections,
                    w.v.inner,
                    w.v.rows,
                    w.v.experts,
                    Some(&vs.dispatch),
                    false,
                )?;
            }
            ops.values(
                stream,
                &vs.projections,
                &vs.weights,
                &mut sc.value,
                w.v.rows,
                k.values_per_token as usize,
            )?;
        } else {
            w.v.prefill_prepared(
                ops,
                stream,
                &sc.normed,
                &vs.ids,
                &mut sc.value,
                &mut vs.gemm,
                prepared,
            )?;
        }
        Ok(())
    }
    fn finish(
        &self,
        stream: &Arc<CudaStream>,
        sc: &mut AttnScratch,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), AttentionBlockError> {
        let ops = &self.kernels.ops;
        ops.gate(stream, &sc.pregate, &sc.gate, &mut sc.gated)?;
        let vs = sc.k2.as_mut().unwrap();
        self.weights.out.prefill(
            ops,
            stream,
            &sc.gated,
            &vs.ids,
            &mut sc.projected,
            &mut vs.gemm,
        )?;
        self.kernels
            .add
            .add(stream, &sc.projected, x, out, out.len())?;
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &self,
        stream: &Arc<CudaStream>,
        sc: &mut AttnScratch,
        x: &CudaSlice<f32>,
        cache: &mut KvCache,
        offset: usize,
        position: &CudaSlice<i32>,
        rope: RopeSource<'_>,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), AttentionBlockError> {
        if offset + self.tokens > cache.max_seq() {
            return Err(AttentionBlockError::CacheExhausted {
                position: offset,
                tokens: self.tokens,
                max_seq: cache.max_seq(),
            });
        }
        self.projections(stream, sc, x)?;
        let RopeSource::Scalar(rope) = rope else {
            return Err(AttentionBlockError::BufferShape {
                what: "K2 text rotary position",
                expected: 1,
                actual: 3,
            });
        };
        if self.weights.layer == 0 {
            if rope.len() != 1 {
                return Err(AttentionBlockError::BufferShape {
                    what: "K2 rotary position",
                    expected: 1,
                    actual: rope.len(),
                });
            }
            let pointer = rope.device_ptr(stream).0;
            // SAFETY: the checked scalar remains live on this pass's stream.
            unsafe {
                sc.k2.as_mut().unwrap().rotary.prepare_raw(
                    stream,
                    &[pointer],
                    self.tokens,
                    0,
                    self.theta,
                )?;
            }
        }
        self.attend(
            stream,
            &sc.query,
            &sc.key,
            &sc.value,
            &mut sc.query_roped,
            &mut sc.key_roped,
            &mut sc.pregate,
            &mut sc.decode[0],
            cache,
            self.tokens,
            offset,
            position,
            &sc.k2.as_ref().unwrap().rotary,
            0,
        )?;
        self.finish(stream, sc, x, out)
    }
    #[allow(clippy::too_many_arguments)]
    fn attend(
        &self,
        stream: &Arc<CudaStream>,
        q: &CudaSlice<f32>,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        qr: &mut CudaSlice<f32>,
        kr: &mut CudaSlice<f32>,
        result: &mut CudaSlice<f32>,
        decode: &mut AttnDecodeScratch,
        cache: &mut KvCache,
        t: usize,
        offset: usize,
        position: &CudaSlice<i32>,
        rotary: &K2Rope,
        rotary_offset: usize,
    ) -> Result<(), AttentionBlockError> {
        let c = &self.config;
        let a = &self.kernels.attention;
        rotary.apply(stream, q, qr, c.attention.q_heads as usize, rotary_offset)?;
        rotary.apply(
            stream,
            key,
            kr,
            c.attention.kv_heads as usize,
            rotary_offset,
        )?;
        a.append_kv(
            stream,
            kr,
            value,
            &mut cache.k,
            &mut cache.v,
            t,
            cache.max_seq,
            position,
        )?;
        a.forward_k2(
            stream,
            decode,
            qr,
            &cache.k,
            &cache.v,
            result,
            t,
            cache.max_seq,
            offset + t,
            position,
        )?;
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batch(
        &self,
        stream: &Arc<CudaStream>,
        sc: &mut AttnScratch,
        x: &CudaSlice<f32>,
        caches: &mut [&mut KvCache],
        chunk: usize,
        offsets: &[usize],
        positions: &[&CudaSlice<i32>],
        rope: &[&CudaSlice<i32>],
        out: &mut CudaSlice<f32>,
    ) -> Result<(), AttentionBlockError> {
        let n = caches.len();
        if n * chunk != self.tokens
            || offsets.len() != n
            || positions.len() != n
            || rope.len() != n
            || chunk == 0
        {
            return Err(AttentionBlockError::BufferShape {
                what: "K2 batch geometry",
                expected: self.tokens,
                actual: n * chunk,
            });
        }
        for (cache, &offset) in caches.iter().zip(offsets) {
            if offset + chunk > cache.max_seq() {
                return Err(AttentionBlockError::CacheExhausted {
                    position: offset,
                    tokens: chunk,
                    max_seq: cache.max_seq(),
                });
            }
        }
        self.projections(stream, sc, x)?;
        let qdim = (self.config.attention.q_heads * self.config.attention.head_dim) as usize;
        let vdim = (self.config.attention.kv_heads * self.config.attention.head_dim) as usize;
        if self.weights.layer == 0 {
            for (batch, positions) in rope.chunks(STEP_SLOTS).enumerate() {
                let mut pointers = [0u64; STEP_SLOTS];
                for (i, position) in positions.iter().enumerate() {
                    if position.len() != 1 {
                        return Err(AttentionBlockError::BufferShape {
                            what: "K2 rotary position",
                            expected: 1,
                            actual: position.len(),
                        });
                    }
                    pointers[i] = position.device_ptr(stream).0;
                }
                // SAFETY: checked scalar slots are live through this launch.
                unsafe {
                    sc.k2.as_mut().unwrap().rotary.prepare_raw(
                        stream,
                        &pointers[..positions.len()],
                        chunk,
                        batch * STEP_SLOTS * chunk,
                        self.theta,
                    )?;
                }
            }
        }
        if chunk == 1 && n <= STEP_SLOTS {
            let a = &self.kernels.attention;
            let (mut ap, mut kc, mut vc) =
                ([0u64; STEP_SLOTS], [0u64; STEP_SLOTS], [0u64; STEP_SLOTS]);
            for i in 0..n {
                if positions[i].len() != 1 || rope[i].len() != 1 {
                    return Err(AttentionBlockError::BufferShape {
                        what: "K2 batch positions",
                        expected: 1,
                        actual: positions[i].len().max(rope[i].len()),
                    });
                }
                ap[i] = positions[i].device_ptr(stream).0;
                kc[i] = caches[i].k.device_ptr(stream).0;
                vc[i] = caches[i].v.device_ptr(stream).0;
            }
            // SAFETY: checked cache capacity and distinct sequence caches are
            // borrowed exclusively above; the slots are live for this launch.
            unsafe {
                sc.k2.as_ref().unwrap().rotary.append_raw(
                    stream,
                    &sc.query,
                    &mut sc.query_roped,
                    &sc.key,
                    &mut sc.key_roped,
                    &sc.value,
                    &ap[..n],
                    &kc[..n],
                    &vc[..n],
                    self.config.attention.q_heads as usize,
                    self.config.attention.kv_heads as usize,
                )
            }?;
            let mma = a.k2_decode_mma_wpo(offsets[0] + 1).filter(|wpo| {
                offsets
                    .iter()
                    .all(|o| a.k2_decode_mma_wpo(o + 1) == Some(*wpo))
            });
            if let Some(wpo) = mma
                && n > 1
            {
                // SAFETY: checked independent cache slots and fixed batch scratch
                // remain live through the launch, just as for scalar attention.
                unsafe {
                    a.decode_batch_k2_mma_raw(
                        stream,
                        sc.k2.as_mut().unwrap().batch_decode.as_mut().unwrap(),
                        &sc.query_roped,
                        &mut sc.pregate,
                        &ap[..n],
                        &kc[..n],
                        &vc[..n],
                        wpo,
                    )
                }?;
            } else if offsets.iter().all(|o| a.k2_batch_uses_warp(o + 1)) {
                // SAFETY: the same checked slots, with one independent window
                // per sequence; scratch was allocated for this physical width.
                unsafe {
                    a.decode_batch_k2_raw(
                        stream,
                        sc.k2.as_mut().unwrap().batch_decode.as_mut().unwrap(),
                        &sc.query_roped,
                        &mut sc.pregate,
                        &ap[..n],
                        &kc[..n],
                        &vc[..n],
                    )
                }?;
            } else {
                for i in 0..n {
                    // SAFETY: sequence slices are disjoint and inside the
                    // validated physical activation buffers.
                    let q = unsafe {
                        crate::viewslice::subslice(stream, &sc.query_roped, i * qdim, qdim)
                    };
                    let mut y =
                        unsafe { crate::viewslice::subslice(stream, &sc.pregate, i * qdim, qdim) };
                    a.forward_k2(
                        stream,
                        &mut sc.decode[0],
                        &q,
                        &caches[i].k,
                        &caches[i].v,
                        &mut y,
                        1,
                        caches[i].max_seq,
                        offsets[i] + 1,
                        positions[i],
                    )?;
                }
            }
            return self.finish(stream, sc, x, out);
        }
        for i in 0..n {
            // SAFETY: all flattened buffers are shape-sized, and i < sequences.
            let q = unsafe {
                crate::viewslice::subslice(stream, &sc.query, i * chunk * qdim, chunk * qdim)
            };
            let key = unsafe {
                crate::viewslice::subslice(stream, &sc.key, i * chunk * vdim, chunk * vdim)
            };
            let v = unsafe {
                crate::viewslice::subslice(stream, &sc.value, i * chunk * vdim, chunk * vdim)
            };
            let mut qr = unsafe {
                crate::viewslice::subslice(stream, &sc.query_roped, i * chunk * qdim, chunk * qdim)
            };
            let mut kr = unsafe {
                crate::viewslice::subslice(stream, &sc.key_roped, i * chunk * vdim, chunk * vdim)
            };
            let mut y = unsafe {
                crate::viewslice::subslice(stream, &sc.pregate, i * chunk * qdim, chunk * qdim)
            };
            self.attend(
                stream,
                &q,
                &key,
                &v,
                &mut qr,
                &mut kr,
                &mut y,
                &mut sc.decode[0],
                caches[i],
                chunk,
                offsets[i],
                positions[i],
                &sc.k2.as_ref().unwrap().rotary,
                i * chunk,
            )?;
        }
        self.finish(stream, sc, x, out)
    }
}

/// Resident K2 FFN tensors for one dense or MoE layer.
pub struct K2FfnWeights {
    layer: u32,
    norm: CudaSlice<f32>,
    gate: Projection,
    up: Projection,
    down: Projection,
    router: Option<(Projection, CudaSlice<f32>)>,
    shared: Option<(Projection, Projection, Projection)>,
}
impl K2FfnWeights {
    pub fn upload(
        stream: &Arc<CudaStream>,
        file: &GgufFile,
        d: &Directory<'_>,
        c: &ModelConfig,
        layer: u32,
    ) -> Result<Self, MoeBlockError> {
        let p = |r| Projection::upload(stream, file, d, r, layer);
        let routed = c.ffn_for_layer(layer).moe().is_some();
        let (gate, up, down, router, shared) = if routed {
            let m = c.moe().unwrap();
            (
                p(Role::MoeGateExps)?,
                p(Role::MoeUpExps)?,
                p(Role::MoeDownExps)?,
                Some((
                    p(Role::MoeRouter)?,
                    vector(
                        stream,
                        file,
                        d,
                        Role::MoeRouterBias,
                        layer,
                        m.num_experts as usize,
                    )?,
                )),
                Some((
                    p(Role::MoeSharedGate)?,
                    p(Role::MoeSharedUp)?,
                    p(Role::MoeSharedDown)?,
                )),
            )
        } else {
            (
                p(Role::FfnGate)?,
                p(Role::FfnUp)?,
                p(Role::FfnDown)?,
                None,
                None,
            )
        };
        Ok(Self {
            layer,
            norm: vector(
                stream,
                file,
                d,
                Role::K2FfnNorm,
                layer,
                c.hidden_size as usize,
            )?,
            gate,
            up,
            down,
            router,
            shared,
        })
    }
    pub fn layer(&self) -> u32 {
        self.layer
    }
    pub fn bytes(&self) -> usize {
        self.norm.len() * 4
            + [&self.gate, &self.up, &self.down]
                .iter()
                .map(|p| p.bytes.len())
                .sum::<usize>()
            + self
                .router
                .as_ref()
                .map_or(0, |(p, b)| p.bytes.len() + b.len() * 4)
            + self
                .shared
                .as_ref()
                .map_or(0, |(g, u, d)| g.bytes.len() + u.bytes.len() + d.bytes.len())
    }
}
/// Fixed scratch and device routing for the mixed K2 feed-forward layers.
struct K2FfnGemmScratch {
    dispatch: K2Dispatch,
    gemm: K2Gemm,
    dense_gemm: K2Gemm,
    gate: CudaSlice<f32>,
    up: CudaSlice<f32>,
    partial: CudaSlice<f32>,
    shared_gate: CudaSlice<f32>,
    shared_up: CudaSlice<f32>,
    shared_inter: CudaSlice<f32>,
}
pub struct K2FfnBlock {
    ops: K2Kernels,
    geometry: MoeGeometry,
    ids: CudaSlice<i32>,
    weights: CudaSlice<f32>,
    gemm: K2FfnGemmScratch,
    layer_ops: LayerOpsKernels,
    config: ModelConfig,
    eps: f32,
    normed: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    shared: CudaSlice<f32>,
    dense_gate: CudaSlice<f32>,
    dense_up: CudaSlice<f32>,
    dense_inter: CudaSlice<f32>,
}
impl K2FfnBlock {
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        c: &ModelConfig,
        g: MoeGeometry,
        eps: f32,
    ) -> Result<Self, MoeBlockError> {
        let dense = c.k2.unwrap().dense_intermediate as usize;
        Ok(Self {
            ops: K2Kernels::new(ctx)?,
            geometry: g,
            ids: stream.alloc_zeros::<i32>(g.max_tokens * g.experts_per_token)?,
            weights: stream.alloc_zeros::<f32>(g.max_tokens * g.experts_per_token)?,
            gemm: K2FfnGemmScratch {
                dispatch: K2Dispatch::new_gemm(
                    stream,
                    g.max_tokens,
                    g.experts_per_token,
                    g.num_experts,
                )?,
                gemm: K2Gemm::new_with_activation_bits(
                    ctx,
                    stream,
                    g.max_tokens,
                    g.experts_per_token,
                    g.num_experts,
                    (g.num_experts * g.hidden * g.intermediate).max(g.hidden * dense),
                    g.hidden.max(g.intermediate),
                    g.hidden.max(g.intermediate),
                    16,
                )?,
                dense_gemm: K2Gemm::new_with_activation_bits(
                    ctx,
                    stream,
                    g.max_tokens,
                    1,
                    1,
                    g.hidden * dense,
                    g.hidden.max(dense),
                    g.hidden.max(dense),
                    16,
                )?,
                gate: stream
                    .alloc_zeros::<f32>(g.max_tokens * g.experts_per_token * g.intermediate)?,
                up: stream
                    .alloc_zeros::<f32>(g.max_tokens * g.experts_per_token * g.intermediate)?,
                shared_gate: stream.alloc_zeros::<f32>(g.max_tokens * g.intermediate)?,
                shared_up: stream.alloc_zeros::<f32>(g.max_tokens * g.intermediate)?,
                shared_inter: stream.alloc_zeros::<f32>(g.max_tokens * g.intermediate)?,
                partial: stream
                    .alloc_zeros::<f32>(g.max_tokens * g.experts_per_token * g.hidden)?,
            },
            layer_ops: LayerOpsKernels::new(ctx)?,
            config: c.clone(),
            eps,
            normed: stream.alloc_zeros::<f32>(g.max_tokens * g.hidden)?,
            logits: stream.alloc_zeros::<f32>(g.max_tokens * g.num_experts)?,
            shared: stream.alloc_zeros::<f32>(g.max_tokens * g.hidden)?,
            dense_gate: stream.alloc_zeros::<f32>(g.max_tokens * dense)?,
            dense_up: stream.alloc_zeros::<f32>(g.max_tokens * dense)?,
            dense_inter: stream.alloc_zeros::<f32>(g.max_tokens * dense)?,
        })
    }
    pub fn geometry(&self) -> MoeGeometry {
        self.geometry
    }
    pub fn normed(&self) -> &CudaSlice<f32> {
        &self.normed
    }
    pub fn publish_tokens(
        &mut self,
        stream: &Arc<CudaStream>,
        tokens: usize,
    ) -> Result<(), MoeBlockError> {
        let _ = stream;
        if tokens > self.geometry.max_tokens {
            return Err(MoeBlockError::FfnKindMismatch {
                block: "K2 token capacity",
                weights: "too many tokens",
            });
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        stream: &Arc<CudaStream>,
        w: &K2FfnWeights,
        x: &CudaSlice<f32>,
        tokens: usize,
        ffn: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), MoeBlockError> {
        let c = &self.config;
        let k = c.k2.unwrap();
        let g = self.geometry;
        if tokens != g.max_tokens {
            return Err(MoeBlockError::FfnKindMismatch {
                block: "fixed K2 token shape",
                weights: "different token shape",
            });
        }
        self.ops.norm(
            stream,
            x,
            &w.norm,
            &mut self.normed,
            c.hidden_size as usize,
            k.norm_groups as usize,
            self.eps,
        )?;
        if let Some((router, bias)) = &w.router {
            router.forward(
                &self.ops,
                stream,
                &self.normed,
                &self.ids,
                &mut self.logits,
                1,
                false,
            )?;
            let (ids, weights) = (&mut self.ids, &mut self.weights);
            self.ops.route(
                stream,
                &self.logits,
                bias,
                ids,
                weights,
                g.experts_per_token,
                k.route_scale(),
            )?;
            let (sg, su, sd) = w.shared.as_ref().unwrap();
            let sc = &mut self.gemm;
            self.ops
                .dispatch_gemm(stream, &self.ids, &mut sc.dispatch)?;
            // The routed and shared gate/up projections share one
            // quantization of the normed input.
            let prepared = sc.gemm.prepare(
                stream,
                &self.normed,
                g.hidden,
                [w.gate.quant, w.up.quant, sg.quant, su.quant].contains(&Some(ExpertQuant::Q4K)),
            )?;
            sc.gemm.project_prepared(
                stream,
                &w.gate.bytes,
                w.gate.quant.unwrap(),
                prepared,
                &mut sc.gate,
                g.hidden,
                g.intermediate,
                g.num_experts,
                Some(&sc.dispatch),
                false,
            )?;
            sc.gemm.project_prepared(
                stream,
                &w.up.bytes,
                w.up.quant.unwrap(),
                prepared,
                &mut sc.up,
                g.hidden,
                g.intermediate,
                g.num_experts,
                Some(&sc.dispatch),
                false,
            )?;
            sg.prefill_prepared(
                &self.ops,
                stream,
                &self.normed,
                &self.ids,
                &mut sc.shared_gate,
                &mut sc.gemm,
                prepared,
            )?;
            su.prefill_prepared(
                &self.ops,
                stream,
                &self.normed,
                &self.ids,
                &mut sc.shared_up,
                &mut sc.gemm,
                prepared,
            )?;
            let prepared = sc.gemm.prepare_swiglu(
                stream,
                &sc.gate,
                &sc.up,
                g.intermediate,
                w.down.quant == Some(ExpertQuant::Q4K),
            )?;
            sc.gemm.project_prepared(
                stream,
                &w.down.bytes,
                w.down.quant.unwrap(),
                prepared,
                &mut sc.partial,
                g.intermediate,
                g.hidden,
                g.num_experts,
                Some(&sc.dispatch),
                true,
            )?;
            sd.prefill_swiglu(
                &self.ops,
                &self.layer_ops,
                stream,
                &sc.shared_gate,
                &sc.shared_up,
                &mut sc.shared_inter,
                &self.ids,
                &mut self.shared,
                &mut sc.dense_gemm,
            )?;
            // Routed plus shared, then the residual: one pass over the
            // outputs instead of a combine and two adds.
            self.ops.combine_residual(
                stream,
                &sc.partial,
                &self.weights,
                &self.shared,
                x,
                ffn,
                out,
                g.hidden,
                g.experts_per_token,
            )?;
        } else {
            let prepared = self.gemm.dense_gemm.prepare(
                stream,
                &self.normed,
                g.hidden,
                w.gate.quant == Some(ExpertQuant::Q4K) || w.up.quant == Some(ExpertQuant::Q4K),
            )?;
            w.gate.prefill_prepared(
                &self.ops,
                stream,
                &self.normed,
                &self.ids,
                &mut self.dense_gate,
                &mut self.gemm.dense_gemm,
                prepared,
            )?;
            w.up.prefill_prepared(
                &self.ops,
                stream,
                &self.normed,
                &self.ids,
                &mut self.dense_up,
                &mut self.gemm.dense_gemm,
                prepared,
            )?;
            w.down.prefill_swiglu(
                &self.ops,
                &self.layer_ops,
                stream,
                &self.dense_gate,
                &self.dense_up,
                &mut self.dense_inter,
                &self.ids,
                ffn,
                &mut self.gemm.dense_gemm,
            )?;
            self.layer_ops.add(stream, x, ffn, out, out.len())?;
        }
        Ok(())
    }
}
