//! K2-Horizon mixer and FFN integration. IFM llama.cpp `src/models/k2-horizon.cpp`
//! is the source for grouped norms, separate softplus gate, routed SiLU values,
//! sigmoid FFN routing and the ungated shared expert.
use super::{
    attention::{AttentionBlockError, AttnScratch, KvCache, RopeSource},
    moe::{MoeBlockError, dense_tensor, quantized_tensor},
};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use llmcuda_cuda::kernels::{
    attention::{AttentionKernels, AttnDecodeScratch},
    k2::K2Kernels,
    layer_ops::LayerOpsKernels,
    moe::{ExpertQuant, MoeBuffers, MoeGeometry, MoeKernels, QuantTensor, to_device_layout},
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
            (stream.clone_htod(&*to_device_layout(q, src))?, Some(q))
        };
        Ok(Self {
            bytes,
            quant,
            inner,
            rows,
            experts,
        })
    }
    fn tensor(&self) -> QuantTensor<'_> {
        QuantTensor {
            bytes: &self.bytes,
            quant: self.quant.expect("expert matrices are quantized"),
        }
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
            logits: stream.alloc_zeros::<f32>(t * k.value_experts as usize)?,
            ids: stream.alloc_zeros::<i32>(t * top)?,
            weights: stream.alloc_zeros::<f32>(t * top)?,
            projections: stream.alloc_zeros::<f32>(t * top * rows)?,
        })
    }
}
pub(crate) struct K2AttentionWeights {
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
        w.q.forward(ops, stream, &sc.normed, &vs.ids, &mut sc.query, 1, false)?;
        w.k.forward(ops, stream, &sc.normed, &vs.ids, &mut sc.key, 1, false)?;
        w.gate
            .forward(ops, stream, &sc.normed, &vs.ids, &mut sc.gate, 1, false)?;
        if let Some((router, bias)) = &w.router {
            router.forward(ops, stream, &sc.normed, &vs.ids, &mut vs.logits, 1, false)?;
            ops.route(
                stream,
                &vs.logits,
                bias,
                &mut vs.ids,
                &mut vs.weights,
                k.values_per_token as usize,
                k.route_scale(),
            )?;
            w.v.forward(
                ops,
                stream,
                &sc.normed,
                &vs.ids,
                &mut vs.projections,
                k.values_per_token as usize,
                true,
            )?;
            ops.values(
                stream,
                &vs.projections,
                &vs.weights,
                &mut sc.value,
                w.v.rows,
                k.values_per_token as usize,
            )?;
        } else {
            w.v.forward(ops, stream, &sc.normed, &vs.ids, &mut sc.value, 1, false)?;
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
        self.weights.out.forward(
            ops,
            stream,
            &sc.gated,
            &sc.k2.as_ref().unwrap().ids,
            &mut sc.projected,
            1,
            false,
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
            rope,
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
        rope: &CudaSlice<i32>,
    ) -> Result<(), AttentionBlockError> {
        let c = &self.config;
        let a = &self.kernels.attention;
        a.rope(
            stream,
            q,
            qr,
            t,
            c.attention.q_heads as usize,
            c.attention.rope_dim as usize,
            rope,
            self.theta,
        )?;
        a.rope(
            stream,
            key,
            kr,
            t,
            c.attention.kv_heads as usize,
            c.attention.rope_dim as usize,
            rope,
            self.theta,
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
        a.forward(
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
                rope[i],
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
pub struct K2FfnBlock {
    ops: K2Kernels,
    moe: MoeKernels,
    buffers: MoeBuffers,
    layer_ops: LayerOpsKernels,
    config: ModelConfig,
    eps: f32,
    normed: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    routed: CudaSlice<f32>,
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
        let mut moe = MoeKernels::new(ctx, g)?;
        moe.disable_tensor_cores();
        let buffers = moe.buffers(stream)?;
        let dense = c.k2.unwrap().dense_intermediate as usize;
        Ok(Self {
            ops: K2Kernels::new(ctx)?,
            moe,
            buffers,
            layer_ops: LayerOpsKernels::new(ctx)?,
            config: c.clone(),
            eps,
            normed: stream.alloc_zeros::<f32>(g.max_tokens * g.hidden)?,
            logits: stream.alloc_zeros::<f32>(g.max_tokens * g.num_experts)?,
            routed: stream.alloc_zeros::<f32>(g.max_tokens * g.hidden)?,
            shared: stream.alloc_zeros::<f32>(g.max_tokens * g.hidden)?,
            dense_gate: stream.alloc_zeros::<f32>(g.max_tokens * dense)?,
            dense_up: stream.alloc_zeros::<f32>(g.max_tokens * dense)?,
            dense_inter: stream.alloc_zeros::<f32>(g.max_tokens * dense)?,
        })
    }
    pub fn geometry(&self) -> MoeGeometry {
        self.moe.geometry()
    }
    pub fn normed(&self) -> &CudaSlice<f32> {
        &self.normed
    }
    pub fn publish_tokens(
        &mut self,
        stream: &Arc<CudaStream>,
        tokens: usize,
    ) -> Result<(), MoeBlockError> {
        self.moe
            .set_valid_tokens(stream, &mut self.buffers, tokens)?;
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
        let g = self.moe.geometry();
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
                self.buffers.topk_ids(),
                &mut self.logits,
                1,
                false,
            )?;
            let (ids, weights) = self.buffers.routing_mut();
            self.ops.route(
                stream,
                &self.logits,
                bias,
                ids,
                weights,
                g.experts_per_token,
                k.route_scale(),
            )?;
            self.moe.build_dispatch(stream, &mut self.buffers)?;
            self.moe.grouped_forward(
                stream,
                &mut self.buffers,
                w.gate.tensor(),
                w.up.tensor(),
                w.down.tensor(),
                &self.normed,
                &mut self.routed,
            )?;
            let (sg, su, sd) = w.shared.as_ref().unwrap();
            self.moe.shared_expert(
                stream,
                &mut self.buffers,
                sg.tensor(),
                su.tensor(),
                sd.tensor(),
                &self.normed,
                &mut self.shared,
            )?;
            self.layer_ops
                .add(stream, &self.routed, &self.shared, ffn, ffn.len())?;
        } else {
            w.gate.forward(
                &self.ops,
                stream,
                &self.normed,
                self.buffers.topk_ids(),
                &mut self.dense_gate,
                1,
                false,
            )?;
            w.up.forward(
                &self.ops,
                stream,
                &self.normed,
                self.buffers.topk_ids(),
                &mut self.dense_up,
                1,
                false,
            )?;
            self.layer_ops.swiglu(
                stream,
                &self.dense_gate,
                &self.dense_up,
                &mut self.dense_inter,
                self.dense_gate.len(),
            )?;
            w.down.forward(
                &self.ops,
                stream,
                &self.dense_inter,
                self.buffers.topk_ids(),
                ffn,
                1,
                false,
            )?;
        }
        self.layer_ops.add(stream, x, ffn, out, out.len())?;
        Ok(())
    }
}
