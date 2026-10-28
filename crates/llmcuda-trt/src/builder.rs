//! Building engines: builder, network definition, config, ONNX parser.
//!
//! Lifetimes follow TensorRT's ownership: a network and a config borrow the
//! builder that made them, tensors borrow their network, and a parser borrows
//! the network it fills (and must live until the build, which reads plugin
//! objects the parser owns).

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_void};
use std::marker::PhantomData;

use crate::ffi;
use crate::plugin::Field;
use crate::types::{
    BuilderFlag, CausalMask, DataType, Dims, ElementWise, MatrixOp, ProfilingVerbosity, Result,
    TrtError, check, clear_last_error, ensure_logger, fail, non_null,
};

fn cstring(s: &str) -> Result<CString> {
    CString::new(s).map_err(|_| TrtError::Invalid(format!("{s:?} contains a NUL byte")))
}

/// `nvinfer1::IBuilder`.
pub struct Builder {
    raw: *mut c_void,
}

impl Builder {
    pub fn new() -> Result<Self> {
        ensure_logger();
        clear_last_error();
        // SAFETY: no arguments; the shim supplies the shared logger.
        let raw = non_null(
            unsafe { ffi::llmtrt_builder_create() },
            "createInferBuilder",
        )?;
        Ok(Self { raw })
    }

    pub fn network(&self) -> Result<Network<'_>> {
        clear_last_error();
        // SAFETY: `self.raw` is a live builder.
        let raw = non_null(
            unsafe { ffi::llmtrt_network_create(self.raw) },
            "createNetworkV2",
        )?;
        Ok(Network {
            raw,
            constants: RefCell::new(Vec::new()),
            _builder: PhantomData,
        })
    }

    pub fn config(&self) -> Result<BuilderConfig<'_>> {
        clear_last_error();
        // SAFETY: `self.raw` is a live builder.
        let raw = non_null(
            unsafe { ffi::llmtrt_config_create(self.raw) },
            "createBuilderConfig",
        )?;
        Ok(BuilderConfig {
            raw,
            builder: self.raw,
            _builder: PhantomData,
        })
    }

    /// Build and serialize. The plan is what a runtime deserializes.
    pub fn build(&self, network: &Network<'_>, config: &BuilderConfig<'_>) -> Result<HostMemory> {
        clear_last_error();
        // SAFETY: all three objects are live and belong to this builder; the
        // network's constant data is owned by `network` and outlives the call.
        let raw = unsafe { ffi::llmtrt_build_serialized(self.raw, network.raw, config.raw) };
        Ok(HostMemory {
            raw: non_null(raw, "buildSerializedNetwork")?,
        })
    }
}

impl Drop for Builder {
    fn drop(&mut self) {
        // SAFETY: created by `llmtrt_builder_create`, dropped once.
        unsafe { ffi::llmtrt_builder_destroy(self.raw) }
    }
}

/// `nvinfer1::IHostMemory`: a serialized plan or timing cache.
pub struct HostMemory {
    raw: *mut c_void,
}

impl std::ops::Deref for HostMemory {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        // SAFETY: TensorRT owns `size` bytes at `data` until destroy.
        unsafe {
            let data = ffi::llmtrt_host_memory_data(self.raw) as *const u8;
            let size = ffi::llmtrt_host_memory_size(self.raw);
            if size == 0 {
                &[]
            } else {
                std::slice::from_raw_parts(data, size)
            }
        }
    }
}

impl Drop for HostMemory {
    fn drop(&mut self) {
        // SAFETY: created by TensorRT, dropped once.
        unsafe { ffi::llmtrt_host_memory_destroy(self.raw) }
    }
}

/// A tensor in a network under construction.
#[derive(Clone, Copy)]
pub struct Tensor<'n> {
    raw: *mut c_void,
    _network: PhantomData<&'n ()>,
}

impl<'n> Tensor<'n> {
    fn wrap(raw: *mut c_void) -> Self {
        Self {
            raw,
            _network: PhantomData,
        }
    }

    pub fn set_name(&self, name: &str) -> Result<()> {
        let name = cstring(name)?;
        // SAFETY: live tensor; TensorRT copies the name.
        unsafe { ffi::llmtrt_tensor_set_name(self.raw, name.as_ptr()) };
        Ok(())
    }

    pub fn name(&self) -> String {
        // SAFETY: live tensor; the name is NUL-terminated and owned by it.
        unsafe { CStr::from_ptr(ffi::llmtrt_tensor_name(self.raw)) }
            .to_string_lossy()
            .into_owned()
    }

    pub fn data_type(&self) -> Option<DataType> {
        // SAFETY: live tensor.
        DataType::from_raw(unsafe { ffi::llmtrt_tensor_type(self.raw) })
    }

    pub fn dims(&self) -> Dims {
        let mut out = Dims::new(&[]);
        // SAFETY: live tensor; `out` is a valid llmtrt_dims.
        unsafe { ffi::llmtrt_tensor_dims(self.raw, &mut out.0) };
        out
    }
}

/// `nvinfer1::INetworkDefinition`, plus the constant data it references.
pub struct Network<'b> {
    raw: *mut c_void,
    /// TensorRT keeps a pointer to constant values until the build, so the
    /// network owns them.
    constants: RefCell<Vec<Vec<u8>>>,
    _builder: PhantomData<&'b Builder>,
}

impl<'b> Network<'b> {
    pub(crate) fn raw(&self) -> *mut c_void {
        self.raw
    }

    fn output(&self, layer: *mut c_void, call: &'static str) -> Result<Tensor<'_>> {
        let layer = non_null(layer, call)?;
        // SAFETY: a live layer of this network; index 0 always exists for
        // the layers this crate creates.
        let out = non_null(unsafe { ffi::llmtrt_layer_output(layer, 0) }, call)?;
        Ok(Tensor::wrap(out))
    }

    pub fn input(
        &self,
        name: &str,
        data_type: DataType,
        dims: impl Into<Dims>,
    ) -> Result<Tensor<'_>> {
        let name = cstring(name)?;
        let dims = dims.into();
        clear_last_error();
        // SAFETY: live network; TensorRT copies the name and dims.
        let raw = unsafe {
            ffi::llmtrt_net_add_input(self.raw, name.as_ptr(), data_type as i32, &dims.0)
        };
        Ok(Tensor::wrap(non_null(raw, "addInput")?))
    }

    pub fn mark_output(&self, tensor: Tensor<'_>) -> Result<()> {
        clear_last_error();
        // SAFETY: both live; the tensor belongs to this network.
        check(
            unsafe { ffi::llmtrt_net_mark_output(self.raw, tensor.raw) },
            "markOutput",
        )
    }

    /// `markDebug`: the tensor's value is reported to a debug listener when
    /// an execution context enables it.
    pub fn mark_debug(&self, tensor: Tensor<'_>) -> Result<()> {
        clear_last_error();
        // SAFETY: both live; the tensor belongs to this network.
        check(
            unsafe { ffi::llmtrt_net_mark_debug(self.raw, tensor.raw) },
            "markDebug",
        )
    }

    pub fn nb_layers(&self) -> i32 {
        // SAFETY: live network.
        unsafe { ffi::llmtrt_net_nb_layers(self.raw) }
    }

    /// A build-time constant. `count` is in elements, which for the packed
    /// 4-bit types is twice the byte count. A `weights_name` makes it
    /// refittable under that name.
    pub fn constant(
        &self,
        dims: impl Into<Dims>,
        data_type: DataType,
        bytes: Vec<u8>,
        weights_name: Option<&str>,
    ) -> Result<Tensor<'_>> {
        let dims = dims.into();
        let count = dims
            .volume()
            .ok_or_else(|| TrtError::Invalid("a constant must have a static shape".into()))?;
        let need = (count as usize * data_type.bits()).div_ceil(8);
        if bytes.len() != need {
            return Err(TrtError::Invalid(format!(
                "constant {dims:?} of {data_type:?} needs {need} bytes, got {}",
                bytes.len()
            )));
        }
        let name = weights_name.map(cstring).transpose()?;
        let ptr = bytes.as_ptr();
        // Moving the Vec into the list does not move its heap buffer, so
        // `ptr` stays valid for the network's life.
        self.constants.borrow_mut().push(bytes);
        clear_last_error();
        // SAFETY: live network; `ptr` covers `count` elements and is owned by
        // `self.constants` until the network drops.
        let layer = unsafe {
            ffi::llmtrt_net_add_constant(
                self.raw,
                &dims.0,
                data_type as i32,
                ptr as *const c_void,
                count,
                name.as_ref().map_or(std::ptr::null(), |n| n.as_ptr()),
            )
        };
        self.output(layer, "addConstant")
    }

    pub fn elementwise(&self, a: Tensor<'_>, b: Tensor<'_>, op: ElementWise) -> Result<Tensor<'_>> {
        clear_last_error();
        // SAFETY: live network and tensors.
        let layer = unsafe { ffi::llmtrt_net_add_elementwise(self.raw, a.raw, b.raw, op as i32) };
        self.output(layer, "addElementWise")
    }

    pub fn matmul(
        &self,
        a: Tensor<'_>,
        op_a: MatrixOp,
        b: Tensor<'_>,
        op_b: MatrixOp,
    ) -> Result<Tensor<'_>> {
        clear_last_error();
        // SAFETY: live network and tensors.
        let layer =
            unsafe { ffi::llmtrt_net_add_matmul(self.raw, a.raw, op_a as i32, b.raw, op_b as i32) };
        self.output(layer, "addMatrixMultiply")
    }

    pub fn cast(&self, tensor: Tensor<'_>, to: DataType) -> Result<Tensor<'_>> {
        clear_last_error();
        // SAFETY: live network and tensor.
        let layer = unsafe { ffi::llmtrt_net_add_cast(self.raw, tensor.raw, to as i32) };
        self.output(layer, "addCast")
    }

    /// `addDequantize`, optionally with an axis and a block shape.
    pub fn dequantize(
        &self,
        input: Tensor<'_>,
        scale: Tensor<'_>,
        out: DataType,
        axis: Option<i32>,
        block: Option<Dims>,
    ) -> Result<Tensor<'_>> {
        clear_last_error();
        // SAFETY: live network and tensors; `block` outlives the call.
        let layer = unsafe {
            ffi::llmtrt_net_add_dequantize(
                self.raw,
                input.raw,
                scale.raw,
                out as i32,
                axis.unwrap_or(-1),
                block.as_ref().map_or(std::ptr::null(), |b| &b.0),
            )
        };
        self.output(layer, "addDequantize")
    }

    /// `addAttentionV2` with softmax normalization.
    pub fn attention(
        &self,
        q: Tensor<'_>,
        k: Tensor<'_>,
        v: Tensor<'_>,
        causal: CausalMask,
        decomposable: bool,
    ) -> Result<Tensor<'_>> {
        clear_last_error();
        // SAFETY: live network and tensors.
        let attention = non_null(
            unsafe {
                ffi::llmtrt_net_add_attention(
                    self.raw,
                    q.raw,
                    k.raw,
                    v.raw,
                    causal as i32,
                    decomposable,
                )
            },
            "addAttentionV2",
        )?;
        // SAFETY: a live IAttention of this network.
        let out = unsafe { ffi::llmtrt_attention_output(attention) };
        Ok(Tensor::wrap(non_null(out, "IAttention::getOutput")?))
    }

    /// Add a plugin made by a creator registered under `name`/`version` in
    /// `namespace`, at the build phase, with `fields`. Returns its outputs.
    pub fn plugin(
        &self,
        inputs: &[Tensor<'_>],
        name: &str,
        version: &str,
        namespace: &str,
        fields: &[Field],
        nb_outputs: usize,
    ) -> Result<Vec<Tensor<'_>>> {
        let (cname, cversion, cns) = (cstring(name)?, cstring(version)?, cstring(namespace)?);
        let raw_inputs: Vec<*mut c_void> = inputs.iter().map(|t| t.raw).collect();
        let raw_fields: Vec<ffi::llmtrt_field> = fields.iter().map(Field::raw).collect();
        clear_last_error();
        // SAFETY: live network and tensors; the field data outlives the
        // call, and the creator copies what it keeps.
        let layer = unsafe {
            ffi::llmtrt_net_add_plugin(
                self.raw,
                raw_inputs.as_ptr(),
                raw_inputs.len() as i32,
                cname.as_ptr(),
                cversion.as_ptr(),
                cns.as_ptr(),
                raw_fields.as_ptr(),
                raw_fields.len() as i32,
            )
        };
        let layer = non_null(layer, "addPluginV3")?;
        (0..nb_outputs)
            .map(|i| {
                // SAFETY: a live layer; TensorRT returns null past the end.
                let t = unsafe { ffi::llmtrt_layer_output(layer, i as i32) };
                non_null(t, "IPluginV3Layer::getOutput").map(Tensor::wrap)
            })
            .collect()
    }
}

impl Drop for Network<'_> {
    fn drop(&mut self) {
        // SAFETY: created by `llmtrt_network_create`, dropped once, before
        // the builder it borrows.
        unsafe { ffi::llmtrt_network_destroy(self.raw) }
    }
}

/// `nvinfer1::IBuilderConfig`.
pub struct BuilderConfig<'b> {
    raw: *mut c_void,
    builder: *mut c_void,
    _builder: PhantomData<&'b Builder>,
}

impl BuilderConfig<'_> {
    pub fn set_flag(&self, flag: BuilderFlag, on: bool) {
        // SAFETY: live config.
        unsafe { ffi::llmtrt_config_set_flag(self.raw, flag as i32, on) }
    }

    /// The design runs every layer and plugin on the caller's stream, so
    /// this is always set to zero by the engine.
    pub fn set_max_aux_streams(&self, n: i32) -> Result<()> {
        clear_last_error();
        // SAFETY: live config.
        check(
            unsafe { ffi::llmtrt_config_set_max_aux_streams(self.raw, n) },
            "setMaxAuxStreams",
        )
    }

    pub fn set_profiling_verbosity(&self, verbosity: ProfilingVerbosity) {
        // SAFETY: live config.
        unsafe { ffi::llmtrt_config_set_profiling_verbosity(self.raw, verbosity as i32) }
    }

    pub fn set_workspace_limit(&self, bytes: u64) {
        // SAFETY: live config.
        unsafe { ffi::llmtrt_config_set_workspace_limit(self.raw, bytes) }
    }

    /// One optimization profile. `(name, min, opt, max)` per dynamic input;
    /// an exact-shape profile passes the same dims three times.
    pub fn add_profile(&self, inputs: &[(&str, Dims, Dims, Dims)]) -> Result<i32> {
        let names: Vec<CString> = inputs.iter().map(|i| cstring(i.0)).collect::<Result<_>>()?;
        let name_ptrs: Vec<_> = names.iter().map(|n| n.as_ptr()).collect();
        let min: Vec<_> = inputs.iter().map(|i| i.1.0).collect();
        let opt: Vec<_> = inputs.iter().map(|i| i.2.0).collect();
        let max: Vec<_> = inputs.iter().map(|i| i.3.0).collect();
        clear_last_error();
        // SAFETY: live builder and config; arrays have `inputs.len()` entries.
        let index = unsafe {
            ffi::llmtrt_config_add_profile(
                self.builder,
                self.raw,
                name_ptrs.as_ptr(),
                min.as_ptr(),
                opt.as_ptr(),
                max.as_ptr(),
                inputs.len() as i32,
            )
        };
        if index < 0 {
            Err(fail("addOptimizationProfile"))
        } else {
            Ok(index)
        }
    }

    /// An exact-shape profile: min = opt = max.
    pub fn add_exact_profile(&self, inputs: &[(&str, Dims)]) -> Result<i32> {
        let full: Vec<_> = inputs.iter().map(|&(n, d)| (n, d, d, d)).collect();
        self.add_profile(&full)
    }

    /// A timing cache, empty or from a serialized blob.
    pub fn timing_cache(&self, blob: &[u8]) -> Result<TimingCache> {
        clear_last_error();
        // SAFETY: live config; TensorRT copies the blob.
        let raw = unsafe {
            ffi::llmtrt_timing_cache_create(self.raw, blob.as_ptr() as *const c_void, blob.len())
        };
        Ok(TimingCache {
            raw: non_null(raw, "createTimingCache")?,
        })
    }

    /// The cache must outlive every build that uses this config.
    pub fn set_timing_cache(&self, cache: &TimingCache, ignore_mismatch: bool) -> Result<()> {
        clear_last_error();
        // SAFETY: both live.
        check(
            unsafe { ffi::llmtrt_config_set_timing_cache(self.raw, cache.raw, ignore_mismatch) },
            "setTimingCache",
        )
    }
}

impl Drop for BuilderConfig<'_> {
    fn drop(&mut self) {
        // SAFETY: created by `llmtrt_config_create`, dropped once.
        unsafe { ffi::llmtrt_config_destroy(self.raw) }
    }
}

/// `nvinfer1::ITimingCache`.
pub struct TimingCache {
    raw: *mut c_void,
}

/// `TimingCacheValue`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimingEntry {
    pub tactic: u64,
    pub ms: f32,
}

impl TimingCache {
    pub fn serialize(&self) -> Result<HostMemory> {
        clear_last_error();
        // SAFETY: live cache.
        let raw = unsafe { ffi::llmtrt_timing_cache_serialize(self.raw) };
        Ok(HostMemory {
            raw: non_null(raw, "ITimingCache::serialize")?,
        })
    }

    pub fn keys(&self) -> Vec<[u8; 16]> {
        // SAFETY: live cache; a null buffer with zero capacity asks for the count.
        let n = unsafe { ffi::llmtrt_timing_cache_keys(self.raw, std::ptr::null_mut(), 0) };
        if n <= 0 {
            return Vec::new();
        }
        let mut keys = vec![[0u8; 16]; n as usize];
        // SAFETY: `keys` holds `n` 16-byte keys.
        let got = unsafe { ffi::llmtrt_timing_cache_keys(self.raw, keys.as_mut_ptr().cast(), n) };
        keys.truncate(got.max(0) as usize);
        keys
    }

    pub fn query(&self, key: &[u8; 16]) -> Option<TimingEntry> {
        let (mut tactic, mut ms) = (0u64, 0f32);
        // SAFETY: live cache; out-pointers are valid.
        let ok =
            unsafe { ffi::llmtrt_timing_cache_query(self.raw, key.as_ptr(), &mut tactic, &mut ms) };
        ok.then_some(TimingEntry { tactic, ms })
    }

    pub fn update(&self, key: &[u8; 16], entry: TimingEntry) -> Result<()> {
        clear_last_error();
        // SAFETY: live cache.
        check(
            unsafe {
                ffi::llmtrt_timing_cache_update(self.raw, key.as_ptr(), entry.tactic, entry.ms)
            },
            "ITimingCache::update",
        )
    }
}

impl Drop for TimingCache {
    fn drop(&mut self) {
        // SAFETY: created by TensorRT, dropped once.
        unsafe { ffi::llmtrt_timing_cache_destroy(self.raw) }
    }
}

/// `nvonnxparser::IParser`, filling a network from an ONNX model.
pub struct OnnxParser<'n> {
    raw: *mut c_void,
    _network: PhantomData<&'n ()>,
}

impl<'n> OnnxParser<'n> {
    pub fn new(network: &'n Network<'_>) -> Result<Self> {
        clear_last_error();
        // SAFETY: live network.
        let raw = non_null(
            unsafe { ffi::llmtrt_parser_create(network.raw()) },
            "createParser",
        )?;
        Ok(Self {
            raw,
            _network: PhantomData,
        })
    }

    pub fn parse(&self, model: &[u8]) -> Result<()> {
        clear_last_error();
        // SAFETY: live parser; TensorRT reads `model` during the call.
        let ok = unsafe {
            ffi::llmtrt_parser_parse(self.raw, model.as_ptr() as *const c_void, model.len())
        };
        if ok {
            return Ok(());
        }
        // SAFETY: live parser; each description is owned by it.
        let n = unsafe { ffi::llmtrt_parser_nb_errors(self.raw) };
        let errors = (0..n)
            .filter_map(|i| {
                let p = unsafe { ffi::llmtrt_parser_error(self.raw, i) };
                (!p.is_null()).then(|| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
            })
            .collect();
        Err(TrtError::Parse(errors))
    }
}

impl Drop for OnnxParser<'_> {
    fn drop(&mut self) {
        // SAFETY: created by `llmtrt_parser_create`, dropped once.
        unsafe { ffi::llmtrt_parser_destroy(self.raw) }
    }
}
