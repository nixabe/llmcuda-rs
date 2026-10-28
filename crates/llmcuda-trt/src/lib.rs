//! TensorRT for llmcuda-rs, over a thin C ABI shim.
//!
//! This crate owns the boundary between Rust and TensorRT 11: builder,
//! network definition, ONNX parser, runtime, engine, execution context,
//! refitter, and IPluginV3 plugins implemented in Rust. Like `llmcuda-cuda`
//! it holds no model knowledge — it does not know what a layer or an expert
//! is, and it should stay that way.
//!
//! ## Feature `tensorrt`
//!
//! Everything that touches TensorRT is behind feature `tensorrt`, off by
//! default. Without it this crate builds on a machine with no TensorRT and no
//! GPU and exports only [`bindings`], which is plain Rust. With it,
//! `build.rs` compiles `cpp/shim.cpp` against `$TENSORRT_ROOT` and links
//! `libnvinfer` and `libnvonnxparser` dynamically; nothing is vendored.
//!
//! The standard TensorRT is used, not TensorRT-RTX: the latter leaves Turing
//! out of its default targets and has no FP32 GEMMs there.
//!
//! ## Streams and contexts
//!
//! TensorRT carries its own static CUDA runtime, which uses whatever context
//! is current. cudarc's `CudaContext::new` retains the device's primary
//! context and makes it current, so the two agree as long as the calling
//! thread has bound that context. `trt_probe` checks this on the device
//! rather than assuming it. Engines are built with no auxiliary streams, so
//! every layer and plugin runs on the stream passed to `enqueueV3`.

pub mod bindings;

#[cfg(feature = "tensorrt")]
mod builder;
#[cfg(feature = "tensorrt")]
mod ffi;
#[cfg(feature = "tensorrt")]
pub mod plugin;
#[cfg(feature = "tensorrt")]
mod runtime;
#[cfg(feature = "tensorrt")]
mod types;

#[cfg(feature = "tensorrt")]
pub use builder::{
    Builder, BuilderConfig, HostMemory, Network, OnnxParser, Tensor, TimingCache, TimingEntry,
};
#[cfg(feature = "tensorrt")]
pub use runtime::{
    DebugListener, DebugTensor, DeviceAllocator, Engine, ExecutionContext, GpuAllocator, Refitter,
    Runtime,
};
#[cfg(feature = "tensorrt")]
pub use types::{
    AllocationStrategy, BuilderFlag, CausalMask, DataType, Dims, ElementWise, MatrixOp,
    ProfilingVerbosity, Result, TensorLocation, TrtError, header_version, library_version,
    onnx_parser_version,
};

/// The plugin ABI version serialized into every plan this crate builds.
#[cfg(feature = "tensorrt")]
pub const PLUGIN_ABI_VERSION: i32 = ffi::ABI_VERSION;

/// Whether this build links TensorRT.
pub const ENABLED: bool = cfg!(feature = "tensorrt");
