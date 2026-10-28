//! Running engines: runtime, engine, execution context, refitter, and the
//! adapters that let Rust own device allocation and debug-tensor capture.

use std::ffi::{CStr, CString, c_char, c_void};
use std::marker::PhantomData;
use std::sync::Arc;

use cudarc::driver::sys::CUstream;

use crate::ffi;
use crate::types::{
    AllocationStrategy, DataType, Dims, Result, TensorLocation, TrtError, check, clear_last_error,
    ensure_logger, fail, non_null,
};

fn cstring(s: &str) -> Result<CString> {
    CString::new(s).map_err(|_| TrtError::Invalid(format!("{s:?} contains a NUL byte")))
}

/// Device memory for TensorRT, supplied by Rust.
///
/// Installed on a [`Runtime`] it serves everything an engine allocates;
/// installed on an [`ExecutionContext`] it serves temporary storage. Errors
/// cannot cross back into TensorRT, so `allocate` returns null on failure and
/// says why through `tracing`.
pub trait DeviceAllocator: Send + Sync + 'static {
    fn allocate(&self, size: u64, alignment: u64, stream: CUstream) -> *mut c_void;
    fn free(&self, memory: *mut c_void, stream: CUstream) -> bool;
}

/// A [`DeviceAllocator`] wrapped as an `IGpuAsyncAllocator`.
pub struct GpuAllocator {
    raw: *mut c_void,
    /// Boxed twice so the pointer handed to C is thin and stable.
    _inner: Box<Arc<dyn DeviceAllocator>>,
}

// SAFETY: the C++ adapter is stateless beyond its callbacks, and the Rust
// allocator behind it is `Send + Sync`.
unsafe impl Send for GpuAllocator {}
unsafe impl Sync for GpuAllocator {}

unsafe extern "C" fn alloc_trampoline(
    user: *mut c_void,
    size: u64,
    alignment: u64,
    _flags: u32,
    stream: *mut c_void,
) -> *mut c_void {
    std::panic::catch_unwind(|| {
        // SAFETY: `user` is the `Arc` boxed by `GpuAllocator::new`, alive
        // while the C++ adapter exists.
        let inner = unsafe { &*(user as *const Arc<dyn DeviceAllocator>) };
        inner.allocate(size, alignment, stream as CUstream)
    })
    .unwrap_or_else(|_| {
        tracing::error!("TensorRT allocator panicked");
        std::ptr::null_mut()
    })
}

unsafe extern "C" fn free_trampoline(
    user: *mut c_void,
    memory: *mut c_void,
    stream: *mut c_void,
) -> bool {
    std::panic::catch_unwind(|| {
        // SAFETY: as in `alloc_trampoline`.
        let inner = unsafe { &*(user as *const Arc<dyn DeviceAllocator>) };
        inner.free(memory, stream as CUstream)
    })
    .unwrap_or_else(|_| {
        tracing::error!("TensorRT allocator panicked on free");
        false
    })
}

impl GpuAllocator {
    pub fn new(inner: Arc<dyn DeviceAllocator>) -> Result<Arc<Self>> {
        ensure_logger();
        let inner = Box::new(inner);
        let user = &*inner as *const Arc<dyn DeviceAllocator> as *mut c_void;
        // SAFETY: the callbacks are valid for the process; `user` outlives
        // the adapter because both drop together.
        let raw = unsafe { ffi::llmtrt_allocator_create(alloc_trampoline, free_trampoline, user) };
        Ok(Arc::new(Self {
            raw: non_null(raw, "IGpuAsyncAllocator")?,
            _inner: inner,
        }))
    }
}

impl Drop for GpuAllocator {
    fn drop(&mut self) {
        // SAFETY: created by `llmtrt_allocator_create`, dropped once, after
        // every runtime and context that referenced it (they hold the Arc).
        unsafe { ffi::llmtrt_allocator_destroy(self.raw) }
    }
}

/// What a debug listener is told about one marked tensor.
pub struct DebugTensor<'a> {
    pub address: *const c_void,
    pub location: TensorLocation,
    pub data_type: Option<DataType>,
    pub shape: Dims,
    pub name: &'a str,
    pub stream: CUstream,
}

type DebugFn = dyn FnMut(&DebugTensor<'_>) -> bool + Send;

/// `nvinfer1::IDebugListener` over a Rust closure.
///
/// The callback runs inside `enqueueV3`, on the enqueueing thread; anything
/// it wants to keep must be copied on `stream` before it returns, because
/// TensorRT may reuse the address for a later tensor.
pub struct DebugListener {
    raw: *mut c_void,
    _inner: Box<Box<DebugFn>>,
}

// SAFETY: the closure is `Send`; TensorRT calls it only from inside enqueue.
unsafe impl Send for DebugListener {}
unsafe impl Sync for DebugListener {}

unsafe extern "C" fn debug_trampoline(
    user: *mut c_void,
    addr: *const c_void,
    location: i32,
    data_type: i32,
    shape: *const ffi::llmtrt_dims,
    name: *const c_char,
    stream: *mut c_void,
) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: `user` is the boxed closure, alive while the listener is;
        // TensorRT calls it serially within one enqueue.
        let f = unsafe { &mut *(user as *mut Box<DebugFn>) };
        // SAFETY: TensorRT passes a valid shape and a NUL-terminated name.
        let (shape, name) = unsafe { (Dims(*shape), CStr::from_ptr(name).to_string_lossy()) };
        f(&DebugTensor {
            address: addr,
            location: if location == 0 {
                TensorLocation::Device
            } else {
                TensorLocation::Host
            },
            data_type: DataType::from_raw(data_type),
            shape,
            name: &name,
            stream: stream as CUstream,
        })
    }))
    .unwrap_or_else(|_| {
        tracing::error!("TensorRT debug listener panicked");
        false
    })
}

impl DebugListener {
    pub fn new(f: impl FnMut(&DebugTensor<'_>) -> bool + Send + 'static) -> Result<Arc<Self>> {
        let mut inner: Box<Box<DebugFn>> = Box::new(Box::new(f));
        let user = &mut *inner as *mut Box<DebugFn> as *mut c_void;
        // SAFETY: as for the allocator.
        let raw = unsafe { ffi::llmtrt_debug_listener_create(debug_trampoline, user) };
        Ok(Arc::new(Self {
            raw: non_null(raw, "IDebugListener")?,
            _inner: inner,
        }))
    }
}

impl Drop for DebugListener {
    fn drop(&mut self) {
        // SAFETY: created by `llmtrt_debug_listener_create`, dropped once.
        unsafe { ffi::llmtrt_debug_listener_destroy(self.raw) }
    }
}

/// `nvinfer1::IRuntime`.
pub struct Runtime {
    raw: *mut c_void,
    allocator: Option<Arc<GpuAllocator>>,
}

impl Runtime {
    pub fn new() -> Result<Self> {
        ensure_logger();
        clear_last_error();
        // SAFETY: no arguments; the shim supplies the shared logger.
        let raw = non_null(
            unsafe { ffi::llmtrt_runtime_create() },
            "createInferRuntime",
        )?;
        Ok(Self {
            raw,
            allocator: None,
        })
    }

    /// Route every allocation engines from this runtime make through
    /// `allocator`. Set it before deserializing.
    pub fn set_allocator(&mut self, allocator: Arc<GpuAllocator>) {
        // SAFETY: both live; the runtime keeps the Arc.
        unsafe { ffi::llmtrt_runtime_set_allocator(self.raw, allocator.raw) };
        self.allocator = Some(allocator);
    }

    pub fn deserialize(&self, plan: &[u8]) -> Result<Engine<'_>> {
        clear_last_error();
        // SAFETY: live runtime; TensorRT reads `plan` during the call.
        let raw = unsafe {
            ffi::llmtrt_engine_deserialize(self.raw, plan.as_ptr() as *const c_void, plan.len())
        };
        Ok(Engine {
            raw: non_null(raw, "deserializeCudaEngine")?,
            _runtime: PhantomData,
        })
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        // SAFETY: created by `llmtrt_runtime_create`, dropped once, after
        // every engine (which borrow it).
        unsafe { ffi::llmtrt_runtime_destroy(self.raw) }
    }
}

/// `nvinfer1::ICudaEngine`.
pub struct Engine<'r> {
    raw: *mut c_void,
    _runtime: PhantomData<&'r Runtime>,
}

impl Engine<'_> {
    /// Activation memory a context needs: for `profile`, or the maximum.
    pub fn device_memory_size(&self, profile: Option<i32>) -> i64 {
        // SAFETY: live engine.
        unsafe { ffi::llmtrt_engine_device_memory_size(self.raw, profile.unwrap_or(-1)) }
    }

    pub fn nb_profiles(&self) -> i32 {
        // SAFETY: live engine.
        unsafe { ffi::llmtrt_engine_nb_profiles(self.raw) }
    }

    pub fn nb_layers(&self) -> i32 {
        // SAFETY: live engine.
        unsafe { ffi::llmtrt_engine_nb_layers(self.raw) }
    }

    pub fn nb_aux_streams(&self) -> i32 {
        // SAFETY: live engine.
        unsafe { ffi::llmtrt_engine_nb_aux_streams(self.raw) }
    }

    /// `IEngineInspector` JSON; layer detail needs a detailed-verbosity build.
    pub fn information(&self) -> Result<String> {
        // SAFETY: live engine; first call sizes, second copies.
        let len = unsafe { ffi::llmtrt_engine_information(self.raw, std::ptr::null_mut(), 0) };
        if len < 0 {
            return Err(fail("IEngineInspector::getEngineInformation"));
        }
        let mut buf = vec![0u8; len as usize];
        unsafe { ffi::llmtrt_engine_information(self.raw, buf.as_mut_ptr().cast(), len) };
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    pub fn context(&self, strategy: AllocationStrategy) -> Result<ExecutionContext<'_>> {
        clear_last_error();
        // SAFETY: live engine.
        let raw = unsafe { ffi::llmtrt_context_create(self.raw, strategy as i32) };
        Ok(ExecutionContext {
            raw: non_null(raw, "createExecutionContext")?,
            temporary: None,
            listener: None,
            _engine: PhantomData,
        })
    }

    pub fn refitter(&self) -> Result<Refitter<'_>> {
        clear_last_error();
        // SAFETY: live engine.
        let raw = unsafe { ffi::llmtrt_refitter_create(self.raw) };
        Ok(Refitter {
            raw: non_null(raw, "createInferRefitter")?,
            _engine: PhantomData,
        })
    }
}

impl Drop for Engine<'_> {
    fn drop(&mut self) {
        // SAFETY: created by deserialization, dropped once, after its
        // contexts and refitters (which borrow it).
        unsafe { ffi::llmtrt_engine_destroy(self.raw) }
    }
}

/// `nvinfer1::IExecutionContext`.
pub struct ExecutionContext<'e> {
    raw: *mut c_void,
    temporary: Option<Arc<GpuAllocator>>,
    listener: Option<Arc<DebugListener>>,
    _engine: PhantomData<&'e ()>,
}

impl ExecutionContext<'_> {
    /// Bind activation memory for a [`AllocationStrategy::UserManaged`]
    /// context.
    ///
    /// # Safety
    /// `memory` must be device memory of at least `size` bytes that stays
    /// valid, and is used by nothing else, while this context can run.
    pub unsafe fn set_device_memory(&self, memory: u64, size: i64) -> Result<()> {
        clear_last_error();
        // SAFETY: the caller upholds the contract above.
        check(
            unsafe { ffi::llmtrt_context_set_device_memory(self.raw, memory as *mut c_void, size) },
            "setDeviceMemoryV2",
        )
    }

    pub fn set_temporary_allocator(&mut self, allocator: Arc<GpuAllocator>) -> Result<()> {
        clear_last_error();
        // SAFETY: both live; the context keeps the Arc.
        check(
            unsafe { ffi::llmtrt_context_set_temporary_allocator(self.raw, allocator.raw) },
            "setTemporaryStorageAllocator",
        )?;
        self.temporary = Some(allocator);
        Ok(())
    }

    pub fn set_profile(&self, index: i32, stream: CUstream) -> Result<()> {
        clear_last_error();
        // SAFETY: live context and stream.
        check(
            unsafe { ffi::llmtrt_context_set_profile(self.raw, index, stream as *mut c_void) },
            "setOptimizationProfileAsync",
        )
    }

    pub fn set_input_shape(&self, name: &str, dims: impl Into<Dims>) -> Result<()> {
        let name = cstring(name)?;
        let dims = dims.into();
        clear_last_error();
        // SAFETY: live context.
        check(
            unsafe { ffi::llmtrt_context_set_input_shape(self.raw, name.as_ptr(), &dims.0) },
            "setInputShape",
        )
    }

    /// # Safety
    /// `address` must be device memory large enough for the tensor's shape at
    /// enqueue, valid until the enqueued work completes.
    pub unsafe fn set_tensor_address(&self, name: &str, address: u64) -> Result<()> {
        let name = cstring(name)?;
        clear_last_error();
        // SAFETY: the caller upholds the contract above.
        check(
            unsafe {
                ffi::llmtrt_context_set_tensor_address(
                    self.raw,
                    name.as_ptr(),
                    address as *mut c_void,
                )
            },
            "setTensorAddress",
        )
    }

    /// `enqueueV3` on `stream`.
    ///
    /// # Safety
    /// Every bound address must stay valid until the work completes on
    /// `stream`, and `stream` must belong to the context TensorRT runs in.
    pub unsafe fn enqueue(&self, stream: CUstream) -> Result<()> {
        clear_last_error();
        // SAFETY: the caller upholds the contract above.
        check(
            unsafe { ffi::llmtrt_context_enqueue(self.raw, stream as *mut c_void) },
            "enqueueV3",
        )
    }

    pub fn set_debug_listener(&mut self, listener: Arc<DebugListener>) -> Result<()> {
        clear_last_error();
        // SAFETY: both live; the context keeps the Arc.
        check(
            unsafe { ffi::llmtrt_context_set_debug_listener(self.raw, listener.raw) },
            "setDebugListener",
        )?;
        self.listener = Some(listener);
        Ok(())
    }

    pub fn set_tensor_debug_state(&self, name: &str, on: bool) -> Result<()> {
        let name = cstring(name)?;
        clear_last_error();
        // SAFETY: live context.
        check(
            unsafe { ffi::llmtrt_context_set_tensor_debug_state(self.raw, name.as_ptr(), on) },
            "setTensorDebugState",
        )
    }
}

impl Drop for ExecutionContext<'_> {
    fn drop(&mut self) {
        // SAFETY: created by `llmtrt_context_create`, dropped once; the
        // allocator and listener it references drop after (fields drop after
        // this body runs).
        unsafe { ffi::llmtrt_context_destroy(self.raw) }
    }
}

/// `nvinfer1::IRefitter`.
pub struct Refitter<'e> {
    raw: *mut c_void,
    _engine: PhantomData<&'e ()>,
}

impl Refitter<'_> {
    /// # Safety
    /// `values` must point to `count` elements of `data_type` at `location`,
    /// valid until [`Self::refit`]'s work completes.
    pub unsafe fn set_named_weights(
        &self,
        name: &str,
        data_type: DataType,
        values: *const c_void,
        count: i64,
        location: TensorLocation,
    ) -> Result<()> {
        let name = cstring(name)?;
        clear_last_error();
        // SAFETY: the caller upholds the contract above.
        check(
            unsafe {
                ffi::llmtrt_refitter_set_named_weights(
                    self.raw,
                    name.as_ptr(),
                    data_type as i32,
                    values,
                    count,
                    location as i32,
                )
            },
            "setNamedWeights",
        )
    }

    /// Weights still missing from a combination TensorRT optimized
    /// together, after some of it was set. Not "weights a stripped plan
    /// needs": for those, see [`Self::all`].
    pub fn missing(&self) -> Vec<String> {
        self.names(ffi::llmtrt_refitter_missing)
    }

    /// Every weight that can be refit, by name.
    pub fn all(&self) -> Vec<String> {
        self.names(ffi::llmtrt_refitter_all)
    }

    fn names(
        &self,
        query: unsafe extern "C" fn(*mut c_void, *mut *const c_char, i32) -> i32,
    ) -> Vec<String> {
        // SAFETY: live refitter; the first call counts.
        let n = unsafe { query(self.raw, std::ptr::null_mut(), 0) };
        if n <= 0 {
            return Vec::new();
        }
        let mut names = vec![std::ptr::null(); n as usize];
        // SAFETY: `names` has room for `n` pointers, owned by the engine.
        unsafe { query(self.raw, names.as_mut_ptr(), n) };
        names
            .into_iter()
            .filter(|p| !p.is_null())
            .map(|p| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
            .collect()
    }

    pub fn refit(&self, stream: CUstream) -> Result<()> {
        clear_last_error();
        // SAFETY: live refitter; stream ordering covers the weights' use.
        check(
            unsafe { ffi::llmtrt_refitter_refit(self.raw, stream as *mut c_void) },
            "refitCudaEngineAsync",
        )
    }
}

impl Drop for Refitter<'_> {
    fn drop(&mut self) {
        // SAFETY: created by `llmtrt_refitter_create`, dropped once.
        unsafe { ffi::llmtrt_refitter_destroy(self.raw) }
    }
}
