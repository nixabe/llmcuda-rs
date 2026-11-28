//! The device, stream, kernels and allocator every check shares.

use std::collections::HashMap;
use std::ffi::{CString, c_void};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::sys::{self, CUfunction, CUstream};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DeviceRepr, result};
use llmcuda_trt::DeviceAllocator;
use tracing::error;

pub type Error = Box<dyn std::error::Error>;

/// Kernels the probe's plugins launch, as raw driver handles.
///
/// Raw because a plugin launches on whatever stream TensorRT hands it, and
/// cudarc launches only on a `CudaStream` it created.
#[derive(Clone, Copy)]
pub struct Kernels {
    /// The engine's `tensor_add`, from its embedded Rust-kernel PTX.
    pub add: CUfunction,
    pub accumulate: CUfunction,
    pub through_workspace: CUfunction,
}

/// Probe-only kernels: a side-effecting accumulator reached through a
/// device pointer table, and a pass through plugin workspace.
const PROBE_SRC: &str = r#"
extern "C" {

__global__ void probe_accumulate(const long long* __restrict__ table, float* __restrict__ out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float* acc = reinterpret_cast<float*>(table[0]);
        float v = *acc + 1.0f;
        *acc = v;
        out[0] = v;
    }
}

__global__ void probe_through_workspace(
    const float* __restrict__ in, float* __restrict__ ws, float* __restrict__ out, long long n
) {
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        ws[i] = in[i] * 2.0f;
        out[i] = ws[i] + 1.0f;
    }
}

}
"#;

fn load(src: &str, name: &str, functions: &[&str]) -> Result<Vec<CUfunction>, Error> {
    let ptx = llmcuda_cuda::kernels::compile(src, name)?;
    load_ptx(&ptx.to_src(), functions)
}

fn load_ptx(ptx: &str, functions: &[&str]) -> Result<Vec<CUfunction>, Error> {
    let image = CString::new(ptx)?;
    // SAFETY: a NUL-terminated PTX image; the module is never unloaded, so
    // its functions stay valid for the process.
    let module = unsafe { result::module::load_data(image.as_ptr() as *const c_void) }?;
    functions
        .iter()
        .map(|f| Ok(unsafe { result::module::get_function(module, CString::new(*f)?) }?))
        .collect()
}

pub struct Env {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub kernels: Kernels,
}

impl Env {
    pub fn new(ordinal: usize) -> Result<Self, Error> {
        let ctx = CudaContext::new(ordinal)?;
        // SAFETY: the probe uses one stream and keeps every buffer alive
        // until it synchronizes; graph capture (P4) requires this.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream()?;
        let add = load_ptx(llmcuda_cuda::kernels::RUST_KERNELS_PTX, &["tensor_add"])?;
        let probe = load(
            PROBE_SRC,
            "trt_probe",
            &["probe_accumulate", "probe_through_workspace"],
        )?;
        Ok(Self {
            ctx,
            stream,
            kernels: Kernels {
                add: add[0],
                accumulate: probe[0],
                through_workspace: probe[1],
            },
        })
    }

    pub fn upload<T: DeviceRepr + Clone>(&self, data: &[T]) -> Result<CudaSlice<T>, Error> {
        Ok(self.stream.clone_htod(data)?)
    }

    pub fn zeros<T: DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        &self,
        n: usize,
    ) -> Result<CudaSlice<T>, Error> {
        Ok(self.stream.alloc_zeros::<T>(n)?)
    }

    pub fn download<T: DeviceRepr + Clone + Default>(
        &self,
        buf: &CudaSlice<T>,
    ) -> Result<Vec<T>, Error> {
        let v = self.stream.clone_dtoh(buf)?;
        self.stream.synchronize()?;
        Ok(v)
    }

    pub fn ptr<T>(&self, buf: &CudaSlice<T>) -> u64 {
        buf.device_ptr(&self.stream).0
    }

    pub fn raw_stream(&self) -> CUstream {
        self.stream.cu_stream()
    }

    pub fn sync(&self) -> Result<(), Error> {
        Ok(self.stream.synchronize()?)
    }

    /// The device's primary context, as the driver reports it.
    pub fn primary_context(&self) -> Result<sys::CUcontext, Error> {
        let device = result::device::get(self.ctx.ordinal() as i32)?;
        // SAFETY: retain then release leaves the refcount unchanged; the
        // context stays alive through cudarc's own retain.
        unsafe {
            let primary = result::primary_ctx::retain(device)?;
            result::primary_ctx::release(device)?;
            Ok(primary)
        }
    }
}

/// `cuMemGetInfo` free bytes.
pub fn free_bytes() -> Result<usize, Error> {
    Ok(result::mem_get_info()?.0)
}

/// Unpooled device memory, from `cuMemAlloc`, freed on drop.
///
/// Not cudarc's pool: a transient that shares the stream-ordered pool with
/// resident data fragments it (BENCHMARKS.md, "Residency: a transient must
/// not share the pool").
pub struct Unpooled {
    pub ptr: u64,
}

impl Unpooled {
    pub fn new(bytes: usize) -> Result<Self, Error> {
        // SAFETY: plain allocation in the current context.
        let ptr = unsafe { result::malloc_sync(bytes.max(1)) }?;
        Ok(Self { ptr })
    }
}

impl Drop for Unpooled {
    fn drop(&mut self) {
        // SAFETY: allocated by `malloc_sync`, freed once.
        if let Err(e) = unsafe { result::free_sync(self.ptr) } {
            error!("cuMemFree: {e:?}");
        }
    }
}

/// A [`DeviceAllocator`] over unpooled `cuMemAlloc`, counting every call.
///
/// `cuMemAlloc` guarantees 256-byte alignment; TensorRT asks for 512 for
/// some buffers, so a larger request is over-allocated and aligned up, and
/// the base kept for the free.
#[derive(Default)]
pub struct CountingAllocator {
    pub allocations: AtomicU64,
    pub frees: AtomicU64,
    pub bytes: AtomicU64,
    /// Aligned address -> `cuMemAlloc` base, for over-aligned blocks.
    bases: Mutex<HashMap<u64, u64>>,
}

impl CountingAllocator {
    pub fn snapshot(&self) -> (u64, u64, u64) {
        (
            self.allocations.load(Ordering::SeqCst),
            self.frees.load(Ordering::SeqCst),
            self.bytes.load(Ordering::SeqCst),
        )
    }
}

/// `cuMemAlloc` aligns to at least this many bytes.
const CU_MEM_ALLOC_ALIGNMENT: u64 = 256;

impl DeviceAllocator for CountingAllocator {
    fn allocate(&self, size: u64, alignment: u64, _stream: CUstream) -> *mut c_void {
        self.allocations.fetch_add(1, Ordering::SeqCst);
        self.bytes.fetch_add(size, Ordering::SeqCst);
        let alignment = alignment.max(1);
        if !alignment.is_power_of_two() {
            error!(size, alignment, "non-power-of-two alignment");
            return std::ptr::null_mut();
        }
        let extra = if alignment > CU_MEM_ALLOC_ALIGNMENT {
            alignment
        } else {
            0
        };
        // SAFETY: plain allocation; TensorRT calls this with the engine's
        // context current.
        let base = match unsafe { result::malloc_sync((size + extra).max(1) as usize) } {
            Ok(p) => p,
            Err(e) => {
                error!(size, "cuMemAlloc failed: {e:?}");
                return std::ptr::null_mut();
            }
        };
        let aligned = base.next_multiple_of(alignment);
        if aligned != base {
            self.bases.lock().unwrap().insert(aligned, base);
        }
        aligned as *mut c_void
    }

    fn free(&self, memory: *mut c_void, _stream: CUstream) -> bool {
        self.frees.fetch_add(1, Ordering::SeqCst);
        if memory.is_null() {
            return true;
        }
        let ptr = memory as u64;
        let base = self.bases.lock().unwrap().remove(&ptr).unwrap_or(ptr);
        // SAFETY: TensorRT frees only what `allocate` returned. A freed
        // block may still be in use by queued work; cuMemFree synchronizes.
        unsafe { result::free_sync(base) }.is_ok()
    }
}

/// The current context, the context `stream` belongs to, and the context
/// `ptr` was allocated in.
pub fn contexts_of(stream: CUstream, ptr: u64) -> (sys::CUcontext, sys::CUcontext, sys::CUcontext) {
    let current = result::ctx::get_current()
        .ok()
        .flatten()
        .unwrap_or(std::ptr::null_mut());
    let mut of_stream: sys::CUcontext = std::ptr::null_mut();
    let mut of_ptr: sys::CUcontext = std::ptr::null_mut();
    // SAFETY: out-pointers are valid; failures leave them null, which the
    // check reports as a mismatch.
    unsafe {
        let _ = sys::cuStreamGetCtx(stream, &mut of_stream);
        let _ = sys::cuPointerGetAttribute(
            &mut of_ptr as *mut sys::CUcontext as *mut c_void,
            sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_CONTEXT,
            ptr,
        );
    }
    (current, of_stream, of_ptr)
}

/// `LayerOpsKernels::add`'s launch shape, so the plugin runs the kernel
/// exactly as the engine does.
pub fn elementwise_grid(n: usize) -> (u32, u32) {
    const BLOCK: usize = 256;
    (n.div_ceil(BLOCK).clamp(1, 1024) as u32, BLOCK as u32)
}
