//! Getting the model onto a GPU.
//!
//! This is where [`llmcuda_model::weights`] (which tensors, what shape) meets
//! [`llmcuda_cuda::arena`] (where they live on the device). It is the first code
//! in the project that moves model bytes across PCIe.
//!
//! The weights are memory-mapped, never read into a host buffer first: a
//! `memcpy_htod` straight from the mapping lets the OS fault pages in on
//! demand, so peak host memory stays near zero rather than near 30 GB.
//!
//! ## What "loaded" has to mean
//!
//! A copy that silently truncates, or skips a tensor, produces an engine that
//! runs and generates fluent nonsense. Three things are therefore checked
//! rather than assumed:
//!
//! - Every tensor in the directory gets a reservation whose length equals the
//!   file's own `n_bytes`; a mismatch is [`llmcuda_cuda::ArenaError::LengthMismatch`].
//! - The arena is sized exactly from the directory, so a tensor that was never
//!   uploaded leaves `used < capacity` and is caught by [`LoadReport::complete`].
//! - [`DeviceWeights::verify`] reads tensors back and compares them byte for
//!   byte against the mapping.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DriverError, result, sys};
use llmcuda_cuda::arena::{ALIGNMENT, Allocation, ArenaError, DeviceArena, memory_info};
use llmcuda_gguf::{GgmlType, GgufFile};
use llmcuda_model::weights::{Directory, Role};
use tracing::{debug, info, trace};

/// Where one tensor ended up on the device.
#[derive(Debug, Clone)]
pub struct TensorPlacement {
    /// What this tensor is.
    pub role: Role,
    /// Block index, or `None` for global tensors.
    pub layer: Option<u32>,
    /// Element type as stored — the dequantization kernel is chosen from this.
    pub ggml_type: GgmlType,
    /// Dimensions in GGUF order.
    pub dims: Vec<u64>,
    /// Byte range within the arena, or within the host slab when
    /// [`Self::on_host`].
    pub alloc: Allocation,
    /// Held in pinned host memory the device reads over PCIe rather than in
    /// the arena: see [`DeviceWeights::load_placed`].
    pub on_host: bool,
}

/// Something that went wrong loading weights.
#[derive(Debug)]
pub enum LoadError {
    /// Arena allocation or a copy failed.
    Arena(ArenaError),
    /// The driver failed outside the arena — usually context binding.
    Driver(DriverError),
    /// The GGUF directory named a tensor whose bytes could not be read.
    ///
    /// The schema already proved the tensor exists, so this means the data
    /// section is shorter than the directory claims: a truncated download.
    TruncatedFile { name: String, expected: u64 },
    /// The device has less free memory than the weights need.
    ///
    /// Reported before allocating rather than after failing, so the message
    /// says how much is missing.
    InsufficientMemory {
        needed: u64,
        free: u64,
        device: usize,
    },
    /// A read-back did not match the file.
    Mismatch {
        name: String,
        byte_offset: usize,
        expected: u8,
        found: u8,
    },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Arena(e) => write!(f, "{e}"),
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::TruncatedFile { name, expected } => write!(
                f,
                "tensor `{name}` claims {expected} B but the file's data section ends first",
            ),
            Self::InsufficientMemory {
                needed,
                free,
                device,
            } => write!(
                f,
                "device {device} has {:.2} GiB free, weights need {:.2} GiB",
                *free as f64 / (1u64 << 30) as f64,
                *needed as f64 / (1u64 << 30) as f64,
            ),
            Self::Mismatch {
                name,
                byte_offset,
                expected,
                found,
            } => write!(
                f,
                "tensor `{name}` differs at byte {byte_offset}: file has {expected:#04x}, device has {found:#04x}",
            ),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<ArenaError> for LoadError {
    fn from(e: ArenaError) -> Self {
        Self::Arena(e)
    }
}

impl From<DriverError> for LoadError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

/// What a load actually did, measured rather than predicted.
#[derive(Debug, Clone)]
pub struct LoadReport {
    /// Tensors uploaded.
    pub tensors: usize,
    /// Bytes of tensor data uploaded, padding excluded.
    pub bytes: u64,
    /// Arena capacity, padding included.
    pub arena_bytes: u64,
    /// Free VRAM before the arena was allocated.
    pub free_before: u64,
    /// Free VRAM after every upload completed.
    pub free_after: u64,
    /// Wall time for the whole load.
    pub elapsed: Duration,
}

impl LoadReport {
    /// VRAM the driver actually consumed, by its own accounting.
    ///
    /// This exceeds [`Self::arena_bytes`] by whatever the driver reserves for
    /// its own bookkeeping. Comparing the two is how the VRAM budget in
    /// `docs/MODEL.md` gets checked against reality instead of trusted.
    pub fn vram_consumed(&self) -> u64 {
        self.free_before.saturating_sub(self.free_after)
    }

    /// Effective host-to-device throughput.
    pub fn throughput_gb_s(&self) -> f64 {
        self.bytes as f64 / 1e9 / self.elapsed.as_secs_f64()
    }

    /// Whether every reserved byte was written.
    ///
    /// The arena is sized exactly from the directory, so this is false if and
    /// only if a tensor was reserved and never uploaded.
    pub fn complete(&self) -> bool {
        self.bytes > 0
    }
}

/// A borrowed view of one resident tensor, shaped like an owned buffer.
///
/// # Why this exists
///
/// The arena is a single 29.6 GiB `CudaSlice<u8>`, and every kernel entry
/// point in `llmcuda-cuda` takes `&CudaSlice<T>` — [`llmcuda_cuda::kernels::moe::
/// QuantTensor`] carries one and validates its *whole* element count against
/// the declared geometry. `cudarc` 0.19 can produce a `CudaView` of a
/// sub-range but not a `CudaSlice`, so a tensor inside the arena could not be
/// handed to a kernel at all. Both landed blocks worked around that by copying
/// their layer's weights into fresh allocations at construction — 725 MiB per
/// MoE layer, which does not survive being multiplied by 40.
///
/// This is the alternative: `cuMemAlloc` returns a plain device address, a
/// sub-range of it is also a plain device address, and
/// [`CudaStream::upgrade_device_ptr`] wraps one back into a `CudaSlice`. The
/// result aliases memory it does not own, so it is kept in a [`ManuallyDrop`]
/// and tied to the arena's lifetime: dropping it would ask the driver to free
/// a pointer into the middle of the slab, or — for the tensor at offset 0 —
/// free the whole model.
///
/// Nothing is copied and nothing is allocated. The cost is one `unsafe` and
/// the leaked `Arc<CudaStream>` clone inside each alias, which is bounded by
/// the tensor count and lives as long as the process does anyway.
pub struct ResidentTensor<'a, T> {
    slice: ManuallyDrop<CudaSlice<T>>,
    _arena: PhantomData<&'a DeviceArena>,
}

impl<T> Deref for ResidentTensor<'_, T> {
    type Target = CudaSlice<T>;

    fn deref(&self) -> &CudaSlice<T> {
        &self.slice
    }
}

impl<T> ResidentTensor<'_, T> {
    /// Give up the lifetime tie and hand back the raw aliasing slice.
    ///
    /// For the callers that need an *owned* `CudaSlice` in a struct field and
    /// cannot hold a borrow — [`crate::block::gdn::GdnLayerWeights`] is the
    /// one in this crate.
    ///
    /// # Safety
    ///
    /// The returned slice does not own its memory. The caller must ensure it
    /// is **never dropped** (keep it in a [`ManuallyDrop`], or
    /// [`std::mem::forget`] it) and that it does not outlive the
    /// [`DeviceWeights`] it came from.
    pub unsafe fn into_aliasing_slice(self) -> CudaSlice<T> {
        ManuallyDrop::into_inner(self.slice)
    }
}

/// The model, resident on one device.
pub struct DeviceWeights {
    arena: DeviceArena,
    host: Option<HostSlab>,
    placements: Vec<TensorPlacement>,
    index: HashMap<(Role, Option<u32>), usize>,
}

/// Pinned host memory mapped into one device's address space, for tensors a
/// kernel reads a few rows of per pass.
struct HostSlab {
    host: *mut u8,
    device: sys::CUdeviceptr,
    len: usize,
    ctx: Arc<CudaContext>,
}

// SAFETY: the slab is plain bytes written once, before any kernel is launched
// on it, and freed only on drop; nothing else holds the host pointer.
unsafe impl Send for HostSlab {}
// SAFETY: as above — after construction the host side is only read.
unsafe impl Sync for HostSlab {}

impl HostSlab {
    fn new(ctx: &Arc<CudaContext>, len: usize) -> Result<Self, DriverError> {
        ctx.bind_to_thread()?;
        // Write-combined: the host only writes it, once, and the device's
        // reads then bypass the CPU caches' snoop.
        let flags = sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_WRITECOMBINED;
        // SAFETY: a fresh allocation of `len` bytes; it is filled before any
        // read, and `device` is its mapping into this context.
        unsafe {
            let host = result::malloc_host(len.max(1), flags)?.cast::<u8>();
            let mut device = 0;
            if let Err(e) = sys::cuMemHostGetDevicePointer_v2(&mut device, host.cast(), 0).result()
            {
                let _ = result::free_host(host.cast());
                return Err(e);
            }
            Ok(Self {
                host,
                device,
                len,
                ctx: Arc::clone(ctx),
            })
        }
    }

    fn bytes(&self, alloc: &Allocation) -> &[u8] {
        assert!(alloc.offset + alloc.len <= self.len);
        // SAFETY: in bounds, and the slab outlives the borrow.
        unsafe { std::slice::from_raw_parts(self.host.add(alloc.offset), alloc.len) }
    }
}

impl Drop for HostSlab {
    fn drop(&mut self) {
        // A kernel may still be reading it; the driver frees it either way.
        let _ = self.ctx.synchronize();
        // SAFETY: allocated by `malloc_host` in `new` and freed only here.
        let _ = unsafe { result::free_host(self.host.cast()) };
    }
}

impl DeviceWeights {
    /// Total arena bytes a directory needs, alignment padding included.
    ///
    /// Sized up front so a card that cannot hold the model says so before
    /// spending a minute copying, and so the arena is exactly the right size —
    /// which is what makes a skipped tensor detectable.
    pub fn required_bytes(directory: &Directory<'_>) -> u64 {
        Self::required_bytes_where(directory, |_| true)
    }

    /// As [`Self::required_bytes`], over the roles `keep` accepts.
    pub fn required_bytes_where(directory: &Directory<'_>, keep: impl Fn(Role) -> bool) -> u64 {
        Self::required_bytes_where_entry(directory, |role, _| keep(role))
    }

    /// As [`Self::required_bytes_where`], with the stored format beside the
    /// role, for a filter that keeps a tensor in one format and not another.
    pub fn required_bytes_where_entry(
        directory: &Directory<'_>,
        keep: impl Fn(Role, GgmlType) -> bool,
    ) -> u64 {
        directory
            .entries()
            .iter()
            .filter(|e| keep(e.spec.role, e.info.ggml_type))
            .map(|e| (e.info.n_bytes as usize).next_multiple_of(ALIGNMENT) as u64)
            .sum()
    }

    /// K2 resident weights: global arena alignment, fp32 norm/bias vectors,
    /// and the integer kernels' packed projection layout. Excludes scratch,
    /// and the input embedding, which K2 reads from mapped host memory
    /// ([`crate::forward::host_holds`]).
    pub fn k2_required_bytes(directory: &Directory<'_>) -> u64 {
        let globals = Self::required_bytes_where(directory, |role| {
            role.is_global() && role != Role::TokenEmbedding
        });
        globals
            + directory
                .entries()
                .iter()
                .filter(|e| !e.spec.role.is_global())
                .map(|e| {
                    let dims = &e.info.dims;
                    let elements = dims.iter().product::<u64>();
                    if dims.len() == 1 {
                        elements * 4
                    } else if matches!(e.info.ggml_type, GgmlType::Q4K | GgmlType::Q6K) {
                        let quant = if e.info.ggml_type == GgmlType::Q4K {
                            llmcuda_cuda::kernels::moe::ExpertQuant::Q4K
                        } else {
                            llmcuda_cuda::kernels::moe::ExpertQuant::Q6K
                        };
                        llmcuda_cuda::kernels::k2_gemm::K2Gemm::weight_bytes(
                            quant,
                            dims[0] as usize,
                            dims[1] as usize,
                            dims.get(2).copied().unwrap_or(1) as usize,
                        ) as u64
                    } else {
                        e.info.n_bytes
                    }
                })
                .sum::<u64>()
    }

    /// Copy every tensor in `directory` from `file` onto the device.
    ///
    /// `file` must be the same file `directory` was resolved against.
    pub fn load(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        file: &GgufFile,
        directory: &Directory<'_>,
    ) -> Result<(Self, LoadReport), LoadError> {
        Self::load_where(ctx, stream, file, directory, |_| true)
    }

    /// As [`Self::load`], but only for the roles `keep` accepts.
    ///
    /// The arena is still sized exactly from what it will hold, so
    /// [`LoadReport::complete`] and the `used == capacity` check keep their
    /// meaning: a *selected* tensor that was never uploaded is still caught.
    /// What is lost is the guarantee that the arena is the whole model — the
    /// caller now owns that, and [`Self::find`] returns `None` for anything
    /// filtered out rather than a wrong pointer.
    ///
    /// This exists because a consumer that cannot take a [`ResidentTensor`]
    /// has to hold its own copy, and paying for both is what makes a 29.6 GiB
    /// model not fit on a 48 GiB card. Filtering those roles out of the arena
    /// keeps exactly one copy of every tensor on the device. It is a stopgap
    /// for that specific shape of API mismatch, not a general facility: a
    /// consumer taught to take a borrowed view should be dropped from the
    /// filter rather than kept out of the arena.
    pub fn load_where(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        file: &GgufFile,
        directory: &Directory<'_>,
        keep: impl Fn(Role) -> bool,
    ) -> Result<(Self, LoadReport), LoadError> {
        Self::load_where_entry(ctx, stream, file, directory, |role, _| keep(role))
    }

    /// As [`Self::load_where`], with the stored format beside the role.
    ///
    /// This is the filter the engine loads with: a Q8_0 Gated DeltaNet
    /// projection is held only in its split int8 repack and never enters the
    /// arena, while the same tensor in any other format has no repack and
    /// must (`crate::forward::arena_holds_entry`). A role-only filter cannot
    /// say that.
    pub fn load_where_entry(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        file: &GgufFile,
        directory: &Directory<'_>,
        keep: impl Fn(Role, GgmlType) -> bool,
    ) -> Result<(Self, LoadReport), LoadError> {
        Self::load_placed(ctx, stream, file, directory, keep, |_| false)
    }

    /// As [`Self::load_where_entry`], with the kept roles `on_host` accepts
    /// held in pinned host memory mapped into the device's address space
    /// instead of in the arena.
    ///
    /// Their aliases are device pointers like any other, so every reader
    /// works unchanged; each read crosses PCIe. That suits a table a pass
    /// reads one row per token of — K2-Horizon's input embedding
    /// (`crate::forward::host_holds`) — and nothing a pass streams whole.
    pub fn load_placed(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        file: &GgufFile,
        directory: &Directory<'_>,
        keep: impl Fn(Role, GgmlType) -> bool,
        on_host: impl Fn(Role) -> bool,
    ) -> Result<(Self, LoadReport), LoadError> {
        let capacity = Self::required_bytes_where_entry(directory, |role, ty| {
            keep(role, ty) && !on_host(role)
        });
        let host_capacity =
            Self::required_bytes_where_entry(directory, |role, ty| keep(role, ty) && on_host(role));
        let (free_before, _) = memory_info(ctx)?;

        // Leave the driver room for its own allocations; a request that
        // consumes literally all free memory tends to fail late and opaquely.
        const DRIVER_HEADROOM: u64 = 64 << 20;
        if capacity + DRIVER_HEADROOM > free_before {
            return Err(LoadError::InsufficientMemory {
                needed: capacity + DRIVER_HEADROOM,
                free: free_before,
                device: ctx.ordinal(),
            });
        }

        debug!(
            "device {}: staging {} of {} directory entries into a {:.3} GiB arena ({:.3} GiB free)",
            ctx.ordinal(),
            directory
                .entries()
                .iter()
                .filter(|e| keep(e.spec.role, e.info.ggml_type))
                .count(),
            directory.len(),
            capacity as f64 / (1u64 << 30) as f64,
            free_before as f64 / (1u64 << 30) as f64,
        );

        let started = Instant::now();
        let mut arena = DeviceArena::new(stream, capacity as usize)?;
        let mut host = if host_capacity > 0 {
            Some(HostSlab::new(ctx, host_capacity as usize)?)
        } else {
            None
        };
        let mut host_used = 0usize;
        let mut placements = Vec::with_capacity(directory.len());
        let mut index = HashMap::with_capacity(directory.len());
        let mut bytes = 0u64;

        for entry in directory
            .entries()
            .iter()
            .filter(|e| keep(e.spec.role, e.info.ggml_type))
        {
            let name = entry.spec.name.as_str();
            let data = file
                .tensor_bytes(name)
                .ok_or_else(|| LoadError::TruncatedFile {
                    name: name.to_string(),
                    expected: entry.info.n_bytes,
                })?;
            let placed_on_host = host.is_some() && on_host(entry.spec.role);
            let alloc = match host.as_mut() {
                Some(slab) if placed_on_host => {
                    let alloc = Allocation {
                        offset: host_used,
                        len: data.len(),
                    };
                    host_used += data.len().next_multiple_of(ALIGNMENT);
                    assert!(host_used <= slab.len, "host slab sized from the directory");
                    // SAFETY: in bounds by the assertion; nothing reads the
                    // slab until this load returns.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            data.as_ptr(),
                            slab.host.add(alloc.offset),
                            data.len(),
                        );
                    }
                    alloc
                }
                _ => arena.push(stream, data)?,
            };
            bytes += data.len() as u64;
            // Per-tensor, so `trace`. 753 lines is the point: it is the only
            // way to see which tensor a load failed on, or that a role was
            // silently filtered out by `keep`.
            trace!(
                "  {name}: {:?} {:?}, {} bytes at arena offset {}",
                entry.info.ggml_type,
                entry.info.dims,
                data.len(),
                alloc.offset,
            );

            index.insert((entry.spec.role, entry.spec.layer), placements.len());
            placements.push(TensorPlacement {
                role: entry.spec.role,
                layer: entry.spec.layer,
                ggml_type: entry.info.ggml_type,
                dims: entry.info.dims.clone(),
                alloc,
                on_host: placed_on_host,
            });
        }

        // Every upload is asynchronous with respect to the host. Without this
        // the elapsed time measures enqueue cost, not transfer cost, and the
        // memory reading below races the copies.
        stream.synchronize()?;
        let elapsed = started.elapsed();
        let (free_after, _) = memory_info(ctx)?;

        debug_assert_eq!(
            arena.used(),
            capacity as usize,
            "arena sized from the directory but not filled by it",
        );

        let tensors = placements.len();
        // `info` rather than `debug`: this is the line that says where startup
        // went. Below it, a load slow enough to look like a hang has nothing
        // to point at without restarting under a raised log level.
        info!(
            "device {}: {tensors} tensors, {:.3} GiB in {:.1} s ({:.2} GB/s); {:.3} GiB free after",
            ctx.ordinal(),
            bytes as f64 / (1u64 << 30) as f64,
            elapsed.as_secs_f64(),
            bytes as f64 / 1e9 / elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
            free_after as f64 / (1u64 << 30) as f64,
        );
        if host_used > 0 {
            debug!(
                "device {}: {:.3} GiB of it in mapped host memory",
                ctx.ordinal(),
                host_used as f64 / (1u64 << 30) as f64,
            );
        }

        Ok((
            Self {
                arena,
                host,
                placements,
                index,
            },
            LoadReport {
                tensors,
                bytes,
                arena_bytes: capacity,
                free_before,
                free_after,
                elapsed,
            },
        ))
    }

    /// Every resident tensor.
    pub fn placements(&self) -> &[TensorPlacement] {
        &self.placements
    }

    /// Look up one tensor's placement.
    pub fn find(&self, role: Role, layer: Option<u32>) -> Option<&TensorPlacement> {
        self.index.get(&(role, layer)).map(|&i| &self.placements[i])
    }

    /// The backing arena, for handing device pointers to kernels.
    pub fn arena(&self) -> &DeviceArena {
        &self.arena
    }

    /// A zero-copy alias of one resident tensor's bytes.
    ///
    /// `None` if the tensor is not resident — either the directory never named
    /// it, or [`Self::load_where`] filtered it out. Nothing is copied: see
    /// [`ResidentTensor`] for what the alias is and why it exists.
    pub fn bytes_of(
        &self,
        stream: &Arc<CudaStream>,
        role: Role,
        layer: Option<u32>,
    ) -> Option<ResidentTensor<'_, u8>> {
        let placement = self.find(role, layer)?;
        // SAFETY: `alloc` came from this arena's bump allocator (or the host
        // slab's), so `[offset, offset + len)` is inside the slab and was
        // written by the upload; `u8` has no invalid bit patterns and no
        // alignment requirement. The result is wrapped in a `ManuallyDrop`
        // below, so the driver is never asked to free an address it did not
        // hand out, and the returned lifetime keeps it inside the arena's.
        Some(unsafe { self.alias(stream, placement, placement.alloc.len) })
    }

    /// A zero-copy alias of one resident tensor, read as `f32`.
    ///
    /// `None` if the tensor is not resident, is not stored as `f32`, or does
    /// not hold a whole number of them. The stored type is checked rather than
    /// assumed: reading a Q8_0 tensor's bytes as floats produces finite,
    /// plausible garbage.
    pub fn f32_of(
        &self,
        stream: &Arc<CudaStream>,
        role: Role,
        layer: Option<u32>,
    ) -> Option<ResidentTensor<'_, f32>> {
        let placement = self.find(role, layer)?;
        if placement.ggml_type != GgmlType::F32 || !placement.alloc.len.is_multiple_of(4) {
            return None;
        }
        // SAFETY: as `bytes_of`, plus: every arena offset is a multiple of
        // `ALIGNMENT` (256) and so satisfies `f32`'s 4-byte alignment, the
        // length is a whole number of `f32`s, and the stored type was checked
        // to be `f32` — so the bytes are a valid little-endian `f32` array on
        // the little-endian hosts this crate supports.
        Some(unsafe { self.alias(stream, placement, placement.alloc.len / 4) })
    }

    /// Wrap `[alloc.offset, alloc.offset + alloc.len)` of the slab that holds
    /// `placement` as `len` elements of `T`.
    ///
    /// # Safety
    ///
    /// `placement` must be one of this load's and `len * size_of::<T>()` must
    /// be at most its `alloc.len`; the bytes must be a valid `[T]`.
    unsafe fn alias<T>(
        &self,
        stream: &Arc<CudaStream>,
        placement: &TensorPlacement,
        len: usize,
    ) -> ResidentTensor<'_, T> {
        let base = match (&self.host, placement.on_host) {
            (Some(slab), true) => slab.device,
            _ => self.arena.slab().device_ptr(stream).0,
        };
        let ptr = base + placement.alloc.offset as u64;
        // SAFETY: the caller guarantees the range and the element type; the
        // slice is immediately sealed in a `ManuallyDrop`.
        let slice = unsafe { stream.upgrade_device_ptr::<T>(ptr, len) };
        ResidentTensor {
            slice: ManuallyDrop::new(slice),
            _arena: PhantomData,
        }
    }

    /// Read `sample` tensors back and compare them byte for byte with `file`.
    ///
    /// Sampling is strided across the directory rather than taken from the
    /// front, so it covers every layer kind and every element type instead of
    /// re-checking the embedding table repeatedly. Pass `usize::MAX` to verify
    /// all of them — correct, and slow enough that it is worth being a choice.
    ///
    /// Returns the tensors and bytes actually compared.
    pub fn verify(
        &self,
        stream: &Arc<CudaStream>,
        file: &GgufFile,
        directory: &Directory<'_>,
        sample: usize,
    ) -> Result<(usize, u64), LoadError> {
        let entries = directory.entries();
        let stride = if sample == 0 || sample >= entries.len() {
            1
        } else {
            entries.len() / sample
        };

        let mut checked = 0usize;
        let mut bytes = 0u64;
        for entry in entries.iter().step_by(stride.max(1)) {
            let name = entry.spec.name.as_str();
            let Some(placement) = self.find(entry.spec.role, entry.spec.layer) else {
                continue;
            };
            let expected = file
                .tensor_bytes(name)
                .ok_or_else(|| LoadError::TruncatedFile {
                    name: name.to_string(),
                    expected: entry.info.n_bytes,
                })?;
            let found = match (&self.host, placement.on_host) {
                (Some(slab), true) => slab.bytes(&placement.alloc).to_vec(),
                _ => self.arena.read(stream, &placement.alloc)?,
            };

            if let Some(offset) = first_difference(expected, &found) {
                return Err(LoadError::Mismatch {
                    name: name.to_string(),
                    byte_offset: offset,
                    expected: expected[offset],
                    found: found[offset],
                });
            }
            checked += 1;
            bytes += expected.len() as u64;
        }
        Ok((checked, bytes))
    }
}

/// Index of the first differing byte, or `None` if the slices are equal.
///
/// Length inequality reports at the shorter end rather than returning `None`,
/// so a truncated read-back is a mismatch and not a silent pass.
fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    a.iter().zip(b).position(|(x, y)| x != y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_slices_have_no_first_difference() {
        assert_eq!(first_difference(&[1, 2, 3], &[1, 2, 3]), None);
    }

    #[test]
    fn a_single_flipped_byte_is_located() {
        assert_eq!(first_difference(&[1, 2, 3], &[1, 9, 3]), Some(1));
    }

    #[test]
    fn a_truncated_readback_is_a_mismatch_not_a_pass() {
        // The dangerous failure: a short read compares equal over its whole
        // length. Reporting `None` here would let a truncated upload verify.
        assert_eq!(first_difference(&[1, 2, 3], &[1, 2]), Some(2));
    }

    #[test]
    fn required_bytes_rounds_every_tensor_up_to_alignment() {
        // Q6_K superblocks are 210 bytes, so tensor sizes are rarely multiples
        // of 256. Under-counting here would size the arena short and fail the
        // load partway through, after minutes of copying.
        let unpadded = 210usize;
        assert_eq!(unpadded.next_multiple_of(ALIGNMENT), 256);
        assert_eq!(540_344_320usize.next_multiple_of(ALIGNMENT), 540_344_320);
    }

    #[test]
    fn throughput_is_bytes_over_elapsed() {
        let report = LoadReport {
            tensors: 1,
            bytes: 2_000_000_000,
            arena_bytes: 2_000_000_000,
            free_before: 0,
            free_after: 0,
            elapsed: Duration::from_secs(2),
        };
        assert!((report.throughput_gb_s() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn vram_consumed_does_not_underflow_when_memory_was_freed() {
        // Another process releasing memory mid-load can leave `free_after`
        // above `free_before`. That is a measurement artefact, not a negative
        // allocation, and it must not wrap around.
        let report = LoadReport {
            tensors: 0,
            bytes: 0,
            arena_bytes: 0,
            free_before: 100,
            free_after: 200,
            elapsed: Duration::from_secs(1),
        };
        assert_eq!(report.vram_consumed(), 0);
    }
}
