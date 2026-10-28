//! TensorRT enums, shapes and errors, as Rust types.
//!
//! Each enum's discriminants are TensorRT 11's integer values; the shim casts
//! them straight through, so a wrong value here is a wrong call there.

use std::ffi::CStr;
use std::sync::Mutex;

use crate::ffi;

/// `nvinfer1::DataType`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    Float = 0,
    Half = 1,
    Int8 = 2,
    Int32 = 3,
    Bool = 4,
    Uint8 = 5,
    Fp8 = 6,
    Bf16 = 7,
    Int64 = 8,
    Int4 = 9,
    Fp4 = 10,
    E8m0 = 11,
}

impl DataType {
    /// The value TensorRT reported, if it is one this crate knows.
    pub fn from_raw(raw: i32) -> Option<Self> {
        use DataType::*;
        Some(match raw {
            0 => Float,
            1 => Half,
            2 => Int8,
            3 => Int32,
            4 => Bool,
            5 => Uint8,
            6 => Fp8,
            7 => Bf16,
            8 => Int64,
            9 => Int4,
            10 => Fp4,
            11 => E8m0,
            _ => return None,
        })
    }

    /// Storage bits per element; the sub-byte types pack two per byte.
    pub fn bits(self) -> usize {
        use DataType::*;
        match self {
            Float | Int32 => 32,
            Half | Bf16 => 16,
            Int8 | Uint8 | Bool | Fp8 | E8m0 => 8,
            Int64 => 64,
            Int4 | Fp4 => 4,
        }
    }
}

/// A tensor shape, `nvinfer1::Dims64`. `-1` marks a dynamic extent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dims(pub(crate) ffi::llmtrt_dims);

impl Dims {
    /// Panics above eight dimensions, which TensorRT cannot represent.
    pub fn new(extents: &[i64]) -> Self {
        assert!(
            extents.len() <= ffi::MAX_DIMS,
            "TensorRT tensors have at most {} dimensions, got {}",
            ffi::MAX_DIMS,
            extents.len()
        );
        let mut d = [0i64; ffi::MAX_DIMS];
        d[..extents.len()].copy_from_slice(extents);
        Self(ffi::llmtrt_dims {
            nb_dims: extents.len() as i32,
            d,
        })
    }

    pub fn as_slice(&self) -> &[i64] {
        let n = self.0.nb_dims.clamp(0, ffi::MAX_DIMS as i32) as usize;
        &self.0.d[..n]
    }

    /// Element count, or `None` while any extent is dynamic.
    pub fn volume(&self) -> Option<i64> {
        self.as_slice()
            .iter()
            .try_fold(1i64, |acc, &e| (e >= 0).then(|| acc * e))
    }
}

impl From<&[i64]> for Dims {
    fn from(extents: &[i64]) -> Self {
        Self::new(extents)
    }
}

impl<const N: usize> From<[i64; N]> for Dims {
    fn from(extents: [i64; N]) -> Self {
        Self::new(&extents)
    }
}

/// `nvinfer1::ElementWiseOperation` (the subset this crate uses).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementWise {
    Sum = 0,
    Prod = 1,
    Max = 2,
    Min = 3,
    Sub = 4,
    Div = 5,
}

/// `nvinfer1::MatrixOperation`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixOp {
    None = 0,
    Transpose = 1,
    Vector = 2,
}

/// `nvinfer1::BuilderFlag` (the subset this crate uses).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuilderFlag {
    Debug = 2,
    Refit = 4,
    DisableTimingCache = 5,
    ErrorOnTimingCacheMiss = 15,
    StripPlan = 18,
    RefitIdentical = 19,
    EditableTimingCache = 26,
}

/// `nvinfer1::ProfilingVerbosity`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfilingVerbosity {
    LayerNamesOnly = 0,
    None = 1,
    Detailed = 2,
}

/// `nvinfer1::ExecutionContextAllocationStrategy`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationStrategy {
    Static = 0,
    OnProfileChange = 1,
    /// The caller binds activation memory with
    /// [`crate::ExecutionContext::set_device_memory`]; TensorRT allocates none.
    UserManaged = 2,
}

/// `nvinfer1::CausalMaskKind`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalMask {
    None = 0,
    UpperLeft = 1,
    LowerRight = 2,
}

/// `nvinfer1::TensorLocation`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorLocation {
    Device = 0,
    Host = 1,
}

/// A failed TensorRT call, with the last error TensorRT logged before it.
///
/// TensorRT reports *why* through its logger and returns only null or false,
/// so the reason is captured from the log stream rather than invented here.
#[derive(Debug, thiserror::Error)]
pub enum TrtError {
    #[error("TensorRT {call} failed: {reason}")]
    Call { call: &'static str, reason: String },
    #[error("ONNX parse failed: {}", .0.join("; "))]
    Parse(Vec<String>),
    #[error("{0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, TrtError>;

/// The most recent error-severity message TensorRT logged, process-wide.
///
/// Builder threads log too, so this is a mutex rather than a thread-local; it
/// is read only on a failure path.
static LAST_ERROR: Mutex<Option<String>> = Mutex::new(None);

pub(crate) fn fail(call: &'static str) -> TrtError {
    let reason = LAST_ERROR
        .lock()
        .ok()
        .and_then(|mut e| e.take())
        .unwrap_or_else(|| "no error was logged".to_owned());
    TrtError::Call { call, reason }
}

pub(crate) fn non_null(
    ptr: *mut std::ffi::c_void,
    call: &'static str,
) -> Result<*mut std::ffi::c_void> {
    if ptr.is_null() {
        Err(fail(call))
    } else {
        Ok(ptr)
    }
}

pub(crate) fn check(ok: bool, call: &'static str) -> Result<()> {
    if ok { Ok(()) } else { Err(fail(call)) }
}

/// Forget any error logged before the call about to be made.
pub(crate) fn clear_last_error() {
    if let Ok(mut e) = LAST_ERROR.lock() {
        *e = None;
    }
}

/// `nvinfer1::ILogger::Severity` routed into `tracing`.
///
/// TensorRT's `kINFO` is chatty at build time, so it lands at `debug`; its
/// errors are also kept for [`TrtError`].
unsafe extern "C" fn log_callback(severity: i32, msg: *const std::ffi::c_char) {
    let _ = std::panic::catch_unwind(|| {
        if msg.is_null() {
            return;
        }
        // SAFETY: TensorRT passes a NUL-terminated string valid for the call.
        let text = unsafe { CStr::from_ptr(msg) }.to_string_lossy();
        match severity {
            0 | 1 => {
                tracing::error!(target: "tensorrt", "{text}");
                if let Ok(mut e) = LAST_ERROR.lock() {
                    *e = Some(text.into_owned());
                }
            }
            2 => tracing::warn!(target: "tensorrt", "{text}"),
            3 => tracing::debug!(target: "tensorrt", "{text}"),
            _ => tracing::trace!(target: "tensorrt", "{text}"),
        }
    });
}

static INIT: std::sync::Once = std::sync::Once::new();

/// Route TensorRT's logger into `tracing`. Idempotent; every constructor in
/// this crate calls it, so callers never need to.
pub(crate) fn ensure_logger() {
    // SAFETY: the callback is a plain function valid for the process.
    INIT.call_once(|| unsafe { ffi::llmtrt_set_log_callback(Some(log_callback)) });
}

/// `(major, minor, patch)` of the loaded libnvinfer.
pub fn library_version() -> (i32, i32, i32) {
    // SAFETY: no arguments, no state.
    split_version(unsafe { ffi::llmtrt_lib_version() })
}

/// `(major, minor, patch)` of the headers the shim was compiled against.
pub fn header_version() -> (i32, i32, i32) {
    // SAFETY: no arguments, no state.
    split_version(unsafe { ffi::llmtrt_header_version() })
}

/// `getNvOnnxParserVersion()` of the loaded libnvonnxparser, raw.
pub fn onnx_parser_version() -> i32 {
    // SAFETY: no arguments, no state.
    unsafe { ffi::llmtrt_onnx_parser_version() }
}

/// TensorRT 10+ encodes `major * 10000 + minor * 100 + patch`.
fn split_version(v: i32) -> (i32, i32, i32) {
    (v / 10000, (v / 100) % 100, v % 100)
}
