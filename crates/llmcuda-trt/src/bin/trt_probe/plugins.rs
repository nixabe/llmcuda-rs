//! The probe's plugins: one creator per op, all built on [`ProbePlugin`].
//!
//! - `ProbeAdd`: `out = a + b` through the engine's NVRTC `tensor_add` (P3,
//!   U5, U8). Carries a `probe_tag` field so U5 can see an ONNX attribute
//!   arrive as a plugin field.
//! - `ProbeCtx`: copies its input and records which CUDA context and stream
//!   TensorRT ran it in (P1).
//! - `ProbeAccumulate`: increments a float through a device pointer table
//!   passed as an INT64 tensor — the state-access shape the trunk's plugins
//!   use — so graph replays are countable (P4).
//! - `ProbeWorkspace`: routes its data through plugin workspace, so P5's
//!   allocation count covers workspace carved from the user-managed block.

use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

use cudarc::driver::result;
use cudarc::driver::sys::{CUcontext, CUstream};
use llmcuda_trt::DataType;
use llmcuda_trt::bindings;
use llmcuda_trt::plugin::{
    DimExpr, DynamicTensorDesc, Enqueue, ExprBuilder, Field, FieldRef, FieldType, Phase, Plugin,
    PluginCreator, PluginError, PluginResult, TensorDesc, register_creator,
};

use crate::env::{Kernels, contexts_of, elementwise_grid};

pub const NAMESPACE: &str = "llmcuda";

/// What plugins read at enqueue: the kernels, installed around every
/// enqueue by [`bindings::install`].
pub struct ProbeBindings {
    pub kernels: Kernels,
}

/// What `ProbeCtx` saw on its last launch.
#[derive(Debug, Clone, Copy)]
pub struct CtxObservation {
    pub current: CUcontext,
    pub of_stream: CUcontext,
    pub of_input: CUcontext,
    pub input: u64,
    pub stream: CUstream,
}

// SAFETY: raw handles compared for identity only, never dereferenced.
unsafe impl Send for CtxObservation {}

pub static CTX_SEEN: Mutex<Option<CtxObservation>> = Mutex::new(None);
/// The last `probe_tag` a creator received, or -1.
pub static LAST_TAG: AtomicI64 = AtomicI64::new(-1);
/// Plugin enqueues, and how many found no bindings installed.
pub static ENQUEUES: AtomicU64 = AtomicU64::new(0);
pub static UNBOUND_ENQUEUES: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Add,
    Ctx,
    Accumulate,
    Workspace,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Self::Add => "ProbeAdd",
            Self::Ctx => "ProbeCtx",
            Self::Accumulate => "ProbeAccumulate",
            Self::Workspace => "ProbeWorkspace",
        }
    }

    fn input_types(self) -> &'static [DataType] {
        match self {
            Self::Add => &[DataType::Float, DataType::Float],
            Self::Ctx | Self::Workspace => &[DataType::Float],
            Self::Accumulate => &[DataType::Int64],
        }
    }
}

#[derive(Clone)]
struct ProbePlugin {
    kind: Kind,
    tag: i64,
}

fn volume(desc: &TensorDesc) -> PluginResult<usize> {
    desc.dims()
        .volume()
        .map(|v| v as usize)
        .ok_or_else(|| PluginError("dynamic extent at enqueue".into()))
}

/// Launch `f` with `params` on `stream`.
///
/// # Safety
/// `params` must match `f`'s signature and every pointer must be valid
/// device memory for the launch.
unsafe fn launch(
    f: cudarc::driver::sys::CUfunction,
    grid: u32,
    block: u32,
    stream: CUstream,
    params: &mut [*mut c_void],
) -> PluginResult<()> {
    // SAFETY: the caller guarantees `params`.
    unsafe { result::launch_kernel(f, (grid, 1, 1), (block, 1, 1), 0, stream, params) }
        .map_err(|e| PluginError(format!("cuLaunchKernel: {e:?}")))
}

impl Plugin for ProbePlugin {
    fn nb_outputs(&self) -> usize {
        1
    }

    fn output_types(&self, inputs: &[DataType]) -> PluginResult<Vec<DataType>> {
        if inputs != self.kind.input_types() {
            return Err(PluginError(format!(
                "{} takes {:?}, got {inputs:?}",
                self.kind.name(),
                self.kind.input_types()
            )));
        }
        Ok(vec![DataType::Float])
    }

    fn output_shapes<'a>(
        &self,
        exprs: &ExprBuilder<'a>,
        inputs: &[Vec<DimExpr<'a>>],
    ) -> PluginResult<Vec<Vec<DimExpr<'a>>>> {
        Ok(vec![match self.kind {
            Kind::Accumulate => vec![exprs.constant(1)],
            _ => inputs[0].clone(),
        }])
    }

    fn supports_format(&self, pos: usize, in_out: &[DynamicTensorDesc], nb_inputs: usize) -> bool {
        let desc = in_out[pos].desc();
        let want = if pos < nb_inputs {
            self.kind.input_types()[pos]
        } else {
            DataType::Float
        };
        desc.format() == 0 && desc.data_type() == Some(want)
    }

    fn workspace_size(
        &self,
        inputs: &[DynamicTensorDesc],
        _outputs: &[DynamicTensorDesc],
    ) -> usize {
        match self.kind {
            Kind::Workspace => inputs[0].max().volume().unwrap_or(0) as usize * size_of::<f32>(),
            _ => 0,
        }
    }

    fn enqueue(&mut self, l: &Enqueue<'_>) -> PluginResult<()> {
        ENQUEUES.fetch_add(1, Ordering::Relaxed);
        let kind = self.kind;
        bindings::with(|b: &ProbeBindings| -> PluginResult<()> {
            let k = &b.kernels;
            match kind {
                Kind::Add => {
                    let n = volume(&l.input_descs[0])?;
                    let (mut a, mut bb, mut out, mut n64) = (
                        l.inputs[0] as u64,
                        l.inputs[1] as u64,
                        l.outputs[0] as u64,
                        n as i64,
                    );
                    let (grid, block) = elementwise_grid(n);
                    let mut params = [
                        &mut a as *mut u64 as *mut c_void,
                        &mut bb as *mut u64 as *mut c_void,
                        &mut out as *mut u64 as *mut c_void,
                        &mut n64 as *mut i64 as *mut c_void,
                    ];
                    // SAFETY: tensor_add(const float*, const float*, float*, long long)
                    // over TensorRT's buffers of `n` floats.
                    unsafe { launch(k.add, grid, block, l.stream, &mut params) }
                }
                Kind::Ctx => {
                    let n = volume(&l.input_descs[0])?;
                    let input = l.inputs[0] as u64;
                    let (current, of_stream, of_input) = contexts_of(l.stream, input);
                    if let Ok(mut seen) = CTX_SEEN.lock() {
                        *seen = Some(CtxObservation {
                            current,
                            of_stream,
                            of_input,
                            input,
                            stream: l.stream,
                        });
                    }
                    // SAFETY: both buffers hold `n` floats on this stream.
                    unsafe {
                        result::memcpy_dtod_async(
                            l.outputs[0] as u64,
                            input,
                            n * size_of::<f32>(),
                            l.stream,
                        )
                    }
                    .map_err(|e| PluginError(format!("cuMemcpyDtoDAsync: {e:?}")))
                }
                Kind::Accumulate => {
                    let (mut table, mut out) = (l.inputs[0] as u64, l.outputs[0] as u64);
                    let mut params = [
                        &mut table as *mut u64 as *mut c_void,
                        &mut out as *mut u64 as *mut c_void,
                    ];
                    // SAFETY: probe_accumulate(const long long*, float*); the
                    // table holds one valid float pointer.
                    unsafe { launch(k.accumulate, 1, 32, l.stream, &mut params) }
                }
                Kind::Workspace => {
                    let n = volume(&l.input_descs[0])?;
                    let (mut input, mut ws, mut out, mut n64) = (
                        l.inputs[0] as u64,
                        l.workspace as u64,
                        l.outputs[0] as u64,
                        n as i64,
                    );
                    if ws == 0 {
                        return Err(PluginError("no workspace was provided".into()));
                    }
                    let (grid, block) = elementwise_grid(n);
                    let mut params = [
                        &mut input as *mut u64 as *mut c_void,
                        &mut ws as *mut u64 as *mut c_void,
                        &mut out as *mut u64 as *mut c_void,
                        &mut n64 as *mut i64 as *mut c_void,
                    ];
                    // SAFETY: probe_through_workspace over `n` floats each;
                    // workspace_size reserved at least that.
                    unsafe { launch(k.through_workspace, grid, block, l.stream, &mut params) }
                }
            }
        })
        .unwrap_or_else(|| {
            UNBOUND_ENQUEUES.fetch_add(1, Ordering::Relaxed);
            Err(PluginError(
                "no ProbeBindings installed on the enqueueing thread".into(),
            ))
        })
    }

    fn clone_plugin(&self) -> Box<dyn Plugin> {
        Box::new(self.clone())
    }

    fn fields(&self) -> Vec<Field> {
        vec![Field::i64s("probe_tag", &[self.tag])]
    }
}

struct ProbeCreator {
    kind: Kind,
}

impl PluginCreator for ProbeCreator {
    fn name(&self) -> &str {
        self.kind.name()
    }

    fn namespace(&self) -> &str {
        NAMESPACE
    }

    fn field_schema(&self) -> Vec<Field> {
        vec![Field::schema("probe_tag", FieldType::Int64)]
    }

    fn create(&self, fields: &[FieldRef<'_>], phase: Phase) -> PluginResult<Box<dyn Plugin>> {
        let tag = fields
            .iter()
            .find(|f| f.name == "probe_tag")
            .and_then(|f| f.as_i64s())
            .and_then(|v| v.first().copied())
            .unwrap_or(0);
        if phase == Phase::Build {
            LAST_TAG.store(tag, Ordering::SeqCst);
        }
        Ok(Box::new(ProbePlugin {
            kind: self.kind,
            tag,
        }))
    }
}

pub fn register_all() -> llmcuda_trt::Result<()> {
    for kind in [Kind::Add, Kind::Ctx, Kind::Accumulate, Kind::Workspace] {
        register_creator(ProbeCreator { kind })?;
    }
    Ok(())
}
