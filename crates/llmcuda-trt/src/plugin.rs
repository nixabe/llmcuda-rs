//! IPluginV3 plugins implemented in Rust.
//!
//! One C++ class, `RustPlugin` in `cpp/shim.cpp`, implements every IPluginV3
//! capability by forwarding to the static vtable here, which dispatches to a
//! boxed [`Plugin`]. A [`PluginCreator`] registered with
//! [`register_creator`] makes them, both when a network is built (from the
//! network API or an ONNX custom op) and when an engine is deserialized.
//!
//! Rules this module enforces rather than documents:
//!
//! - **Every entry point catches panics.** A panic is logged and becomes a
//!   non-zero status; it never unwinds into TensorRT.
//! - **Serialized fields carry geometry only, plus the ABI version.** The
//!   trampoline appends `llmtrt_abi` on serialization and, at runtime
//!   deserialization, refuses a plugin whose ABI differs from this build's.
//! - **Enqueue does not allocate.** Descriptor and pointer arrays are
//!   TensorRT's own, borrowed as slices.

use std::ffi::{CStr, CString, c_void};
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};

use cudarc::driver::sys::CUstream;

use crate::ffi;
use crate::types::{DataType, Dims, Result, TrtError};

const ABI_FIELD: &str = "llmtrt_abi";

/// `nvinfer1::PluginFieldType`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldType {
    Float16 = 0,
    Float32 = 1,
    Float64 = 2,
    Int8 = 3,
    Int16 = 4,
    Int32 = 5,
    Char = 6,
    Dims = 7,
    Unknown = 8,
    Bf16 = 9,
    Int64 = 10,
}

impl FieldType {
    fn from_raw(raw: i32) -> Self {
        match raw {
            0 => Self::Float16,
            1 => Self::Float32,
            2 => Self::Float64,
            3 => Self::Int8,
            4 => Self::Int16,
            5 => Self::Int32,
            6 => Self::Char,
            7 => Self::Dims,
            9 => Self::Bf16,
            10 => Self::Int64,
            _ => Self::Unknown,
        }
    }
}

/// One plugin field: a name and a typed array of `length` elements.
#[derive(Debug, Clone)]
pub struct Field {
    name: CString,
    kind: FieldType,
    data: Vec<u8>,
    length: i32,
}

impl Field {
    pub fn i32s(name: &str, values: &[i32]) -> Self {
        Self::new(
            name,
            FieldType::Int32,
            values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
            values.len(),
        )
    }

    pub fn i64s(name: &str, values: &[i64]) -> Self {
        Self::new(
            name,
            FieldType::Int64,
            values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
            values.len(),
        )
    }

    pub fn f32s(name: &str, values: &[f32]) -> Self {
        Self::new(
            name,
            FieldType::Float32,
            values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
            values.len(),
        )
    }

    /// A field's name and type with no data: what a creator advertises.
    pub fn schema(name: &str, kind: FieldType) -> Self {
        Self::new(name, kind, Vec::new(), 1)
    }

    fn new(name: &str, kind: FieldType, data: Vec<u8>, length: usize) -> Self {
        Self {
            name: CString::new(name).expect("plugin field names contain no NUL"),
            kind,
            data,
            length: length as i32,
        }
    }

    pub(crate) fn raw(&self) -> ffi::llmtrt_field {
        ffi::llmtrt_field {
            name: self.name.as_ptr(),
            data: if self.data.is_empty() {
                std::ptr::null()
            } else {
                self.data.as_ptr() as *const c_void
            },
            field_type: self.kind as i32,
            length: self.length,
        }
    }
}

/// A field as TensorRT hands it to a creator: borrowed, possibly from an
/// ONNX attribute, possibly from a serialized plan.
#[derive(Debug, Clone, Copy)]
pub struct FieldRef<'a> {
    pub name: &'a str,
    pub kind: FieldType,
    data: *const c_void,
    pub length: usize,
    _lifetime: PhantomData<&'a [u8]>,
}

impl<'a> FieldRef<'a> {
    fn element_bytes(&self) -> usize {
        match self.kind {
            FieldType::Int8 | FieldType::Char => 1,
            FieldType::Float16 | FieldType::Int16 | FieldType::Bf16 => 2,
            FieldType::Float32 | FieldType::Int32 => 4,
            FieldType::Float64 | FieldType::Int64 => 8,
            FieldType::Dims | FieldType::Unknown => 0,
        }
    }

    fn bytes(&self) -> &'a [u8] {
        let n = self.length * self.element_bytes();
        if self.data.is_null() || n == 0 {
            return &[];
        }
        // SAFETY: TensorRT guarantees `length` elements of `kind` at `data`
        // for the duration of the creator call, which bounds `'a`.
        unsafe { std::slice::from_raw_parts(self.data as *const u8, n) }
    }

    /// The values as `i64`, widening `Int32`. ONNX integer attributes arrive
    /// as one or the other depending on the parser, so creators read both.
    pub fn as_i64s(&self) -> Option<Vec<i64>> {
        let b = self.bytes();
        match self.kind {
            FieldType::Int32 => Some(
                b.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| i64::from(i32::from_ne_bytes(*c)))
                    .collect(),
            ),
            FieldType::Int64 => Some(
                b.as_chunks::<8>()
                    .0
                    .iter()
                    .map(|c| i64::from_ne_bytes(*c))
                    .collect(),
            ),
            _ => None,
        }
    }

    pub fn as_f32s(&self) -> Option<Vec<f32>> {
        (self.kind == FieldType::Float32).then(|| {
            self.bytes()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_ne_bytes(*c))
                .collect()
        })
    }
}

/// `nvinfer1::TensorRTPhase`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Network construction (network API or ONNX import).
    Build,
    /// Engine deserialization.
    Runtime,
}

/// `PluginTensorDesc`: a tensor as the plugin will see it at enqueue.
#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct TensorDesc(ffi::llmtrt_tensor_desc);

impl TensorDesc {
    pub fn dims(&self) -> Dims {
        Dims(self.0.dims)
    }
    pub fn data_type(&self) -> Option<DataType> {
        DataType::from_raw(self.0.data_type)
    }
    /// `TensorFormat`; 0 is `kLINEAR`, the only one plugins here accept.
    pub fn format(&self) -> i32 {
        self.0.format
    }
}

/// `DynamicPluginTensorDesc`: a tensor's type, format and shape range at
/// build time.
#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct DynamicTensorDesc(ffi::llmtrt_dyn_desc);

impl DynamicTensorDesc {
    pub fn desc(&self) -> TensorDesc {
        TensorDesc(self.0.desc)
    }
    pub fn min(&self) -> Dims {
        Dims(self.0.min)
    }
    pub fn max(&self) -> Dims {
        Dims(self.0.max)
    }
}

/// One dimension expression, valid during `output_shapes`.
#[derive(Debug, Clone, Copy)]
pub struct DimExpr<'a> {
    raw: *const c_void,
    _lifetime: PhantomData<&'a ()>,
}

impl DimExpr<'_> {
    pub fn constant_value(&self) -> Option<i64> {
        // SAFETY: a live expression for the duration of output_shapes.
        unsafe {
            ffi::llmtrt_expr_is_constant(self.raw)
                .then(|| ffi::llmtrt_expr_constant_value(self.raw))
        }
    }
}

/// `nvinfer1::DimensionOperation`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimOp {
    Sum = 0,
    Prod = 1,
    Max = 2,
    Min = 3,
    Sub = 4,
    Equal = 5,
    Less = 6,
    FloorDiv = 7,
    CeilDiv = 8,
}

/// `nvinfer1::IExprBuilder`, valid during `output_shapes`.
pub struct ExprBuilder<'a> {
    raw: *mut c_void,
    _lifetime: PhantomData<&'a ()>,
}

impl<'a> ExprBuilder<'a> {
    pub fn constant(&self, value: i64) -> DimExpr<'a> {
        DimExpr {
            // SAFETY: a live builder for the duration of output_shapes.
            raw: unsafe { ffi::llmtrt_expr_constant(self.raw, value) },
            _lifetime: PhantomData,
        }
    }

    pub fn op(&self, op: DimOp, a: DimExpr<'a>, b: DimExpr<'a>) -> DimExpr<'a> {
        DimExpr {
            // SAFETY: as above; the shim rejects null operands.
            raw: unsafe { ffi::llmtrt_expr_operation(self.raw, op as i32, a.raw, b.raw) },
            _lifetime: PhantomData,
        }
    }
}

/// Everything one plugin launch sees.
pub struct Enqueue<'a> {
    pub input_descs: &'a [TensorDesc],
    pub output_descs: &'a [TensorDesc],
    pub inputs: &'a [*const c_void],
    pub outputs: &'a [*mut c_void],
    pub workspace: *mut c_void,
    /// The stream `enqueueV3` was called with when the engine was built with
    /// no auxiliary streams.
    pub stream: CUstream,
}

/// A plugin's error, logged by the trampoline before TensorRT sees a status.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct PluginError(pub String);

pub type PluginResult<T> = std::result::Result<T, PluginError>;

/// One TensorRT plugin, implemented in Rust.
///
/// The methods mirror `IPluginV3OneBuild` and `IPluginV3OneRuntime`. Build
/// methods may allocate; [`Plugin::enqueue`] must not.
pub trait Plugin: Send + 'static {
    fn nb_outputs(&self) -> usize;
    fn output_types(&self, inputs: &[DataType]) -> PluginResult<Vec<DataType>>;
    fn output_shapes<'a>(
        &self,
        exprs: &ExprBuilder<'a>,
        inputs: &[Vec<DimExpr<'a>>],
    ) -> PluginResult<Vec<Vec<DimExpr<'a>>>>;
    /// Whether `in_out[pos]` may take the type and format it now has, given
    /// the choices already made for `in_out[..pos]`.
    fn supports_format(&self, pos: usize, in_out: &[DynamicTensorDesc], nb_inputs: usize) -> bool;
    fn configure(
        &mut self,
        _inputs: &[DynamicTensorDesc],
        _outputs: &[DynamicTensorDesc],
    ) -> PluginResult<()> {
        Ok(())
    }
    fn workspace_size(
        &self,
        _inputs: &[DynamicTensorDesc],
        _outputs: &[DynamicTensorDesc],
    ) -> usize {
        0
    }
    fn on_shape_change(
        &mut self,
        _inputs: &[TensorDesc],
        _outputs: &[TensorDesc],
    ) -> PluginResult<()> {
        Ok(())
    }
    fn enqueue(&mut self, launch: &Enqueue<'_>) -> PluginResult<()>;
    fn clone_plugin(&self) -> Box<dyn Plugin>;
    /// What a plan stores to recreate this plugin. Geometry only — never a
    /// pointer: a plan outlives the process that built it.
    fn fields(&self) -> Vec<Field>;
}

/// Makes [`Plugin`]s for one op name.
pub trait PluginCreator: Send + Sync + 'static {
    fn name(&self) -> &str;
    fn version(&self) -> &str {
        "1"
    }
    fn namespace(&self) -> &str;
    /// Advertised field schema. The ONNX parser converts a custom node's
    /// attributes into fields by these names.
    fn field_schema(&self) -> Vec<Field>;
    fn create(&self, fields: &[FieldRef<'_>], phase: Phase) -> PluginResult<Box<dyn Plugin>>;
}

/// The Rust object behind one `RustPlugin`: the plugin, the name it logs
/// under, and the storage its serialized fields live in between calls.
struct Slot {
    plugin: Box<dyn Plugin>,
    name: String,
    fields: Vec<Field>,
    raw_fields: Vec<ffi::llmtrt_field>,
}

struct CreatorSlot {
    creator: Box<dyn PluginCreator>,
    name: String,
}

/// Run `f` on the slot, turning an error or a panic into `failure`.
///
/// # Safety
/// `this` must be a `Slot` made by this module and not yet destroyed.
unsafe fn with_slot<R>(
    this: *mut c_void,
    what: &str,
    failure: R,
    f: impl FnOnce(&mut Slot) -> PluginResult<R>,
) -> R {
    // SAFETY: the caller guarantees `this` is a live Slot; TensorRT does not
    // call one plugin object concurrently.
    let result = catch_unwind(AssertUnwindSafe(|| f(unsafe { &mut *(this as *mut Slot) })));
    // The name is read only on failure: this wraps enqueue, which must not
    // touch the host allocator.
    // SAFETY: as above; the closure's borrow has ended.
    let name = || unsafe { &(*(this as *const Slot)).name };
    match result {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            tracing::error!(plugin = %name(), "{what}: {e}");
            failure
        }
        Err(_) => {
            tracing::error!(plugin = %name(), "{what} panicked");
            failure
        }
    }
}

/// Borrow a TensorRT array as a slice; null with zero length is empty.
///
/// # Safety
/// `ptr` must point to `n` valid `T` for the returned lifetime.
unsafe fn slice<'a, T>(ptr: *const T, n: i32) -> &'a [T] {
    if ptr.is_null() || n <= 0 {
        &[]
    } else {
        // SAFETY: the caller guarantees the extent.
        unsafe { std::slice::from_raw_parts(ptr, n as usize) }
    }
}

unsafe extern "C" fn nb_outputs(this: *mut c_void) -> i32 {
    // SAFETY: TensorRT passes the object it was given.
    unsafe {
        with_slot(this, "getNbOutputs", -1, |s| {
            Ok(s.plugin.nb_outputs() as i32)
        })
    }
}

unsafe extern "C" fn output_types(
    this: *mut c_void,
    out: *mut i32,
    nb_out: i32,
    input: *const i32,
    nb_in: i32,
) -> i32 {
    // SAFETY: TensorRT passes arrays of the stated lengths.
    unsafe {
        with_slot(this, "getOutputDataTypes", 1, |s| {
            let inputs: Vec<DataType> = slice(input, nb_in)
                .iter()
                .map(|&t| {
                    DataType::from_raw(t)
                        .ok_or_else(|| PluginError(format!("unknown data type {t}")))
                })
                .collect::<PluginResult<_>>()?;
            let types = s.plugin.output_types(&inputs)?;
            if types.len() != nb_out as usize {
                return Err(PluginError(format!(
                    "{} output types for {nb_out} outputs",
                    types.len()
                )));
            }
            for (i, t) in types.into_iter().enumerate() {
                *out.add(i) = t as i32;
            }
            Ok(0)
        })
    }
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn output_shapes(
    this: *mut c_void,
    exprs: *mut c_void,
    in_dims: *const *const c_void,
    in_nb_dims: *const i32,
    nb_in: i32,
    out_dims: *mut *const c_void,
    out_nb_dims: *mut i32,
    nb_out: i32,
) -> i32 {
    // SAFETY: the shim passes nb_in * MAX_DIMS input expressions and room for
    // nb_out * MAX_DIMS outputs.
    unsafe {
        with_slot(this, "getOutputShapes", 1, |s| {
            let builder = ExprBuilder {
                raw: exprs,
                _lifetime: PhantomData,
            };
            let ranks = slice(in_nb_dims, nb_in);
            let inputs: Vec<Vec<DimExpr<'_>>> = ranks
                .iter()
                .enumerate()
                .map(|(i, &rank)| {
                    (0..rank as usize)
                        .map(|j| DimExpr {
                            raw: *in_dims.add(i * ffi::MAX_DIMS + j),
                            _lifetime: PhantomData,
                        })
                        .collect()
                })
                .collect();
            let shapes = s.plugin.output_shapes(&builder, &inputs)?;
            if shapes.len() != nb_out as usize {
                return Err(PluginError(format!(
                    "{} shapes for {nb_out} outputs",
                    shapes.len()
                )));
            }
            for (i, shape) in shapes.iter().enumerate() {
                if shape.len() > ffi::MAX_DIMS {
                    return Err(PluginError(format!("output {i} has rank {}", shape.len())));
                }
                *out_nb_dims.add(i) = shape.len() as i32;
                for (j, e) in shape.iter().enumerate() {
                    *out_dims.add(i * ffi::MAX_DIMS + j) = e.raw;
                }
            }
            Ok(0)
        })
    }
}

unsafe extern "C" fn supports_format(
    this: *mut c_void,
    pos: i32,
    in_out: *const ffi::llmtrt_dyn_desc,
    nb_in: i32,
    nb_out: i32,
) -> i32 {
    // SAFETY: TensorRT passes nb_in + nb_out descriptors.
    unsafe {
        with_slot(this, "supportsFormatCombination", 0, |s| {
            let descs = slice(in_out as *const DynamicTensorDesc, nb_in + nb_out);
            Ok(i32::from(s.plugin.supports_format(
                pos as usize,
                descs,
                nb_in as usize,
            )))
        })
    }
}

unsafe extern "C" fn configure(
    this: *mut c_void,
    input: *const ffi::llmtrt_dyn_desc,
    nb_in: i32,
    out: *const ffi::llmtrt_dyn_desc,
    nb_out: i32,
) -> i32 {
    // SAFETY: TensorRT passes arrays of the stated lengths.
    unsafe {
        with_slot(this, "configurePlugin", 1, |s| {
            s.plugin.configure(
                slice(input as *const DynamicTensorDesc, nb_in),
                slice(out as *const DynamicTensorDesc, nb_out),
            )?;
            Ok(0)
        })
    }
}

unsafe extern "C" fn workspace_size(
    this: *mut c_void,
    input: *const ffi::llmtrt_dyn_desc,
    nb_in: i32,
    out: *const ffi::llmtrt_dyn_desc,
    nb_out: i32,
) -> usize {
    // SAFETY: TensorRT passes arrays of the stated lengths.
    unsafe {
        with_slot(this, "getWorkspaceSize", 0, |s| {
            Ok(s.plugin.workspace_size(
                slice(input as *const DynamicTensorDesc, nb_in),
                slice(out as *const DynamicTensorDesc, nb_out),
            ))
        })
    }
}

unsafe extern "C" fn on_shape_change(
    this: *mut c_void,
    input: *const ffi::llmtrt_tensor_desc,
    nb_in: i32,
    out: *const ffi::llmtrt_tensor_desc,
    nb_out: i32,
) -> i32 {
    // SAFETY: TensorRT passes arrays of the stated lengths.
    unsafe {
        with_slot(this, "onShapeChange", 1, |s| {
            s.plugin.on_shape_change(
                slice(input as *const TensorDesc, nb_in),
                slice(out as *const TensorDesc, nb_out),
            )?;
            Ok(0)
        })
    }
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn enqueue(
    this: *mut c_void,
    in_desc: *const ffi::llmtrt_tensor_desc,
    nb_in: i32,
    out_desc: *const ffi::llmtrt_tensor_desc,
    nb_out: i32,
    inputs: *const *const c_void,
    outputs: *const *mut c_void,
    workspace: *mut c_void,
    stream: *mut c_void,
) -> i32 {
    // SAFETY: TensorRT passes arrays of the counts it gave onShapeChange.
    unsafe {
        with_slot(this, "enqueue", 1, |s| {
            s.plugin.enqueue(&Enqueue {
                input_descs: slice(in_desc as *const TensorDesc, nb_in),
                output_descs: slice(out_desc as *const TensorDesc, nb_out),
                inputs: slice(inputs, nb_in),
                outputs: slice(outputs, nb_out),
                workspace,
                stream: stream as CUstream,
            })?;
            Ok(0)
        })
    }
}

unsafe extern "C" fn clone(this: *mut c_void) -> *mut c_void {
    // SAFETY: TensorRT passes the object it was given.
    unsafe {
        with_slot(this, "clone", std::ptr::null_mut(), |s| {
            Ok(Box::into_raw(Box::new(Slot {
                plugin: s.plugin.clone_plugin(),
                name: s.name.clone(),
                fields: Vec::new(),
                raw_fields: Vec::new(),
            })) as *mut c_void)
        })
    }
}

unsafe extern "C" fn fields(this: *mut c_void, out: *mut *const ffi::llmtrt_field) -> i32 {
    // SAFETY: TensorRT passes the object it was given and a valid out-pointer.
    unsafe {
        with_slot(this, "getFieldsToSerialize", -1, |s| {
            let mut fields = s.plugin.fields();
            if fields
                .iter()
                .any(|f| f.name.to_bytes() == ABI_FIELD.as_bytes())
            {
                return Err(PluginError(format!("{ABI_FIELD} is reserved")));
            }
            fields.push(Field::i32s(ABI_FIELD, &[ffi::ABI_VERSION]));
            s.raw_fields = fields.iter().map(Field::raw).collect();
            s.fields = fields;
            *out = s.raw_fields.as_ptr();
            Ok(s.raw_fields.len() as i32)
        })
    }
}

unsafe extern "C" fn destroy(this: *mut c_void) {
    if this.is_null() {
        return;
    }
    // SAFETY: made by `Box::into_raw` in this module; TensorRT destroys once.
    let slot = unsafe { Box::from_raw(this as *mut Slot) };
    if catch_unwind(AssertUnwindSafe(|| drop(slot))).is_err() {
        tracing::error!("plugin destructor panicked");
    }
}

static VTABLE: ffi::llmtrt_plugin_vtable = ffi::llmtrt_plugin_vtable {
    nb_outputs,
    output_types,
    output_shapes,
    supports_format,
    configure,
    workspace_size,
    on_shape_change,
    enqueue,
    clone,
    fields,
    destroy,
};

unsafe extern "C" fn create(
    user: *mut c_void,
    raw_fields: *const ffi::llmtrt_field,
    nb_fields: i32,
    phase: i32,
) -> *mut c_void {
    let result = catch_unwind(AssertUnwindSafe(|| -> PluginResult<*mut c_void> {
        // SAFETY: `user` is the leaked CreatorSlot from `register_creator`.
        let slot = unsafe { &*(user as *const CreatorSlot) };
        let phase = if phase == 1 {
            Phase::Runtime
        } else {
            Phase::Build
        };
        // SAFETY: TensorRT passes `nb_fields` fields valid for this call.
        let raw = unsafe { slice(raw_fields, nb_fields) };
        let mut abi = None;
        let mut fields = Vec::with_capacity(raw.len());
        for f in raw {
            if f.name.is_null() {
                continue;
            }
            // SAFETY: field names are NUL-terminated.
            let name = unsafe { CStr::from_ptr(f.name) }
                .to_str()
                .map_err(|_| PluginError("non-UTF-8 field name".into()))?;
            let field = FieldRef {
                name,
                kind: FieldType::from_raw(f.field_type),
                data: f.data,
                length: f.length.max(0) as usize,
                _lifetime: PhantomData,
            };
            if name == ABI_FIELD {
                abi = field.as_i64s().and_then(|v| v.first().copied());
            } else {
                fields.push(field);
            }
        }
        if phase == Phase::Runtime && abi != Some(i64::from(ffi::ABI_VERSION)) {
            return Err(PluginError(format!(
                "plan was serialized with plugin ABI {abi:?}, this build is {}; rebuild the engine",
                ffi::ABI_VERSION
            )));
        }
        let plugin = slot.creator.create(&fields, phase)?;
        Ok(Box::into_raw(Box::new(Slot {
            plugin,
            name: slot.name.clone(),
            fields: Vec::new(),
            raw_fields: Vec::new(),
        })) as *mut c_void)
    }));
    match result {
        Ok(Ok(p)) => p,
        Ok(Err(e)) => {
            tracing::error!("plugin creation failed: {e}");
            std::ptr::null_mut()
        }
        Err(_) => {
            tracing::error!("plugin creation panicked");
            std::ptr::null_mut()
        }
    }
}

/// Register `creator` in TensorRT's global plugin registry, for the rest of
/// the process.
pub fn register_creator(creator: impl PluginCreator) -> Result<()> {
    crate::types::ensure_logger();
    let to_c =
        |s: &str| CString::new(s).map_err(|_| TrtError::Invalid(format!("{s:?} contains NUL")));
    let (name, version, namespace) = (
        to_c(creator.name())?,
        to_c(creator.version())?,
        to_c(creator.namespace())?,
    );
    let schema = creator.field_schema();
    let raw_schema: Vec<ffi::llmtrt_field> = schema.iter().map(Field::raw).collect();
    let slot = Box::new(CreatorSlot {
        name: creator.name().to_owned(),
        creator: Box::new(creator),
    });
    // The registry keeps the creator for the process, so its Rust half lives
    // as long: leaked on purpose.
    let user = Box::into_raw(slot) as *mut c_void;
    crate::types::clear_last_error();
    // SAFETY: strings and schema outlive the call (the creator copies them);
    // `user` and `VTABLE` live for the process.
    let ok = unsafe {
        ffi::llmtrt_register_creator(
            name.as_ptr(),
            version.as_ptr(),
            namespace.as_ptr(),
            raw_schema.as_ptr(),
            raw_schema.len() as i32,
            create,
            user,
            &VTABLE,
        )
    };
    if ok {
        Ok(())
    } else {
        // SAFETY: registration failed, so nothing else holds `user`.
        drop(unsafe { Box::from_raw(user as *mut CreatorSlot) });
        Err(crate::types::fail("IPluginRegistry::registerCreator"))
    }
}
