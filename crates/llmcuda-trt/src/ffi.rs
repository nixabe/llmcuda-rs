//! Raw declarations for `cpp/shim.h`. Nothing outside this crate sees them.

#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_void};

pub const ABI_VERSION: i32 = 1;
pub const MAX_DIMS: usize = 8;

/// `nvinfer1::Dims64`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct llmtrt_dims {
    pub nb_dims: i32,
    pub d: [i64; MAX_DIMS],
}

/// `nvinfer1::PluginTensorDesc`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct llmtrt_tensor_desc {
    pub dims: llmtrt_dims,
    pub data_type: i32,
    pub format: i32,
    pub scale: f32,
}

/// `nvinfer1::DynamicPluginTensorDesc`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct llmtrt_dyn_desc {
    pub desc: llmtrt_tensor_desc,
    pub min: llmtrt_dims,
    pub max: llmtrt_dims,
    pub opt: llmtrt_dims,
}

/// `nvinfer1::PluginField`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct llmtrt_field {
    pub name: *const c_char,
    pub data: *const c_void,
    pub field_type: i32,
    pub length: i32,
}

pub type llmtrt_log_fn = unsafe extern "C" fn(severity: i32, msg: *const c_char);
pub type llmtrt_create_fn = unsafe extern "C" fn(
    user: *mut c_void,
    fields: *const llmtrt_field,
    nb_fields: i32,
    phase: i32,
) -> *mut c_void;
pub type llmtrt_alloc_fn = unsafe extern "C" fn(
    user: *mut c_void,
    size: u64,
    alignment: u64,
    flags: u32,
    stream: *mut c_void,
) -> *mut c_void;
pub type llmtrt_free_fn =
    unsafe extern "C" fn(user: *mut c_void, memory: *mut c_void, stream: *mut c_void) -> bool;
pub type llmtrt_debug_fn = unsafe extern "C" fn(
    user: *mut c_void,
    addr: *const c_void,
    location: i32,
    data_type: i32,
    shape: *const llmtrt_dims,
    name: *const c_char,
    stream: *mut c_void,
) -> bool;

#[repr(C)]
pub struct llmtrt_plugin_vtable {
    pub nb_outputs: unsafe extern "C" fn(this: *mut c_void) -> i32,
    pub output_types: unsafe extern "C" fn(
        this: *mut c_void,
        out: *mut i32,
        nb_out: i32,
        input: *const i32,
        nb_in: i32,
    ) -> i32,
    pub output_shapes: unsafe extern "C" fn(
        this: *mut c_void,
        exprs: *mut c_void,
        in_dims: *const *const c_void,
        in_nb_dims: *const i32,
        nb_in: i32,
        out_dims: *mut *const c_void,
        out_nb_dims: *mut i32,
        nb_out: i32,
    ) -> i32,
    pub supports_format: unsafe extern "C" fn(
        this: *mut c_void,
        pos: i32,
        in_out: *const llmtrt_dyn_desc,
        nb_in: i32,
        nb_out: i32,
    ) -> i32,
    pub configure: unsafe extern "C" fn(
        this: *mut c_void,
        input: *const llmtrt_dyn_desc,
        nb_in: i32,
        out: *const llmtrt_dyn_desc,
        nb_out: i32,
    ) -> i32,
    pub workspace_size: unsafe extern "C" fn(
        this: *mut c_void,
        input: *const llmtrt_dyn_desc,
        nb_in: i32,
        out: *const llmtrt_dyn_desc,
        nb_out: i32,
    ) -> usize,
    pub on_shape_change: unsafe extern "C" fn(
        this: *mut c_void,
        input: *const llmtrt_tensor_desc,
        nb_in: i32,
        out: *const llmtrt_tensor_desc,
        nb_out: i32,
    ) -> i32,
    pub enqueue: unsafe extern "C" fn(
        this: *mut c_void,
        in_desc: *const llmtrt_tensor_desc,
        nb_in: i32,
        out_desc: *const llmtrt_tensor_desc,
        nb_out: i32,
        inputs: *const *const c_void,
        outputs: *const *mut c_void,
        workspace: *mut c_void,
        stream: *mut c_void,
    ) -> i32,
    pub clone: unsafe extern "C" fn(this: *mut c_void) -> *mut c_void,
    pub fields: unsafe extern "C" fn(this: *mut c_void, fields: *mut *const llmtrt_field) -> i32,
    pub destroy: unsafe extern "C" fn(this: *mut c_void),
}

unsafe extern "C" {
    pub fn llmtrt_lib_version() -> i32;
    pub fn llmtrt_header_version() -> i32;
    pub fn llmtrt_onnx_parser_version() -> i32;
    pub fn llmtrt_set_log_callback(f: Option<llmtrt_log_fn>);

    pub fn llmtrt_register_creator(
        name: *const c_char,
        version: *const c_char,
        plugin_namespace: *const c_char,
        field_names: *const llmtrt_field,
        nb_field_names: i32,
        create: llmtrt_create_fn,
        user: *mut c_void,
        vtable: *const llmtrt_plugin_vtable,
    ) -> bool;
    pub fn llmtrt_expr_constant(exprs: *mut c_void, value: i64) -> *const c_void;
    pub fn llmtrt_expr_operation(
        exprs: *mut c_void,
        op: i32,
        first: *const c_void,
        second: *const c_void,
    ) -> *const c_void;
    pub fn llmtrt_expr_is_constant(expr: *const c_void) -> bool;
    pub fn llmtrt_expr_constant_value(expr: *const c_void) -> i64;

    pub fn llmtrt_builder_create() -> *mut c_void;
    pub fn llmtrt_builder_destroy(builder: *mut c_void);
    pub fn llmtrt_network_create(builder: *mut c_void) -> *mut c_void;
    pub fn llmtrt_network_destroy(network: *mut c_void);
    pub fn llmtrt_config_create(builder: *mut c_void) -> *mut c_void;
    pub fn llmtrt_config_destroy(config: *mut c_void);
    pub fn llmtrt_config_set_flag(config: *mut c_void, flag: i32, on: bool);
    pub fn llmtrt_config_set_max_aux_streams(config: *mut c_void, n: i32) -> bool;
    pub fn llmtrt_config_set_profiling_verbosity(config: *mut c_void, verbosity: i32);
    pub fn llmtrt_config_set_workspace_limit(config: *mut c_void, bytes: u64);
    pub fn llmtrt_config_add_profile(
        builder: *mut c_void,
        config: *mut c_void,
        names: *const *const c_char,
        min: *const llmtrt_dims,
        opt: *const llmtrt_dims,
        max: *const llmtrt_dims,
        n: i32,
    ) -> i32;
    pub fn llmtrt_timing_cache_create(
        config: *mut c_void,
        blob: *const c_void,
        size: usize,
    ) -> *mut c_void;
    pub fn llmtrt_config_set_timing_cache(
        config: *mut c_void,
        cache: *mut c_void,
        ignore_mismatch: bool,
    ) -> bool;
    pub fn llmtrt_timing_cache_serialize(cache: *mut c_void) -> *mut c_void;
    pub fn llmtrt_timing_cache_keys(cache: *mut c_void, keys16: *mut u8, capacity: i64) -> i64;
    pub fn llmtrt_timing_cache_query(
        cache: *mut c_void,
        key16: *const u8,
        tactic: *mut u64,
        ms: *mut f32,
    ) -> bool;
    pub fn llmtrt_timing_cache_update(
        cache: *mut c_void,
        key16: *const u8,
        tactic: u64,
        ms: f32,
    ) -> bool;
    pub fn llmtrt_timing_cache_destroy(cache: *mut c_void);
    pub fn llmtrt_build_serialized(
        builder: *mut c_void,
        network: *mut c_void,
        config: *mut c_void,
    ) -> *mut c_void;
    pub fn llmtrt_host_memory_data(memory: *mut c_void) -> *const c_void;
    pub fn llmtrt_host_memory_size(memory: *mut c_void) -> usize;
    pub fn llmtrt_host_memory_destroy(memory: *mut c_void);

    pub fn llmtrt_net_add_input(
        network: *mut c_void,
        name: *const c_char,
        data_type: i32,
        dims: *const llmtrt_dims,
    ) -> *mut c_void;
    pub fn llmtrt_net_mark_output(network: *mut c_void, tensor: *mut c_void) -> bool;
    pub fn llmtrt_net_mark_debug(network: *mut c_void, tensor: *mut c_void) -> bool;
    pub fn llmtrt_net_nb_layers(network: *mut c_void) -> i32;
    pub fn llmtrt_tensor_set_name(tensor: *mut c_void, name: *const c_char);
    pub fn llmtrt_tensor_name(tensor: *mut c_void) -> *const c_char;
    pub fn llmtrt_tensor_type(tensor: *mut c_void) -> i32;
    pub fn llmtrt_tensor_dims(tensor: *mut c_void, out: *mut llmtrt_dims);
    pub fn llmtrt_layer_output(layer: *mut c_void, index: i32) -> *mut c_void;
    pub fn llmtrt_net_add_constant(
        network: *mut c_void,
        dims: *const llmtrt_dims,
        data_type: i32,
        values: *const c_void,
        count: i64,
        weights_name: *const c_char,
    ) -> *mut c_void;
    pub fn llmtrt_net_add_elementwise(
        network: *mut c_void,
        a: *mut c_void,
        b: *mut c_void,
        op: i32,
    ) -> *mut c_void;
    pub fn llmtrt_net_add_matmul(
        network: *mut c_void,
        a: *mut c_void,
        op_a: i32,
        b: *mut c_void,
        op_b: i32,
    ) -> *mut c_void;
    pub fn llmtrt_net_add_cast(
        network: *mut c_void,
        tensor: *mut c_void,
        data_type: i32,
    ) -> *mut c_void;
    pub fn llmtrt_net_add_dequantize(
        network: *mut c_void,
        input: *mut c_void,
        scale: *mut c_void,
        out_type: i32,
        axis: i32,
        block: *const llmtrt_dims,
    ) -> *mut c_void;
    pub fn llmtrt_net_add_attention(
        network: *mut c_void,
        q: *mut c_void,
        k: *mut c_void,
        v: *mut c_void,
        causal_kind: i32,
        decomposable: bool,
    ) -> *mut c_void;
    pub fn llmtrt_attention_output(attention: *mut c_void) -> *mut c_void;
    pub fn llmtrt_net_add_plugin(
        network: *mut c_void,
        inputs: *const *mut c_void,
        nb_inputs: i32,
        name: *const c_char,
        version: *const c_char,
        plugin_namespace: *const c_char,
        fields: *const llmtrt_field,
        nb_fields: i32,
    ) -> *mut c_void;

    pub fn llmtrt_parser_create(network: *mut c_void) -> *mut c_void;
    pub fn llmtrt_parser_parse(parser: *mut c_void, data: *const c_void, size: usize) -> bool;
    pub fn llmtrt_parser_nb_errors(parser: *mut c_void) -> i32;
    pub fn llmtrt_parser_error(parser: *mut c_void, index: i32) -> *const c_char;
    pub fn llmtrt_parser_destroy(parser: *mut c_void);

    pub fn llmtrt_allocator_create(
        alloc: llmtrt_alloc_fn,
        free: llmtrt_free_fn,
        user: *mut c_void,
    ) -> *mut c_void;
    pub fn llmtrt_allocator_destroy(allocator: *mut c_void);
    pub fn llmtrt_debug_listener_create(f: llmtrt_debug_fn, user: *mut c_void) -> *mut c_void;
    pub fn llmtrt_debug_listener_destroy(listener: *mut c_void);

    pub fn llmtrt_runtime_create() -> *mut c_void;
    pub fn llmtrt_runtime_set_allocator(runtime: *mut c_void, allocator: *mut c_void);
    pub fn llmtrt_runtime_destroy(runtime: *mut c_void);
    pub fn llmtrt_engine_deserialize(
        runtime: *mut c_void,
        data: *const c_void,
        size: usize,
    ) -> *mut c_void;
    pub fn llmtrt_engine_destroy(engine: *mut c_void);
    pub fn llmtrt_engine_device_memory_size(engine: *mut c_void, profile: i32) -> i64;
    pub fn llmtrt_engine_nb_profiles(engine: *mut c_void) -> i32;
    pub fn llmtrt_engine_nb_layers(engine: *mut c_void) -> i32;
    pub fn llmtrt_engine_nb_aux_streams(engine: *mut c_void) -> i32;
    pub fn llmtrt_engine_information(
        engine: *mut c_void,
        buffer: *mut c_char,
        capacity: i64,
    ) -> i64;

    pub fn llmtrt_context_create(engine: *mut c_void, strategy: i32) -> *mut c_void;
    pub fn llmtrt_context_destroy(context: *mut c_void);
    pub fn llmtrt_context_set_device_memory(
        context: *mut c_void,
        memory: *mut c_void,
        size: i64,
    ) -> bool;
    pub fn llmtrt_context_set_temporary_allocator(
        context: *mut c_void,
        allocator: *mut c_void,
    ) -> bool;
    pub fn llmtrt_context_set_profile(
        context: *mut c_void,
        index: i32,
        stream: *mut c_void,
    ) -> bool;
    pub fn llmtrt_context_set_input_shape(
        context: *mut c_void,
        name: *const c_char,
        dims: *const llmtrt_dims,
    ) -> bool;
    pub fn llmtrt_context_set_tensor_address(
        context: *mut c_void,
        name: *const c_char,
        address: *mut c_void,
    ) -> bool;
    pub fn llmtrt_context_enqueue(context: *mut c_void, stream: *mut c_void) -> bool;
    pub fn llmtrt_context_set_debug_listener(context: *mut c_void, listener: *mut c_void) -> bool;
    pub fn llmtrt_context_set_tensor_debug_state(
        context: *mut c_void,
        name: *const c_char,
        on: bool,
    ) -> bool;

    pub fn llmtrt_refitter_create(engine: *mut c_void) -> *mut c_void;
    pub fn llmtrt_refitter_set_named_weights(
        refitter: *mut c_void,
        name: *const c_char,
        data_type: i32,
        values: *const c_void,
        count: i64,
        location: i32,
    ) -> bool;
    pub fn llmtrt_refitter_missing(
        refitter: *mut c_void,
        names: *mut *const c_char,
        capacity: i32,
    ) -> i32;
    pub fn llmtrt_refitter_all(
        refitter: *mut c_void,
        names: *mut *const c_char,
        capacity: i32,
    ) -> i32;
    pub fn llmtrt_refitter_refit(refitter: *mut c_void, stream: *mut c_void) -> bool;
    pub fn llmtrt_refitter_destroy(refitter: *mut c_void);
}
