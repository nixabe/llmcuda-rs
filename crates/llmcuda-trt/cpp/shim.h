// C ABI over standard TensorRT 11, for llmcuda-trt.
//
// Thin by design: every object is an opaque pointer, every enum is the
// TensorRT enum's integer value, and every TensorRT struct that crosses the
// boundary has a layout-identical mirror here (shim.cpp static_asserts each
// one). The Rust side owns all policy. The shim's own logic is the generic
// plugin trampoline (`RustPlugin`), which forwards every IPluginV3 call to a
// Rust vtable, and the adapters that do the same for the logger, the GPU
// allocator and the debug listener.
//
// No function here throws or allocates device memory. Functions returning a
// pointer return NULL on failure; TensorRT's reason goes to the logger.

#ifndef LLMCUDA_TRT_SHIM_H
#define LLMCUDA_TRT_SHIM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Bumped whenever a signature or mirrored layout below changes. Serialized
// plugin fields carry it, so an engine built against another ABI is refused.
#define LLMTRT_ABI_VERSION 1
#define LLMTRT_MAX_DIMS 8
// Upper bound on a plugin's inputs plus outputs, so the trampoline never
// allocates on enqueue.
#define LLMTRT_MAX_PLUGIN_IO 32

// nvinfer1::Dims64.
typedef struct {
    int32_t nb_dims;
    int64_t d[LLMTRT_MAX_DIMS];
} llmtrt_dims;

// nvinfer1::PluginTensorDesc.
typedef struct {
    llmtrt_dims dims;
    int32_t type;
    int32_t format;
    float scale;
} llmtrt_tensor_desc;

// nvinfer1::DynamicPluginTensorDesc.
typedef struct {
    llmtrt_tensor_desc desc;
    llmtrt_dims min;
    llmtrt_dims max;
    llmtrt_dims opt;
} llmtrt_dyn_desc;

// nvinfer1::PluginField.
typedef struct {
    const char* name;
    const void* data;
    int32_t type;
    int32_t length;
} llmtrt_field;

// ---- versions and logging --------------------------------------------------

int32_t llmtrt_lib_version(void);
int32_t llmtrt_header_version(void);
int32_t llmtrt_onnx_parser_version(void);

typedef void (*llmtrt_log_fn)(int32_t severity, const char* msg);
// Install before creating any builder or runtime; all of them share one logger.
void llmtrt_set_log_callback(llmtrt_log_fn fn);

// ---- plugins ---------------------------------------------------------------

// Every entry returns 0 on success unless documented otherwise. `self` is the
// Rust plugin object returned by the creator or by `clone`.
typedef struct {
    int32_t (*nb_outputs)(void* self);
    int32_t (*output_types)(void* self, int32_t* out, int32_t nb_out, const int32_t* in, int32_t nb_in);
    // `in_dims` holds nb_in * LLMTRT_MAX_DIMS expression pointers; the plugin
    // writes nb_out * LLMTRT_MAX_DIMS into `out_dims` and the ranks into
    // `out_nb_dims`, building expressions through `exprs`.
    int32_t (*output_shapes)(void* self, void* exprs, const void* const* in_dims, const int32_t* in_nb_dims,
        int32_t nb_in, const void** out_dims, int32_t* out_nb_dims, int32_t nb_out);
    // 1 if supported, 0 if not.
    int32_t (*supports_format)(void* self, int32_t pos, const llmtrt_dyn_desc* in_out, int32_t nb_in, int32_t nb_out);
    int32_t (*configure)(void* self, const llmtrt_dyn_desc* in, int32_t nb_in, const llmtrt_dyn_desc* out, int32_t nb_out);
    size_t (*workspace_size)(void* self, const llmtrt_dyn_desc* in, int32_t nb_in, const llmtrt_dyn_desc* out, int32_t nb_out);
    int32_t (*on_shape_change)(void* self, const llmtrt_tensor_desc* in, int32_t nb_in, const llmtrt_tensor_desc* out, int32_t nb_out);
    int32_t (*enqueue)(void* self, const llmtrt_tensor_desc* in_desc, int32_t nb_in, const llmtrt_tensor_desc* out_desc,
        int32_t nb_out, const void* const* inputs, void* const* outputs, void* workspace, void* stream);
    // NULL on failure.
    void* (*clone)(void* self);
    // Number of fields, or -1. The array stays valid until the next call or destroy.
    int32_t (*fields)(void* self, const llmtrt_field** fields);
    void (*destroy)(void* self);
} llmtrt_plugin_vtable;

// Returns a new plugin object (`self` for the vtable) or NULL.
typedef void* (*llmtrt_create_fn)(void* user, const llmtrt_field* fields, int32_t nb_fields, int32_t phase);

// Registers one IPluginCreatorV3One in the global plugin registry. The
// creator lives for the rest of the process. `field_names` (name and type
// only) is what the creator advertises, which the ONNX parser uses to turn
// node attributes into fields.
bool llmtrt_register_creator(const char* name, const char* version, const char* plugin_namespace,
    const llmtrt_field* field_names, int32_t nb_field_names, llmtrt_create_fn create, void* user,
    const llmtrt_plugin_vtable* vtable);

const void* llmtrt_expr_constant(void* exprs, int64_t value);
const void* llmtrt_expr_operation(void* exprs, int32_t op, const void* first, const void* second);
bool llmtrt_expr_is_constant(const void* expr);
int64_t llmtrt_expr_constant_value(const void* expr);

// ---- builder ---------------------------------------------------------------

void* llmtrt_builder_create(void);
void llmtrt_builder_destroy(void* builder);
void* llmtrt_network_create(void* builder);
void llmtrt_network_destroy(void* network);
void* llmtrt_config_create(void* builder);
void llmtrt_config_destroy(void* config);
void llmtrt_config_set_flag(void* config, int32_t flag, bool on);
bool llmtrt_config_set_max_aux_streams(void* config, int32_t n);
void llmtrt_config_set_profiling_verbosity(void* config, int32_t verbosity);
void llmtrt_config_set_workspace_limit(void* config, uint64_t bytes);
// One optimization profile over `n` inputs. Returns its index or -1.
int32_t llmtrt_config_add_profile(void* builder, void* config, const char* const* names, const llmtrt_dims* min,
    const llmtrt_dims* opt, const llmtrt_dims* max, int32_t n);

void* llmtrt_timing_cache_create(void* config, const void* blob, size_t size);
bool llmtrt_config_set_timing_cache(void* config, void* cache, bool ignore_mismatch);
void* llmtrt_timing_cache_serialize(void* cache);
int64_t llmtrt_timing_cache_keys(void* cache, uint8_t* keys16, int64_t capacity);
bool llmtrt_timing_cache_query(void* cache, const uint8_t* key16, uint64_t* tactic, float* ms);
bool llmtrt_timing_cache_update(void* cache, const uint8_t* key16, uint64_t tactic, float ms);
void llmtrt_timing_cache_destroy(void* cache);

// Returns an IHostMemory handle holding the plan.
void* llmtrt_build_serialized(void* builder, void* network, void* config);
const void* llmtrt_host_memory_data(void* memory);
size_t llmtrt_host_memory_size(void* memory);
void llmtrt_host_memory_destroy(void* memory);

// ---- network definition ----------------------------------------------------
// Tensors (ITensor*) and layers (ILayer*) belong to their network.

void* llmtrt_net_add_input(void* network, const char* name, int32_t type, const llmtrt_dims* dims);
bool llmtrt_net_mark_output(void* network, void* tensor);
bool llmtrt_net_mark_debug(void* network, void* tensor);
int32_t llmtrt_net_nb_layers(void* network);
void llmtrt_tensor_set_name(void* tensor, const char* name);
const char* llmtrt_tensor_name(void* tensor);
int32_t llmtrt_tensor_type(void* tensor);
void llmtrt_tensor_dims(void* tensor, llmtrt_dims* out);
void* llmtrt_layer_output(void* layer, int32_t index);
// `values` must outlive the build. A non-NULL `weights_name` makes the
// constant refittable by that name.
void* llmtrt_net_add_constant(void* network, const llmtrt_dims* dims, int32_t type, const void* values,
    int64_t count, const char* weights_name);
void* llmtrt_net_add_elementwise(void* network, void* a, void* b, int32_t op);
void* llmtrt_net_add_matmul(void* network, void* a, int32_t op_a, void* b, int32_t op_b);
void* llmtrt_net_add_cast(void* network, void* tensor, int32_t type);
// `axis` < 0 leaves the axis unset; `block` may be NULL.
void* llmtrt_net_add_dequantize(void* network, void* input, void* scale, int32_t out_type, int32_t axis,
    const llmtrt_dims* block);
// Returns the IAttention helper, not a layer.
void* llmtrt_net_add_attention(void* network, void* q, void* k, void* v, int32_t causal_kind, bool decomposable);
void* llmtrt_attention_output(void* attention);
// Creates the plugin from the registered creator (build phase) and adds it.
void* llmtrt_net_add_plugin(void* network, void* const* inputs, int32_t nb_inputs, const char* name,
    const char* version, const char* plugin_namespace, const llmtrt_field* fields, int32_t nb_fields);

// ---- ONNX parser -----------------------------------------------------------

void* llmtrt_parser_create(void* network);
bool llmtrt_parser_parse(void* parser, const void* data, size_t size);
int32_t llmtrt_parser_nb_errors(void* parser);
const char* llmtrt_parser_error(void* parser, int32_t index);
void llmtrt_parser_destroy(void* parser);

// ---- runtime ---------------------------------------------------------------

typedef void* (*llmtrt_alloc_fn)(void* user, uint64_t size, uint64_t alignment, uint32_t flags, void* stream);
typedef bool (*llmtrt_free_fn)(void* user, void* memory, void* stream);
void* llmtrt_allocator_create(llmtrt_alloc_fn alloc, llmtrt_free_fn free, void* user);
void llmtrt_allocator_destroy(void* allocator);

typedef bool (*llmtrt_debug_fn)(void* user, const void* addr, int32_t location, int32_t type,
    const llmtrt_dims* shape, const char* name, void* stream);
void* llmtrt_debug_listener_create(llmtrt_debug_fn fn, void* user);
void llmtrt_debug_listener_destroy(void* listener);

void* llmtrt_runtime_create(void);
void llmtrt_runtime_set_allocator(void* runtime, void* allocator);
void llmtrt_runtime_destroy(void* runtime);

void* llmtrt_engine_deserialize(void* runtime, const void* data, size_t size);
void llmtrt_engine_destroy(void* engine);
// `profile` < 0 means the maximum over all profiles.
int64_t llmtrt_engine_device_memory_size(void* engine, int32_t profile);
int32_t llmtrt_engine_nb_profiles(void* engine);
int32_t llmtrt_engine_nb_layers(void* engine);
int32_t llmtrt_engine_nb_aux_streams(void* engine);
// IEngineInspector JSON. Returns the full length; copies at most `capacity`.
int64_t llmtrt_engine_information(void* engine, char* buffer, int64_t capacity);

// strategy: ExecutionContextAllocationStrategy.
void* llmtrt_context_create(void* engine, int32_t strategy);
void llmtrt_context_destroy(void* context);
bool llmtrt_context_set_device_memory(void* context, void* memory, int64_t size);
bool llmtrt_context_set_temporary_allocator(void* context, void* allocator);
bool llmtrt_context_set_profile(void* context, int32_t index, void* stream);
bool llmtrt_context_set_input_shape(void* context, const char* name, const llmtrt_dims* dims);
bool llmtrt_context_set_tensor_address(void* context, const char* name, void* address);
bool llmtrt_context_enqueue(void* context, void* stream);
bool llmtrt_context_set_debug_listener(void* context, void* listener);
bool llmtrt_context_set_tensor_debug_state(void* context, const char* name, bool on);

void* llmtrt_refitter_create(void* engine);
// location: TensorLocation (0 device, 1 host).
bool llmtrt_refitter_set_named_weights(void* refitter, const char* name, int32_t type, const void* values,
    int64_t count, int32_t location);
// Each writes up to `capacity` names (owned by the engine) and returns the
// total: weights still missing from a combination, and all refittable weights.
int32_t llmtrt_refitter_missing(void* refitter, const char** names, int32_t capacity);
int32_t llmtrt_refitter_all(void* refitter, const char** names, int32_t capacity);
bool llmtrt_refitter_refit(void* refitter, void* stream);
void llmtrt_refitter_destroy(void* refitter);

#ifdef __cplusplus
}
#endif

#endif // LLMCUDA_TRT_SHIM_H
