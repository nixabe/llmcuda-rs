// Implementation of shim.h. See that file for the contract.

#include "shim.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <atomic>
#include <cstddef>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using namespace nvinfer1;

// ---- layout mirrors ----------------------------------------------------------
// The plugin trampoline hands TensorRT's own descriptor arrays to Rust without
// copying, so each mirror must match byte for byte.

static_assert(sizeof(llmtrt_dims) == sizeof(Dims64), "llmtrt_dims != Dims64");
static_assert(offsetof(llmtrt_dims, d) == offsetof(Dims64, d), "llmtrt_dims.d");
static_assert(sizeof(llmtrt_tensor_desc) == sizeof(PluginTensorDesc), "llmtrt_tensor_desc");
static_assert(offsetof(llmtrt_tensor_desc, type) == offsetof(PluginTensorDesc, type), "desc.type");
static_assert(offsetof(llmtrt_tensor_desc, format) == offsetof(PluginTensorDesc, format), "desc.format");
static_assert(offsetof(llmtrt_tensor_desc, scale) == offsetof(PluginTensorDesc, scale), "desc.scale");
static_assert(sizeof(llmtrt_dyn_desc) == sizeof(DynamicPluginTensorDesc), "llmtrt_dyn_desc");
static_assert(offsetof(llmtrt_dyn_desc, min) == offsetof(DynamicPluginTensorDesc, min), "dyn.min");
static_assert(offsetof(llmtrt_dyn_desc, max) == offsetof(DynamicPluginTensorDesc, max), "dyn.max");
static_assert(offsetof(llmtrt_dyn_desc, opt) == offsetof(DynamicPluginTensorDesc, opt), "dyn.opt");
static_assert(sizeof(llmtrt_field) == sizeof(PluginField), "llmtrt_field");
static_assert(offsetof(llmtrt_field, data) == offsetof(PluginField, data), "field.data");
static_assert(offsetof(llmtrt_field, type) == offsetof(PluginField, type), "field.type");
static_assert(offsetof(llmtrt_field, length) == offsetof(PluginField, length), "field.length");
static_assert(sizeof(DataType) == sizeof(int32_t), "DataType is int32");
static_assert(sizeof(TensorFormat) == sizeof(int32_t), "TensorFormat is int32");
static_assert(sizeof(PluginFieldType) == sizeof(int32_t), "PluginFieldType is int32");

namespace
{

Dims64 toDims(llmtrt_dims const& in)
{
    Dims64 out{};
    out.nbDims = in.nb_dims;
    for (int32_t i = 0; i < LLMTRT_MAX_DIMS; ++i)
    {
        out.d[i] = in.d[i];
    }
    return out;
}

llmtrt_dims fromDims(Dims64 const& in)
{
    llmtrt_dims out{};
    out.nb_dims = in.nbDims;
    for (int32_t i = 0; i < LLMTRT_MAX_DIMS; ++i)
    {
        out.d[i] = in.d[i];
    }
    return out;
}

// ---- logger -----------------------------------------------------------------

std::atomic<llmtrt_log_fn> gLogFn{nullptr};

class RustLogger final : public ILogger
{
public:
    void log(Severity severity, AsciiChar const* msg) noexcept override
    {
        llmtrt_log_fn fn = gLogFn.load(std::memory_order_acquire);
        if (fn != nullptr)
        {
            fn(static_cast<int32_t>(severity), msg);
        }
    }
};

RustLogger& logger()
{
    static RustLogger instance;
    return instance;
}

void logError(char const* msg)
{
    logger().log(ILogger::Severity::kERROR, msg);
}

// ---- plugin trampoline --------------------------------------------------------

class RustPlugin final : public IPluginV3,
                         public IPluginV3OneCore,
                         public IPluginV3OneBuild,
                         public IPluginV3OneRuntime
{
public:
    RustPlugin(std::string name, std::string version, std::string ns, void* self, llmtrt_plugin_vtable const* vt)
        : mName(std::move(name))
        , mVersion(std::move(version))
        , mNamespace(std::move(ns))
        , mSelf(self)
        , mVt(vt)
    {
    }

    ~RustPlugin() override
    {
        if (mSelf != nullptr)
        {
            mVt->destroy(mSelf);
        }
    }

    RustPlugin(RustPlugin const&) = delete;
    RustPlugin& operator=(RustPlugin const&) = delete;

    // IPluginV3
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override
    {
        switch (type)
        {
        case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
        case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }

    IPluginV3* clone() noexcept override
    {
        void* copy = mVt->clone(mSelf);
        if (copy == nullptr)
        {
            return nullptr;
        }
        try
        {
            auto* plugin = new RustPlugin(mName, mVersion, mNamespace, copy, mVt);
            plugin->mNbIn = mNbIn;
            plugin->mNbOut = mNbOut;
            return plugin;
        }
        catch (...)
        {
            mVt->destroy(copy);
            return nullptr;
        }
    }

    // IPluginV3OneCore
    AsciiChar const* getPluginName() const noexcept override
    {
        return mName.c_str();
    }
    AsciiChar const* getPluginVersion() const noexcept override
    {
        return mVersion.c_str();
    }
    AsciiChar const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

    // IPluginV3OneBuild
    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override
    {
        mNbIn = nbInputs;
        mNbOut = nbOutputs;
        return mVt->configure(mSelf, reinterpret_cast<llmtrt_dyn_desc const*>(in), nbInputs,
            reinterpret_cast<llmtrt_dyn_desc const*>(out), nbOutputs);
    }

    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept override
    {
        return mVt->output_types(mSelf, reinterpret_cast<int32_t*>(outputTypes), nbOutputs,
            reinterpret_cast<int32_t const*>(inputTypes), nbInputs);
    }

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* /*shapeInputs*/,
        int32_t /*nbShapeInputs*/, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override
    {
        if (nbInputs + nbOutputs > LLMTRT_MAX_PLUGIN_IO)
        {
            logError("llmcuda-trt: plugin has more inputs and outputs than LLMTRT_MAX_PLUGIN_IO");
            return 1;
        }
        void const* inDims[LLMTRT_MAX_PLUGIN_IO * LLMTRT_MAX_DIMS] = {};
        int32_t inNb[LLMTRT_MAX_PLUGIN_IO] = {};
        for (int32_t i = 0; i < nbInputs; ++i)
        {
            inNb[i] = inputs[i].nbDims;
            for (int32_t j = 0; j < inputs[i].nbDims; ++j)
            {
                inDims[i * LLMTRT_MAX_DIMS + j] = inputs[i].d[j];
            }
        }
        void const* outDims[LLMTRT_MAX_PLUGIN_IO * LLMTRT_MAX_DIMS] = {};
        int32_t outNb[LLMTRT_MAX_PLUGIN_IO] = {};
        int32_t status = mVt->output_shapes(mSelf, &exprBuilder, inDims, inNb, nbInputs, outDims, outNb, nbOutputs);
        if (status != 0)
        {
            return status;
        }
        for (int32_t i = 0; i < nbOutputs; ++i)
        {
            if (outNb[i] < 0 || outNb[i] > LLMTRT_MAX_DIMS)
            {
                return 1;
            }
            outputs[i].nbDims = outNb[i];
            for (int32_t j = 0; j < outNb[i]; ++j)
            {
                auto const* e = static_cast<IDimensionExpr const*>(outDims[i * LLMTRT_MAX_DIMS + j]);
                if (e == nullptr)
                {
                    return 1;
                }
                outputs[i].d[j] = e;
            }
        }
        return 0;
    }

    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        return mVt->supports_format(mSelf, pos, reinterpret_cast<llmtrt_dyn_desc const*>(inOut), nbInputs, nbOutputs)
            == 1;
    }

    int32_t getNbOutputs() const noexcept override
    {
        return mVt->nb_outputs(mSelf);
    }

    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override
    {
        return mVt->workspace_size(mSelf, reinterpret_cast<llmtrt_dyn_desc const*>(inputs), nbInputs,
            reinterpret_cast<llmtrt_dyn_desc const*>(outputs), nbOutputs);
    }

    // IPluginV3OneRuntime
    int32_t onShapeChange(
        PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept override
    {
        mNbIn = nbInputs;
        mNbOut = nbOutputs;
        return mVt->on_shape_change(mSelf, reinterpret_cast<llmtrt_tensor_desc const*>(in), nbInputs,
            reinterpret_cast<llmtrt_tensor_desc const*>(out), nbOutputs);
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override
    {
        // IPluginV3OneRuntime::enqueue carries no counts; they were recorded
        // by onShapeChange, which TensorRT calls before any enqueue.
        if (mNbIn < 0 || mNbOut < 0)
        {
            logError("llmcuda-trt: plugin enqueue before onShapeChange");
            return 1;
        }
        return mVt->enqueue(mSelf, reinterpret_cast<llmtrt_tensor_desc const*>(inputDesc), mNbIn,
            reinterpret_cast<llmtrt_tensor_desc const*>(outputDesc), mNbOut, inputs, outputs, workspace,
            static_cast<void*>(stream));
    }

    IPluginV3* attachToContext(IPluginResourceContext* /*context*/) noexcept override
    {
        return clone();
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override
    {
        llmtrt_field const* fields = nullptr;
        int32_t n = mVt->fields(mSelf, &fields);
        if (n < 0)
        {
            return nullptr;
        }
        try
        {
            mFields.assign(reinterpret_cast<PluginField const*>(fields), reinterpret_cast<PluginField const*>(fields) + n);
        }
        catch (...)
        {
            return nullptr;
        }
        mCollection.nbFields = n;
        mCollection.fields = mFields.data();
        return &mCollection;
    }

private:
    std::string mName;
    std::string mVersion;
    std::string mNamespace;
    void* mSelf;
    llmtrt_plugin_vtable const* mVt;
    int32_t mNbIn{-1};
    int32_t mNbOut{-1};
    std::vector<PluginField> mFields;
    PluginFieldCollection mCollection{};
};

class RustCreator final : public IPluginCreatorV3One
{
public:
    RustCreator(char const* name, char const* version, char const* ns, llmtrt_field const* fieldNames,
        int32_t nbFieldNames, llmtrt_create_fn create, void* user, llmtrt_plugin_vtable const* vt)
        : mName(name)
        , mVersion(version)
        , mNamespace(ns)
        , mCreate(create)
        , mUser(user)
        , mVt(vt)
    {
        mFieldNameStorage.reserve(nbFieldNames);
        for (int32_t i = 0; i < nbFieldNames; ++i)
        {
            mFieldNameStorage.emplace_back(fieldNames[i].name);
        }
        for (int32_t i = 0; i < nbFieldNames; ++i)
        {
            mFieldNames.emplace_back(mFieldNameStorage[i].c_str(), nullptr,
                static_cast<PluginFieldType>(fieldNames[i].type), fieldNames[i].length);
        }
        mCollection.nbFields = nbFieldNames;
        mCollection.fields = mFieldNames.data();
    }

    IPluginV3* createPlugin(AsciiChar const* /*name*/, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
        override
    {
        int32_t const nb = fc != nullptr ? fc->nbFields : 0;
        llmtrt_field const* fields = fc != nullptr ? reinterpret_cast<llmtrt_field const*>(fc->fields) : nullptr;
        void* self = mCreate(mUser, fields, nb, static_cast<int32_t>(phase));
        if (self == nullptr)
        {
            return nullptr;
        }
        try
        {
            return new RustPlugin(mName, mVersion, mNamespace, self, mVt);
        }
        catch (...)
        {
            mVt->destroy(self);
            return nullptr;
        }
    }

    PluginFieldCollection const* getFieldNames() noexcept override
    {
        return &mCollection;
    }
    AsciiChar const* getPluginName() const noexcept override
    {
        return mName.c_str();
    }
    AsciiChar const* getPluginVersion() const noexcept override
    {
        return mVersion.c_str();
    }
    AsciiChar const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

private:
    std::string mName;
    std::string mVersion;
    std::string mNamespace;
    llmtrt_create_fn mCreate;
    void* mUser;
    llmtrt_plugin_vtable const* mVt;
    std::vector<std::string> mFieldNameStorage;
    std::vector<PluginField> mFieldNames;
    PluginFieldCollection mCollection{};
};

// Creators live for the process: the registry holds references to them, and
// engines deserialized at any point may ask for them.
std::mutex gCreatorsMutex;
std::vector<std::unique_ptr<RustCreator>>& creators()
{
    static std::vector<std::unique_ptr<RustCreator>> instance;
    return instance;
}

// ---- allocator, debug listener ------------------------------------------------

class RustAllocator final : public IGpuAsyncAllocator
{
public:
    RustAllocator(llmtrt_alloc_fn alloc, llmtrt_free_fn free, void* user)
        : mAlloc(alloc)
        , mFree(free)
        , mUser(user)
    {
    }

    void* allocateAsync(uint64_t const size, uint64_t const alignment, AllocatorFlags const flags,
        cudaStream_t stream) noexcept override
    {
        return mAlloc(mUser, size, alignment, flags, static_cast<void*>(stream));
    }

    bool deallocateAsync(void* const memory, cudaStream_t stream) noexcept override
    {
        return mFree(mUser, memory, static_cast<void*>(stream));
    }

private:
    llmtrt_alloc_fn mAlloc;
    llmtrt_free_fn mFree;
    void* mUser;
};

class RustDebugListener final : public IDebugListener
{
public:
    RustDebugListener(llmtrt_debug_fn fn, void* user)
        : mFn(fn)
        , mUser(user)
    {
    }

    bool processDebugTensor(void const* addr, TensorLocation location, DataType type, Dims const& shape,
        char const* name, cudaStream_t stream) override
    {
        llmtrt_dims dims = fromDims(shape);
        return mFn(mUser, addr, static_cast<int32_t>(location), static_cast<int32_t>(type), &dims, name,
            static_cast<void*>(stream));
    }

private:
    llmtrt_debug_fn mFn;
    void* mUser;
};

// A network definition and the plugins added to it. TensorRT does not say
// whether addPluginV3 copies the plugin, so the network keeps each one alive.
struct Network
{
    INetworkDefinition* def;
    std::vector<std::unique_ptr<IPluginV3>> plugins;
};

INetworkDefinition* net(void* network)
{
    return static_cast<Network*>(network)->def;
}

// An execution context together with the runtime config it was created from.
struct Context
{
    IRuntimeConfig* config;
    IExecutionContext* context;
};

} // namespace

// ---- versions and logging ------------------------------------------------------

extern "C" int32_t llmtrt_lib_version(void)
{
    return getInferLibVersion();
}

extern "C" int32_t llmtrt_header_version(void)
{
    return NV_TENSORRT_VERSION;
}

extern "C" int32_t llmtrt_onnx_parser_version(void)
{
    return getNvOnnxParserVersion();
}

extern "C" void llmtrt_set_log_callback(llmtrt_log_fn fn)
{
    gLogFn.store(fn, std::memory_order_release);
}

// ---- plugins ---------------------------------------------------------------------

extern "C" bool llmtrt_register_creator(char const* name, char const* version, char const* pluginNamespace,
    llmtrt_field const* fieldNames, int32_t nbFieldNames, llmtrt_create_fn create, void* user,
    llmtrt_plugin_vtable const* vtable)
{
    try
    {
        std::lock_guard<std::mutex> lock(gCreatorsMutex);
        auto creator = std::make_unique<RustCreator>(
            name, version, pluginNamespace, fieldNames, nbFieldNames, create, user, vtable);
        if (!getPluginRegistry()->registerCreator(*creator, pluginNamespace))
        {
            return false;
        }
        creators().push_back(std::move(creator));
        return true;
    }
    catch (...)
    {
        return false;
    }
}

extern "C" void const* llmtrt_expr_constant(void* exprs, int64_t value)
{
    return static_cast<IExprBuilder*>(exprs)->constant(value);
}

extern "C" void const* llmtrt_expr_operation(void* exprs, int32_t op, void const* first, void const* second)
{
    if (first == nullptr || second == nullptr)
    {
        return nullptr;
    }
    return static_cast<IExprBuilder*>(exprs)->operation(static_cast<DimensionOperation>(op),
        *static_cast<IDimensionExpr const*>(first), *static_cast<IDimensionExpr const*>(second));
}

extern "C" bool llmtrt_expr_is_constant(void const* expr)
{
    return static_cast<IDimensionExpr const*>(expr)->isConstant();
}

extern "C" int64_t llmtrt_expr_constant_value(void const* expr)
{
    return static_cast<IDimensionExpr const*>(expr)->getConstantValue();
}

// ---- builder ---------------------------------------------------------------------

extern "C" void* llmtrt_builder_create(void)
{
    return createInferBuilder(logger());
}

extern "C" void llmtrt_builder_destroy(void* builder)
{
    delete static_cast<IBuilder*>(builder);
}

extern "C" void* llmtrt_network_create(void* builder)
{
    // TensorRT 11 networks are always strongly typed; no flag is needed.
    INetworkDefinition* def = static_cast<IBuilder*>(builder)->createNetworkV2(0);
    if (def == nullptr)
    {
        return nullptr;
    }
    try
    {
        return new Network{def, {}};
    }
    catch (...)
    {
        delete def;
        return nullptr;
    }
}

extern "C" void llmtrt_network_destroy(void* network)
{
    auto* n = static_cast<Network*>(network);
    delete n->def;
    delete n;
}

extern "C" void* llmtrt_config_create(void* builder)
{
    return static_cast<IBuilder*>(builder)->createBuilderConfig();
}

extern "C" void llmtrt_config_destroy(void* config)
{
    delete static_cast<IBuilderConfig*>(config);
}

extern "C" void llmtrt_config_set_flag(void* config, int32_t flag, bool on)
{
    auto* c = static_cast<IBuilderConfig*>(config);
    if (on)
    {
        c->setFlag(static_cast<BuilderFlag>(flag));
    }
    else
    {
        c->clearFlag(static_cast<BuilderFlag>(flag));
    }
}

extern "C" bool llmtrt_config_set_max_aux_streams(void* config, int32_t n)
{
    return static_cast<IBuilderConfig*>(config)->setMaxAuxStreams(n);
}

extern "C" void llmtrt_config_set_profiling_verbosity(void* config, int32_t verbosity)
{
    static_cast<IBuilderConfig*>(config)->setProfilingVerbosity(static_cast<ProfilingVerbosity>(verbosity));
}

extern "C" void llmtrt_config_set_workspace_limit(void* config, uint64_t bytes)
{
    static_cast<IBuilderConfig*>(config)->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, bytes);
}

extern "C" int32_t llmtrt_config_add_profile(void* builder, void* config, char const* const* names,
    llmtrt_dims const* min, llmtrt_dims const* opt, llmtrt_dims const* max, int32_t n)
{
    IOptimizationProfile* profile = static_cast<IBuilder*>(builder)->createOptimizationProfile();
    if (profile == nullptr)
    {
        return -1;
    }
    for (int32_t i = 0; i < n; ++i)
    {
        if (!profile->setDimensions(names[i], OptProfileSelector::kMIN, toDims(min[i]))
            || !profile->setDimensions(names[i], OptProfileSelector::kOPT, toDims(opt[i]))
            || !profile->setDimensions(names[i], OptProfileSelector::kMAX, toDims(max[i])))
        {
            return -1;
        }
    }
    return static_cast<IBuilderConfig*>(config)->addOptimizationProfile(profile);
}

extern "C" void* llmtrt_timing_cache_create(void* config, void const* blob, size_t size)
{
    return static_cast<IBuilderConfig*>(config)->createTimingCache(blob, size);
}

extern "C" bool llmtrt_config_set_timing_cache(void* config, void* cache, bool ignoreMismatch)
{
    return static_cast<IBuilderConfig*>(config)->setTimingCache(*static_cast<ITimingCache*>(cache), ignoreMismatch);
}

extern "C" void* llmtrt_timing_cache_serialize(void* cache)
{
    return static_cast<ITimingCache*>(cache)->serialize();
}

extern "C" int64_t llmtrt_timing_cache_keys(void* cache, uint8_t* keys16, int64_t capacity)
{
    static_assert(sizeof(TimingCacheKey) == 16, "TimingCacheKey is 16 bytes");
    return static_cast<ITimingCache*>(cache)->queryKeys(reinterpret_cast<TimingCacheKey*>(keys16), capacity);
}

extern "C" bool llmtrt_timing_cache_query(void* cache, uint8_t const* key16, uint64_t* tactic, float* ms)
{
    TimingCacheKey key{};
    std::memcpy(key.data, key16, sizeof(key.data));
    TimingCacheValue value = static_cast<ITimingCache*>(cache)->query(key);
    *tactic = value.tacticHash;
    *ms = value.timingMSec;
    return value.tacticHash != TimingCacheValue::kINVALID_TACTIC_HASH;
}

extern "C" bool llmtrt_timing_cache_update(void* cache, uint8_t const* key16, uint64_t tactic, float ms)
{
    TimingCacheKey key{};
    std::memcpy(key.data, key16, sizeof(key.data));
    TimingCacheValue value{};
    value.tacticHash = tactic;
    value.timingMSec = ms;
    return static_cast<ITimingCache*>(cache)->update(key, value);
}

extern "C" void llmtrt_timing_cache_destroy(void* cache)
{
    delete static_cast<ITimingCache*>(cache);
}

extern "C" void* llmtrt_build_serialized(void* builder, void* network, void* config)
{
    return static_cast<IBuilder*>(builder)->buildSerializedNetwork(*net(network), *static_cast<IBuilderConfig*>(config));
}

extern "C" void const* llmtrt_host_memory_data(void* memory)
{
    return static_cast<IHostMemory*>(memory)->data();
}

extern "C" size_t llmtrt_host_memory_size(void* memory)
{
    return static_cast<IHostMemory*>(memory)->size();
}

extern "C" void llmtrt_host_memory_destroy(void* memory)
{
    delete static_cast<IHostMemory*>(memory);
}

// ---- network definition ----------------------------------------------------------

extern "C" void* llmtrt_net_add_input(void* network, char const* name, int32_t type, llmtrt_dims const* dims)
{
    return net(network)->addInput(name, static_cast<DataType>(type), toDims(*dims));
}

extern "C" bool llmtrt_net_mark_output(void* network, void* tensor)
{
    net(network)->markOutput(*static_cast<ITensor*>(tensor));
    return static_cast<ITensor*>(tensor)->isNetworkOutput();
}

extern "C" bool llmtrt_net_mark_debug(void* network, void* tensor)
{
    return net(network)->markDebug(*static_cast<ITensor*>(tensor));
}

extern "C" int32_t llmtrt_net_nb_layers(void* network)
{
    return net(network)->getNbLayers();
}

extern "C" void llmtrt_tensor_set_name(void* tensor, char const* name)
{
    static_cast<ITensor*>(tensor)->setName(name);
}

extern "C" char const* llmtrt_tensor_name(void* tensor)
{
    return static_cast<ITensor*>(tensor)->getName();
}

extern "C" int32_t llmtrt_tensor_type(void* tensor)
{
    return static_cast<int32_t>(static_cast<ITensor*>(tensor)->getType());
}

extern "C" void llmtrt_tensor_dims(void* tensor, llmtrt_dims* out)
{
    *out = fromDims(static_cast<ITensor*>(tensor)->getDimensions());
}

extern "C" void* llmtrt_layer_output(void* layer, int32_t index)
{
    return static_cast<ILayer*>(layer)->getOutput(index);
}

extern "C" void* llmtrt_net_add_constant(void* network, llmtrt_dims const* dims, int32_t type, void const* values,
    int64_t count, char const* weightsName)
{
    INetworkDefinition* def = net(network);
    Weights weights{static_cast<DataType>(type), values, count};
    IConstantLayer* layer = def->addConstant(toDims(*dims), weights);
    if (layer != nullptr && weightsName != nullptr && !def->setWeightsName(weights, weightsName))
    {
        return nullptr;
    }
    return static_cast<ILayer*>(layer);
}

extern "C" void* llmtrt_net_add_elementwise(void* network, void* a, void* b, int32_t op)
{
    return static_cast<ILayer*>(net(network)->addElementWise(
        *static_cast<ITensor*>(a), *static_cast<ITensor*>(b), static_cast<ElementWiseOperation>(op)));
}

extern "C" void* llmtrt_net_add_matmul(void* network, void* a, int32_t opA, void* b, int32_t opB)
{
    return static_cast<ILayer*>(net(network)->addMatrixMultiply(*static_cast<ITensor*>(a),
        static_cast<MatrixOperation>(opA), *static_cast<ITensor*>(b), static_cast<MatrixOperation>(opB)));
}

extern "C" void* llmtrt_net_add_cast(void* network, void* tensor, int32_t type)
{
    return static_cast<ILayer*>(net(network)->addCast(*static_cast<ITensor*>(tensor), static_cast<DataType>(type)));
}

extern "C" void* llmtrt_net_add_dequantize(
    void* network, void* input, void* scale, int32_t outType, int32_t axis, llmtrt_dims const* block)
{
    IDequantizeLayer* layer = net(network)->addDequantize(
        *static_cast<ITensor*>(input), *static_cast<ITensor*>(scale), static_cast<DataType>(outType));
    if (layer == nullptr)
    {
        return nullptr;
    }
    if (axis >= 0)
    {
        layer->setAxis(axis);
    }
    if (block != nullptr && !layer->setBlockShape(toDims(*block)))
    {
        return nullptr;
    }
    return static_cast<ILayer*>(layer);
}

extern "C" void* llmtrt_net_add_attention(void* network, void* q, void* k, void* v, int32_t causalKind, bool decomposable)
{
    IAttention* attention = net(network)->addAttentionV2(*static_cast<ITensor*>(q),
        *static_cast<ITensor*>(k), *static_cast<ITensor*>(v), AttentionNormalizationOp::kSOFTMAX,
        static_cast<CausalMaskKind>(causalKind));
    if (attention == nullptr)
    {
        return nullptr;
    }
    if (decomposable && !attention->setDecomposable(true))
    {
        return nullptr;
    }
    return attention;
}

extern "C" void* llmtrt_attention_output(void* attention)
{
    return static_cast<IAttention*>(attention)->getOutput(0);
}

extern "C" void* llmtrt_net_add_plugin(void* network, void* const* inputs, int32_t nbInputs, char const* name,
    char const* version, char const* pluginNamespace, llmtrt_field const* fields, int32_t nbFields)
{
    auto* creator = dynamic_cast<IPluginCreatorV3One*>(getPluginRegistry()->getCreator(name, version, pluginNamespace));
    if (creator == nullptr)
    {
        logError("llmcuda-trt: no IPluginCreatorV3One registered under that name, version and namespace");
        return nullptr;
    }
    PluginFieldCollection fc{nbFields, reinterpret_cast<PluginField const*>(fields)};
    IPluginV3* plugin = creator->createPlugin(name, &fc, TensorRTPhase::kBUILD);
    if (plugin == nullptr)
    {
        return nullptr;
    }
    auto* n = static_cast<Network*>(network);
    try
    {
        n->plugins.emplace_back(plugin);
    }
    catch (...)
    {
        delete plugin;
        return nullptr;
    }
    return static_cast<ILayer*>(
        n->def->addPluginV3(reinterpret_cast<ITensor* const*>(inputs), nbInputs, nullptr, 0, *plugin));
}

// ---- ONNX parser -----------------------------------------------------------------

extern "C" void* llmtrt_parser_create(void* network)
{
    return nvonnxparser::createParser(*net(network), logger());
}

extern "C" bool llmtrt_parser_parse(void* parser, void const* data, size_t size)
{
    return static_cast<nvonnxparser::IParser*>(parser)->parse(data, size);
}

extern "C" int32_t llmtrt_parser_nb_errors(void* parser)
{
    return static_cast<nvonnxparser::IParser*>(parser)->getNbErrors();
}

extern "C" char const* llmtrt_parser_error(void* parser, int32_t index)
{
    nvonnxparser::IParserError const* error = static_cast<nvonnxparser::IParser*>(parser)->getError(index);
    return error != nullptr ? error->desc() : nullptr;
}

extern "C" void llmtrt_parser_destroy(void* parser)
{
    delete static_cast<nvonnxparser::IParser*>(parser);
}

// ---- runtime ---------------------------------------------------------------------

extern "C" void* llmtrt_allocator_create(llmtrt_alloc_fn alloc, llmtrt_free_fn free, void* user)
{
    try
    {
        return new RustAllocator(alloc, free, user);
    }
    catch (...)
    {
        return nullptr;
    }
}

extern "C" void llmtrt_allocator_destroy(void* allocator)
{
    delete static_cast<RustAllocator*>(allocator);
}

extern "C" void* llmtrt_debug_listener_create(llmtrt_debug_fn fn, void* user)
{
    try
    {
        return new RustDebugListener(fn, user);
    }
    catch (...)
    {
        return nullptr;
    }
}

extern "C" void llmtrt_debug_listener_destroy(void* listener)
{
    delete static_cast<RustDebugListener*>(listener);
}

extern "C" void* llmtrt_runtime_create(void)
{
    return createInferRuntime(logger());
}

extern "C" void llmtrt_runtime_set_allocator(void* runtime, void* allocator)
{
    static_cast<IRuntime*>(runtime)->setGpuAllocator(static_cast<RustAllocator*>(allocator));
}

extern "C" void llmtrt_runtime_destroy(void* runtime)
{
    delete static_cast<IRuntime*>(runtime);
}

extern "C" void* llmtrt_engine_deserialize(void* runtime, void const* data, size_t size)
{
    return static_cast<IRuntime*>(runtime)->deserializeCudaEngine(data, size);
}

extern "C" void llmtrt_engine_destroy(void* engine)
{
    delete static_cast<ICudaEngine*>(engine);
}

extern "C" int64_t llmtrt_engine_device_memory_size(void* engine, int32_t profile)
{
    auto* e = static_cast<ICudaEngine*>(engine);
    return profile < 0 ? e->getDeviceMemorySizeV2() : e->getDeviceMemorySizeForProfileV2(profile);
}

extern "C" int32_t llmtrt_engine_nb_profiles(void* engine)
{
    return static_cast<ICudaEngine*>(engine)->getNbOptimizationProfiles();
}

extern "C" int32_t llmtrt_engine_nb_layers(void* engine)
{
    return static_cast<ICudaEngine*>(engine)->getNbLayers();
}

extern "C" int32_t llmtrt_engine_nb_aux_streams(void* engine)
{
    return static_cast<ICudaEngine*>(engine)->getNbAuxStreams();
}

extern "C" int64_t llmtrt_engine_information(void* engine, char* buffer, int64_t capacity)
{
    std::unique_ptr<IEngineInspector> inspector(static_cast<ICudaEngine*>(engine)->createEngineInspector());
    if (!inspector)
    {
        return -1;
    }
    char const* info = inspector->getEngineInformation(LayerInformationFormat::kJSON);
    if (info == nullptr)
    {
        return -1;
    }
    int64_t const length = static_cast<int64_t>(std::strlen(info));
    if (buffer != nullptr && capacity > 0)
    {
        int64_t const n = length < capacity ? length : capacity;
        std::memcpy(buffer, info, static_cast<size_t>(n));
    }
    return length;
}

extern "C" void* llmtrt_context_create(void* engine, int32_t strategy)
{
    auto* e = static_cast<ICudaEngine*>(engine);
    IRuntimeConfig* config = e->createRuntimeConfig();
    if (config == nullptr)
    {
        return nullptr;
    }
    config->setExecutionContextAllocationStrategy(static_cast<ExecutionContextAllocationStrategy>(strategy));
    IExecutionContext* context = e->createExecutionContext(config);
    if (context == nullptr)
    {
        delete config;
        return nullptr;
    }
    try
    {
        return new Context{config, context};
    }
    catch (...)
    {
        delete context;
        delete config;
        return nullptr;
    }
}

extern "C" void llmtrt_context_destroy(void* context)
{
    auto* c = static_cast<Context*>(context);
    delete c->context;
    delete c->config;
    delete c;
}

extern "C" bool llmtrt_context_set_device_memory(void* context, void* memory, int64_t size)
{
    // setDeviceMemoryV2 returns void; failures are reported through the
    // logger and surface at enqueue.
    static_cast<Context*>(context)->context->setDeviceMemoryV2(memory, size);
    return true;
}

extern "C" bool llmtrt_context_set_temporary_allocator(void* context, void* allocator)
{
    return static_cast<Context*>(context)->context->setTemporaryStorageAllocator(
        static_cast<RustAllocator*>(allocator));
}

extern "C" bool llmtrt_context_set_profile(void* context, int32_t index, void* stream)
{
    return static_cast<Context*>(context)->context->setOptimizationProfileAsync(
        index, static_cast<cudaStream_t>(stream));
}

extern "C" bool llmtrt_context_set_input_shape(void* context, char const* name, llmtrt_dims const* dims)
{
    return static_cast<Context*>(context)->context->setInputShape(name, toDims(*dims));
}

extern "C" bool llmtrt_context_set_tensor_address(void* context, char const* name, void* address)
{
    return static_cast<Context*>(context)->context->setTensorAddress(name, address);
}

extern "C" bool llmtrt_context_enqueue(void* context, void* stream)
{
    return static_cast<Context*>(context)->context->enqueueV3(static_cast<cudaStream_t>(stream));
}

extern "C" bool llmtrt_context_set_debug_listener(void* context, void* listener)
{
    return static_cast<Context*>(context)->context->setDebugListener(static_cast<RustDebugListener*>(listener));
}

extern "C" bool llmtrt_context_set_tensor_debug_state(void* context, char const* name, bool on)
{
    return static_cast<Context*>(context)->context->setTensorDebugState(name, on);
}

extern "C" void* llmtrt_refitter_create(void* engine)
{
    return createInferRefitter(*static_cast<ICudaEngine*>(engine), logger());
}

extern "C" bool llmtrt_refitter_set_named_weights(
    void* refitter, char const* name, int32_t type, void const* values, int64_t count, int32_t location)
{
    Weights weights{static_cast<DataType>(type), values, count};
    return static_cast<IRefitter*>(refitter)->setNamedWeights(name, weights, static_cast<TensorLocation>(location));
}

extern "C" int32_t llmtrt_refitter_missing(void* refitter, char const** names, int32_t capacity)
{
    auto* r = static_cast<IRefitter*>(refitter);
    int32_t const total = r->getMissingWeights(0, nullptr);
    if (names != nullptr && capacity > 0)
    {
        r->getMissingWeights(capacity < total ? capacity : total, names);
    }
    return total;
}

extern "C" int32_t llmtrt_refitter_all(void* refitter, char const** names, int32_t capacity)
{
    auto* r = static_cast<IRefitter*>(refitter);
    int32_t const total = r->getAllWeights(0, nullptr);
    if (names != nullptr && capacity > 0)
    {
        r->getAllWeights(capacity < total ? capacity : total, names);
    }
    return total;
}

extern "C" bool llmtrt_refitter_refit(void* refitter, void* stream)
{
    return static_cast<IRefitter*>(refitter)->refitCudaEngineAsync(static_cast<cudaStream_t>(stream));
}

extern "C" void llmtrt_refitter_destroy(void* refitter)
{
    delete static_cast<IRefitter*>(refitter);
}
