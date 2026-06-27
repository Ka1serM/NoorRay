#include "Kernels/cuda/OptiXSetup.h"

#include <array>
#include <cstring>
#include <stdexcept>

#include <cuda_runtime_api.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "GPU/Checks.h"
#include "Kernels/OptixLaunchParams.h"
#include "NoorRayOptixIr.h"

namespace
{
template <typename Data = uint32_t>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord
{
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    Data data{};
};

void optixLogCallback(unsigned int level, const char* tag, const char* message, void*)
{
    if (level <= 2)
        fprintf(stderr, "[OptiX][%s] %s\n", tag, message);
}

OptixProgramGroup makeRaygenGroup(
    const OptixDeviceContext context,
    const OptixModule module,
    const char* entry)
{
    OptixProgramGroupDesc description{};
    description.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    description.raygen.module = module;
    description.raygen.entryFunctionName = entry;
    OptixProgramGroupOptions options{};
    std::array<char, 4096> log{};
    size_t logSize = log.size();
    OptixProgramGroup group{};
    const OptixResult result = optixProgramGroupCreate(
        context, &description, 1, &options, log.data(), &logSize, &group);
    if (result != OPTIX_SUCCESS)
        throw std::runtime_error(std::string("OptiX raygen program creation failed: ") + log.data());
    return group;
}

CUdeviceptr uploadRecord(const OptixProgramGroup group)
{
    SbtRecord<> record{};
    NR_OPTIX_CHECK(optixSbtRecordPackHeader(group, &record));
    void* deviceRecord = nullptr;
    NR_GPU_CHECK(cudaMalloc(&deviceRecord, sizeof(record)));
    NR_GPU_CHECK(cudaMemcpy(deviceRecord, &record, sizeof(record), cudaMemcpyHostToDevice));
    return reinterpret_cast<CUdeviceptr>(deviceRecord);
}
}

OptiXSetup::~OptiXSetup()
{
    destroy();
}

void OptiXSetup::create(CUcontext cudaContext)
{
    destroy();
    NR_OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions contextOptions{};
    contextOptions.logCallbackFunction = optixLogCallback;
    contextOptions.logCallbackLevel = 3;
    NR_OPTIX_CHECK(optixDeviceContextCreate(cudaContext, &contextOptions, &context));

    OptixModuleCompileOptions moduleOptions{};
    moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
#if !defined(NDEBUG)
    moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE;
#else
    moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#endif

    OptixPipelineCompileOptions pipelineOptions{};
    pipelineOptions.usesMotionBlur = false;
    pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineOptions.numPayloadValues = 0;
    pipelineOptions.numAttributeValues = 2;
    pipelineOptions.pipelineLaunchParamsVariableName = "params";
    pipelineOptions.pipelineLaunchParamsSizeInBytes = sizeof(OptixLaunchParams);
    pipelineOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    std::array<char, 8192> log{};
    size_t logSize = log.size();
    OptixResult result = optixModuleCreate(
        context,
        &moduleOptions,
        &pipelineOptions,
        reinterpret_cast<const char*>(noorRayOptixIr),
        noorRayOptixIrLength,
        log.data(),
        &logSize,
        &module);
    if (result != OPTIX_SUCCESS)
        throw std::runtime_error(std::string("OptiX module creation failed: ") + log.data());

    extendGroup = makeRaygenGroup(context, module, "__raygen__extend");
    connectGroup = makeRaygenGroup(context, module, "__raygen__connect");

    OptixProgramGroupDesc triangleDescription{};
    triangleDescription.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    OptixProgramGroupOptions groupOptions{};
    logSize = log.size();
    NR_OPTIX_CHECK(optixProgramGroupCreate(
        context, &triangleDescription, 1, &groupOptions, log.data(), &logSize, &triangleGroup));

    OptixProgramGroupDesc missDescription{};
    missDescription.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missDescription.miss.module = nullptr;
    missDescription.miss.entryFunctionName = nullptr;
    logSize = log.size();
    NR_OPTIX_CHECK(optixProgramGroupCreate(
        context, &missDescription, 1, &groupOptions, log.data(), &logSize, &missGroup));

    const std::array groups{extendGroup, connectGroup, triangleGroup, missGroup};
    OptixPipelineLinkOptions linkOptions{};
    linkOptions.maxTraceDepth = 1;
    logSize = log.size();
    result = optixPipelineCreate(
        context,
        &pipelineOptions,
        &linkOptions,
        groups.data(),
        static_cast<unsigned int>(groups.size()),
        log.data(),
        &logSize,
        &pipeline);
    if (result != OPTIX_SUCCESS)
        throw std::runtime_error(std::string("OptiX pipeline creation failed: ") + log.data());
    NR_OPTIX_CHECK(optixPipelineSetStackSize(pipeline, 0, 0, 0, 2));

    extendRaygenRecord = uploadRecord(extendGroup);
    connectRaygenRecord = uploadRecord(connectGroup);
    hitgroupRecord = uploadRecord(triangleGroup);
    missRecord = uploadRecord(missGroup);

    extendSbt.raygenRecord = extendRaygenRecord;
    extendSbt.missRecordBase = missRecord;
    extendSbt.missRecordStrideInBytes = sizeof(SbtRecord<>);
    extendSbt.missRecordCount = 1;
    extendSbt.hitgroupRecordBase = hitgroupRecord;
    extendSbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<>);
    extendSbt.hitgroupRecordCount = 1;
    connectSbt = extendSbt;
    connectSbt.raygenRecord = connectRaygenRecord;
}

void OptiXSetup::destroy() noexcept
{
    if (extendRaygenRecord != 0)
        cudaFree(reinterpret_cast<void*>(extendRaygenRecord));
    if (connectRaygenRecord != 0)
        cudaFree(reinterpret_cast<void*>(connectRaygenRecord));
    if (hitgroupRecord != 0)
        cudaFree(reinterpret_cast<void*>(hitgroupRecord));
    if (missRecord != 0)
        cudaFree(reinterpret_cast<void*>(missRecord));
    extendRaygenRecord = connectRaygenRecord = hitgroupRecord = missRecord = 0;
    if (pipeline != nullptr)
        optixPipelineDestroy(pipeline);
    if (missGroup != nullptr)
        optixProgramGroupDestroy(missGroup);
    if (triangleGroup != nullptr)
        optixProgramGroupDestroy(triangleGroup);
    if (connectGroup != nullptr)
        optixProgramGroupDestroy(connectGroup);
    if (extendGroup != nullptr)
        optixProgramGroupDestroy(extendGroup);
    if (module != nullptr)
        optixModuleDestroy(module);
    if (context != nullptr)
        optixDeviceContextDestroy(context);
    context = nullptr;
    module = nullptr;
    extendGroup = connectGroup = triangleGroup = missGroup = nullptr;
    pipeline = nullptr;
    extendSbt = {};
    connectSbt = {};
}
