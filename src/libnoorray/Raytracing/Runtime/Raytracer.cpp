#include "Raytracing/Runtime/Raytracer.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "stb_image_write.h"

#include <cuda_runtime.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <nvtx3/nvToolsExt.h>

#include "Camera/CameraInstance.h"
#include "CUDA/Checks.h"
#include "CUDA/rstd/Memory.h"
#include "Raytracing/Gpu/SceneData.h"
#include "Log.h"
#include "Mesh/Assets/MeshAsset.h"
#include "Scene/MeshInstance.h"
#include "Scene/Scene.h"
#include "Vulkan/Context.h"

extern const float sRGBToSpectrumTable_Scale[64];
extern const float sRGBToSpectrumTable_Data[3][64][64][64][3];
#include "Shading/Spectrum.h"

namespace
{
constexpr unsigned char noorRayOptixIr[] = {
    #embed "../../../../build/generated/NoorRayOptixIr.ptx"
};
constexpr std::size_t noorRayOptixIrLength = sizeof(noorRayOptixIr);

constexpr unsigned char noorRayTrainingOptixIr[] = {
    #embed "../../../../build/generated/NoorRayTrainingOptixIr.ptx"
};
constexpr std::size_t noorRayTrainingOptixIrLength = sizeof(noorRayTrainingOptixIr);

template <typename Data = uint32_t>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord
{
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    Data data{};
};

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

nr::cuda::UniqueDeviceBuffer uploadRecord(const OptixProgramGroup group)
{
    SbtRecord<> record{};
    NR_OPTIX_CHECK(optixSbtRecordPackHeader(group, &record));
    nr::cuda::UniqueDeviceBuffer deviceRecord(sizeof(record));
    NR_GPU_CHECK(cudaMemcpy(deviceRecord.get(), &record, sizeof(record), cudaMemcpyHostToDevice));
    return deviceRecord;
}

OptixModuleCompileOptions makeModuleCompileOptions()
{
    OptixModuleCompileOptions options{};
    options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
#if !defined(NDEBUG)
    options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE;
#else
    options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#endif
    // The fixed SVM interpreter deliberately puts all MaterialX node families
    // in one OptiX module.  Leaving the driver compiler unconstrained makes
    // it try to retain the entire switch dispatch graph in registers; on the
    // Ada driver this has produced a multi-thousand-register raygen and an
    // abort inside libnvidia-gpucomp while creating the module.  128 is the
    // occupancy/per-thread-state balance for this interpreter.  A runtime
    // override remains available for profiling particular scenes.
    options.maxRegisterCount = 128;
    if (const char* value = std::getenv("NR_OPTIX_MAX_REGISTERS")) {
        const long parsed = std::strtol(value, nullptr, 10);
        if (parsed > 0)
            options.maxRegisterCount =
                static_cast<int>(std::clamp(parsed, 32L, 255L));
    }
    return options;
}

class NvtxRange
{
public:
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }
};

}

extern NR_GPU_KERNEL void postProcessKernel(KernelParams, float*, uint32_t);
extern NR_GPU_KERNEL void writeDenoisedOutputKernel(
    KernelParams, const glm::vec4*);
extern NR_GPU_KERNEL void writeDenoiserGuidesKernel(
    KernelParams, float3*, float3*);

Raytracer::Raytracer(
    Context& context,
    Scene& scene)
    : context(context), scene(scene)
{
    stream = context.getCudaStream();
    optixCtx = context.getOptixContext();
    if (stream == nullptr || optixCtx == nullptr)
        throw std::runtime_error("CUDA/OptiX not initialized in Context");

    OptixModuleCompileOptions moduleOptions = makeModuleCompileOptions();
    // OSL has already specialized the graph, and OptiX's final optimization
    // substantially reduces both the callable and its eventual pipeline-link
    // work. Runtime compilation is performed off the UI/render thread.
    pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipelineCompileOptions.numPayloadValues = 3;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
    pipelineCompileOptions.pipelineLaunchParamsSizeInBytes = sizeof(KernelParams);
    pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    const OptixPipelineCompileOptions& pipelineOptions = pipelineCompileOptions;

    std::array<char, 8192> log{};
    size_t logSize = log.size();
    OptixResult result = optixModuleCreate(
        optixCtx, &moduleOptions, &pipelineOptions,
        reinterpret_cast<const char*>(noorRayOptixIr), noorRayOptixIrLength,
        log.data(), &logSize, optixModule.put());
    if (result != OPTIX_SUCCESS)
        throw std::runtime_error(std::string("OptiX module creation failed: ") + log.data());

    optixPathTraceGroup.reset(makeRaygenGroup(
        optixCtx, optixModule.get(), "__raygen__pathTrace"));
    optixAovGroup.reset(makeRaygenGroup(
        optixCtx, optixModule.get(), "__raygen__aov"));
    optixProxyOverdrawGroup.reset(makeRaygenGroup(
        optixCtx, optixModule.get(), "__raygen__gaussianProxyOverdraw"));

    OptixProgramGroupOptions groupOptions{};

    OptixProgramGroupDesc triDesc{};
    triDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    logSize = log.size();
    NR_OPTIX_CHECK(optixProgramGroupCreate(
        optixCtx, &triDesc, 1, &groupOptions, log.data(), &logSize, optixTriangleGroup.put()));

    OptixProgramGroupDesc gaussianDesc{};
    gaussianDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    gaussianDesc.hitgroup.moduleAH = optixModule.get();
    gaussianDesc.hitgroup.entryFunctionNameAH = "__anyhit__gaussian";
    logSize = log.size();
    NR_OPTIX_CHECK(optixProgramGroupCreate(
        optixCtx, &gaussianDesc, 1, &groupOptions, log.data(), &logSize, optixGaussianHitGroup.put()));

    OptixProgramGroupDesc proxyOverdrawDesc{};
    proxyOverdrawDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    proxyOverdrawDesc.hitgroup.moduleAH = optixModule.get();
    proxyOverdrawDesc.hitgroup.entryFunctionNameAH = "__anyhit__gaussianProxyOverdraw";
    logSize = log.size();
    NR_OPTIX_CHECK(optixProgramGroupCreate(
        optixCtx, &proxyOverdrawDesc, 1, &groupOptions, log.data(), &logSize,
        optixProxyOverdrawHitGroup.put()));

    OptixProgramGroupDesc missDesc{};
    missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missDesc.miss.module = nullptr;
    missDesc.miss.entryFunctionName = nullptr;
    logSize = log.size();
    NR_OPTIX_CHECK(optixProgramGroupCreate(
        optixCtx, &missDesc, 1, &groupOptions, log.data(), &logSize, optixMissGroup.put()));



    rebuildPipeline();

    optixPathTraceRecord = uploadRecord(optixPathTraceGroup.get());
    optixAovRecord = uploadRecord(optixAovGroup.get());
    rebuildMeshHitgroupSbt();
    optixMissRecord = uploadRecord(optixMissGroup.get());

    optixPathTraceSbt.raygenRecord = optixPathTraceRecord.devicePtr();
    optixPathTraceSbt.missRecordBase = optixMissRecord.devicePtr();
    optixPathTraceSbt.missRecordStrideInBytes = sizeof(SbtRecord<>);
    optixPathTraceSbt.missRecordCount = 1;
    optixAovSbt = optixPathTraceSbt;
    optixAovSbt.raygenRecord = optixAovRecord.devicePtr();

    m_startEvent.create();
    m_stopEvent.create();
    NR_GPU_CHECK(cudaMallocHost(&m_noiseVarianceSumHost, sizeof(float)));
    *m_noiseVarianceSumHost = std::numeric_limits<float>::infinity();

    // Upload Jakob & Hanika sRGB-to-spectrum table once (9 MB coefficients + 64-element scale).
    constexpr size_t kScaleBytes = 64 * sizeof(float);
    constexpr size_t kCoeffBytes = sizeof(sRGBToSpectrumTable_Data);
    spectrumTableScaleDevice.allocate(kScaleBytes);
    spectrumTableCoeffsDevice.allocate(kCoeffBytes);
    NR_GPU_CHECK(cudaMemcpy(spectrumTableScaleDevice.get(), sRGBToSpectrumTable_Scale, kScaleBytes, cudaMemcpyHostToDevice));
    NR_GPU_CHECK(cudaMemcpy(spectrumTableCoeffsDevice.get(), sRGBToSpectrumTable_Data, kCoeffBytes, cudaMemcpyHostToDevice));
    gpuCache.data.spectrumTableScale  = static_cast<float*>(spectrumTableScaleDevice.get());
    gpuCache.data.spectrumTableCoeffs = static_cast<float*>(spectrumTableCoeffsDevice.get());

    std::vector<float4> filteredCoefficients(3 * 64 * 64 * 64);
    for (size_t i = 0; i < filteredCoefficients.size(); ++i)
    {
        const float* source = &sRGBToSpectrumTable_Data[0][0][0][0][0] + i * 3;
        filteredCoefficients[i] = make_float4(source[0], source[1], source[2], 0.0f);
    }
    spectrumTableTexture = nr::cuda::UniqueTexture::uploadFloat4Lut3D(
        filteredCoefficients.data(), 64, 64, 3 * 64, stream);
    gpuCache.data.energyLuts = energyLutStorage.upload(stream);
    NR_GPU_CHECK(cudaStreamSynchronize(stream));
    gpuCache.data.spectrumTableTexture = spectrumTableTexture.getObject();

    constexpr size_t kD65Bytes = NrD65Samples * sizeof(float);
    d65Device.allocate(kD65Bytes);
    NR_GPU_CHECK(cudaMemcpy(d65Device.get(), NrD65, kD65Bytes, cudaMemcpyHostToDevice));
    gpuCache.data.d65 = static_cast<float*>(d65Device.get());

    // Upload CIE 1931 2-degree CMF tables (471 floats × 3, ~5.5 KB).
    constexpr size_t kCieBytes = NrCIESamples * sizeof(float);
    cieXDevice.allocate(kCieBytes);
    cieYDevice.allocate(kCieBytes);
    cieZDevice.allocate(kCieBytes);
    NR_GPU_CHECK(cudaMemcpy(cieXDevice.get(), NrCIE_X, kCieBytes, cudaMemcpyHostToDevice));
    NR_GPU_CHECK(cudaMemcpy(cieYDevice.get(), NrCIE_Y, kCieBytes, cudaMemcpyHostToDevice));
    NR_GPU_CHECK(cudaMemcpy(cieZDevice.get(), NrCIE_Z, kCieBytes, cudaMemcpyHostToDevice));
    gpuCache.data.cieX = static_cast<float*>(cieXDevice.get());
    gpuCache.data.cieY = static_cast<float*>(cieYDevice.get());
    gpuCache.data.cieZ = static_cast<float*>(cieZDevice.get());

    {
        optixLaunchParamsDevice.allocate(std::max(
            sizeof(KernelParams), sizeof(GaussianTrainingKernelParams)));
    }

    scene.setMutationBarrier([this] { waitForRender(); });
}

void Raytracer::ensureSemaphores()
{
    if (renderReady)
        return;

    auto createSemaphore = [&]() -> vk::UniqueSemaphore {
        vk::ExportSemaphoreCreateInfo exportInfo{};
        exportInfo.handleTypes = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd;
        vk::SemaphoreTypeCreateInfo timelineInfo{};
        timelineInfo.pNext = &exportInfo;
        timelineInfo.semaphoreType = vk::SemaphoreType::eTimeline;
        timelineInfo.initialValue = 0;
        return context.getDevice().createSemaphoreUnique({vk::SemaphoreCreateFlags{}, &timelineInfo});
    };
    renderReady = createSemaphore();
    bufferReleased = createSemaphore();

    auto importSemaphore = [&](const vk::Semaphore sem) -> cudaExternalSemaphore_t {
        vk::SemaphoreGetFdInfoKHR fdInfo{};
        fdInfo.semaphore = sem;
        fdInfo.handleType = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd;
        const int fd = context.getDevice().getSemaphoreFdKHR(fdInfo);
        cudaExternalSemaphoreHandleDesc desc{};
        desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
        desc.handle.fd = fd;
        cudaExternalSemaphore_t result{};
        NR_GPU_CHECK(cudaImportExternalSemaphore(&result, &desc));
        return result;
    };
    cudaRenderReady.reset(importSemaphore(renderReady.get()));
    cudaBufferReleased.reset(importSemaphore(bufferReleased.get()));
}

void Raytracer::ensureTrainingResources()
{
    if (optixTrainingModule)
        return;

    OptixModuleCompileOptions moduleOptions = makeModuleCompileOptions();

    OptixPipelineCompileOptions trainingPipelineOptions{};
    trainingPipelineOptions.usesMotionBlur = false;
    trainingPipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    trainingPipelineOptions.numPayloadValues = 2;
    trainingPipelineOptions.numAttributeValues = 2;
    trainingPipelineOptions.pipelineLaunchParamsVariableName = "params";
    trainingPipelineOptions.pipelineLaunchParamsSizeInBytes =
        sizeof(GaussianTrainingKernelParams);
    trainingPipelineOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    std::array<char, 8192> log{};
    size_t logSize = log.size();
    const OptixResult result = optixModuleCreate(
        optixCtx, &moduleOptions, &trainingPipelineOptions,
        reinterpret_cast<const char*>(noorRayTrainingOptixIr),
        noorRayTrainingOptixIrLength,
        log.data(), &logSize, optixTrainingModule.put());
    if (result != OPTIX_SUCCESS)
        throw std::runtime_error(
            std::string("OptiX training module creation failed: ") + log.data());

    optixTrainingExtendGroup.reset(makeRaygenGroup(
        optixCtx, optixTrainingModule.get(), "__raygen__trainingPath"));
    OptixProgramGroupOptions groupOptions{};
    OptixProgramGroupDesc trainingGaussianDesc{};
    trainingGaussianDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    trainingGaussianDesc.hitgroup.moduleAH = optixTrainingModule.get();
    trainingGaussianDesc.hitgroup.entryFunctionNameAH = "__anyhit__trainingGaussian";
    logSize = log.size();
    NR_OPTIX_CHECK(optixProgramGroupCreate(
        optixCtx, &trainingGaussianDesc, 1, &groupOptions,
        log.data(), &logSize, optixTrainingGaussianHitGroup.put()));

    OptixPipelineLinkOptions linkOptions{};
    linkOptions.maxTraceDepth = 1;
    const std::array trainingGroups{
        optixTrainingExtendGroup.get(), optixTriangleGroup.get(),
        optixTrainingGaussianHitGroup.get(), optixMissGroup.get()};
    logSize = log.size();
    const auto pipelineResult = optixPipelineCreate(
        optixCtx, &trainingPipelineOptions, &linkOptions,
        trainingGroups.data(), static_cast<unsigned int>(trainingGroups.size()),
        log.data(), &logSize, optixTrainingPipeline.put());
    if (pipelineResult != OPTIX_SUCCESS)
        throw std::runtime_error(
            std::string("OptiX training pipeline creation failed: ") + log.data());

    OptixStackSizes stackSizes{};
    for (const OptixProgramGroup group : trainingGroups)
        NR_OPTIX_CHECK(optixUtilAccumulateStackSizes(
            group, &stackSizes, optixTrainingPipeline.get()));
    unsigned int directCallableStackSizeFromTraversal = 0;
    unsigned int directCallableStackSizeFromState = 0;
    unsigned int continuationStackSize = 0;
    NR_OPTIX_CHECK(optixUtilComputeStackSizes(
        &stackSizes, linkOptions.maxTraceDepth, 0, 0,
        &directCallableStackSizeFromTraversal, &directCallableStackSizeFromState,
        &continuationStackSize));
    NR_OPTIX_CHECK(optixPipelineSetStackSize(optixTrainingPipeline.get(),
        directCallableStackSizeFromTraversal, directCallableStackSizeFromState,
        continuationStackSize, 3));

    optixTrainingExtendRecord = uploadRecord(optixTrainingExtendGroup.get());
    {
        SbtRecord<> meshSbt{};
        NR_OPTIX_CHECK(optixSbtRecordPackHeader(optixTriangleGroup.get(), &meshSbt));
        SbtRecord<> gaussianSbt{};
        NR_OPTIX_CHECK(optixSbtRecordPackHeader(
            optixTrainingGaussianHitGroup.get(), &gaussianSbt));
        const std::array records{meshSbt, gaussianSbt};
        optixTrainingHitgroupRecord.allocate(sizeof(records));
        NR_GPU_CHECK(cudaMemcpy(optixTrainingHitgroupRecord.get(),
            records.data(), sizeof(records), cudaMemcpyHostToDevice));
    }
    optixTrainingExtendSbt = optixPathTraceSbt;
    optixTrainingExtendSbt.raygenRecord = optixTrainingExtendRecord.devicePtr();
    optixTrainingExtendSbt.hitgroupRecordBase = optixTrainingHitgroupRecord.devicePtr();
    // Training links its own pipeline without the material callable, so it must
    // not inherit a callables table describing a program group that pipeline
    // never saw.
    optixTrainingExtendSbt.callablesRecordBase = 0;
    optixTrainingExtendSbt.callablesRecordStrideInBytes = 0;
    optixTrainingExtendSbt.callablesRecordCount = 0;
}

void Raytracer::ensureProxyOverdrawResources()
{
    if (optixProxyOverdrawPipeline)
        return;

    OptixPipelineCompileOptions pipelineOptions{};
    pipelineOptions.usesMotionBlur = false;
    pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipelineOptions.numPayloadValues = 3;
    pipelineOptions.numAttributeValues = 2;
    pipelineOptions.pipelineLaunchParamsVariableName = "params";
    pipelineOptions.pipelineLaunchParamsSizeInBytes = sizeof(KernelParams);
    pipelineOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    OptixPipelineLinkOptions linkOptions{};
    linkOptions.maxTraceDepth = 1;
    const std::array proxyOverdrawGroups{
        optixProxyOverdrawGroup.get(), optixTriangleGroup.get(),
        optixProxyOverdrawHitGroup.get(), optixMissGroup.get()};
    std::array<char, 8192> log{};
    size_t logSize = log.size();
    const auto result = optixPipelineCreate(
        optixCtx, &pipelineOptions, &linkOptions,
        proxyOverdrawGroups.data(), static_cast<unsigned int>(proxyOverdrawGroups.size()),
        log.data(), &logSize, optixProxyOverdrawPipeline.put());
    if (result != OPTIX_SUCCESS)
        throw std::runtime_error(
            std::string("OptiX proxy-overdraw pipeline creation failed: ") + log.data());

    OptixStackSizes stackSizes{};
    for (const OptixProgramGroup group : proxyOverdrawGroups)
        NR_OPTIX_CHECK(optixUtilAccumulateStackSizes(
            group, &stackSizes, optixProxyOverdrawPipeline.get()));
    unsigned int directCallableStackSizeFromTraversal = 0;
    unsigned int directCallableStackSizeFromState = 0;
    unsigned int continuationStackSize = 0;
    NR_OPTIX_CHECK(optixUtilComputeStackSizes(
        &stackSizes, linkOptions.maxTraceDepth, 0, 0,
        &directCallableStackSizeFromTraversal, &directCallableStackSizeFromState,
        &continuationStackSize));
    NR_OPTIX_CHECK(optixPipelineSetStackSize(optixProxyOverdrawPipeline.get(),
        directCallableStackSizeFromTraversal, directCallableStackSizeFromState,
        continuationStackSize, 3));

    optixProxyOverdrawRecord = uploadRecord(optixProxyOverdrawGroup.get());
    {
        SbtRecord<> meshSbt{};
        NR_OPTIX_CHECK(optixSbtRecordPackHeader(optixTriangleGroup.get(), &meshSbt));
        SbtRecord<> gaussianSbt{};
        NR_OPTIX_CHECK(optixSbtRecordPackHeader(optixProxyOverdrawHitGroup.get(), &gaussianSbt));
        const std::array records{meshSbt, gaussianSbt};
        optixProxyOverdrawHitgroupRecord.allocate(sizeof(records));
        NR_GPU_CHECK(cudaMemcpy(optixProxyOverdrawHitgroupRecord.get(), records.data(), sizeof(records), cudaMemcpyHostToDevice));
    }
    optixProxyOverdrawSbt = optixPathTraceSbt;
    optixProxyOverdrawSbt.raygenRecord = optixProxyOverdrawRecord.devicePtr();
    optixProxyOverdrawSbt.hitgroupRecordBase = optixProxyOverdrawHitgroupRecord.devicePtr();
    // As for training: a separate pipeline without the material callable.
    optixProxyOverdrawSbt.callablesRecordBase = 0;
    optixProxyOverdrawSbt.callablesRecordStrideInBytes = 0;
    optixProxyOverdrawSbt.callablesRecordCount = 0;
}

nr::cuda::UniqueOptixPipeline Raytracer::prepareSvmPipeline() const
{
    std::array<char, 8192> log{};
    size_t logSize = log.size();

    // Collect all program groups for the pipeline.
    std::vector<OptixProgramGroup> allGroups;
    allGroups.reserve(32);
    allGroups.push_back(optixPathTraceGroup.get());
    allGroups.push_back(optixAovGroup.get());
    allGroups.push_back(optixTriangleGroup.get());
    allGroups.push_back(optixGaussianHitGroup.get());
    allGroups.push_back(optixMissGroup.get());

    nr::cuda::UniqueOptixPipeline pipeline;
    OptixPipelineLinkOptions linkOptions{};
    linkOptions.maxTraceDepth = 1;
    logSize = log.size();
    OptixResult result = optixPipelineCreate(
        optixCtx, &pipelineCompileOptions, &linkOptions,
        allGroups.data(), static_cast<unsigned int>(allGroups.size()),
        log.data(), &logSize, pipeline.put());
    if (result != OPTIX_SUCCESS)
        throw std::runtime_error(
            std::string("OptiX pipeline creation failed: ") + log.data());

    OptixStackSizes stackSizes{};
    for (const OptixProgramGroup group : allGroups)
        NR_OPTIX_CHECK(
            optixUtilAccumulateStackSizes(group, &stackSizes, pipeline.get()));

    const unsigned int maxDirectCallableDepth = 0;
    unsigned int directCallableStackSizeFromTraversal = 0;
    unsigned int directCallableStackSizeFromState = 0;
    unsigned int continuationStackSize = 0;
    NR_OPTIX_CHECK(optixUtilComputeStackSizes(
        &stackSizes, linkOptions.maxTraceDepth,
        0, maxDirectCallableDepth,
        &directCallableStackSizeFromTraversal, &directCallableStackSizeFromState,
        &continuationStackSize));
    NR_OPTIX_CHECK(optixPipelineSetStackSize(pipeline.get(),
        directCallableStackSizeFromTraversal, directCallableStackSizeFromState,
        continuationStackSize, 3));

    if (std::getenv("NR_OPTIX_LOG_LEVEL") != nullptr)
    {
        LOG_INFO("Path-trace pipeline stack: dcFromTraversal="
            << directCallableStackSizeFromTraversal
            << " dcFromState=" << directCallableStackSizeFromState
            << " continuation=" << continuationStackSize
            << " (accumulated cssRG=" << stackSizes.cssRG
            << " cssCH=" << stackSizes.cssCH
            << " cssAH=" << stackSizes.cssAH
            << " dssDC=" << stackSizes.dssDC << ')');
    }
    return pipeline;
}

void Raytracer::rebuildMeshHitgroupSbt()
{
    // SVM has one fixed hitgroup. Per-face material selection is carried by
    // Surface::material->materialxProgramIndex and indexes the global SVM
    // program table, so the SBT only routes geometry to this common program.
    SbtRecord<> meshRecord{};
    NR_OPTIX_CHECK(optixSbtRecordPackHeader(optixTriangleGroup.get(), &meshRecord));
    constexpr uint32_t meshRecordCount = 1;
    std::vector<SbtRecord<>> records(meshRecordCount + 1, meshRecord);

    SbtRecord<> gaussianRecord{};
    NR_OPTIX_CHECK(optixSbtRecordPackHeader(optixGaussianHitGroup.get(), &gaussianRecord));
    records[meshRecordCount] = gaussianRecord;

    const size_t recordBytes = records.size() * sizeof(SbtRecord<>);
    meshHitgroupRecordBase.allocate(recordBytes);
    NR_GPU_CHECK(cudaMemcpy(meshHitgroupRecordBase.get(), records.data(),
        recordBytes, cudaMemcpyHostToDevice));

    optixPathTraceSbt.hitgroupRecordBase = meshHitgroupRecordBase.devicePtr();
    optixPathTraceSbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<>);
    optixPathTraceSbt.hitgroupRecordCount = static_cast<uint32_t>(records.size());
    optixAovSbt.hitgroupRecordBase = optixPathTraceSbt.hitgroupRecordBase;
    optixAovSbt.hitgroupRecordStrideInBytes = optixPathTraceSbt.hitgroupRecordStrideInBytes;
    optixAovSbt.hitgroupRecordCount = optixPathTraceSbt.hitgroupRecordCount;

    optixPathTraceSbt.callablesRecordBase = 0;
    optixPathTraceSbt.callablesRecordStrideInBytes = 0;
    optixPathTraceSbt.callablesRecordCount = 0;
    optixAovSbt.callablesRecordBase = 0;
    optixAovSbt.callablesRecordStrideInBytes = 0;
    optixAovSbt.callablesRecordCount = 0;
}

void Raytracer::installSvmPipeline(
    nr::cuda::UniqueOptixPipeline pipeline)
{
    // The old pipeline may still be referenced by the last launch.
    NR_GPU_CHECK(cudaStreamSynchronize(stream));
    optixPipeline = std::move(pipeline);
    rebuildMeshHitgroupSbt();
}

void Raytracer::rebuildPipeline()
{
    installSvmPipeline(prepareSvmPipeline());
}

void Raytracer::uploadSvmPrograms()
{
    const auto upload = [](nr::cuda::UniqueDeviceBuffer& device,
                            const auto& host) {
        using Element = typename std::remove_reference_t<decltype(host)>::value_type;
        if (host.empty()) {
            device.reset();
            return;
        }
        device.allocate(sizeof(Element) * host.size());
        NR_GPU_CHECK(cudaMemcpy(device.get(), host.data(),
            sizeof(Element) * host.size(), cudaMemcpyHostToDevice));
    };

    upload(svmWordsDevice, materialxPrograms.words());
    upload(svmTextureIndicesDevice, materialxPrograms.textureIndices());
    gpuCache.data.svmWords = static_cast<const std::uint32_t*>(svmWordsDevice.get());
    gpuCache.data.svmTextureIndices = static_cast<const std::uint32_t*>(
        svmTextureIndicesDevice.get());
}

nr::svm::SvmProgramRecord Raytracer::registerMaterialXProgram(
    const nr::svm::CompiledSvmProgram& program)
{
    const uint32_t index = materialxPrograms.append(program);
    const nr::svm::SvmProgramRecord record = materialxPrograms.records()[index];
    uploadSvmPrograms();
    return record;
}

nr::svm::SvmProgramRecord Raytracer::replaceMaterialXProgram(
    const nr::svm::CompiledSvmProgram& program)
{
    const uint32_t index = materialxPrograms.append(program);
    const nr::svm::SvmProgramRecord record = materialxPrograms.records()[index];
    uploadSvmPrograms();
    return record;
}

void Raytracer::releaseMaterialXProgram(const uint32_t index)
{
    // Program slots are material indices stored in GPU-resident Material
    // records. They stay stable for a scene lifetime, just as Cycles' shader
    // table offsets do; clearMaterialXPrograms() reclaims the table wholesale.
    if (index >= materialxPrograms.records().size())
        return;
}

void Raytracer::clearMaterialXPrograms()
{
    materialxPrograms.clear();
    uploadSvmPrograms();
}

Raytracer::~Raytracer()
{
    scene.setMutationBarrier({});
    if (stream != nullptr)
        cudaStreamSynchronize(stream);
    if (m_noiseVarianceSumHost != nullptr)
        cudaFreeHost(m_noiseVarianceSumHost);
    m_noiseVarianceSumHost = nullptr;
    denoiser.reset();
    tlas.reset();
    freeSceneData();
    freeScratchBuffers();
    optixLaunchParamsDevice.reset();
    if (stream != nullptr)
        cudaStreamSynchronize(stream);
    m_startEvent.reset();
    m_stopEvent.reset();
    cudaRenderReady.reset();
    cudaBufferReleased.reset();
    optixPathTraceRecord.reset();
    optixAovRecord.reset();
    optixProxyOverdrawRecord.reset();
    optixTrainingExtendRecord.reset();
    meshHitgroupRecordBase.reset();
    optixTrainingHitgroupRecord.reset();
    optixProxyOverdrawHitgroupRecord.reset();
    optixMissRecord.reset();
    clearMaterialXPrograms();

    optixPipeline.reset();
    optixProxyOverdrawPipeline.reset();
    optixTrainingPipeline.reset();
    optixMissGroup.reset();
    optixGaussianHitGroup.reset();
    optixProxyOverdrawHitGroup.reset();
    optixTrainingGaussianHitGroup.reset();
    optixTriangleGroup.reset();
    optixProxyOverdrawGroup.reset();
    optixPathTraceGroup.reset();
    optixAovGroup.reset();
    optixTrainingExtendGroup.reset();
    optixModule.reset();
    optixTrainingModule.reset();
    optixPathTraceSbt = {};
    optixAovSbt = {};
    optixProxyOverdrawSbt = {};
    optixTrainingExtendSbt = {};
}

void Raytracer::setAovEnabled(const bool enabled)
{
    if (enabled == aovEnabled)
        return;
    aovEnabled = enabled;
    if (!enabled)
    {
        // The images stay allocated: a host that bound them into a descriptor
        // set keeps that binding valid. Only the per-frame AOV launch stops.
        aovAvailable = false;
        aovStale = true;
        return;
    }
    aovStale = true;
    // Allocate up front when the frame size is already known, so a caller can
    // bind the AOV images immediately after asking for them.
    ensureAovImages();
}

bool Raytracer::getAovEnabled() const
{
    return aovEnabled;
}

void Raytracer::waitForRender() const
{
    if (stream != nullptr)
        NR_GPU_CHECK(cudaStreamSynchronize(stream));
}

void Raytracer::resize(const uint32_t newWidth, const uint32_t newHeight)
{
    if (newWidth == width && newHeight == height)
        return;
    NR_GPU_CHECK(cudaStreamSynchronize(stream));
    width = newWidth;
    height = newHeight;
    aovImagesCreated = false;
    freeScratchBuffers();
    denoiser.reset();
    aovAvailable = false;
    aovStale = true;

    color.create(context, width, height, vk::Format::eR32G32B32A32Sfloat);
    // Drop AOV images that no longer match the frame. When AOVs are enabled
    // they are rebuilt right away, so a host that bound them can rebind at the
    // same point it rebinds the colour image.
    releaseAovImages();
    ensureAovImages();

    allocateScratchBuffers();
    // Scene textures, mesh buffers, and their GPU cache entries do not depend
    // on framebuffer dimensions. Re-uploading them here made every viewport
    // resize recreate every CUDA texture (multiple gigabytes in large Blender
    // scenes). Dirty scene resources are handled once below in renderFrame().
    NR_GPU_CHECK(cudaStreamSynchronize(stream));
    lastReadyValue = 0;
}

void Raytracer::ensureAovImages()
{
    // Only a host that explicitly asked for AOVs pays for them, and only once
    // the frame size is known: renderFrame() sizes the raytracer from the
    // active camera, so earlier callers can still see a zero-sized frame.
    if (aovImagesCreated || !aovEnabled || width == 0 || height == 0)
        return;
    albedo.create(context, width, height, vk::Format::eR8G8B8A8Unorm);
    normal.create(context, width, height, vk::Format::eR16G16B16A16Sfloat);
    cryptomatte.create(context, width, height, vk::Format::eR32Uint);
    position.create(context, width, height, vk::Format::eR16G16B16A16Sfloat);
    aovImagesCreated = true;
}

void Raytracer::releaseAovImages() noexcept
{
    albedo.reset();
    normal.reset();
    cryptomatte.reset();
    position.reset();
    aovImagesCreated = false;
    aovAvailable = false;
    aovStale = true;
}

void Raytracer::allocateScratchBuffers()
{
    scratchCapacity = width * height;
    accumulationBuffer.allocate(sizeof(glm::vec4) * static_cast<size_t>(width) * height, stream);
    noiseVarianceSumBuffer.allocate(sizeof(float), stream);
    NR_GPU_CHECK(cudaMemsetAsync(
        accumulationBuffer.get(), 0, sizeof(glm::vec4) * width * height, stream));
}

void Raytracer::freeScratchBuffers() noexcept
{
    accumulationBuffer.reset();
    denoiserAlbedoGuideBuffer.reset();
    denoiserNormalGuideBuffer.reset();
    noiseMomentsBuffer.reset();
    noiseVarianceSumBuffer.reset();
    scratchCapacity = 0;
}

void Raytracer::freeSceneData() noexcept
{
    scene.getEnvironment().destroyCdf();
    spectrumTableScaleDevice.reset();
    spectrumTableCoeffsDevice.reset();
    spectrumTableTexture.reset();
    d65Device.reset();
    cieXDevice.reset();
    cieYDevice.reset();
    cieZDevice.reset();
    analyticLightAliasDevice.reset();
    gpuCache.clearSceneResources();
}

void Raytracer::updateTextures()
{
    const auto& cpuTextures = scene.getTextures();
    const auto& registry = scene.getTextureRegistry();
    const size_t count = cpuTextures.size();
    auto& cudaTextures = gpuCache.textures;
    auto& mirroredHandles = gpuCache.textureHandles;

    bool changed =
        cudaTextures.size() != count || mirroredHandles.size() != count;
    const size_t commonCount = std::min(mirroredHandles.size(), count);
    for (size_t i = 0; i < commonCount && !changed; ++i)
        changed = mirroredHandles[i] != registry.handleAt(
            static_cast<uint32_t>(i));

    if (!changed) {
        gpuCache.textureRegistryRevision = registry.revision();
        return;
    }

    // Resize can relocate the managed wrapper array, and changed slots can
    // destroy CUDA texture objects. Batch that lifetime boundary into one
    // synchronization, then leave every unchanged texture array intact.
    NR_GPU_CHECK(cudaStreamSynchronize(stream));
    cudaTextures.resize(count);
    mirroredHandles.resize(count);

    for (size_t i = 0; i < count; ++i) {
        const TextureHandle current =
            registry.handleAt(static_cast<uint32_t>(i));
        if (mirroredHandles[i] == current)
            continue;

        cudaTextures[i].reset();
        mirroredHandles[i] = current;
        // A released texture leaves an empty slot behind. It keeps its index so
        // the live textures around it stay addressable, but there is nothing to
        // upload and nothing left referencing it.
        if (!current.isValid() || cpuTextures[i].getWidth() <= 0
            || cpuTextures[i].getHeight() <= 0) {
            continue;
        }
        if (cpuTextures[i].usesByteStorage()) {
            cudaTextures[i] =
                nr::cuda::UniqueTexture::uploadNormalizedUInt8x4(
                    cpuTextures[i].getBytePixels().data(),
                    cpuTextures[i].getWidth(), cpuTextures[i].getHeight(),
                    stream,
                    cpuTextures[i].getEncoding()
                        == TextureEncoding::Srgb8);
        } else if (cpuTextures[i].usesHalfStorage()) {
            cudaTextures[i] = nr::cuda::UniqueTexture::uploadHalf4(
                cpuTextures[i].getHalfPixels().data(),
                cpuTextures[i].getWidth(), cpuTextures[i].getHeight(),
                stream);
        } else {
            cudaTextures[i] = nr::cuda::UniqueTexture(
                cpuTextures[i].getPixels().data(),
                cpuTextures[i].getWidth(), cpuTextures[i].getHeight(),
                stream);
        }
    }
    gpuCache.data.textures = cudaTextures.data();
    gpuCache.data.textureCount = static_cast<uint32_t>(cpuTextures.size());
    gpuCache.textureRegistryRevision = registry.revision();

    // SVM instructions store scene texture indices, not CUDA texture objects.
    // Reloading therefore only updates the scene texture array above; no
    // material program or OptiX pipeline needs rebuilding.
}

void Raytracer::updateEnvironmentCdf()
{
    NR_GPU_CHECK(cudaStreamSynchronize(stream));
    Environment& env = scene.getEnvironment();
    gpuCache.data.environment = &env;
    env.destroyCdf();
    env.cdfDirty = 0;

    const int textureIndex = env.textureIndex;
    const auto& textures = scene.getTextures();
    if (textureIndex < 0 || textureIndex >= static_cast<int>(textures.size()))
        return;

    const Texture& texture = textures[textureIndex];
    if (texture.getWidth() <= 0 || texture.getHeight() <= 0)
        return;
    env.cdfWidth = texture.getWidth();
    env.cdfHeight = texture.getHeight();
    double integral = 0.0;
    const auto& pixels = texture.getPixels();
    for (int y = 0; y < env.cdfHeight; ++y) {
        const double solidAngleWeight = env.mapping == EnvironmentMapping::EqualArea
            ? 1.0 : std::sin((static_cast<double>(y) + 0.5)
                / static_cast<double>(env.cdfHeight) * 3.14159265358979323846);
        for (int x = 0; x < env.cdfWidth; ++x) {
            const size_t offset = static_cast<size_t>(y * env.cdfWidth + x) * 4;
            integral += (0.2126 * pixels[offset] + 0.7152 * pixels[offset + 1]
                + 0.0722 * pixels[offset + 2]) * solidAngleWeight;
        }
    }
    integral *= (env.mapping == EnvironmentMapping::EqualArea
        ? 4.0 * 3.14159265358979323846
        : 2.0 * 3.14159265358979323846 * 3.14159265358979323846)
            / static_cast<double>(env.cdfWidth * env.cdfHeight);
    const float colorLuminance = std::max(
        0.2126f * env.color.r + 0.7152f * env.color.g + 0.0722f * env.color.b, 0.0f);
    env.importanceWeight = static_cast<float>(integral) * colorLuminance
        * std::max(env.lightingExposureScale, 0.0f);
    const std::vector<float> cdf = Environment::computeCdf(
        texture.getPixels().data(), texture.getWidth(), texture.getHeight(), env.mapping);
    env.cdfTexture = nr::cuda::UniqueTexture::uploadFloat4(
        cdf.data(),
        texture.getWidth(),
        texture.getHeight(),
        stream,
        cudaAddressModeClamp,
        cudaFilterModePoint);
}

void Raytracer::updateMeshes()
{
    gpuCache.data.meshes = scene.getMeshAssets().data();
    gpuCache.materials = scene.getMaterials();
    gpuCache.data.materials = gpuCache.materials.data();
    // Mesh/material *bindings* (which program a given mesh slot points at)
    // can change without a new program ever being compiled (setMaterial()
    // reusing an already-registered index, a mesh gaining slots via
    // replaceGeometry(), ...) -- those changes only set the Meshes dirty
    // flag, not materialxSbtDirty (registerMaterialXProgram/
    // replaceMaterialXProgram's own flag, for when the *pipeline* itself
    // needs relinking). Rebuild the hitgroup SBT here too so bindings never
    // go stale waiting for an unrelated program compile.
    rebuildMeshHitgroupSbt();
}

void Raytracer::updateTLAS()
{
    NR_GPU_CHECK(cudaStreamSynchronize(stream));
    const uint32_t gaussianCount = scene.getGaussianCount();
    const bool rebuildGaussianData = scene.isDirty(GaussianData)
        || gpuCache.data.gaussianCount != gaussianCount;

    // Build geometry first. The temporary OptixInstance array is 80 bytes per
    // Gaussian and is released by Tlas before the packed SH data is prefetched,
    // substantially lowering first-load peak VRAM.
    auto& gpuInstances = gpuCache.instances;
    tlas.build(optixCtx, stream, scene, gpuInstances);
    if (rebuildGaussianData)
    {
        scene.buildGaussianRenderData();
        gpuCache.data.gaussianOpacities = scene.getGaussianOpacities();
        gpuCache.data.gaussianShCoeffs = scene.getGaussianShCoeffs();
        gpuCache.data.gaussianShCoefficientCount = scene.getGaussianShCoefficientCount();
        // A single Gaussian instance already uses global IDs as its IAS indices.
        // Keep the offset pointer null in that common case so the any-hit
        // program can avoid walking the nested IAS transform list per proxy.
        gpuCache.data.gaussianInstanceOffsets = scene.getGaussianInstances().size() > 1
            ? scene.getGaussianInstanceOffsets()
            : nullptr;

        int cudaDevice = 0;
        NR_GPU_CHECK(cudaGetDevice(&cudaDevice));
        const cudaMemLocation deviceLocation{
            .type = cudaMemLocationTypeDevice,
            .id = cudaDevice,
        };
        const auto prefetchReadOnly = [&](const void* pointer, const size_t bytes) {
            if (pointer == nullptr || bytes == 0)
                return;
            NR_GPU_CHECK(cudaMemAdvise(
                pointer, bytes, cudaMemAdviseSetReadMostly, deviceLocation));
            NR_GPU_CHECK(cudaMemPrefetchAsync(pointer, bytes, deviceLocation, 0, stream));
        };
        prefetchReadOnly(gpuCache.data.gaussianOpacities,
            sizeof(float) * static_cast<size_t>(gaussianCount));
        prefetchReadOnly(gpuCache.data.gaussianShCoeffs,
            sizeof(__half) * SphericalHarmonicsChannelCount
                * gpuCache.data.gaussianShCoefficientCount
                * static_cast<size_t>(gaussianCount));
        prefetchReadOnly(gpuCache.data.gaussianInstanceOffsets,
            sizeof(uint32_t) * scene.getGaussianInstances().size());
    }
    for (GaussianAsset& asset : scene.getGaussianAssets())
        asset.clearDirtyFlag();
    scene.clearDirtyMeshInstanceIndices();
    scene.clearDirtyGaussianInstanceIndices();
    gpuCache.data.tlasHandle = tlas.getTraversable();
    gpuCache.data.instances = gpuInstances.data();
    gpuCache.data.meshInstanceCount = static_cast<uint32_t>(gpuInstances.size());

    gpuCache.data.gaussianCount = gaussianCount;
}

void Raytracer::updateLights()
{
    // VMA-mapped host addresses and CUDA-imported device addresses are distinct.
    NR_GPU_CHECK(cudaStreamSynchronize(stream));
    gpuCache.data.pointLights = scene.getPointLightsDevice();
    gpuCache.data.spotLights = scene.getSpotLightsDevice();
    gpuCache.data.rectLights = scene.getRectLightsDevice();
    gpuCache.data.directionalLights = scene.getDirectionalLightsDevice();
    gpuCache.data.pointLightCount = scene.getPointLightCount();
    gpuCache.data.spotLightCount = scene.getSpotLightCount();
    gpuCache.data.rectLightCount = scene.getRectLightCount();
    gpuCache.data.directionalLightCount = scene.getDirectionalLightCount();

    const PointLight* pointLights = scene.getPointLights();
    const SpotLight* spotLights = scene.getSpotLights();
    const RectLight* rectLights = scene.getRectLights();
    const DirectionalLight* directionalLights = scene.getDirectionalLights();

    std::vector<float> weights;
    weights.reserve(static_cast<size_t>(gpuCache.data.pointLightCount)
        + gpuCache.data.spotLightCount + gpuCache.data.rectLightCount
        + gpuCache.data.directionalLightCount);
    for (uint32_t i = 0; i < gpuCache.data.pointLightCount; ++i)
        weights.push_back(fmaxf(pointLights[i].selectionWeight(), 0.0f));
    for (uint32_t i = 0; i < gpuCache.data.spotLightCount; ++i)
        weights.push_back(fmaxf(spotLights[i].selectionWeight(), 0.0f));
    for (uint32_t i = 0; i < gpuCache.data.rectLightCount; ++i)
        weights.push_back(fmaxf(rectLights[i].selectionWeight(), 0.0f));
    for (uint32_t i = 0; i < gpuCache.data.directionalLightCount; ++i)
        weights.push_back(fmaxf(directionalLights[i].selectionWeight(), 0.0f));

    float analyticWeight = 0.0f;
    for (const float weight : weights)
        analyticWeight += weight;
    gpuCache.data.analyticLightSelectionWeight = analyticWeight;

    analyticLightAliasDevice.reset();
    gpuCache.data.analyticLightAliases = nullptr;
    gpuCache.data.analyticLightAliasCount = 0;
    if (weights.empty() || analyticWeight <= 0.0f)
        return;

    const uint32_t count = static_cast<uint32_t>(weights.size());
    std::vector<AnalyticLightAliasEntry> aliases(count);
    std::vector<float> scaled(count);
    std::vector<uint32_t> small;
    std::vector<uint32_t> large;
    small.reserve(count);
    large.reserve(count);
    for (uint32_t i = 0; i < count; ++i)
    {
        aliases[i].alias = i;
        aliases[i].selectionPdf = weights[i] / analyticWeight;
        scaled[i] = aliases[i].selectionPdf * static_cast<float>(count);
        (scaled[i] < 1.0f ? small : large).push_back(i);
    }
    while (!small.empty() && !large.empty())
    {
        const uint32_t low = small.back();
        small.pop_back();
        const uint32_t high = large.back();
        large.pop_back();
        aliases[low].threshold = scaled[low];
        aliases[low].alias = high;
        scaled[high] = scaled[high] + scaled[low] - 1.0f;
        (scaled[high] < 1.0f ? small : large).push_back(high);
    }
    for (const uint32_t index : large)
        aliases[index].threshold = 1.0f;
    for (const uint32_t index : small)
        aliases[index].threshold = 1.0f;

    analyticLightAliasDevice.allocate(sizeof(AnalyticLightAliasEntry) * aliases.size());
    NR_GPU_CHECK(cudaMemcpy(analyticLightAliasDevice.get(), aliases.data(),
        sizeof(AnalyticLightAliasEntry) * aliases.size(), cudaMemcpyHostToDevice));
    gpuCache.data.analyticLightAliases =
        static_cast<const AnalyticLightAliasEntry*>(analyticLightAliasDevice.get());
    gpuCache.data.analyticLightAliasCount = count;
}

void Raytracer::launchPostProcess(
    const KernelParams& params, const cudaStream_t stream, const uint32_t flags) const
{
    constexpr uint32_t blockSize = 256;
    const uint32_t count = params.frame.width * params.frame.height;
    const dim3 grid((count + blockSize - 1) / blockSize, 1, 1);
    float* varianceSum = nullptr;
    if ((flags & PostProcessNoiseVariance) != 0)
    {
        varianceSum = noiseVarianceSumBuffer.as<float>();
        NR_GPU_CHECK(cudaMemsetAsync(varianceSum, 0, sizeof(float), stream));
    }
    uint32_t launchFlags = flags;
    void* args[] = {const_cast<KernelParams*>(&params), &varianceSum,
        &launchFlags};
    NR_GPU_CHECK(cudaLaunchKernel(
        reinterpret_cast<const void*>(&postProcessKernel),
        grid, blockSize, args, 0, stream));
    if ((flags & PostProcessNoiseVariance) != 0)
        NR_GPU_CHECK(cudaMemcpyAsync(m_noiseVarianceSumHost, varianceSum,
            sizeof(float), cudaMemcpyDeviceToHost, stream));
}

void Raytracer::launchDenoiser(
    KernelParams const& params, const cudaStream_t stream,
    const bool useAovGuides)
{
    constexpr uint32_t blockSize = 256;
    const uint32_t count = width * height;
    const dim3 grid((count + blockSize - 1) / blockSize, 1, 1);
    const void* albedoGuide = nullptr;
    const void* normalGuide = nullptr;
    if (useAovGuides)
    {
        if (!denoiserAlbedoGuideBuffer)
            denoiserAlbedoGuideBuffer.allocate(
                sizeof(float3) * static_cast<size_t>(count), stream);
        if (!denoiserNormalGuideBuffer)
            denoiserNormalGuideBuffer.allocate(
                sizeof(float3) * static_cast<size_t>(count), stream);
        float3* albedo = denoiserAlbedoGuideBuffer.as<float3>();
        float3* normal = denoiserNormalGuideBuffer.as<float3>();
        void* guideArgs[] = {
            const_cast<KernelParams*>(&params), &albedo, &normal};
        NR_GPU_CHECK(cudaLaunchKernel(
            reinterpret_cast<const void*>(&writeDenoiserGuidesKernel),
            grid, blockSize, guideArgs, 0, stream));
        albedoGuide = albedo;
        normalGuide = normal;
    }
    const glm::vec4* denoised = static_cast<const glm::vec4*>(
        denoiser.run(optixCtx, stream, params.accumulation, albedoGuide,
            normalGuide, width, height));
    void* args[] = {const_cast<KernelParams*>(&params), &denoised};
    NR_GPU_CHECK(cudaLaunchKernel(
        reinterpret_cast<const void*>(&writeDenoisedOutputKernel),
        grid, blockSize, args, 0, stream));
}

void Raytracer::prepareSensorFrame(Sensor& sensor, KernelParams& params, const bool resetAccumulation)
{
    params.psfGatherBuckets = nullptr;
    params.psfBinCount = 0;

    if (resetAccumulation)
        NR_GPU_CHECK(cudaMemsetAsync(
            accumulationBuffer.get(), 0, sizeof(glm::vec4) * width * height, stream));

    if (sensor.Is<ScatterPsfSensor>()) {
        return;
    }

    if (auto* gather = sensor.CastOrNullptr<GatherPsfSensor>())
        gather->prepareFrame(width, height, resetAccumulation, stream,
            params.psfGatherBuckets, params.psfBinCount);
}

void Raytracer::launchGaussianTrainPath(
    const GaussianTrainingKernelParams& params, const uint32_t launchCount,
    const cudaStream_t stream) const
{
    NR_GPU_CHECK(cudaMemcpyAsync(optixLaunchParamsDevice.get(),
        &params, sizeof(params), cudaMemcpyHostToDevice, stream));
    NR_OPTIX_CHECK(optixLaunch(optixTrainingPipeline.get(), stream,
        optixLaunchParamsDevice.devicePtr(), sizeof(GaussianTrainingKernelParams),
        &optixTrainingExtendSbt, launchCount, 1, 1));
}


void Raytracer::launchPathTrace(
    const KernelParams& params, const cudaStream_t stream) const
{
    const NvtxRange profileRange("NoorRay/PathTrace");
    NR_GPU_CHECK(cudaMemcpyAsync(optixLaunchParamsDevice.get(),
        &params, sizeof(params), cudaMemcpyHostToDevice, stream));
    NR_OPTIX_CHECK(optixLaunch(optixPipeline.get(), stream,
        optixLaunchParamsDevice.devicePtr(), sizeof(KernelParams),
        &optixPathTraceSbt, params.frame.width, params.frame.height, 1));
}

void Raytracer::launchAov(
    const KernelParams& params, const cudaStream_t stream) const
{
    NR_GPU_CHECK(cudaMemcpyAsync(optixLaunchParamsDevice.get(),
        &params, sizeof(params), cudaMemcpyHostToDevice, stream));
    NR_OPTIX_CHECK(optixLaunch(optixPipeline.get(), stream,
        optixLaunchParamsDevice.devicePtr(), sizeof(KernelParams),
        &optixAovSbt, params.frame.width, params.frame.height, 1));
}

void Raytracer::renderGaussianTrainForward(
    const GaussianTrainParams& trainParams,
    const uint32_t renderWidth, const uint32_t renderHeight)
{
    ensureTrainingResources();
    const size_t pixelCount = static_cast<size_t>(renderWidth) * renderHeight;
    if (pixelCount > scratchCapacity)
        throw std::runtime_error("Training image exceeds the scratch-buffer capacity");
    CameraInstance* activeCamera = scene.getActiveCamera();
    if (activeCamera == nullptr)
        throw std::runtime_error("Gaussian training requires an active camera");

    GaussianTrainingKernelParams params{};
    params.scene = gpuCache.data;
    params.scene.camera = activeCamera->getGpuCamera();
    params.train = trainParams;
    params.frame.width = renderWidth;
    params.frame.height = renderHeight;
    params.frame.cutoffDistanceSq = scene.getRenderSettings().gaussianCutoffSigma
        * scene.getRenderSettings().gaussianCutoffSigma;

    const size_t outputBytes = static_cast<size_t>(renderWidth) * renderHeight
        * 3 * sizeof(float);
    NR_GPU_CHECK(cudaMemsetAsync(params.train.outputColor, 0, outputBytes, stream));

    for (uint32_t sample = 0; sample < trainParams.samplesPerPixel; ++sample)
    {
        params.frame.totalAccumulated = sample;
        launchGaussianTrainPath(params, static_cast<uint32_t>(pixelCount), stream);
    }
    NR_GPU_CHECK(cudaStreamSynchronize(stream));
}

void Raytracer::renderGaussianTrainBackward(
    const GaussianTrainParams& trainParams,
    const uint32_t renderWidth, const uint32_t renderHeight)
{
    ensureTrainingResources();
    const size_t pixelCount = static_cast<size_t>(renderWidth) * renderHeight;
    if (pixelCount > scratchCapacity)
        throw std::runtime_error("Training image exceeds the scratch-buffer capacity");
    CameraInstance* activeCamera = scene.getActiveCamera();
    if (activeCamera == nullptr)
        throw std::runtime_error("Gaussian training requires an active camera");

    GaussianTrainingKernelParams params{};
    params.scene = gpuCache.data;
    params.scene.camera = activeCamera->getGpuCamera();
    params.train = trainParams;
    params.frame.width = renderWidth;
    params.frame.height = renderHeight;
    params.frame.cutoffDistanceSq = scene.getRenderSettings().gaussianCutoffSigma
        * scene.getRenderSettings().gaussianCutoffSigma;

    const uint32_t gaussianCount = params.train.gaussianCount;
    NR_GPU_CHECK(cudaMemsetAsync(params.train.dPosition, 0,
        sizeof(glm::vec3) * gaussianCount, stream));
    NR_GPU_CHECK(cudaMemsetAsync(params.train.dLogScale, 0,
        sizeof(glm::vec3) * gaussianCount, stream));
    NR_GPU_CHECK(cudaMemsetAsync(params.train.dRotation, 0,
        sizeof(glm::vec4) * gaussianCount, stream));
    NR_GPU_CHECK(cudaMemsetAsync(params.train.dOpacityLogit, 0,
        sizeof(float) * gaussianCount, stream));
    NR_GPU_CHECK(cudaMemsetAsync(params.train.dColorRgb, 0,
        sizeof(glm::vec3) * gaussianCount, stream));

    for (uint32_t sample = 0; sample < trainParams.samplesPerPixel; ++sample)
    {
        params.frame.totalAccumulated = sample;
        launchGaussianTrainPath(params, static_cast<uint32_t>(pixelCount), stream);
    }
    NR_GPU_CHECK(cudaStreamSynchronize(stream));
}

void Raytracer::launchProxyOverdraw(
    const KernelParams& params, const cudaStream_t stream) const
{
    NR_GPU_CHECK(cudaMemcpyAsync(optixLaunchParamsDevice.get(),
        &params, sizeof(params), cudaMemcpyHostToDevice, stream));
    NR_OPTIX_CHECK(optixLaunch(optixProxyOverdrawPipeline.get(), stream,
        optixLaunchParamsDevice.devicePtr(), sizeof(KernelParams),
        &optixProxyOverdrawSbt, params.frame.width * params.frame.height, 1, 1));
}

void Raytracer::renderFrame(
    const uint32_t frameIndex, const uint32_t accumulatedSamples)
{
    // Synchronize the GPU stream once per frame if any mutation occurred since
    // the last render. This replaces the old per-mutation waitForRender barrier
    // with a single sync point, batching all Hydra Sync() calls together.
    if (scene.consumeGpuSync())
        NR_GPU_CHECK(cudaStreamSynchronize(stream));

    const bool textureRegistryChanged =
        gpuCache.textureRegistryRevision
        != scene.getTextureRegistry().revision();
    const bool sceneDirty = scene.isAnyDirty() || textureRegistryChanged;

    CameraInstance* activeCamera = scene.getRenderCamera();
    if (!activeCamera)
        return;

    const glm::uvec2 resolution = activeCamera->getCamera()->getSensor().resolution();
    if (resolution.x == 0 || resolution.y == 0)
        throw std::runtime_error("Cannot render with a zero-sized camera sensor");
    if (resolution.x != width || resolution.y != height)
        resize(resolution.x, resolution.y);

    // AOVs are decided after the frame is sized: their images must match the
    // resolution this frame renders at, and before the first resize there is no
    // resolution to allocate them for.
    //
    // Refresh AOVs immediately whenever scene data changes so viewport guides,
    // picking, outlines, and denoiser inputs stay aligned with moved cameras
    // and objects. Unchanged progressive frames continue reusing the last AOVs.
    const bool useAovs = aovEnabled && scene.getRenderSettings().aovEnabled;
    if (useAovs)
        ensureAovImages();
    else
    {
        aovAvailable = false;
        aovStale = true;
    }
    const bool renderAov = useAovs && aovImagesCreated && (sceneDirty || aovStale);

    if (scene.isDirty(Meshes)) updateMeshes();
    if (scene.isDirty(Textures) || textureRegistryChanged)
        updateTextures();
    if (scene.isDirty(EnvironmentCdf))
        updateEnvironmentCdf();
    if (scene.isDirty(Lights)) updateLights();
    if (scene.isDirty(TLAS) || scene.isDirty(GaussianData)) updateTLAS();
    if (scene.isDirty(CameraState)) activeCamera->rebuildCamera();

    activeCamera->getCamera()->prepareForRender();

    const RenderSettings& renderSettings = scene.getRenderSettings();
    if (scene.getEnvironment().cdfDirty != 0)
        updateEnvironmentCdf();

    gpuCache.data.renderSettings = scene.getRenderSettings();
    const uint64_t frameValue = ++submittedFrame;
    const bool useViewportInterop = !context.isHeadless();
    if (useViewportInterop)
        ensureSemaphores();
    if (useViewportInterop && lastReadyValue != 0)
    {
        cudaExternalSemaphoreWaitParams waitParams{};
        waitParams.params.fence.value = lastReadyValue;
        cudaExternalSemaphore_t semaphore = cudaBufferReleased.get();
        NR_GPU_CHECK(cudaWaitExternalSemaphoresAsync(&semaphore, &waitParams, 1, stream));
    }

    KernelParams params{};
    params.scene = gpuCache.data;
    params.scene.renderSettings = scene.getRenderSettings();
    params.scene.camera = activeCamera->getGpuCamera();
    params.output = {
        color.getSurface(),
        albedo.getSurface(),
        normal.getSurface(),
        cryptomatte.getSurface(),
        position.getSurface(),
        width, height};
    params.frame.width = width;
    params.frame.height = height;
    params.frame.frameIndex = frameIndex;
    params.frame.visibilityMask = params.scene.gaussianCount > 0
        ? SceneVisibility
        : MeshVisibility;
    params.accumulation = accumulationBuffer.as<glm::vec4>();

    if (renderSettings.noiseLimitEnabled && !noiseMomentsBuffer)
        noiseMomentsBuffer.allocate(
            sizeof(glm::vec2) * static_cast<size_t>(width) * height, stream);
    params.noiseMoments = renderSettings.noiseLimitEnabled
        ? noiseMomentsBuffer.as<glm::vec2>() : nullptr;
    if (params.noiseMoments != nullptr && frameIndex == 0)
        NR_GPU_CHECK(cudaMemsetAsync(
            params.noiseMoments, 0, sizeof(glm::vec2) * width * height, stream));
    params.frame.cutoffDistanceSq = renderSettings.gaussianCutoffSigma
                                 * renderSettings.gaussianCutoffSigma;
    const uint32_t configuredSamplesPerFrame =
        static_cast<uint32_t>(std::max(1, renderSettings.samples));
    const uint32_t maxSamples = context.isHeadless()
        ? configuredSamplesPerFrame
        : static_cast<uint32_t>(std::max(1, renderSettings.maxSamples));
    const uint32_t remainingSamples = accumulatedSamples < maxSamples
        ? maxSamples - accumulatedSamples
        : configuredSamplesPerFrame;
    const uint32_t samplesPerFrame = std::min(configuredSamplesPerFrame, remainingSamples);
    const uint32_t requestedMaxShaderBounces = std::min(MaxBounces - 1,
        static_cast<uint32_t>(std::max(renderSettings.maxBounces, 1)));
    const uint32_t maxShaderBounces = requestedMaxShaderBounces;
    params.depth = maxShaderBounces;

    Sensor& activeSensor = activeCamera->getCamera()->getSensor();
    prepareSensorFrame(activeSensor, params, frameIndex == 0);

    if (m_timingEnabled)
        NR_GPU_CHECK(cudaEventRecord(m_startEvent.get(), stream));

    if (renderSettings.gaussianProxyOverdrawVisualization)
    {
        ensureProxyOverdrawResources();
        kernelStats.time("ProxyOverdraw", stream,
            [&] { launchProxyOverdraw(params, stream); });
    }
    else
    {
        if (renderAov)
        {
            // The AOV raygen traces both queries in this one OptiX launch.
            params.frame.totalAccumulated = accumulatedSamples;
            params.frame.aovQuery = 1;
            kernelStats.time("AOV", stream,
                [&] { launchAov(params, stream); });
            params.frame.aovQuery = 0;
            aovAvailable = true;
            aovStale = false;
        }
        for (uint32_t s = 0; s < samplesPerFrame; ++s)
        {
            params.frame.totalAccumulated = accumulatedSamples + s;
            kernelStats.time("PathTrace", stream,
                [&] { launchPathTrace(params, stream); });
        }
        const uint32_t accumulatedBeforeFrame = accumulatedSamples;
        const uint32_t accumulatedAfterFrame = accumulatedSamples + samplesPerFrame;
        const bool finalSample = accumulatedBeforeFrame < maxSamples && accumulatedAfterFrame >= maxSamples;
        uint32_t postProcessFlags = 0;
        if (activeSensor.Is<ScatterPsfSensor>())
            postProcessFlags |= PostProcessPsf;
        else if (activeSensor.Is<GatherPsfSensor>() && finalSample)
        {
            if (params.psfGatherBuckets == nullptr || params.psfBinCount == 0)
            {
                std::cerr << "[PSF] Gather final resolve skipped: buckets="
                          << static_cast<const void*>(params.psfGatherBuckets)
                          << " bins=" << params.psfBinCount << std::endl;
            }
            else
            {
                std::cerr << "[PSF] Gather final resolve applying " << params.psfBinCount
                          << " PSF bins" << std::endl;
                postProcessFlags |= PostProcessPsf;
            }
        }
        const bool directSensor = !activeSensor.Is<ScatterPsfSensor>()
            && !activeSensor.Is<GatherPsfSensor>();
        const uint32_t denoiserMinSamples = static_cast<uint32_t>(
            std::max(1, renderSettings.optixDenoiserMinSamples));
        if (renderSettings.optixDenoiserEnabled && directSensor
            && accumulatedAfterFrame >= denoiserMinSamples)
            kernelStats.time("OptixDenoiser", stream,
                [&] { launchDenoiser(params, stream, aovAvailable); });
        if (renderSettings.noiseLimitEnabled)
        {
            m_noiseResultSampleCount = accumulatedAfterFrame;
            postProcessFlags |= PostProcessNoiseVariance;
        }
        if (postProcessFlags != 0)
            kernelStats.time("PostProcess", stream,
                [&] { launchPostProcess(params, stream, postProcessFlags); });
    }
    NR_GPU_CHECK(cudaPeekAtLastError());

    if (m_timingEnabled)
    {
        NR_GPU_CHECK(cudaEventRecord(m_stopEvent.get(), stream));
        m_eventsRecorded = true;
    }

    if (useViewportInterop)
    {
        cudaExternalSemaphoreSignalParams signalParams{};
        signalParams.params.fence.value = frameValue;
        cudaExternalSemaphore_t semaphore = cudaRenderReady.get();
        NR_GPU_CHECK(cudaSignalExternalSemaphoresAsync(&semaphore, &signalParams, 1, stream));
    }
    lastReadyValue = frameValue;
    scene.clearDirtyFlags();
}

void Raytracer::debugSave(const std::string& path) const
{
    NR_GPU_CHECK(cudaStreamSynchronize(stream));

    LOG_DEBUG("meshCount=" << scene.getMeshAssets().size()
              << " instanceCount=" << gpuCache.instances.size()
              << " traversable=" << tlas.getTraversable()
              << " size=" << width << "x" << height);

    const size_t pixelCount = static_cast<size_t>(width) * height;
    std::vector<glm::vec4> host(pixelCount);
    NR_GPU_CHECK(cudaMemcpy(
        host.data(), accumulationBuffer.get(), pixelCount * sizeof(glm::vec4), cudaMemcpyDeviceToHost));

    // Per-channel stats
    float maxR = 0, maxG = 0, maxB = 0, maxA = 0, minA = 1.0f;
    float sumR = 0, sumG = 0, sumB = 0;
    for (const auto& p : host) {
        maxR = std::max(maxR, p.x); maxG = std::max(maxG, p.y);
        maxB = std::max(maxB, p.z); maxA = std::max(maxA, p.w);
        minA = std::min(minA, p.w);
        sumR += p.x; sumG += p.y; sumB += p.z;
    }
    const float n = static_cast<float>(pixelCount);
    LOG_DEBUG("R: max=" << maxR << " avg=" << sumR/n
              << "  G: max=" << maxG << " avg=" << sumG/n
              << "  B: max=" << maxB << " avg=" << sumB/n
              << "  A: min=" << minA << " max=" << maxA);
    // Print first 4 pixel values
    for (int i = 0; i < 4; ++i)
        LOG_DEBUG("pixel[" << i << "] r=" << host[i].x << " g=" << host[i].y
                  << " b=" << host[i].z << " a=" << host[i].w);
    const float maxVal = std::max({maxR, maxG, maxB, maxA});

    std::vector<uint8_t> pixels(pixelCount * 4);
    for (size_t i = 0; i < pixelCount; ++i)
    {
        // Simple linear → sRGB (gamma 2.2 approximation) + clamp
        auto toU8 = [](float v) -> uint8_t {
            v = std::max(0.0f, std::min(1.0f, std::pow(v, 1.0f / 2.2f)));
            return static_cast<uint8_t>(v * 255.0f + 0.5f);
        };
        pixels[i * 4 + 0] = toU8(host[i].x);
        pixels[i * 4 + 1] = toU8(host[i].y);
        pixels[i * 4 + 2] = toU8(host[i].z);
        pixels[i * 4 + 3] = static_cast<uint8_t>(std::min(1.0f, host[i].w) * 255.0f + 0.5f);
    }
    if (stbi_write_png(path.c_str(), static_cast<int>(width), static_cast<int>(height), 4,
                       pixels.data(), static_cast<int>(width) * 4))
        LOG_DEBUG("Saved " << path);
    else
        LOG_ERROR("Failed to write " << path);
}

InteropFrame Raytracer::getInteropFrame() const
{
    return {0, lastReadyValue, renderReady.get(), bufferReleased.get()};
}

bool Raytracer::isRenderInFlight() const
{
    return lastReadyValue != 0 && renderReady
        && context.getDevice().getSemaphoreCounterValue(renderReady.get()) < lastReadyValue;
}

bool Raytracer::isFrameReady() const
{
    return lastReadyValue != 0 && renderReady
        && context.getDevice().getSemaphoreCounterValue(renderReady.get()) >= lastReadyValue;
}

float Raytracer::getGpuTimeMs()
{
    if (m_timingEnabled && m_eventsRecorded)
    {
        const cudaError_t status = cudaEventQuery(m_stopEvent.get());
        if (status == cudaErrorNotReady)
            return m_gpuTimeMs;
        NR_GPU_CHECK(status);
        NR_GPU_CHECK(cudaEventElapsedTime(&m_gpuTimeMs, m_startEvent.get(), m_stopEvent.get()));
    }
    return m_gpuTimeMs;
}


float Raytracer::getAverageNoiseVariance() const
{
    if (m_noiseVarianceSumHost == nullptr || m_noiseResultSampleCount < 2 || width == 0 || height == 0)
        return std::numeric_limits<float>::infinity();
    return *m_noiseVarianceSumHost / static_cast<float>(width * height);
}
