#include "Backend/OptiX/Runtime/Raytracer.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "stb_image_write.h"

#include <cuda_runtime.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <nvtx3/nvToolsExt.h>

#include "Rendering/Camera/CameraInstance.h"
#include "Rendering/Camera/GatherPsfSensor.h"
#include "Rendering/Camera/ScatterPsfSensor.h"
#include "Backend/CUDA/Checks.h"
#include "Backend/CUDA/rstd/Memory.h"
#include "Backend/OptiX/ABI/SceneData.h"
#include "Backend/OptiX/LightTreeBuilder.h"
#include "Log.h"
#include "Geometry/Mesh/Assets/MeshAsset.h"
#include "Scene/Objects/MeshInstance.h"
#include "Scene/Scene.h"
#include "Backend/Vulkan/Runtime/Context.h"

extern const float sRGBToSpectrumTable_Scale[64];
extern const float sRGBToSpectrumTable_Data[3][64][64][64][3];
#include "Materials/Shading/Spectrum.h"

namespace
{
constexpr unsigned char noorRayOptixIr[] = {
    #embed "../../../../../build/generated/NoorRayOptixIr.ptx"
};
constexpr std::size_t noorRayOptixIrLength = sizeof(noorRayOptixIr);

constexpr unsigned char noorRayOptixMissIr[] = {
    #embed "../../../../../build/generated/NoorRayOptixMissIr.ptx"
};
constexpr std::size_t noorRayOptixMissIrLength = sizeof(noorRayOptixMissIr);

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

Raytracer::Raytracer(
    Context& context,
    Scene& scene)
    : context(context), scene(scene)
{
    stream = context.getCudaStream();
    optixCtx = context.getOptixContext();
    if (stream == nullptr || optixCtx == nullptr)
        throw std::runtime_error("Backend/CUDA/OptiX not initialized in Context");

    OptixModuleCompileOptions moduleOptions = makeModuleCompileOptions();
    // The fixed SVM device program is compiled once; material graphs are
    // uploaded as bytecode records. Runtime work stays off the UI/render thread.
    pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipelineCompileOptions.numPayloadValues = 3;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
    pipelineCompileOptions.pipelineLaunchParamsSizeInBytes = sizeof(KernelParams);
    pipelineCompileOptions.usesPrimitiveTypeFlags =
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE
        | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    const OptixPipelineCompileOptions& pipelineOptions = pipelineCompileOptions;

    std::array<char, 8192> log{};
    size_t logSize = log.size();
    OptixResult result = optixModuleCreate(
        optixCtx, &moduleOptions, &pipelineOptions,
        reinterpret_cast<const char*>(noorRayOptixIr), noorRayOptixIrLength,
        log.data(), &logSize, optixModule.put());
    if (result != OPTIX_SUCCESS)
        throw std::runtime_error(std::string("OptiX module creation failed: ") + log.data());

    logSize = log.size();
    result = optixModuleCreate(
        optixCtx, &moduleOptions, &pipelineOptions,
        reinterpret_cast<const char*>(noorRayOptixMissIr), noorRayOptixMissIrLength,
        log.data(), &logSize, optixPathMissModule.put());
    if (result != OPTIX_SUCCESS)
        throw std::runtime_error(std::string("OptiX miss module creation failed: ") + log.data());

    optixPathTraceGroup.reset(makeRaygenGroup(
        optixCtx, optixModule.get(), "__raygen__pathTrace"));
    optixProxyOverdrawGroup.reset(makeRaygenGroup(
        optixCtx, optixModule.get(), "__raygen__gaussianProxyOverdraw"));

    OptixProgramGroupOptions groupOptions{};

    OptixProgramGroupDesc pathMeshDesc{};
    pathMeshDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pathMeshDesc.hitgroup.moduleCH = optixModule.get();
    pathMeshDesc.hitgroup.entryFunctionNameCH = "__closesthit__mesh";
    logSize = log.size();
    NR_OPTIX_CHECK(optixProgramGroupCreate(
        optixCtx, &pathMeshDesc, 1, &groupOptions, log.data(), &logSize,
        optixPathMeshHitGroup.put()));

    OptixProgramGroupDesc pathGaussianDesc{};
    pathGaussianDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pathGaussianDesc.hitgroup.moduleCH = optixModule.get();
    pathGaussianDesc.hitgroup.entryFunctionNameCH = "__closesthit__gaussian";
    pathGaussianDesc.hitgroup.moduleAH = optixModule.get();
    pathGaussianDesc.hitgroup.entryFunctionNameAH = "__anyhit__gaussian_path";
    logSize = log.size();
    NR_OPTIX_CHECK(optixProgramGroupCreate(
        optixCtx, &pathGaussianDesc, 1, &groupOptions, log.data(), &logSize,
        optixPathGaussianHitGroup.put()));

    OptixProgramGroupDesc lightDesc{};
    lightDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    lightDesc.hitgroup.moduleIS = optixModule.get();
    lightDesc.hitgroup.entryFunctionNameIS = "__intersection__analyticLight";
    lightDesc.hitgroup.moduleCH = optixModule.get();
    lightDesc.hitgroup.entryFunctionNameCH = "__closesthit__analyticLight";
    logSize = log.size();
    NR_OPTIX_CHECK(optixProgramGroupCreate(
        optixCtx, &lightDesc, 1, &groupOptions, log.data(), &logSize,
        optixAnalyticLightHitGroup.put()));

    OptixProgramGroupDesc triDesc{};
    triDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    logSize = log.size();
    NR_OPTIX_CHECK(optixProgramGroupCreate(
        optixCtx, &triDesc, 1, &groupOptions, log.data(), &logSize, optixTriangleGroup.put()));

    OptixProgramGroupDesc gaussianDesc{};
    gaussianDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    gaussianDesc.hitgroup.moduleAH = optixModule.get();
    gaussianDesc.hitgroup.entryFunctionNameAH = "__anyhit__gaussian_query";
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

    OptixProgramGroupDesc pathMissDesc{};
    pathMissDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pathMissDesc.miss.module = optixPathMissModule.get();
    pathMissDesc.miss.entryFunctionName = "__miss__pathTrace";
    logSize = log.size();
    NR_OPTIX_CHECK(optixProgramGroupCreate(
        optixCtx, &pathMissDesc, 1, &groupOptions, log.data(), &logSize,
        optixPathMissGroup.put()));



    rebuildPipeline();

    optixPathTraceRecord = uploadRecord(optixPathTraceGroup.get());
    rebuildMeshHitgroupSbt();
    optixMissRecord = uploadRecord(optixMissGroup.get());
    optixPathMissRecord = uploadRecord(optixPathMissGroup.get());

    optixPathTraceSbt.raygenRecord = optixPathTraceRecord.devicePtr();
    optixPathTraceSbt.missRecordBase = optixPathMissRecord.devicePtr();
    optixPathTraceSbt.missRecordStrideInBytes = sizeof(SbtRecord<>);
    optixPathTraceSbt.missRecordCount = 1;

    m_startEvent.create();
    m_stopEvent.create();
    m_frameCompleteEvent.create();

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

    optixLaunchParamsDevice.allocate(sizeof(KernelParams));

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
    pipelineOptions.usesPrimitiveTypeFlags =
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE
        | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

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
    optixProxyOverdrawSbt.missRecordBase = optixMissRecord.devicePtr();
    optixProxyOverdrawSbt.hitgroupRecordBase = optixProxyOverdrawHitgroupRecord.devicePtr();
    // This SBT has only mesh and proxy-Gaussian records. Do not inherit the
    // four-record count from the combined beauty/query SBT.
    optixProxyOverdrawSbt.hitgroupRecordCount = 2;
    // This proxy-overdraw pipeline does not use material callables.
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
    allGroups.push_back(optixPathMeshHitGroup.get());
    allGroups.push_back(optixPathGaussianHitGroup.get());
    allGroups.push_back(optixAnalyticLightHitGroup.get());
    allGroups.push_back(optixTriangleGroup.get());
    allGroups.push_back(optixGaussianHitGroup.get());
    allGroups.push_back(optixMissGroup.get());
    allGroups.push_back(optixPathMissGroup.get());

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
    SbtRecord<> pathMeshRecord{};
    NR_OPTIX_CHECK(optixSbtRecordPackHeader(
        optixPathMeshHitGroup.get(), &pathMeshRecord));
    SbtRecord<> pathGaussianRecord{};
    NR_OPTIX_CHECK(optixSbtRecordPackHeader(
        optixPathGaussianHitGroup.get(), &pathGaussianRecord));
    SbtRecord<> pathAnalyticLightRecord{};
    NR_OPTIX_CHECK(optixSbtRecordPackHeader(
        optixAnalyticLightHitGroup.get(), &pathAnalyticLightRecord));
    SbtRecord<> queryMeshRecord{};
    NR_OPTIX_CHECK(optixSbtRecordPackHeader(
        optixTriangleGroup.get(), &queryMeshRecord));
    SbtRecord<> queryGaussianRecord{};
    NR_OPTIX_CHECK(optixSbtRecordPackHeader(
        optixGaussianHitGroup.get(), &queryGaussianRecord));
    const std::array records{
        pathMeshRecord, pathGaussianRecord, pathAnalyticLightRecord,
        queryMeshRecord, queryGaussianRecord};
    pathHitgroupRecordBase.allocate(sizeof(records));
    NR_GPU_CHECK(cudaMemcpy(pathHitgroupRecordBase.get(), records.data(),
        sizeof(records), cudaMemcpyHostToDevice));
    optixPathTraceSbt.hitgroupRecordBase = pathHitgroupRecordBase.devicePtr();
    optixPathTraceSbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<>);
    optixPathTraceSbt.hitgroupRecordCount = static_cast<uint32_t>(records.size());

    optixPathTraceSbt.callablesRecordBase = 0;
    optixPathTraceSbt.callablesRecordStrideInBytes = 0;
    optixPathTraceSbt.callablesRecordCount = 0;
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
    // Program replacement can change opacity nodes without changing mesh or
    // light dirty flags. Disable per-material shadow shortcuts until the next
    // light/mesh upload recomputes their conservative classifications.
    gpuCache.data.allMaterialsOpaque = 0u;
    for (Material& material : gpuCache.materials)
        material.shadowOpaque = 0u;
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
    denoiser.reset();
    denoisedOutputAvailable = false;
    tlas.reset();
    freeSceneData();
    freeScratchBuffers();
    optixLaunchParamsDevice.reset();
    if (stream != nullptr)
        cudaStreamSynchronize(stream);
    m_startEvent.reset();
    m_stopEvent.reset();
    m_frameCompleteEvent.reset();
    cudaRenderReady.reset();
    cudaBufferReleased.reset();
    optixPathTraceRecord.reset();
    optixProxyOverdrawRecord.reset();
    pathHitgroupRecordBase.reset();
    optixProxyOverdrawHitgroupRecord.reset();
    optixMissRecord.reset();
    optixPathMissRecord.reset();
    clearMaterialXPrograms();

    optixPipeline.reset();
    optixProxyOverdrawPipeline.reset();
    optixPathMissGroup.reset();
    optixMissGroup.reset();
    optixPathGaussianHitGroup.reset();
    optixPathMeshHitGroup.reset();
    optixGaussianHitGroup.reset();
    optixProxyOverdrawHitGroup.reset();
    optixTriangleGroup.reset();
    optixProxyOverdrawGroup.reset();
    optixPathTraceGroup.reset();
    optixModule.reset();
    optixPathMissModule.reset();
    optixPathTraceSbt = {};
    optixProxyOverdrawSbt = {};
}

void Raytracer::setAovEnabled(const bool enabled)
{
    if (enabled == aovEnabled)
        return;
    aovEnabled = enabled;
    if (!enabled)
    {
        // The images stay allocated: a host that bound them into a descriptor
        // set keeps that binding valid. Only the per-frame AOV refresh stops.
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
    consumedReadyValue = 0;
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
    denoised.create(context,
        sizeof(glm::vec4) * static_cast<size_t>(width) * height,
        vk::BufferUsageFlagBits::eStorageBuffer
        | vk::BufferUsageFlagBits::eTransferSrc
        | vk::BufferUsageFlagBits::eTransferDst);
    accumulationBuffer.allocate(sizeof(glm::vec4) * static_cast<size_t>(width) * height, stream);
    NR_GPU_CHECK(cudaMemsetAsync(
        accumulationBuffer.get(), 0, sizeof(glm::vec4) * width * height, stream));
}

void Raytracer::freeScratchBuffers() noexcept
{
    denoised.reset();
    accumulationBuffer.reset();
    denoiserAlbedoGuideBuffer.reset();
    denoiserNormalGuideBuffer.reset();
    denoiserGuidesStale = true;
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
    lightAliasDevice.reset();
    lightTreeDevice.reset();
    directLightCandidateDevice.reset();
    meshLightGeometryDevice.reset();
    meshLightCandidateOffsetsDevice.reset();
    meshLightCandidateIndicesDevice.reset();
    analyticLightBvhCandidateDevice.reset();
    meshLightBvhCandidateDevice.reset();
    analyticLightBlas.reset();
    meshLightBlas.reset();
    gpuCache.clearSceneResources();
}

void Raytracer::updateTextures()
{
    const auto& cpuTextures = scene.getTextures();
    const size_t count = cpuTextures.size();
    auto& cudaTextures = gpuCache.textures;
    auto& mirroredHandles = gpuCache.textureHandles;

    bool changed =
        cudaTextures.size() != count || mirroredHandles.size() != count;
    const size_t commonCount = std::min(mirroredHandles.size(), count);
    for (size_t i = 0; i < commonCount && !changed; ++i)
        changed = mirroredHandles[i] != scene.getTextureHandle(
            static_cast<uint32_t>(i));

    if (!changed) {
        gpuCache.textureRevision = scene.getTextureRevision();
        return;
    }

    // Resize can relocate the managed wrapper array, and changed slots can
    // destroy CUDA texture objects. Batch that lifetime boundary into one
    // synchronization, then leave every unchanged texture array intact.
    NR_GPU_CHECK(cudaStreamSynchronize(stream));
    cudaTextures.resize(count);
    mirroredHandles.resize(count);

    for (size_t i = 0; i < count; ++i) {
        const TextureHandle current = scene.getTextureHandle(
            static_cast<uint32_t>(i));
        if (mirroredHandles[i] == current)
            continue;

        cudaTextures[i].reset();
        mirroredHandles[i] = current;
        // Texture slots are scene-owned and remain dense until the scene is
        // cleared, but keep this validity check for stale handles during a
        // scene-generation transition.
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
    gpuCache.textureRevision = scene.getTextureRevision();

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
    // Geometry/Mesh/material *bindings* (which program a given mesh slot points at)
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
    tlas.build(optixCtx, stream, scene, gpuInstances,
        analyticLightBlas.getTraversable(),
        gpuCache.data.analyticLightBvhPrimitiveCount,
        meshLightBlas.getTraversable(),
        gpuCache.data.meshLightBvhPrimitiveCount);
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
    const PointLight* pointLights = scene.getPointLights();
    const SpotLight* spotLights = scene.getSpotLights();
    const RectLight* rectLights = scene.getRectLights();
    const DirectionalLight* directionalLights = scene.getDirectionalLights();
    const uint32_t pointLightCount = scene.getPointLightCount();
    const uint32_t spotLightCount = scene.getSpotLightCount();
    const uint32_t rectLightCount = scene.getRectLightCount();
    const uint32_t directionalLightCount = scene.getDirectionalLightCount();
    gpuCache.data.directionalLightCount = directionalLightCount;
    gpuCache.data.directionalLightCandidateOffset = pointLightCount
        + spotLightCount + rectLightCount;

    std::vector<DirectLightCandidate> candidates;
    std::vector<MeshLightGeometry> meshLightGeometry;
    std::vector<float> weights;
    std::vector<OptixAabb> analyticLightBvhAabbs;
    std::vector<uint32_t> analyticLightBvhCandidateIndices;
    std::vector<OptixAabb> meshLightBvhAabbs;
    std::vector<uint32_t> meshLightBvhCandidateIndices;
    candidates.reserve(static_cast<size_t>(pointLightCount)
        + spotLightCount + rectLightCount + directionalLightCount);
    meshLightGeometry.reserve(candidates.capacity());
    weights.reserve(candidates.capacity());
    analyticLightBvhAabbs.reserve(candidates.capacity());
    analyticLightBvhCandidateIndices.reserve(candidates.capacity());
    const auto appendLightBvhAabb = [&](std::vector<OptixAabb>& aabbs,
                                        std::vector<uint32_t>& candidateIndices,
                                        const glm::vec3 minimum,
                                        const glm::vec3 maximum,
                                        const uint32_t candidateIndex) {
        if (!nr::isFinite(minimum) || !nr::isFinite(maximum)
            || candidateIndex == InvalidIndex)
            return;
        constexpr float epsilon = 1.0e-5f;
        aabbs.push_back({
            minimum.x - epsilon, minimum.y - epsilon, minimum.z - epsilon,
            maximum.x + epsilon, maximum.y + epsilon, maximum.z + epsilon});
        candidateIndices.push_back(candidateIndex);
    };
    const auto appendCandidate = [&](const DirectLightType type,
                                     const uint32_t index,
                                     const glm::vec3 position,
                                     const glm::vec3 normal,
                                     const float area,
                                     const float powerBound,
                                     const float spatialRadius,
                                     const float orientationBound,
                                     const glm::vec3 tangent = glm::vec3(0.0f),
                                     const float width = 0.0f,
                                     const float height = 0.0f,
                                     const uint32_t twoSided = 0u) {
        const float weight = fmaxf(powerBound, 0.0f);
        DirectLightCandidate candidate{};
        candidate.type = type;
        candidate.index = index;
        candidate.instanceIndex = InvalidIndex;
        candidate.primitiveIndex = InvalidIndex;
        candidate.position = position;
        candidate.area = area;
        candidate.normal = normal;
        candidate.selectionWeight = weight;
        candidate.powerBound = weight;
        candidate.spatialRadius = spatialRadius;
        candidate.orientationBound = orientationBound;
        candidate.tangent = tangent;
        candidate.width = width;
        candidate.height = height;
        candidate.twoSided = twoSided;
        candidates.push_back(candidate);
        meshLightGeometry.push_back({});
        weights.push_back(weight);
        return static_cast<uint32_t>(candidates.size() - 1u);
    };
    for (uint32_t i = 0; i < pointLightCount; ++i)
    {
        const uint32_t candidateIndex = appendCandidate(
            DirectLightType::Point, i, pointLights[i].position,
            glm::vec3(0.0f), 0.0f, pointLights[i].selectionWeight(),
            fmaxf(pointLights[i].softRadius, 0.0f), 1.0f);
        candidates[candidateIndex].color = pointLights[i].color;
        candidates[candidateIndex].intensity = pointLights[i].intensity;
        const float radius = fmaxf(pointLights[i].softRadius, 0.0f);
        if (radius > 0.0f)
            appendLightBvhAabb(
                analyticLightBvhAabbs, analyticLightBvhCandidateIndices,
                pointLights[i].position - glm::vec3(radius),
                pointLights[i].position + glm::vec3(radius), candidateIndex);
    }
    for (uint32_t i = 0; i < spotLightCount; ++i)
    {
        const uint32_t candidateIndex = appendCandidate(
            DirectLightType::Spot, i, spotLights[i].position,
            glm::normalize(spotLights[i].direction), 0.0f,
            spotLights[i].selectionWeight(),
            fmaxf(spotLights[i].softRadius, 0.0f), 1.0f);
        DirectLightCandidate& candidate = candidates[candidateIndex];
        candidate.color = spotLights[i].color;
        candidate.intensity = spotLights[i].intensity;
        const float innerAngle = fminf(
            spotLights[i].innerConeAngle, spotLights[i].outerConeAngle)
            * LightPi / 180.0f;
        const float outerAngle = fmaxf(
            spotLights[i].innerConeAngle, spotLights[i].outerConeAngle)
            * LightPi / 180.0f;
        candidate.innerCos = cosf(innerAngle);
        candidate.outerCos = cosf(outerAngle);
        candidate.invConeCosineRange = 1.0f
            / fmaxf(candidate.innerCos - candidate.outerCos, 1.0e-5f);
        const float radius = fmaxf(spotLights[i].softRadius, 0.0f);
        if (radius > 0.0f)
            appendLightBvhAabb(
                analyticLightBvhAabbs, analyticLightBvhCandidateIndices,
                spotLights[i].position - glm::vec3(radius),
                spotLights[i].position + glm::vec3(radius), candidateIndex);
    }
    for (uint32_t i = 0; i < rectLightCount; ++i)
    {
        const glm::vec3 normal = nr::safeNormalize(rectLights[i].direction);
        const glm::vec3 tangent = nr::safeNormalize(rectLights[i].tangent);
        const glm::vec3 bitangent = nr::safeNormalize(glm::cross(normal, tangent));
        const uint32_t candidateIndex = appendCandidate(
            DirectLightType::Rect, i, rectLights[i].position,
            normal,
            fmaxf(rectLights[i].width * rectLights[i].height, 0.0f),
            rectLights[i].selectionWeight(),
            0.5f * std::sqrt(
                rectLights[i].width * rectLights[i].width
                + rectLights[i].height * rectLights[i].height), 1.0f,
            tangent,
            fmaxf(rectLights[i].width, 0.0f),
            fmaxf(rectLights[i].height, 0.0f),
            rectLights[i].twoSided != 0 ? 1u : 0u);
        DirectLightCandidate& candidate = candidates[candidateIndex];
        candidate.bitangent = bitangent;
        candidate.color = rectLights[i].color;
        candidate.intensity = rectLights[i].intensity;
        candidate.barnDoorLength = fmaxf(rectLights[i].barnDoorLength, 0.0f);
        candidate.barnDoorEnabled = candidate.barnDoorLength > 0.0f
            && rectLights[i].barnDoorAngle < 89.9f ? 1u : 0u;
        candidate.barnDoorExpansion = candidate.barnDoorEnabled != 0u
            ? candidate.barnDoorLength * tanf(fmaxf(
                rectLights[i].barnDoorAngle, 0.0f) * LightPi / 180.0f) : 0.0f;
        if (rectLights[i].width > 0.0f && rectLights[i].height > 0.0f)
        {
            const glm::vec3 halfU = tangent * (0.5f * rectLights[i].width);
            const glm::vec3 halfV = bitangent * (0.5f * rectLights[i].height);
            const std::array corners{
                rectLights[i].position + halfU + halfV,
                rectLights[i].position + halfU - halfV,
                rectLights[i].position - halfU + halfV,
                rectLights[i].position - halfU - halfV};
            glm::vec3 minimum = corners[0];
            glm::vec3 maximum = corners[0];
            for (size_t corner = 1; corner < corners.size(); ++corner)
            {
                minimum = glm::min(minimum, corners[corner]);
                maximum = glm::max(maximum, corners[corner]);
            }
            appendLightBvhAabb(analyticLightBvhAabbs,
                analyticLightBvhCandidateIndices, minimum, maximum,
                candidateIndex);
        }
    }
    for (uint32_t i = 0; i < directionalLightCount; ++i)
    {
        const uint32_t candidateIndex = appendCandidate(
            DirectLightType::Directional, i, glm::vec3(0.0f),
            -glm::normalize(directionalLights[i].direction), 0.0f,
            directionalLights[i].selectionWeight(), 0.0f, 1.0f);
        DirectLightCandidate& candidate = candidates[candidateIndex];
        candidate.color = directionalLights[i].color;
        candidate.intensity = directionalLights[i].intensity;
        const float halfAngle = 0.5f * fminf(
            fmaxf(directionalLights[i].softAngle, 0.0f), 180.0f)
            * LightPi / 180.0f;
        candidate.coneOneMinusCosine = oneMinusCosine(halfAngle);
        candidate.coneProjectedArea = LightPi * sinf(halfAngle) * sinf(halfAngle);
    }

    // Match Cycles' non-light-tree distribution: sample emissive mesh
    // triangles proportional to world-space area, sample analytic lights
    // uniformly, and split probability equally between those two groups when
    // both are present.  In particular, do not use a conservative shader
    // bound as the proposal: textures and procedurals routinely make that
    // bound many orders of magnitude larger than the actual emission.
    const uint32_t analyticCandidateCount = static_cast<uint32_t>(candidates.size());

    // MaterialX emission can be texture- or position-dependent, so the host
    // cannot reduce this list to a single scalar per material.  It can still
    // identify programs that contain an EDF and put every such triangle in
    // the candidate set; the kernel evaluates the actual emission at the
    // sampled barycentric point.
    const auto& meshInstances = scene.getMeshInstances();
    const auto& materials = scene.getMaterials();
    const auto& words = materialxPrograms.words();
    bool allMaterialsOpaque = true;
    constexpr float MaxEmitterLuminance = 1.0e6f;
    constexpr float MaxSelectionWeight = 1.0e30f;
    const auto inputAbsBound = [&](const uint32_t word) {
        if (nr::svm::isStackOffset(word))
            return MaxEmitterLuminance;
        const float value = std::bit_cast<float>(word);
        return std::isfinite(value)
            ? std::min(std::abs(value), MaxEmitterLuminance) : MaxEmitterLuminance;
    };
    const auto colorAbsBound = [&](const uint32_t x, const uint32_t y,
                                   const uint32_t z) {
        return std::max({inputAbsBound(x), inputAbsBound(y), inputAbsBound(z)});
    };
    const auto nodeWordCount = [&](const nr::svm::NodeType type) -> size_t {
        using nr::svm::NodeType;
        switch (type)
        {
        case NodeType::End: return 1;
        case NodeType::Math: return 1 + sizeof(nr::svm::NodeMath) / sizeof(uint32_t);
        case NodeType::VectorMath: return 1 + sizeof(nr::svm::NodeVectorMath) / sizeof(uint32_t);
        case NodeType::Mix: return 1 + sizeof(nr::svm::NodeMix) / sizeof(uint32_t);
        case NodeType::Clamp: return 1 + sizeof(nr::svm::NodeClamp) / sizeof(uint32_t);
        case NodeType::RemapRange: return 1 + sizeof(nr::svm::NodeRemapRange) / sizeof(uint32_t);
        case NodeType::Range: return 1 + sizeof(nr::svm::NodeRange) / sizeof(uint32_t);
        case NodeType::Range4: return 1 + sizeof(nr::svm::NodeRange4) / sizeof(uint32_t);
        case NodeType::Gamma: return 1 + sizeof(nr::svm::NodeGamma) / sizeof(uint32_t);
        case NodeType::HsvAdjust: return 1 + sizeof(nr::svm::NodeHsvAdjust) / sizeof(uint32_t);
        case NodeType::Invert: return 1 + sizeof(nr::svm::NodeInvert) / sizeof(uint32_t);
        case NodeType::Contrast: return 1 + sizeof(nr::svm::NodeContrast) / sizeof(uint32_t);
        case NodeType::Saturate: return 1 + sizeof(nr::svm::NodeSaturate) / sizeof(uint32_t);
        case NodeType::Blackbody: return 1 + sizeof(nr::svm::NodeBlackbody) / sizeof(uint32_t);
        case NodeType::SeparateColor: return 1 + sizeof(nr::svm::NodeSeparateColor) / sizeof(uint32_t);
        case NodeType::CombineColor: return 1 + sizeof(nr::svm::NodeCombineColor) / sizeof(uint32_t);
        case NodeType::CombineColor2: return 1 + sizeof(nr::svm::NodeCombineColor2) / sizeof(uint32_t);
        case NodeType::CombineColor4: return 1 + sizeof(nr::svm::NodeCombineColor4) / sizeof(uint32_t);
        case NodeType::SeparateColor4: return 1 + sizeof(nr::svm::NodeSeparateColor4) / sizeof(uint32_t);
        case NodeType::Premultiply:
        case NodeType::Unpremultiply: return 1 + sizeof(nr::svm::NodeColor4Op) / sizeof(uint32_t);
        case NodeType::MatrixValue: return 1 + sizeof(nr::svm::NodeMatrixValue) / sizeof(uint32_t);
        case NodeType::MatrixCompose: return 1 + sizeof(nr::svm::NodeMatrixCompose) / sizeof(uint32_t);
        case NodeType::TransformMatrix: return 1 + sizeof(nr::svm::NodeTransformMatrix) / sizeof(uint32_t);
        case NodeType::MatrixBinary: return 1 + sizeof(nr::svm::NodeMatrixBinary) / sizeof(uint32_t);
        case NodeType::MatrixUnary: return 1 + sizeof(nr::svm::NodeMatrixUnary) / sizeof(uint32_t);
        case NodeType::MatrixDeterminant: return 1 + sizeof(nr::svm::NodeMatrixDeterminant) / sizeof(uint32_t);
        case NodeType::MatrixSelect: return 1 + sizeof(nr::svm::NodeMatrixSelect) / sizeof(uint32_t);
        case NodeType::Transform: return 1 + sizeof(nr::svm::NodeTransform) / sizeof(uint32_t);
        case NodeType::TexCoord: return 1 + sizeof(nr::svm::NodeTexCoord) / sizeof(uint32_t);
        case NodeType::Mapping: return 1 + sizeof(nr::svm::NodeMapping) / sizeof(uint32_t);
        case NodeType::GradientTexture: return 1 + sizeof(nr::svm::NodeGradientTexture) / sizeof(uint32_t);
        case NodeType::Rotate2d: return 1 + sizeof(nr::svm::NodeRotate2d) / sizeof(uint32_t);
        case NodeType::Time: return 1 + sizeof(nr::svm::NodeTime) / sizeof(uint32_t);
        case NodeType::Rotate3d: return 1 + sizeof(nr::svm::NodeRotate3d) / sizeof(uint32_t);
        case NodeType::ImageTexture: return 1 + sizeof(nr::svm::NodeImageTexture) / sizeof(uint32_t);
        case NodeType::FractalNoiseTexture: return 1 + sizeof(nr::svm::NodeProceduralTexture) / sizeof(uint32_t);
        case NodeType::NoiseTexture: return 1 + sizeof(nr::svm::NodeNoiseTexture) / sizeof(uint32_t);
        case NodeType::WorleyNoiseTexture: return 1 + sizeof(nr::svm::NodeWorleyNoiseTexture) / sizeof(uint32_t);
        case NodeType::CellNoiseTexture: return 1 + sizeof(nr::svm::NodeCellNoiseTexture) / sizeof(uint32_t);
        case NodeType::UnifiedNoiseTexture: return 1 + sizeof(nr::svm::NodeUnifiedNoiseTexture) / sizeof(uint32_t);
        case NodeType::CheckerTexture: return 1 + sizeof(nr::svm::NodeCheckerTexture) / sizeof(uint32_t);
        case NodeType::NormalMap: return 1 + sizeof(nr::svm::NodeNormalMap) / sizeof(uint32_t);
        case NodeType::Bump: return 1 + sizeof(nr::svm::NodeBump) / sizeof(uint32_t);
        case NodeType::ClosureWeight: return 1 + sizeof(nr::svm::NodeClosureWeight) / sizeof(uint32_t);
        case NodeType::ClosureDiffuseBsdf: return 1 + sizeof(nr::svm::NodeClosureDiffuseBsdf) / sizeof(uint32_t);
        case NodeType::ClosureConductorBsdf: return 1 + sizeof(nr::svm::NodeClosureConductorBsdf) / sizeof(uint32_t);
        case NodeType::ClosureDielectricBsdf: return 1 + sizeof(nr::svm::NodeClosureDielectricBsdf) / sizeof(uint32_t);
        case NodeType::ClosureSheenBsdf: return 1 + sizeof(nr::svm::NodeClosureSheenBsdf) / sizeof(uint32_t);
        case NodeType::ClosureSubsurfaceBsdf: return 1 + sizeof(nr::svm::NodeClosureSubsurfaceBsdf) / sizeof(uint32_t);
        case NodeType::ClosureUniformEdf: return 1 + sizeof(nr::svm::NodeClosureUniformEdf) / sizeof(uint32_t);
        case NodeType::ClosureOpenPbrSurface: return 1 + sizeof(nr::svm::NodeClosureOpenPbrSurface) / sizeof(uint32_t);
        case NodeType::ArtisticIor: return 1 + sizeof(nr::svm::NodeArtisticIor) / sizeof(uint32_t);
        case NodeType::JumpIfZero:
        case NodeType::JumpIfOne: return 1 + sizeof(nr::svm::NodeJump) / sizeof(uint32_t);
        case NodeType::SurfaceOutput: return 1 + sizeof(nr::svm::NodeSurfaceOutput) / sizeof(uint32_t);
        default: return 0;
        }
    };
    const auto literalEquals = [&](const uint32_t word, const float expected) {
        if (nr::svm::isStackOffset(word))
            return false;
        const float value = std::bit_cast<float>(word);
        return std::isfinite(value) && value == expected;
    };
    const auto shadowOpaqueFor = [&](const uint32_t materialIndex) {
        if (materialIndex >= materials.size())
            return false;
        const Material& material = materials[materialIndex];
        if (material.svmBytecodeLength == 0)
            return true;
        if (material.svmBytecodeOffset > words.size()
            || material.svmBytecodeLength > words.size() - material.svmBytecodeOffset)
            return false;
        const uint32_t* begin = words.data() + material.svmBytecodeOffset;
        const size_t length = material.svmBytecodeLength;
        for (size_t i = 0; i < length;)
        {
            const auto type = static_cast<nr::svm::NodeType>(begin[i]);
            const size_t wordsForNode = nodeWordCount(type);
            if (wordsForNode == 0 || wordsForNode > length - i)
                return false;
            if (type == nr::svm::NodeType::SurfaceOutput)
            {
                nr::svm::NodeSurfaceOutput node{};
                std::memcpy(&node, begin + i + 1u, sizeof(node));
                if (!literalEquals(node.opacity, 1.0f))
                    return false;
            }
            else if (type == nr::svm::NodeType::ClosureOpenPbrSurface)
            {
                nr::svm::NodeClosureOpenPbrSurface node{};
                std::memcpy(&node, begin + i + 1u, sizeof(node));
                if (!literalEquals(node.opacity, 1.0f)
                    || !literalEquals(node.transmissionWeight, 0.0f))
                    return false;
            }
            else if (type == nr::svm::NodeType::ClosureDielectricBsdf)
            {
                nr::svm::NodeClosureDielectricBsdf node{};
                std::memcpy(&node, begin + i + 1u, sizeof(node));
                if (!literalEquals(node.transmission, 0.0f))
                    return false;
            }
            else if (type == nr::svm::NodeType::ClosureDiffuseBsdf)
            {
                nr::svm::NodeClosureDiffuseBsdf node{};
                std::memcpy(&node, begin + i + 1u, sizeof(node));
                if (node.translucent != 0u)
                    return false;
            }
            if (type == nr::svm::NodeType::End)
                break;
            i += wordsForNode;
        }
        return true;
    };
    for (uint32_t materialIndex = 0;
         materialIndex < materials.size()
         && materialIndex < gpuCache.materials.size(); ++materialIndex)
        gpuCache.materials[materialIndex].shadowOpaque =
            shadowOpaqueFor(materialIndex) ? 1u : 0u;

    const auto emissionBound = [&](const uint32_t materialIndex) {
        if (materialIndex >= materials.size())
            return 0.0f;
        const Material& material = materials[materialIndex];
        if (material.svmBytecodeOffset > words.size()
            || material.svmBytecodeLength > words.size() - material.svmBytecodeOffset)
            return 0.0f;
        const uint32_t* begin = words.data() + material.svmBytecodeOffset;
        const size_t length = material.svmBytecodeLength;
        float bound = 0.0f;
        float closureWeightBound = 1.0f;
        for (size_t i = 0; i < length;)
        {
            const auto type = static_cast<nr::svm::NodeType>(begin[i]);
            const size_t wordsForNode = nodeWordCount(type);
            if (wordsForNode == 0 || wordsForNode > length - i)
                return MaxEmitterLuminance;
            if (type == nr::svm::NodeType::SurfaceOutput
                || type == nr::svm::NodeType::ClosureOpenPbrSurface)
                allMaterialsOpaque = false;
            if (type == nr::svm::NodeType::End)
                break;
            if (type == nr::svm::NodeType::ClosureWeight)
            {
                nr::svm::NodeClosureWeight node{};
                std::memcpy(&node, begin + i + 1u, sizeof(node));
                closureWeightBound = colorAbsBound(
                    node.weightX, node.weightY, node.weightZ);
            }
            else if (type == nr::svm::NodeType::ClosureUniformEdf)
            {
                nr::svm::NodeClosureUniformEdf node{};
                std::memcpy(&node, begin + i + 1u, sizeof(node));
                const float estimate = colorAbsBound(
                    node.colorX, node.colorY, node.colorZ)
                    * inputAbsBound(node.strength) * closureWeightBound;
                bound = std::min(MaxEmitterLuminance, bound + estimate);
            }
            else if (type == nr::svm::NodeType::ClosureOpenPbrSurface)
            {
                nr::svm::NodeClosureOpenPbrSurface node{};
                std::memcpy(&node, begin + i + 1u, sizeof(node));
                const float estimate = colorAbsBound(
                    node.emissionColorX, node.emissionColorY,
                    node.emissionColorZ) * inputAbsBound(node.emissionLuminance)
                    * closureWeightBound;
                bound = std::min(MaxEmitterLuminance, bound + estimate);
            }
            i += wordsForNode;
        }
        return std::isfinite(bound) ? bound : MaxEmitterLuminance;
    };
    std::vector<float> materialEmissionBounds(materials.size(), 0.0f);
    for (uint32_t materialIndex = 0; materialIndex < materials.size(); ++materialIndex)
        materialEmissionBounds[materialIndex] = emissionBound(materialIndex);
    std::vector<uint32_t> meshCandidateOffsets;
    std::vector<uint32_t> meshCandidateIndices;
    meshCandidateOffsets.reserve(meshInstances.size() + 1u);
    meshCandidateOffsets.push_back(0);
    for (uint32_t instanceIndex = 0;
         instanceIndex < meshInstances.size(); ++instanceIndex)
    {
        const MeshInstance& instance = *meshInstances[instanceIndex];
        const MeshAsset& mesh = instance.getMeshAsset();
        const glm::mat4 objectToWorld = instance.getWorldTransform().getMatrix();
        const auto& vertices = mesh.getVertices();
        const auto& indices = mesh.getIndices();
        const auto& faces = mesh.getFaces();
        const uint32_t lookupBegin = static_cast<uint32_t>(meshCandidateIndices.size());
        meshCandidateIndices.resize(meshCandidateIndices.size() + faces.size(), InvalidIndex);
        for (uint32_t primitive = 0; primitive * 3u + 2u < indices.size()
             && primitive < faces.size(); ++primitive)
        {
            const Face face = faces[primitive];
            if (face.materialIndex < 0
                || static_cast<size_t>(face.materialIndex) >= mesh.getMaterialCount()
                || materialEmissionBounds[mesh.getMaterialIds()[face.materialIndex]] <= 0.0f)
                continue;
            const uint32_t ia = indices[primitive * 3u];
            const uint32_t ib = indices[primitive * 3u + 1u];
            const uint32_t ic = indices[primitive * 3u + 2u];
            if (ia >= vertices.size() || ib >= vertices.size() || ic >= vertices.size())
                continue;
            const glm::vec3 a = glm::vec3(objectToWorld * glm::vec4(vertices[ia].position, 1.0f));
            const glm::vec3 b = glm::vec3(objectToWorld * glm::vec4(vertices[ib].position, 1.0f));
            const glm::vec3 c = glm::vec3(objectToWorld * glm::vec4(vertices[ic].position, 1.0f));
            const glm::vec3 crossProduct = glm::cross(b - a, c - a);
            const float area = 0.5f * glm::length(crossProduct);
            if (area <= 0.0f)
                continue;
            const glm::vec3 center = (a + b + c) / 3.0f;
            const float radius = std::max({
                glm::length(a - center), glm::length(b - center),
                glm::length(c - center)});
            const float powerBound = area * LightPi * materialEmissionBounds[
                mesh.getMaterialIds()[face.materialIndex]];
            DirectLightCandidate candidate{};
            candidate.type = DirectLightType::MeshTriangle;
            candidate.index = static_cast<uint32_t>(candidates.size());
            candidate.instanceIndex = instanceIndex;
            candidate.primitiveIndex = primitive;
            candidate.position = center;
            candidate.area = area;
            candidate.normal = glm::normalize(crossProduct);
            candidate.selectionWeight = fmaxf(powerBound, 0.0f);
            candidate.powerBound = fmaxf(powerBound, 0.0f);
            candidate.spatialRadius = radius;
            candidate.orientationBound = 1.0f;
            meshLightGeometry.push_back({a, b, c});
            meshCandidateIndices[lookupBegin + primitive] =
                static_cast<uint32_t>(candidates.size());
            const uint32_t candidateIndex =
                static_cast<uint32_t>(candidates.size());
            candidates.push_back(candidate);
            weights.push_back(fmaxf(powerBound, 0.0f));
            appendLightBvhAabb(
                meshLightBvhAabbs, meshLightBvhCandidateIndices,
                glm::min(glm::min(a, b), c),
                glm::max(glm::max(a, b), c), candidateIndex);
        }
        meshCandidateOffsets.push_back(static_cast<uint32_t>(meshCandidateIndices.size()));
    }

    // Keep the proposal distribution radiometric. The previous area-only
    // mesh fallback discarded the material emission bound here, making a
    // small bright triangle and a large dim triangle equally likely per unit
    // area. Conservative bounds are still clamped below, while zero-power
    // candidates retain a tiny geometric fallback so they cannot poison the
    // alias/tree construction.
    for (size_t i = 0; i < candidates.size(); ++i) {
        const float fallback = candidates[i].type == DirectLightType::MeshTriangle
            ? candidates[i].area : 1.0f;
        weights[i] = candidates[i].powerBound > 0.0f
            ? candidates[i].powerBound : fallback;
    }

    double finiteWeightDouble = 0.0;
    for (float& weight : weights)
    {
        weight = std::isfinite(weight)
            ? std::clamp(weight, 0.0f, MaxSelectionWeight) : 0.0f;
        finiteWeightDouble += static_cast<double>(weight);
    }
    if (finiteWeightDouble > MaxSelectionWeight)
    {
        const float scale = static_cast<float>(MaxSelectionWeight / finiteWeightDouble);
        finiteWeightDouble = 0.0;
        for (float& weight : weights)
        {
            weight *= scale;
            finiteWeightDouble += static_cast<double>(weight);
        }
    }
    for (size_t i = 0; i < candidates.size(); ++i)
        candidates[i].selectionWeight = weights[i];
    std::vector<LightTreeNode> lightTreeNodes;
    nr::light_tree::build(candidates, lightTreeNodes);
    const float finiteWeight = static_cast<float>(finiteWeightDouble);
    gpuCache.data.lightSelectionWeight = finiteWeight;
    const float environmentWeight = std::max(
        scene.getEnvironment().importanceWeight, 0.0f);
    const float totalLightWeight = finiteWeight + environmentWeight;
    gpuCache.data.finiteLightProbability = totalLightWeight > 0.0f
        ? finiteWeight / totalLightWeight : 0.0f;
    gpuCache.data.environmentLightProbability = totalLightWeight > 0.0f
        ? environmentWeight / totalLightWeight : 0.0f;
    gpuCache.data.allMaterialsOpaque = allMaterialsOpaque ? 1u : 0u;

    lightAliasDevice.reset();
    lightTreeDevice.reset();
    gpuCache.data.lightAliases = nullptr;
    gpuCache.data.lightAliasCount = 0;
    gpuCache.data.lightTreeNodes = nullptr;
    gpuCache.data.lightTreeNodeCount = 0;
    directLightCandidateDevice.reset();
    meshLightGeometryDevice.reset();
    meshLightCandidateOffsetsDevice.reset();
    meshLightCandidateIndicesDevice.reset();
    analyticLightBvhCandidateDevice.reset();
    meshLightBvhCandidateDevice.reset();
    analyticLightBlas.reset();
    meshLightBlas.reset();
    gpuCache.data.meshLightCandidateOffsets = nullptr;
    gpuCache.data.meshLightCandidateIndices = nullptr;
    gpuCache.data.meshLightCandidateIndexCount = 0;
    gpuCache.data.meshLightInstanceCount = 0;
    gpuCache.data.directLightCandidates = nullptr;
    gpuCache.data.meshLightGeometry = nullptr;
    gpuCache.data.directLightCandidateCount = static_cast<uint32_t>(candidates.size());
    gpuCache.data.analyticLightBvhCandidateIndices = nullptr;
    gpuCache.data.meshLightBvhCandidateIndices = nullptr;
    gpuCache.data.analyticLightBvhPrimitiveCount = 0;
    gpuCache.data.meshLightBvhPrimitiveCount = 0;
    if (!lightTreeNodes.empty())
    {
        lightTreeDevice.allocate(sizeof(LightTreeNode) * lightTreeNodes.size());
        NR_GPU_CHECK(cudaMemcpy(lightTreeDevice.get(), lightTreeNodes.data(),
            sizeof(LightTreeNode) * lightTreeNodes.size(), cudaMemcpyHostToDevice));
        gpuCache.data.lightTreeNodes =
            static_cast<const LightTreeNode*>(lightTreeDevice.get());
        gpuCache.data.lightTreeNodeCount =
            static_cast<uint32_t>(lightTreeNodes.size());
    }
    if (!candidates.empty())
    {
        directLightCandidateDevice.allocate(sizeof(DirectLightCandidate) * candidates.size());
        NR_GPU_CHECK(cudaMemcpy(directLightCandidateDevice.get(), candidates.data(),
            sizeof(DirectLightCandidate) * candidates.size(), cudaMemcpyHostToDevice));
        gpuCache.data.directLightCandidates =
            static_cast<const DirectLightCandidate*>(directLightCandidateDevice.get());
        meshLightGeometryDevice.allocate(
            sizeof(MeshLightGeometry) * meshLightGeometry.size());
        NR_GPU_CHECK(cudaMemcpy(meshLightGeometryDevice.get(), meshLightGeometry.data(),
            sizeof(MeshLightGeometry) * meshLightGeometry.size(),
            cudaMemcpyHostToDevice));
        gpuCache.data.meshLightGeometry =
            static_cast<const MeshLightGeometry*>(meshLightGeometryDevice.get());
    }
    if (!meshCandidateOffsets.empty())
    {
        meshLightCandidateOffsetsDevice.allocate(
            sizeof(uint32_t) * meshCandidateOffsets.size());
        NR_GPU_CHECK(cudaMemcpy(meshLightCandidateOffsetsDevice.get(),
            meshCandidateOffsets.data(), sizeof(uint32_t) * meshCandidateOffsets.size(),
            cudaMemcpyHostToDevice));
        gpuCache.data.meshLightCandidateOffsets =
            static_cast<const uint32_t*>(meshLightCandidateOffsetsDevice.get());
        gpuCache.data.meshLightInstanceCount = static_cast<uint32_t>(meshInstances.size());
    }
    if (!meshCandidateIndices.empty())
    {
        meshLightCandidateIndicesDevice.allocate(
            sizeof(uint32_t) * meshCandidateIndices.size());
        NR_GPU_CHECK(cudaMemcpy(meshLightCandidateIndicesDevice.get(),
            meshCandidateIndices.data(), sizeof(uint32_t) * meshCandidateIndices.size(),
            cudaMemcpyHostToDevice));
        gpuCache.data.meshLightCandidateIndices =
            static_cast<const uint32_t*>(meshLightCandidateIndicesDevice.get());
        gpuCache.data.meshLightCandidateIndexCount =
            static_cast<uint32_t>(meshCandidateIndices.size());
    }
    if (!analyticLightBvhCandidateIndices.empty())
    {
        analyticLightBvhCandidateDevice.allocate(
            sizeof(uint32_t) * analyticLightBvhCandidateIndices.size());
        NR_GPU_CHECK(cudaMemcpy(analyticLightBvhCandidateDevice.get(),
            analyticLightBvhCandidateIndices.data(),
            sizeof(uint32_t) * analyticLightBvhCandidateIndices.size(),
            cudaMemcpyHostToDevice));
        gpuCache.data.analyticLightBvhCandidateIndices =
            static_cast<const uint32_t*>(analyticLightBvhCandidateDevice.get());
        gpuCache.data.analyticLightBvhPrimitiveCount =
            static_cast<uint32_t>(analyticLightBvhCandidateIndices.size());
        analyticLightBlas.build(optixCtx, stream, analyticLightBvhAabbs);
    }
    if (!meshLightBvhCandidateIndices.empty())
    {
        meshLightBvhCandidateDevice.allocate(
            sizeof(uint32_t) * meshLightBvhCandidateIndices.size());
        NR_GPU_CHECK(cudaMemcpy(meshLightBvhCandidateDevice.get(),
            meshLightBvhCandidateIndices.data(),
            sizeof(uint32_t) * meshLightBvhCandidateIndices.size(),
            cudaMemcpyHostToDevice));
        gpuCache.data.meshLightBvhCandidateIndices =
            static_cast<const uint32_t*>(meshLightBvhCandidateDevice.get());
        gpuCache.data.meshLightBvhPrimitiveCount =
            static_cast<uint32_t>(meshLightBvhCandidateIndices.size());
        meshLightBlas.build(optixCtx, stream, meshLightBvhAabbs);
    }
    if (weights.empty() || finiteWeight <= 0.0f)
        return;

    const uint32_t count = static_cast<uint32_t>(candidates.size());
    std::vector<LightAliasEntry> aliases(count);
    std::vector<float> scaled(count);
    std::vector<uint32_t> small;
    std::vector<uint32_t> large;
    small.reserve(count);
    large.reserve(count);
    for (uint32_t i = 0; i < count; ++i)
    {
        aliases[i].alias = i;
        aliases[i].selectionPdf = weights[i] / finiteWeight;
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

    lightAliasDevice.allocate(sizeof(LightAliasEntry) * aliases.size());
    NR_GPU_CHECK(cudaMemcpy(lightAliasDevice.get(), aliases.data(),
        sizeof(LightAliasEntry) * aliases.size(), cudaMemcpyHostToDevice));
    gpuCache.data.lightAliases =
        static_cast<const LightAliasEntry*>(lightAliasDevice.get());
    gpuCache.data.lightAliasCount = count;
}

void Raytracer::launchPsfResolve(
    const KernelParams& params, const cudaStream_t stream) const
{
    const Sensor& sensor = params.scene.camera->Dispatch(
        [](const auto* camera) -> const Sensor& { return camera->getSensor(); });

    if (const auto* scatter = sensor.CastOrNullptr<ScatterPsfSensor>())
    {
        NR_GPU_CHECK(launchScatterPsfResolveKernel(
            scatter, params.accumulation, params.output.color,
            params.frame.width, params.frame.height, stream));
        return;
    }

    if (const auto* gather = sensor.CastOrNullptr<GatherPsfSensor>())
    {
        NR_GPU_CHECK(launchGatherPsfResolveKernel(
            gather, params.accumulation, params.output.color,
            params.frame.width, params.frame.height, params.psfBinCount,
            params.psfGatherBuckets, stream));
    }
}

void Raytracer::launchDenoiser(
    KernelParams const& params, const cudaStream_t stream,
    const bool useAovGuides)
{
    const void* albedoGuide = useAovGuides
        ? params.denoiserAlbedoGuide : nullptr;
    const void* normalGuide = useAovGuides
        ? params.denoiserNormalGuide : nullptr;
    denoiser.run(optixCtx, stream, params.accumulation, albedoGuide,
        normalGuide, denoised.cudaPointer(), width, height);
    denoisedOutputAvailable = true;
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

void Raytracer::launchPathTrace(
    const KernelParams& params, const cudaStream_t stream,
    const bool uploadStaticParams) const
{
    const NvtxRange profileRange("NoorRay/PathTrace");
    if (uploadStaticParams)
        NR_GPU_CHECK(cudaMemcpyAsync(optixLaunchParamsDevice.get(),
            &params, sizeof(params), cudaMemcpyHostToDevice, stream));
    else
        NR_GPU_CHECK(cudaMemcpyAsync(
            static_cast<char*>(optixLaunchParamsDevice.get())
                + offsetof(KernelParams, frame),
            &params.frame, sizeof(params.frame),
            cudaMemcpyHostToDevice, stream));
    NR_OPTIX_CHECK(optixLaunch(optixPipeline.get(), stream,
        optixLaunchParamsDevice.devicePtr(), sizeof(KernelParams),
        &optixPathTraceSbt, params.frame.width, params.frame.height, 1));
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
    const uint32_t frameIndex, const uint32_t accumulatedSamples,
    const uint32_t sampleSeed)
{
    // Synchronize the GPU stream once per frame if any mutation occurred since
    // the last render. This replaces the old per-mutation waitForRender barrier
    // with a single sync point, batching all Hydra Sync() calls together.
    if (scene.consumeGpuSync())
        NR_GPU_CHECK(cudaStreamSynchronize(stream));

    const bool textureLibraryChanged =
        gpuCache.textureRevision != scene.getTextureRevision();
    const bool sceneDirty = scene.isAnyDirty() || textureLibraryChanged;
    // Camera motion changes which geometry each pixel sees, so the AOV images
    // must be regenerated along with the beauty accumulation.
    const bool aovSceneDirty = textureLibraryChanged
        || scene.isDirty(TLAS) || scene.isDirty(Meshes)
        || scene.isDirty(Textures) || scene.isDirty(EnvironmentCdf)
        || scene.isDirty(Lights) || scene.isDirty(CameraState)
        || scene.isDirty(GaussianData);

    CameraInstance* activeCamera = scene.getRenderCamera();
    if (!activeCamera)
        return;

    const glm::uvec2 resolution = activeCamera->getCamera()->getSensor().resolution();
    if (resolution.x == 0 || resolution.y == 0)
        throw std::runtime_error("Cannot render with a zero-sized camera sensor");
    if (resolution.x != width || resolution.y != height)
        resize(resolution.x, resolution.y);

    const RenderSettings& renderSettings = scene.getRenderSettings();
    denoisedOutputAvailable = false;

    // AOVs are decided after the frame is sized: their images must match the
    // resolution this frame renders at, and before the first resize there is no
    // resolution to allocate them for.
    //
    // Refresh AOVs immediately whenever scene data changes so viewport guides,
    // picking, outlines, and denoiser inputs stay aligned with moved cameras
    // and objects. Unchanged progressive frames continue reusing the last AOVs.
    const bool runDenoiser = runsOptixDenoiser(renderSettings);
    const bool proxyOverdrawView = rendersProxyOverdraw(renderSettings);
    const bool useAovs = aovEnabled && scene.getRenderSettings().aovEnabled;
    if (useAovs)
        ensureAovImages();
    else
    {
        aovAvailable = false;
        aovStale = true;
    }
    const bool useDenoiserGuides = useAovs && runDenoiser;
    const size_t denoiserGuideBytes = sizeof(float3)
        * static_cast<size_t>(width) * height;
    if (useDenoiserGuides)
    {
        bool guideBuffersReallocated = false;
        if (!denoiserAlbedoGuideBuffer
            || denoiserAlbedoGuideBuffer.size() != denoiserGuideBytes)
        {
            denoiserAlbedoGuideBuffer.allocate(denoiserGuideBytes, stream);
            guideBuffersReallocated = true;
        }
        if (!denoiserNormalGuideBuffer
            || denoiserNormalGuideBuffer.size() != denoiserGuideBytes)
        {
            denoiserNormalGuideBuffer.allocate(denoiserGuideBytes, stream);
            guideBuffersReallocated = true;
        }
        if (guideBuffersReallocated)
            denoiserGuidesStale = true;
    }
    const bool renderAov = useAovs && aovImagesCreated
        && (aovSceneDirty || aovStale || (useDenoiserGuides && denoiserGuidesStale));

    if (scene.isDirty(Meshes)) updateMeshes();
    if (scene.isDirty(Textures) || textureLibraryChanged)
        updateTextures();
    if (scene.isDirty(EnvironmentCdf))
        updateEnvironmentCdf();
    if (scene.isDirty(Lights) || scene.isDirty(Meshes)
        || scene.isDirty(TLAS))
        updateLights();
    // MaterialX updates use the Meshes flag to refresh material records and
    // hit-group bindings, but they do not change geometry. In particular,
    // rebuilding a 20-million-Gaussian TLAS for every IOR/transmission edit
    // creates a second multi-gigabyte acceleration structure and can trigger
    // the OS OOM killer. Geometry mutations set TLAS explicitly.
    if (scene.isDirty(TLAS) || scene.isDirty(GaussianData)
        || scene.isDirty(Lights))
        updateTLAS();
    if (scene.isDirty(CameraState)) activeCamera->rebuildCamera();

    activeCamera->getCamera()->prepareForRender();

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
    params.frame.sampleSeed = sampleSeed;
    // SER has a fixed reorder/invoke cost. It pays off once several distinct
    // SVM programs compete in the same launch, but is measurably slower for
    // one-to-three-material scenes where rays are already coherent. Keep the
    // normal policy adaptive while retaining explicit profiling overrides.
    const bool forceSer = std::getenv("NR_OPTIX_FORCE_SER") != nullptr;
    const bool disableSer = std::getenv("NR_OPTIX_DISABLE_SER") != nullptr;
    params.frame.serEnabled = !disableSer
        && (forceSer || materialxPrograms.records().size() >= 4);
    if (std::getenv("NR_OPTIX_LOG_LEVEL") != nullptr)
        LOG_INFO("Path SER: " << (params.frame.serEnabled ? "enabled" : "disabled")
            << " (SVM programs=" << materialxPrograms.records().size() << ')');
    params.frame.visibilityMask = params.scene.gaussianCount > 0
        ? SceneVisibility
        : MeshVisibility;
    params.accumulation = accumulationBuffer.as<glm::vec4>();
    if (useDenoiserGuides)
    {
        params.denoiserAlbedoGuide = denoiserAlbedoGuideBuffer.as<float3>();
        params.denoiserNormalGuide = denoiserNormalGuideBuffer.as<float3>();
    }
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

    if (proxyOverdrawView)
    {
        ensureProxyOverdrawResources();
        kernelStats.time("ProxyOverdraw", stream,
            [&] { launchProxyOverdraw(params, stream); });
    }
    else
    {
        for (uint32_t s = 0; s < samplesPerFrame; ++s)
        {
            params.frame.totalAccumulated = accumulatedSamples + s;
            params.frame.writeOutput = s + 1u == samplesPerFrame ? 1u : 0u;
            // AOV queries run inline in the first beauty raygen launch. The
            // combined SBT routes their traversal calls to query records.
            params.frame.aovQuery = renderAov && s == 0 ? 1u : 0u;
            kernelStats.time("PathTrace", stream,
                [&] { launchPathTrace(params, stream, s == 0); });
        }
        params.frame.aovQuery = 0;
        if (renderAov)
        {
            aovAvailable = true;
            aovStale = false;
            if (useDenoiserGuides)
                denoiserGuidesStale = false;
            else
                denoiserGuidesStale = true;
        }
        const uint32_t accumulatedBeforeFrame = accumulatedSamples;
        const uint32_t accumulatedAfterFrame = accumulatedSamples + samplesPerFrame;
        const bool finalSample = accumulatedBeforeFrame < maxSamples && accumulatedAfterFrame >= maxSamples;
        bool resolvePsf = activeSensor.Is<ScatterPsfSensor>();
        if (activeSensor.Is<GatherPsfSensor>() && finalSample)
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
                resolvePsf = true;
            }
        }
        const bool directSensor = !activeSensor.Is<ScatterPsfSensor>()
            && !activeSensor.Is<GatherPsfSensor>();
        const uint32_t denoiserMinSamples = static_cast<uint32_t>(
            std::max(1, renderSettings.optixDenoiserMinSamples));
        if (runDenoiser && directSensor
            && accumulatedAfterFrame >= denoiserMinSamples)
            kernelStats.time("OptixDenoiser", stream,
                [&] { launchDenoiser(params, stream, aovAvailable); });
        if (resolvePsf)
            kernelStats.time("PsfResolve", stream,
                [&] { launchPsfResolve(params, stream); });
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
    // Hydra's Blender integration uses a headless NoorRay Vulkan context, so
    // it has no Vulkan timeline semaphore to poll. This CUDA event is the
    // completion signal for that path and is also useful as a precise GPU
    // ordering point for the retained viewport image.
    NR_GPU_CHECK(cudaEventRecord(m_frameCompleteEvent.get(), stream));
    m_frameCompleteRecorded = true;
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

std::optional<InteropFrame> Raytracer::consumeInteropFrame()
{
    if (!isFrameReady() || lastReadyValue <= consumedReadyValue)
        return std::nullopt;
    consumedReadyValue = lastReadyValue;
    return getInteropFrame();
}

bool Raytracer::isRenderInFlight() const
{
    if (context.isHeadless())
    {
        if (!m_frameCompleteRecorded)
            return false;
        return cudaEventQuery(m_frameCompleteEvent.get()) == cudaErrorNotReady;
    }
    return lastReadyValue != 0 && renderReady
        && context.getDevice().getSemaphoreCounterValue(renderReady.get()) < lastReadyValue;
}

bool Raytracer::isFrameReady() const
{
    if (context.isHeadless())
    {
        if (!m_frameCompleteRecorded)
            return false;
        const cudaError_t status = cudaEventQuery(m_frameCompleteEvent.get());
        if (status == cudaSuccess)
            return true;
        if (status == cudaErrorNotReady)
            return false;
        NR_GPU_CHECK(status);
    }
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
