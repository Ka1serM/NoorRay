#include "Raytracing/Raytracer.h"

#include <algorithm>
#include <array>
#include <cmath>
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

#include "Camera/CameraInstance.h"
#include "CUDA/Checks.h"
#include "CUDA/rstd/Memory.h"
#include "Log.h"
#include "NoorRayOptixIr.h"
#include "Mesh/GaussianCutoff.h"
#include "Mesh/MeshAsset.h"
#include "Scene/MeshInstance.h"
#include "Scene/Scene.h"
#include "Vulkan/Context.h"

extern const float sRGBToSpectrumTable_Scale[64];
extern const float sRGBToSpectrumTable_Data[3][64][64][64][3];
#include "Raytracing/Spectrum.h"

namespace
{
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

}

extern NR_GPU_KERNEL void generateKernel(KernelParams);
extern NR_GPU_KERNEL void finalizeKernel(KernelParams);
extern NR_GPU_KERNEL void reduceNoiseVarianceKernel(KernelParams, float*);
extern NR_GPU_KERNEL void resolveScatterPsfKernel(KernelParams);
extern NR_GPU_KERNEL void applyGatherPsfKernel(KernelParams);
extern NR_GPU_KERNEL void shadeKernel(KernelParams);
extern NR_GPU_KERNEL void generateAovKernel(KernelParams);
extern NR_GPU_KERNEL void shadeAovKernel(KernelParams);

Raytracer::Raytracer(
    Context& context,
    Scene& scene)
    : context(context), scene(scene)
{
    stream = context.getCudaStream();
    optixCtx = context.getOptixContext();
    if (stream == nullptr || optixCtx == nullptr)
        throw std::runtime_error("CUDA/OptiX not initialized in Context");

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
    pipelineOptions.numPayloadValues = 4;
    pipelineOptions.numAttributeValues = 2;
    pipelineOptions.pipelineLaunchParamsVariableName = "params";
    pipelineOptions.pipelineLaunchParamsSizeInBytes = sizeof(KernelParams);
    pipelineOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    std::array<char, 8192> log{};
    size_t logSize = log.size();
    OptixResult result = optixModuleCreate(
        optixCtx, &moduleOptions, &pipelineOptions,
        reinterpret_cast<const char*>(noorRayOptixIr), noorRayOptixIrLength,
        log.data(), &logSize, optixModule.put());
    if (result != OPTIX_SUCCESS)
        throw std::runtime_error(std::string("OptiX module creation failed: ") + log.data());

    optixExtendGroup.reset(makeRaygenGroup(optixCtx, optixModule.get(), "__raygen__extend"));
    optixConnectGroup.reset(makeRaygenGroup(optixCtx, optixModule.get(), "__raygen__connect"));
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

    const std::array groups{
        optixExtendGroup.get(), optixConnectGroup.get(), optixTriangleGroup.get(),
        optixGaussianHitGroup.get(), optixMissGroup.get()};
    OptixPipelineLinkOptions linkOptions{};
    linkOptions.maxTraceDepth = 1;
    logSize = log.size();
    result = optixPipelineCreate(
        optixCtx, &pipelineOptions, &linkOptions,
        groups.data(), static_cast<unsigned int>(groups.size()),
        log.data(), &logSize, optixPipeline.put());
    if (result != OPTIX_SUCCESS)
        throw std::runtime_error(std::string("OptiX pipeline creation failed: ") + log.data());

    OptixStackSizes stackSizes{};
    for (const OptixProgramGroup group : groups)
        NR_OPTIX_CHECK(optixUtilAccumulateStackSizes(group, &stackSizes, optixPipeline.get()));
    unsigned int directCallableStackSizeFromTraversal = 0;
    unsigned int directCallableStackSizeFromState = 0;
    unsigned int continuationStackSize = 0;
    NR_OPTIX_CHECK(optixUtilComputeStackSizes(
        &stackSizes, linkOptions.maxTraceDepth, 0, 0,
        &directCallableStackSizeFromTraversal, &directCallableStackSizeFromState,
        &continuationStackSize));
    NR_OPTIX_CHECK(optixPipelineSetStackSize(optixPipeline.get(),
        directCallableStackSizeFromTraversal, directCallableStackSizeFromState,
        continuationStackSize, 2));

    // Keep diagnostics in a separate pipeline. Its larger raygen program and
    // stack requirements therefore cannot change normal-render occupancy.
    const std::array proxyOverdrawGroups{
        optixProxyOverdrawGroup.get(), optixTriangleGroup.get(),
        optixProxyOverdrawHitGroup.get(), optixMissGroup.get()};
    logSize = log.size();
    result = optixPipelineCreate(
        optixCtx, &pipelineOptions, &linkOptions,
        proxyOverdrawGroups.data(), static_cast<unsigned int>(proxyOverdrawGroups.size()),
        log.data(), &logSize, optixProxyOverdrawPipeline.put());
    if (result != OPTIX_SUCCESS)
        throw std::runtime_error(
            std::string("OptiX proxy-overdraw pipeline creation failed: ") + log.data());

    OptixStackSizes proxyOverdrawStackSizes{};
    for (const OptixProgramGroup group : proxyOverdrawGroups)
        NR_OPTIX_CHECK(optixUtilAccumulateStackSizes(
            group, &proxyOverdrawStackSizes, optixProxyOverdrawPipeline.get()));
    NR_OPTIX_CHECK(optixUtilComputeStackSizes(
        &proxyOverdrawStackSizes, linkOptions.maxTraceDepth, 0, 0,
        &directCallableStackSizeFromTraversal, &directCallableStackSizeFromState,
        &continuationStackSize));
    NR_OPTIX_CHECK(optixPipelineSetStackSize(optixProxyOverdrawPipeline.get(),
        directCallableStackSizeFromTraversal, directCallableStackSizeFromState,
        continuationStackSize, 2));

    optixExtendRecord = uploadRecord(optixExtendGroup.get());
    optixConnectRecord = uploadRecord(optixConnectGroup.get());
    optixProxyOverdrawRecord = uploadRecord(optixProxyOverdrawGroup.get());

    // Upload two hitgroup records contiguously: mesh (sbtOffset=0) and Gaussian (sbtOffset=1)
    {
        SbtRecord<> meshSbt{};
        NR_OPTIX_CHECK(optixSbtRecordPackHeader(optixTriangleGroup.get(), &meshSbt));
        SbtRecord<> gaussianSbt{};
        NR_OPTIX_CHECK(optixSbtRecordPackHeader(optixGaussianHitGroup.get(), &gaussianSbt));
        const std::array records{meshSbt, gaussianSbt};
        optixHitgroupRecord.allocate(sizeof(records));
        NR_GPU_CHECK(cudaMemcpy(optixHitgroupRecord.get(), records.data(), sizeof(records), cudaMemcpyHostToDevice));
    }

    // Diagnostic SBT: mesh instances retain an empty hit group while Gaussian
    // instances use the counting any-hit program at their existing sbtOffset 1.
    {
        SbtRecord<> meshSbt{};
        NR_OPTIX_CHECK(optixSbtRecordPackHeader(optixTriangleGroup.get(), &meshSbt));
        SbtRecord<> gaussianSbt{};
        NR_OPTIX_CHECK(optixSbtRecordPackHeader(optixProxyOverdrawHitGroup.get(), &gaussianSbt));
        const std::array records{meshSbt, gaussianSbt};
        optixProxyOverdrawHitgroupRecord.allocate(sizeof(records));
        NR_GPU_CHECK(cudaMemcpy(optixProxyOverdrawHitgroupRecord.get(), records.data(), sizeof(records), cudaMemcpyHostToDevice));
    }

    optixMissRecord = uploadRecord(optixMissGroup.get());

    optixExtendSbt.raygenRecord = optixExtendRecord.devicePtr();
    optixExtendSbt.missRecordBase = optixMissRecord.devicePtr();
    optixExtendSbt.missRecordStrideInBytes = sizeof(SbtRecord<>);
    optixExtendSbt.missRecordCount = 1;
    optixExtendSbt.hitgroupRecordBase = optixHitgroupRecord.devicePtr();
    optixExtendSbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<>);
    optixExtendSbt.hitgroupRecordCount = 2;
    optixConnectSbt = optixExtendSbt;
    optixConnectSbt.raygenRecord = optixConnectRecord.devicePtr();
    optixProxyOverdrawSbt = optixExtendSbt;
    optixProxyOverdrawSbt.raygenRecord = optixProxyOverdrawRecord.devicePtr();
    optixProxyOverdrawSbt.hitgroupRecordBase = optixProxyOverdrawHitgroupRecord.devicePtr();

    m_startEvent.create();
    m_stopEvent.create();
    NR_GPU_CHECK(cudaMallocHost(&m_noiseVarianceSumHost, sizeof(float)));
    *m_noiseVarianceSumHost = std::numeric_limits<float>::infinity();

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

    // Upload Jakob & Hanika sRGB-to-spectrum table once (9 MB coefficients + 64-element scale).
    constexpr size_t kScaleBytes = 64 * sizeof(float);
    constexpr size_t kCoeffBytes = sizeof(sRGBToSpectrumTable_Data);
    spectrumTableScaleDevice.allocate(kScaleBytes);
    spectrumTableCoeffsDevice.allocate(kCoeffBytes);
    NR_GPU_CHECK(cudaMemcpy(spectrumTableScaleDevice.get(), sRGBToSpectrumTable_Scale, kScaleBytes, cudaMemcpyHostToDevice));
    NR_GPU_CHECK(cudaMemcpy(spectrumTableCoeffsDevice.get(), sRGBToSpectrumTable_Data, kCoeffBytes, cudaMemcpyHostToDevice));
    gpuScene.spectrumTableScale  = static_cast<float*>(spectrumTableScaleDevice.get());
    gpuScene.spectrumTableCoeffs = static_cast<float*>(spectrumTableCoeffsDevice.get());

    constexpr size_t kD65Bytes = NrD65Samples * sizeof(float);
    d65Device.allocate(kD65Bytes);
    NR_GPU_CHECK(cudaMemcpy(d65Device.get(), NrD65, kD65Bytes, cudaMemcpyHostToDevice));
    gpuScene.d65 = static_cast<float*>(d65Device.get());

    // Upload CIE 1931 2-degree CMF tables (471 floats × 3, ~5.5 KB).
    constexpr size_t kCieBytes = NrCIESamples * sizeof(float);
    cieXDevice.allocate(kCieBytes);
    cieYDevice.allocate(kCieBytes);
    cieZDevice.allocate(kCieBytes);
    NR_GPU_CHECK(cudaMemcpy(cieXDevice.get(), NrCIE_X, kCieBytes, cudaMemcpyHostToDevice));
    NR_GPU_CHECK(cudaMemcpy(cieYDevice.get(), NrCIE_Y, kCieBytes, cudaMemcpyHostToDevice));
    NR_GPU_CHECK(cudaMemcpy(cieZDevice.get(), NrCIE_Z, kCieBytes, cudaMemcpyHostToDevice));
    gpuScene.cieX = static_cast<float*>(cieXDevice.get());
    gpuScene.cieY = static_cast<float*>(cieYDevice.get());
    gpuScene.cieZ = static_cast<float*>(cieZDevice.get());

    // Upload OpenPBR opaque-dielectric energy-compensation LUTs as hardware-filtered textures.
    nr::openpbr::uploadEnergyLuts(openPbrLutStorage, gpuScene.openPbrLuts, stream);

    {
        optixLaunchParamsDevice.allocate(sizeof(KernelParams));
    }

    auto* cam = scene.getRenderCamera();
    auto* c = cam ? cam->getCamera() : nullptr;
    auto res = c ? c->getSensor().resolution() : glm::uvec2(1280, 720);
    resize(res.x, res.y);
    updateLights();
    updateTLAS();
}

Raytracer::~Raytracer()
{
    if (stream != nullptr)
        cudaStreamSynchronize(stream);
    if (m_noiseVarianceSumHost != nullptr)
        cudaFreeHost(m_noiseVarianceSumHost);
    m_noiseVarianceSumHost = nullptr;
    tlas.reset();
    freeSceneData();
    freeQueues();
    optixLaunchParamsDevice.reset();
    if (stream != nullptr)
        cudaStreamSynchronize(stream);
    m_startEvent.reset();
    m_stopEvent.reset();
    cudaRenderReady.reset();
    cudaBufferReleased.reset();
    optixExtendRecord.reset();
    optixConnectRecord.reset();
    optixProxyOverdrawRecord.reset();
    optixHitgroupRecord.reset();
    optixProxyOverdrawHitgroupRecord.reset();
    optixMissRecord.reset();
    optixPipeline.reset();
    optixProxyOverdrawPipeline.reset();
    optixMissGroup.reset();
    optixGaussianHitGroup.reset();
    optixProxyOverdrawHitGroup.reset();
    optixTriangleGroup.reset();
    optixConnectGroup.reset();
    optixProxyOverdrawGroup.reset();
    optixExtendGroup.reset();
    optixModule.reset();
    optixExtendSbt = {};
    optixConnectSbt = {};
    optixProxyOverdrawSbt = {};
}

void Raytracer::resize(const uint32_t newWidth, const uint32_t newHeight)
{
    if (newWidth == width && newHeight == height)
        return;
    NR_GPU_CHECK(cudaStreamSynchronize(stream));
    width = newWidth;
    height = newHeight;
    freeQueues();

    auto createImages = [&](nr::cuda::UniqueSharedImage (&arr)[2], const vk::Format format)
    {
        for (auto& img : arr)
            img.create(context, width, height, format);
    };
    createImages(color,        vk::Format::eR32G32B32A32Sfloat);
    createImages(albedo,       vk::Format::eR8G8B8A8Unorm);
    createImages(normal,       vk::Format::eR16G16B16A16Sfloat);
    createImages(cryptomatte,  vk::Format::eR32Uint);
    createImages(position,     vk::Format::eR16G16B16A16Sfloat);

    allocateQueues();
    updateTextures();
    updateMeshes();
    NR_GPU_CHECK(cudaStreamSynchronize(stream));
    nextBuffer = 0;
    lastLaunched = 0;
    lastReadyValue = 0;
    lastUseValue = {};
}

void Raytracer::allocateQueues()
{
    queues.capacity = width * height;
    rayCountBuffer.allocate(sizeof(uint32_t) * MaxBounces, stream);
    pathStateBuffer.allocate(sizeof(PathState) * queues.capacity, stream);
    rayQueueBuffers[0].allocate(sizeof(PathRayWorkItem) * queues.capacity, stream);
    rayQueueBuffers[1].allocate(sizeof(PathRayWorkItem) * queues.capacity, stream);
    hitQueueBuffer.allocate(sizeof(HitWorkItem) * queues.capacity, stream);
    shadowQueueBuffer.allocate(sizeof(ShadowWorkItem) * queues.capacity, stream);
    aovRayQueueBuffer.allocate(sizeof(PathRayWorkItem) * queues.capacity, stream);
    aovHitQueueBuffer.allocate(sizeof(HitWorkItem) * queues.capacity, stream);
    accumulationBuffer.allocate(sizeof(glm::vec4) * static_cast<size_t>(width) * height, stream);
    noiseVarianceSumBuffer.allocate(sizeof(float), stream);

    queues.rayCounts = rayCountBuffer.as<uint32_t>();
    queues.pathStates = pathStateBuffer.as<PathState>();
    queues.rayQueues[0] = rayQueueBuffers[0].as<PathRayWorkItem>();
    queues.rayQueues[1] = rayQueueBuffers[1].as<PathRayWorkItem>();
    queues.hitQueue = hitQueueBuffer.as<HitWorkItem>();
    queues.shadowQueue = shadowQueueBuffer.as<ShadowWorkItem>();
    queues.aovRayQueue = aovRayQueueBuffer.as<PathRayWorkItem>();
    queues.aovHitQueue = aovHitQueueBuffer.as<HitWorkItem>();
    NR_GPU_CHECK(cudaMemsetAsync(
        accumulationBuffer.get(), 0, sizeof(glm::vec4) * width * height, stream));
}

void Raytracer::freeQueues() noexcept
{
    rayCountBuffer.reset();
    pathStateBuffer.reset();
    rayQueueBuffers[0].reset();
    rayQueueBuffers[1].reset();
    hitQueueBuffer.reset();
    shadowQueueBuffer.reset();
    aovRayQueueBuffer.reset();
    aovHitQueueBuffer.reset();
    accumulationBuffer.reset();
    noiseMomentsBuffer.reset();
    noiseVarianceSumBuffer.reset();
    queues = {};
}

void Raytracer::freeSceneData() noexcept
{
    scene.getCudaTexturesRef().clear();
    scene.getEnvironment().destroyCdf();
    scene.getGpuInstancesRef().clear();
    spectrumTableScaleDevice.reset();
    spectrumTableCoeffsDevice.reset();
    d65Device.reset();
    cieXDevice.reset();
    cieYDevice.reset();
    cieZDevice.reset();
    openPbrLutStorage.reset();
    gpuScene = {};
}

void Raytracer::updateTextures()
{
    NR_GPU_CHECK(cudaStreamSynchronize(stream));

    const auto& cpuTextures = scene.getTextures();
    const size_t count = cpuTextures.size();

    auto& cudaTextures = scene.getCudaTexturesRef();
    cudaTextures.clear();
    for (size_t i = 0; i < count; ++i)
        cudaTextures.emplace_back(cpuTextures[i].getPixels().data(), cpuTextures[i].getWidth(), cpuTextures[i].getHeight(), stream);
    gpuScene.textures = cudaTextures.data();
    gpuScene.textureCount = static_cast<uint32_t>(cpuTextures.size());

    for (MeshAsset& asset : scene.getMeshAssets())
        asset.updateMaterials();

    updateEnvironmentCdf();
}

void Raytracer::updateEnvironmentCdf()
{
    NR_GPU_CHECK(cudaStreamSynchronize(stream));
    Environment& env = scene.getEnvironment();
    gpuScene.environment = &env;
    env.destroyCdf();
    env.cdfDirty = 0;

    const int textureIndex = env.textureIndex;
    const auto& textures = scene.getTextures();
    if (textureIndex < 0 || textureIndex >= static_cast<int>(textures.size()))
        return;

    const Texture& texture = textures[textureIndex];
    env.cdfWidth = texture.getWidth();
    env.cdfHeight = texture.getHeight();
    double integral = 0.0;
    const auto& pixels = texture.getPixels();
    for (int y = 0; y < env.cdfHeight; ++y) {
        const double sinTheta = std::sin((static_cast<double>(y) + 0.5)
            / static_cast<double>(env.cdfHeight) * 3.14159265358979323846);
        for (int x = 0; x < env.cdfWidth; ++x) {
            const size_t offset = static_cast<size_t>(y * env.cdfWidth + x) * 4;
            integral += (0.2126 * pixels[offset] + 0.7152 * pixels[offset + 1]
                + 0.0722 * pixels[offset + 2]) * sinTheta;
        }
    }
    integral *= 2.0 * 3.14159265358979323846 * 3.14159265358979323846
        / static_cast<double>(env.cdfWidth * env.cdfHeight);
    const float colorLuminance = std::max(
        0.2126f * env.color.r + 0.7152f * env.color.g + 0.0722f * env.color.b, 0.0f);
    env.importanceWeight = static_cast<float>(integral) * colorLuminance
        * std::max(env.lightingExposureScale, 0.0f);
    const std::vector<float> cdf = Environment::computeCdf(
        texture.getPixels().data(), texture.getWidth(), texture.getHeight());
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
    gpuScene.meshes = scene.getMeshAssets().data();
}

void Raytracer::updateTLAS()
{
    NR_GPU_CHECK(cudaStreamSynchronize(stream));
    auto& gpuInstances = scene.getGpuInstancesRef();
    tlas.build(optixCtx, stream, scene, gpuInstances);
    scene.clearDirtyMeshInstanceIndices();
    gpuScene.tlasHandle = tlas.getTraversable();
    gpuScene.instances = gpuInstances.data();
    gpuScene.meshInstanceCount = static_cast<uint32_t>(gpuInstances.size());

    scene.buildGaussianRenderData();
    gpuScene.gaussianOpacities = scene.getGaussianOpacities();
    gpuScene.gaussianShCoeffs = scene.getGaussianShCoeffs();
    gpuScene.gaussianCount = scene.getGaussianCount();
    NR_GPU_CHECK(cudaStreamSynchronize(stream));
}

void Raytracer::updateLights()
{
    // Light arrays live in a Vulkan-allocated buffer imported into CUDA (see
    // nr::cuda::SharedVector), so the host and device see the same content through
    // two different pointers. Kernels get the device pointer; the host-side weight
    // sum below must use the host pointer instead of dereferencing gpuScene's fields.
    NR_GPU_CHECK(cudaStreamSynchronize(stream));
    gpuScene.pointLights = scene.getPointLightsDevice();
    gpuScene.spotLights = scene.getSpotLightsDevice();
    gpuScene.rectLights = scene.getRectLightsDevice();
    gpuScene.directionalLights = scene.getDirectionalLightsDevice();
    gpuScene.pointLightCount = scene.getPointLightCount();
    gpuScene.spotLightCount = scene.getSpotLightCount();
    gpuScene.rectLightCount = scene.getRectLightCount();
    gpuScene.directionalLightCount = scene.getDirectionalLightCount();

    const PointLight* pointLights = scene.getPointLights();
    const SpotLight* spotLights = scene.getSpotLights();
    const RectLight* rectLights = scene.getRectLights();
    const DirectionalLight* directionalLights = scene.getDirectionalLights();

    float analyticWeight = 0.0f;
    for (uint32_t i = 0; i < gpuScene.pointLightCount; ++i)
        analyticWeight += pointLights[i].selectionWeight();
    for (uint32_t i = 0; i < gpuScene.spotLightCount; ++i)
        analyticWeight += spotLights[i].selectionWeight();
    for (uint32_t i = 0; i < gpuScene.rectLightCount; ++i)
        analyticWeight += rectLights[i].selectionWeight();
    for (uint32_t i = 0; i < gpuScene.directionalLightCount; ++i)
        analyticWeight += directionalLights[i].selectionWeight();
    gpuScene.analyticLightSelectionWeight = analyticWeight;
}

void Raytracer::launchGenerate(const KernelParams& params, const cudaStream_t stream) const
{
    constexpr uint32_t blockSize = 256;
    const uint32_t count = params.frame.width * params.frame.height;
    const dim3 grid((count + blockSize - 1) / blockSize, 1, 1);
    void* args[] = { const_cast<KernelParams*>(&params) };
    NR_GPU_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(&generateKernel), grid, blockSize, args, 0, stream));
}

void Raytracer::launchFinalize(const KernelParams& params, const cudaStream_t stream) const
{
    constexpr uint32_t blockSize = 256;
    const uint32_t count = params.frame.width * params.frame.height;
    const dim3 grid((count + blockSize - 1) / blockSize, 1, 1);
    void* args[] = { const_cast<KernelParams*>(&params) };
    NR_GPU_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(&finalizeKernel), grid, blockSize, args, 0, stream));
}

void Raytracer::launchNoiseReduction(const KernelParams& params, const cudaStream_t stream) const
{
    constexpr uint32_t blockSize = 256;
    const uint32_t count = params.frame.width * params.frame.height;
    const dim3 grid((count + blockSize - 1) / blockSize, 1, 1);
    float* varianceSum = noiseVarianceSumBuffer.as<float>();
    NR_GPU_CHECK(cudaMemsetAsync(varianceSum, 0, sizeof(float), stream));
    void* args[] = { const_cast<KernelParams*>(&params), &varianceSum };
    NR_GPU_CHECK(cudaLaunchKernel(
        reinterpret_cast<const void*>(&reduceNoiseVarianceKernel),
        grid, blockSize, args, 0, stream));
    NR_GPU_CHECK(cudaMemcpyAsync(
        m_noiseVarianceSumHost, varianceSum, sizeof(float), cudaMemcpyDeviceToHost, stream));
}

void Raytracer::launchResolveScatterPsf(const KernelParams& params, const cudaStream_t stream) const
{
    constexpr uint32_t blockSize = 256;
    const uint32_t count = params.frame.width * params.frame.height;
    const dim3 grid((count + blockSize - 1) / blockSize, 1, 1);
    void* args[] = { const_cast<KernelParams*>(&params) };
    NR_GPU_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(&resolveScatterPsfKernel), grid, blockSize, args, 0, stream));
}

void Raytracer::launchApplyGatherPsf(const KernelParams& params, const cudaStream_t stream) const
{
    constexpr uint32_t blockSize = 256;
    const uint32_t count = params.frame.width * params.frame.height;
    const dim3 grid((count + blockSize - 1) / blockSize, 1, 1);
    void* args[] = { const_cast<KernelParams*>(&params) };
    NR_GPU_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(&applyGatherPsfKernel), grid, blockSize, args, 0, stream));
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

void Raytracer::launchSensorAddSample(
    const Sensor& sensor, const KernelParams& params, const cudaStream_t stream)
{
    if (sensor.Is<ScatterPsfSensor>())
        kernelStats.time("FinalizePsfScatter", stream, [&] { launchFinalize(params, stream); });
    else if (sensor.Is<GatherPsfSensor>())
        kernelStats.time("FinalizePsfGatherBuckets", stream, [&] { launchFinalize(params, stream); });
    else
        kernelStats.time("FinalizeDirect", stream, [&] { launchFinalize(params, stream); });
}

void Raytracer::applySensorAfterFrame(
    const Sensor& sensor, const KernelParams& params, const cudaStream_t stream, const bool finalSample)
{
    if (sensor.Is<ScatterPsfSensor>()) {
        kernelStats.time("ResolvePsfScatter", stream, [&] { launchResolveScatterPsf(params, stream); });
        return;
    }

    if (!sensor.Is<GatherPsfSensor>())
        return;

    if (!finalSample)
        return;

    if (params.psfGatherBuckets == nullptr || params.psfBinCount == 0) {
        std::cerr << "[PSF] Gather final resolve skipped: buckets="
                  << static_cast<const void*>(params.psfGatherBuckets)
                  << " bins=" << params.psfBinCount << std::endl;
        return;
    }

    std::cerr << "[PSF] Gather final resolve applying " << params.psfBinCount
              << " PSF bins" << std::endl;
    kernelStats.time("ApplyPsfGather", stream, [&] { launchApplyGatherPsf(params, stream); });
}

void Raytracer::launchShade(
    const KernelParams& params, const uint32_t launchCount, const cudaStream_t stream) const
{
    constexpr uint32_t blockSize = 256;
    const dim3 grid((launchCount + blockSize - 1) / blockSize, 1, 1);
    void* args[] = { const_cast<KernelParams*>(&params) };
    NR_GPU_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(&shadeKernel), grid, blockSize, args, 0, stream));
}


void Raytracer::launchExtend(
    const KernelParams& params, const uint32_t launchCount, const cudaStream_t stream) const
{
    NR_GPU_CHECK(cudaMemcpyAsync(optixLaunchParamsDevice.get(),
        &params, sizeof(params), cudaMemcpyHostToDevice, stream));
    NR_OPTIX_CHECK(optixLaunch(optixPipeline.get(), stream,
        optixLaunchParamsDevice.devicePtr(), sizeof(KernelParams),
        &optixExtendSbt, launchCount, 1, 1));
}

void Raytracer::renderGaussianTrainForward(
    const GaussianTrainParams& trainParams,
    const uint32_t renderWidth, const uint32_t renderHeight)
{
    const size_t pixelCount = static_cast<size_t>(renderWidth) * renderHeight;
    if (pixelCount > queues.capacity)
        throw std::runtime_error("Training image exceeds the raytracing queue capacity");
    CameraInstance* activeCamera = scene.getActiveCamera();
    if (activeCamera == nullptr)
        throw std::runtime_error("Gaussian training requires an active camera");

    KernelParams params{};
    params.scene = gpuScene;
    params.scene.camera = activeCamera->getGpuCamera();
    params.queues = queues;
    params.train = trainParams;
    params.train.enabled = 1;
    params.train.mode = RenderMode::Forward;
    params.frame.width = renderWidth;
    params.frame.height = renderHeight;
    params.frame.cutoffDistanceSq = scene.getRenderSettings().gaussianCutoffSigma
        * scene.getRenderSettings().gaussianCutoffSigma;

    const size_t outputBytes = static_cast<size_t>(renderWidth) * renderHeight
        * 3 * sizeof(float);
    NR_GPU_CHECK(cudaMemsetAsync(params.train.outputColor, 0, outputBytes, stream));

    for (uint32_t sample = 0; sample < trainParams.samplesPerPixel; ++sample)
    {
        NR_GPU_CHECK(cudaMemsetAsync(queues.rayCounts, 0,
            sizeof(uint32_t) * MaxBounces, stream));
        params.frame.totalAccumulated = sample;
        params.depth = 0;
        launchGenerate(params, stream);
        launchExtend(params, queues.capacity, stream);
        launchShade(params, queues.capacity, stream);
        launchFinalize(params, stream);
    }
    NR_GPU_CHECK(cudaStreamSynchronize(stream));
}

void Raytracer::renderGaussianTrainBackward(
    const GaussianTrainParams& trainParams,
    const uint32_t renderWidth, const uint32_t renderHeight)
{
    const size_t pixelCount = static_cast<size_t>(renderWidth) * renderHeight;
    if (pixelCount > queues.capacity)
        throw std::runtime_error("Training image exceeds the raytracing queue capacity");
    CameraInstance* activeCamera = scene.getActiveCamera();
    if (activeCamera == nullptr)
        throw std::runtime_error("Gaussian training requires an active camera");

    KernelParams params{};
    params.scene = gpuScene;
    params.scene.camera = activeCamera->getGpuCamera();
    params.queues = queues;
    params.train = trainParams;
    params.train.enabled = 1;
    params.train.mode = RenderMode::Backward;
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
        NR_GPU_CHECK(cudaMemsetAsync(queues.rayCounts, 0,
            sizeof(uint32_t) * MaxBounces, stream));
        params.frame.totalAccumulated = sample;
        params.depth = 0;
        launchGenerate(params, stream);
        launchExtend(params, queues.capacity, stream);
        launchShade(params, queues.capacity, stream);
    }
    NR_GPU_CHECK(cudaStreamSynchronize(stream));
}

void Raytracer::launchConnect(
    const KernelParams& params, const uint32_t launchCount, const cudaStream_t stream) const
{
    NR_GPU_CHECK(cudaMemcpyAsync(optixLaunchParamsDevice.get(),
        &params, sizeof(params), cudaMemcpyHostToDevice, stream));
    NR_OPTIX_CHECK(optixLaunch(optixPipeline.get(), stream,
        optixLaunchParamsDevice.devicePtr(), sizeof(KernelParams),
        &optixConnectSbt, launchCount, 1, 1));
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

void Raytracer::launchGenerateAov(const KernelParams& params, const cudaStream_t stream) const
{
    constexpr uint32_t blockSize = 256;
    const uint32_t count = params.frame.width * params.frame.height;
    const dim3 grid((count + blockSize - 1) / blockSize, 1, 1);
    void* args[] = { const_cast<KernelParams*>(&params) };
    NR_GPU_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(&generateAovKernel), grid, blockSize, args, 0, stream));
}

// Reuses __raygen__extend/optixExtendSbt with the AOV's own dense ray/hit buffers
// substituted in for depth 0, so no dedicated OptiX program group is needed.
void Raytracer::launchExtendAov(const KernelParams& params, const cudaStream_t stream) const
{
    NR_GPU_CHECK(cudaMemcpyAsync(params.queues.rayCounts, &params.queues.capacity,
        sizeof(uint32_t), cudaMemcpyHostToDevice, stream));

    KernelParams aovParams = params;
    aovParams.depth = 0;
    aovParams.frame.visibilityMask = 0x01; // mesh only, no gaussians
    aovParams.queues.rayQueues[0] = params.queues.aovRayQueue;
    aovParams.queues.hitQueue = params.queues.aovHitQueue;

    NR_GPU_CHECK(cudaMemcpyAsync(optixLaunchParamsDevice.get(),
        &aovParams, sizeof(aovParams), cudaMemcpyHostToDevice, stream));
    NR_OPTIX_CHECK(optixLaunch(optixPipeline.get(), stream,
        optixLaunchParamsDevice.devicePtr(), sizeof(KernelParams),
        &optixExtendSbt, params.queues.capacity, 1, 1));
}

void Raytracer::launchShadeAov(const KernelParams& params, const cudaStream_t stream) const
{
    constexpr uint32_t blockSize = 256;
    const uint32_t count = params.frame.width * params.frame.height;
    const dim3 grid((count + blockSize - 1) / blockSize, 1, 1);
    void* args[] = { const_cast<KernelParams*>(&params) };
    NR_GPU_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(&shadeAovKernel), grid, blockSize, args, 0, stream));
}

void Raytracer::renderFrame(const PushData& pushData)
{
    CameraInstance* activeCamera = scene.getRenderCamera();
    if (!activeCamera)
        return;

    const glm::uvec2 resolution = activeCamera->getCamera()->getSensor().resolution();
    if (resolution.x == 0 || resolution.y == 0)
        throw std::runtime_error("Cannot render with a zero-sized camera sensor");
    if (resolution.x != width || resolution.y != height)
        resize(resolution.x, resolution.y);

    if (scene.isDirty(Meshes)) updateMeshes();
    if (scene.isDirty(Textures)) updateTextures();
    else if (scene.isDirty(EnvironmentCdf)) updateEnvironmentCdf();
    if (scene.isDirty(Lights)) updateLights();
    if (scene.isDirty(TLAS)) updateTLAS();
    if (scene.isDirty(CameraState)) activeCamera->rebuildCamera();

    activeCamera->getCamera()->prepareForRender();

    if (scene.getEnvironment().cdfDirty != 0)
        updateEnvironmentCdf();

    gpuScene.renderSettings = scene.getRenderSettings();
    const uint32_t buffer = nextBuffer;
    const uint64_t frameValue = ++submittedFrame;
    const bool useViewportInterop = !context.isHeadless();
    if (useViewportInterop && lastUseValue[buffer] != 0)
    {
        cudaExternalSemaphoreWaitParams waitParams{};
        waitParams.params.fence.value = lastUseValue[buffer];
        cudaExternalSemaphore_t semaphore = cudaBufferReleased.get();
        NR_GPU_CHECK(cudaWaitExternalSemaphoresAsync(&semaphore, &waitParams, 1, stream));
    }

    if (m_timingEnabled && m_eventsRecorded) {
        NR_GPU_CHECK(cudaEventSynchronize(m_stopEvent.get()));
        NR_GPU_CHECK(cudaEventElapsedTime(&m_gpuTimeMs, m_startEvent.get(), m_stopEvent.get()));
    }

    KernelParams params{};
    params.scene = gpuScene;
    params.scene.renderSettings = scene.getRenderSettings();
    params.scene.camera = activeCamera->getGpuCamera();
    params.queues = queues;
    params.output = {
        color[buffer].getSurface(),
        albedo[buffer].getSurface(),
        normal[buffer].getSurface(),
        cryptomatte[buffer].getSurface(),
        position[buffer].getSurface(),
        width, height};
    params.frame.width = width;
    params.frame.height = height;
    params.frame.frameIndex = pushData.frame;
    params.accumulation = accumulationBuffer.as<glm::vec4>();

    const RenderSettings& renderSettings = scene.getRenderSettings();
    if (renderSettings.noiseLimitEnabled && !noiseMomentsBuffer)
        noiseMomentsBuffer.allocate(
            sizeof(glm::vec2) * static_cast<size_t>(width) * height, stream);
    params.noiseMoments = renderSettings.noiseLimitEnabled
        ? noiseMomentsBuffer.as<glm::vec2>() : nullptr;
    if (params.noiseMoments != nullptr && pushData.frame == 0)
        NR_GPU_CHECK(cudaMemsetAsync(
            params.noiseMoments, 0, sizeof(glm::vec2) * width * height, stream));
    params.frame.cutoffDistanceSq = renderSettings.gaussianCutoffSigma
                                 * renderSettings.gaussianCutoffSigma;
    const bool gaussianDirectColor =
        renderSettings.gaussianShadingMode == GaussianShadingMode::DirectColor;
    const uint32_t configuredSamplesPerFrame =
        static_cast<uint32_t>(std::max(1, renderSettings.samples));
    const uint32_t maxSamples = context.isHeadless()
        ? configuredSamplesPerFrame
        : static_cast<uint32_t>(std::max(1, renderSettings.maxSamples));
    const uint32_t remainingSamples = pushData.accumulatedSampleOffset < maxSamples
        ? maxSamples - pushData.accumulatedSampleOffset
        : configuredSamplesPerFrame;
    const uint32_t samplesPerFrame = std::min(configuredSamplesPerFrame, remainingSamples);
    const uint32_t requestedMaxShaderBounces = std::min(MaxBounces - 1,
        static_cast<uint32_t>(std::max(renderSettings.maxBounces, 1)));
    const uint32_t maxShaderBounces = gaussianDirectColor && gpuScene.meshInstanceCount == 0
        ? 1u
        : requestedMaxShaderBounces;

    Sensor& activeSensor = activeCamera->getCamera()->getSensor();
    prepareSensorFrame(activeSensor, params, pushData.frame == 0);

    // Shade only ever populates the shadow queue from analytic lights (both
    // mesh and Gaussian NEE) or environment importance sampling on a mesh
    // hit — never for Gaussians, which sample the environment through the
    // scattered ray instead. With no analytic lights and no meshes, Connect
    // would launch a full-width raygen every bounce just to find an empty
    // queue, so skip it entirely in that case.
    const bool mayGenerateShadowRays = gpuScene.meshInstanceCount > 0 ||
        (!gaussianDirectColor &&
            (gpuScene.pointLightCount > 0 || gpuScene.spotLightCount > 0 ||
             gpuScene.rectLightCount > 0 || gpuScene.directionalLightCount > 0));

    if (m_timingEnabled)
        NR_GPU_CHECK(cudaEventRecord(m_startEvent.get(), stream));

    if (pushData.frame == 0)
        aovStaleBuffers = 2;
    if (aovEnabled && aovStaleBuffers > 0)
    {
        kernelStats.time("GenerateAov", stream, [&] { launchGenerateAov(params, stream); });
        NR_GPU_CHECK(cudaGetLastError());
        kernelStats.time("ExtendAov", stream, [&] { launchExtendAov(params, stream); });
        NR_GPU_CHECK(cudaGetLastError());
        kernelStats.time("ShadeAov", stream, [&] { launchShadeAov(params, stream); });
        NR_GPU_CHECK(cudaGetLastError());
        --aovStaleBuffers;
    }

    if (renderSettings.gaussianProxyOverdrawVisualization)
    {
        kernelStats.time("GenerateProxyOverdrawRays", stream,
            [&] { launchGenerateAov(params, stream); });
        kernelStats.time("ProxyOverdraw", stream,
            [&] { launchProxyOverdraw(params, stream); });
    }
    else
    {
        for (uint32_t s = 0; s < samplesPerFrame; ++s)
        {
            NR_GPU_CHECK(cudaMemsetAsync(queues.rayCounts, 0, sizeof(uint32_t) * MaxBounces, stream));
            params.frame.totalAccumulated = pushData.accumulatedSampleOffset + s;

            kernelStats.time("Generate", stream, [&] { launchGenerate(params, stream); });
            NR_GPU_CHECK(cudaGetLastError());
            // pbrt-style wavefront: always launch over full capacity; each kernel
            // reads its device-side queue count and early-exits when
            // index >= count.  No CPU-GPU synchronization in the hot loop.
            for (uint32_t depth = 0; depth < maxShaderBounces; ++depth)
            {
                params.depth = depth;
                kernelStats.time("Extend", stream, [&] { launchExtend(params, queues.capacity, stream); });
                kernelStats.time("Shade", stream, [&] { launchShade(params, queues.capacity, stream); });
                if (mayGenerateShadowRays)
                    kernelStats.time("Connect", stream, [&] { launchConnect(params, queues.capacity, stream); });
            }
            launchSensorAddSample(activeSensor, params, stream);
        }
        const uint32_t accumulatedBeforeFrame = pushData.accumulatedSampleOffset;
        const uint32_t accumulatedAfterFrame = pushData.accumulatedSampleOffset + samplesPerFrame;
        const bool finalSample = accumulatedBeforeFrame < maxSamples && accumulatedAfterFrame >= maxSamples;
        applySensorAfterFrame(activeSensor, params, stream, finalSample);
        if (renderSettings.noiseLimitEnabled)
        {
            m_noiseResultSampleCount = accumulatedAfterFrame;
            kernelStats.time("NoiseVariance", stream,
                [&] { launchNoiseReduction(params, stream); });
        }
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
    lastLaunched = buffer;
    lastReadyValue = frameValue;
    lastUseValue[buffer] = frameValue;
    nextBuffer = 1 - buffer;
    scene.clearDirtyFlags();
}

void Raytracer::debugSave(const std::string& path) const
{
    NR_GPU_CHECK(cudaStreamSynchronize(stream));

    // Read back ray count for depth 0 to check if any rays were generated
    uint32_t rayCount0 = 0;
    NR_GPU_CHECK(cudaMemcpy(&rayCount0, queues.rayCounts, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    LOG_DEBUG("rayCounts[0]=" << rayCount0
              << " meshCount=" << scene.getMeshAssets().size()
              << " instanceCount=" << scene.getGpuInstanceCount()
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

FrameInfo Raytracer::getFrameInfo() const
{
    return {lastLaunched, lastReadyValue, renderReady.get(), bufferReleased.get()};
}

bool Raytracer::isRenderInFlight() const
{
    return lastReadyValue != 0
        && context.getDevice().getSemaphoreCounterValue(renderReady.get()) < lastReadyValue;
}

bool Raytracer::isFrameReady() const
{
    return lastReadyValue != 0
        && context.getDevice().getSemaphoreCounterValue(renderReady.get()) >= lastReadyValue;
}

float Raytracer::getGpuTimeMs()
{
    if (m_timingEnabled && m_eventsRecorded)
    {
        NR_GPU_CHECK(cudaEventSynchronize(m_stopEvent.get()));
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
