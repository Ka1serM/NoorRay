#include "Raytracing/Raytracer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "stb_image_write.h"

#include <cuda_runtime.h>
#include <optix_stubs.h>

#include "Camera/CameraInstance.h"
#include "CUDA/Checks.h"
#include "CUDA/rstd/Memory.h"
#include "Raytracing/Shade.h"
#include "NoorRayOptixIr.h"
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

extern NR_GPU_KERNEL void generateKernel(KernelParams);
extern NR_GPU_KERNEL void finalizeKernel(KernelParams);
extern NR_GPU_KERNEL void shadeKernel(KernelParams);

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
    pipelineOptions.numPayloadValues = 0;
    pipelineOptions.numAttributeValues = 2;
    pipelineOptions.pipelineLaunchParamsVariableName = "params";
    pipelineOptions.pipelineLaunchParamsSizeInBytes = sizeof(KernelParams);
    pipelineOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    std::array<char, 8192> log{};
    size_t logSize = log.size();
    OptixResult result = optixModuleCreate(
        optixCtx, &moduleOptions, &pipelineOptions,
        reinterpret_cast<const char*>(noorRayOptixIr), noorRayOptixIrLength,
        log.data(), &logSize, &optixModule);
    if (result != OPTIX_SUCCESS)
        throw std::runtime_error(std::string("OptiX module creation failed: ") + log.data());

    optixExtendGroup = makeRaygenGroup(optixCtx, optixModule, "__raygen__extend");
    optixConnectGroup = makeRaygenGroup(optixCtx, optixModule, "__raygen__connect");

    OptixProgramGroupDesc triDesc{};
    triDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    OptixProgramGroupOptions groupOptions{};
    logSize = log.size();
    NR_OPTIX_CHECK(optixProgramGroupCreate(
        optixCtx, &triDesc, 1, &groupOptions, log.data(), &logSize, &optixTriangleGroup));

    OptixProgramGroupDesc missDesc{};
    missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missDesc.miss.module = nullptr;
    missDesc.miss.entryFunctionName = nullptr;
    logSize = log.size();
    NR_OPTIX_CHECK(optixProgramGroupCreate(
        optixCtx, &missDesc, 1, &groupOptions, log.data(), &logSize, &optixMissGroup));

    const std::array groups{optixExtendGroup, optixConnectGroup, optixTriangleGroup, optixMissGroup};
    OptixPipelineLinkOptions linkOptions{};
    linkOptions.maxTraceDepth = 1;
    logSize = log.size();
    result = optixPipelineCreate(
        optixCtx, &pipelineOptions, &linkOptions,
        groups.data(), static_cast<unsigned int>(groups.size()),
        log.data(), &logSize, &optixPipeline);
    if (result != OPTIX_SUCCESS)
        throw std::runtime_error(std::string("OptiX pipeline creation failed: ") + log.data());
    NR_OPTIX_CHECK(optixPipelineSetStackSize(optixPipeline, 0, 0, 0, 2));

    optixExtendRecord = uploadRecord(optixExtendGroup);
    optixConnectRecord = uploadRecord(optixConnectGroup);
    optixHitgroupRecord = uploadRecord(optixTriangleGroup);
    optixMissRecord = uploadRecord(optixMissGroup);

    optixExtendSbt.raygenRecord = optixExtendRecord;
    optixExtendSbt.missRecordBase = optixMissRecord;
    optixExtendSbt.missRecordStrideInBytes = sizeof(SbtRecord<>);
    optixExtendSbt.missRecordCount = 1;
    optixExtendSbt.hitgroupRecordBase = optixHitgroupRecord;
    optixExtendSbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<>);
    optixExtendSbt.hitgroupRecordCount = 1;
    optixConnectSbt = optixExtendSbt;
    optixConnectSbt.raygenRecord = optixConnectRecord;

    NR_GPU_CHECK(cudaEventCreate(&m_startEvent));
    NR_GPU_CHECK(cudaEventCreate(&m_stopEvent));

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
    cudaRenderReady = importSemaphore(renderReady.get());
    cudaBufferReleased = importSemaphore(bufferReleased.get());

    // Upload Jakob & Hanika sRGB-to-spectrum table once (9 MB coefficients + 64-element scale).
    constexpr size_t kScaleBytes = 64 * sizeof(float);
    constexpr size_t kCoeffBytes = sizeof(sRGBToSpectrumTable_Data);
    NR_GPU_CHECK(cudaMalloc(&spectrumTableScaleDevice, kScaleBytes));
    NR_GPU_CHECK(cudaMalloc(&spectrumTableCoeffsDevice, kCoeffBytes));
    NR_GPU_CHECK(cudaMemcpy(spectrumTableScaleDevice, sRGBToSpectrumTable_Scale, kScaleBytes, cudaMemcpyHostToDevice));
    NR_GPU_CHECK(cudaMemcpy(spectrumTableCoeffsDevice, sRGBToSpectrumTable_Data, kCoeffBytes, cudaMemcpyHostToDevice));
    gpuScene.spectrumTableScale  = spectrumTableScaleDevice;
    gpuScene.spectrumTableCoeffs = spectrumTableCoeffsDevice;

    constexpr size_t kD65Bytes = NrD65Samples * sizeof(float);
    NR_GPU_CHECK(cudaMalloc(&d65Device, kD65Bytes));
    NR_GPU_CHECK(cudaMemcpy(d65Device, NrD65, kD65Bytes, cudaMemcpyHostToDevice));
    gpuScene.d65 = d65Device;

    // Upload CIE 1931 2-degree CMF tables (471 floats × 3, ~5.5 KB).
    constexpr size_t kCieBytes = NrCIESamples * sizeof(float);
    NR_GPU_CHECK(cudaMalloc(&cieXDevice, kCieBytes));
    NR_GPU_CHECK(cudaMalloc(&cieYDevice, kCieBytes));
    NR_GPU_CHECK(cudaMalloc(&cieZDevice, kCieBytes));
    NR_GPU_CHECK(cudaMemcpy(cieXDevice, NrCIE_X, kCieBytes, cudaMemcpyHostToDevice));
    NR_GPU_CHECK(cudaMemcpy(cieYDevice, NrCIE_Y, kCieBytes, cudaMemcpyHostToDevice));
    NR_GPU_CHECK(cudaMemcpy(cieZDevice, NrCIE_Z, kCieBytes, cudaMemcpyHostToDevice));
    gpuScene.cieX = cieXDevice;
    gpuScene.cieY = cieYDevice;
    gpuScene.cieZ = cieZDevice;

    auto* cam = scene.getActiveCamera();
    auto* c = cam ? cam->getCamera() : nullptr;
    auto res = c ? c->getSensor().resolution() : glm::uvec2(1280, 720);
    resize(res.x, res.y);
}

Raytracer::~Raytracer()
{
    if (stream != nullptr)
        cudaStreamSynchronize(stream);
    tlas.destroy(stream);
    freeSceneData();
    freeQueues();
    if (stream != nullptr)
        cudaStreamSynchronize(stream);
    cudaEventDestroy(m_startEvent);
    cudaEventDestroy(m_stopEvent);
    for (auto& img : color) img.destroy();
    for (auto& img : albedo) img.destroy();
    for (auto& img : normal) img.destroy();
    for (auto& img : cryptomatte) img.destroy();
    for (auto& img : position) img.destroy();
    if (cudaRenderReady != nullptr)
        cudaDestroyExternalSemaphore(cudaRenderReady);
    if (cudaBufferReleased != nullptr)
        cudaDestroyExternalSemaphore(cudaBufferReleased);

    auto freeRecord = [](CUdeviceptr& ptr) {
        if (ptr != 0) { cudaFree(reinterpret_cast<void*>(ptr)); ptr = 0; }
    };
    freeRecord(optixExtendRecord);
    freeRecord(optixConnectRecord);
    freeRecord(optixHitgroupRecord);
    freeRecord(optixMissRecord);
    if (optixPipeline != nullptr) { optixPipelineDestroy(optixPipeline); optixPipeline = nullptr; }
    if (optixMissGroup != nullptr) { optixProgramGroupDestroy(optixMissGroup); optixMissGroup = nullptr; }
    if (optixTriangleGroup != nullptr) { optixProgramGroupDestroy(optixTriangleGroup); optixTriangleGroup = nullptr; }
    if (optixConnectGroup != nullptr) { optixProgramGroupDestroy(optixConnectGroup); optixConnectGroup = nullptr; }
    if (optixExtendGroup != nullptr) { optixProgramGroupDestroy(optixExtendGroup); optixExtendGroup = nullptr; }
    if (optixModule != nullptr) { optixModuleDestroy(optixModule); optixModule = nullptr; }
    optixExtendSbt = {};
    optixConnectSbt = {};
}

void Raytracer::resize(const uint32_t newWidth, const uint32_t newHeight)
{
    if (newWidth == width && newHeight == height)
        return;
    NR_GPU_CHECK(cudaStreamSynchronize(stream));
    width = newWidth;
    height = newHeight;
    freeQueues();

    auto createImages = [&](SharedImage (&arr)[2], const vk::Format format)
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
    submittedFrame = 0;
    lastUseValue = {};
}

void Raytracer::allocateQueues()
{
    queues.capacity = width * height;
    queues.rayCounts = nr::rstd::allocate_device<uint32_t>(MaxBounces, stream);
    queues.pathStates = nr::rstd::allocate_device<PathState>(queues.capacity, stream);
    queues.primaryStates = nr::rstd::allocate_device<PrimaryState>(queues.capacity, stream);
    queues.rayQueues[0] = nr::rstd::allocate_device<PathRayWorkItem>(queues.capacity, stream);
    queues.rayQueues[1] = nr::rstd::allocate_device<PathRayWorkItem>(queues.capacity, stream);
    queues.hitQueue = nr::rstd::allocate_device<HitWorkItem>(queues.capacity, stream);
    queues.shadowQueue = nr::rstd::allocate_device<ShadowWorkItem>(queues.capacity, stream);
    accumulation = nr::rstd::allocate_device<glm::vec4>(static_cast<size_t>(width) * height, stream);
    adaptiveState = nr::rstd::allocate_device<glm::vec4>(static_cast<size_t>(width) * height, stream);
    NR_GPU_CHECK(cudaMemsetAsync(accumulation, 0, sizeof(glm::vec4) * width * height, stream));
    NR_GPU_CHECK(cudaMemsetAsync(adaptiveState, 0, sizeof(glm::vec4) * width * height, stream));
}

void Raytracer::freeQueues() noexcept
{
    nr::rstd::deallocate_device(queues.rayCounts, stream);
    nr::rstd::deallocate_device(queues.pathStates, stream);
    nr::rstd::deallocate_device(queues.primaryStates, stream);
    nr::rstd::deallocate_device(queues.rayQueues[0], stream);
    nr::rstd::deallocate_device(queues.rayQueues[1], stream);
    nr::rstd::deallocate_device(queues.hitQueue, stream);
    nr::rstd::deallocate_device(queues.shadowQueue, stream);
    nr::rstd::deallocate_device(accumulation, stream);
    nr::rstd::deallocate_device(adaptiveState, stream);
    queues = {};
    accumulation = nullptr;
    adaptiveState = nullptr;
}

void Raytracer::freeSceneData() noexcept
{
    scene.getCudaTexturesRef().clear();
    scene.getEnvironment().destroyCdf();
    scene.getGpuInstancesRef().clear();
    if (spectrumTableScaleDevice)  { cudaFree(spectrumTableScaleDevice);  spectrumTableScaleDevice  = nullptr; }
    if (spectrumTableCoeffsDevice) { cudaFree(spectrumTableCoeffsDevice); spectrumTableCoeffsDevice = nullptr; }
    if (d65Device) { cudaFree(d65Device); d65Device = nullptr; }
    if (cieXDevice) { cudaFree(cieXDevice); cieXDevice = nullptr; }
    if (cieYDevice) { cudaFree(cieYDevice); cieYDevice = nullptr; }
    if (cieZDevice) { cudaFree(cieZDevice); cieZDevice = nullptr; }
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
    const cudaChannelFormatDesc format = cudaCreateChannelDesc<float4>();
    NR_GPU_CHECK(cudaMallocArray(
        &env.cdfArray, &format, texture.getWidth(), texture.getHeight()));
    NR_GPU_CHECK(cudaMemcpy2DToArrayAsync(
        env.cdfArray, 0, 0, cdf.data(), texture.getWidth() * sizeof(float4),
        texture.getWidth() * sizeof(float4), texture.getHeight(),
        cudaMemcpyHostToDevice, stream));

    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = env.cdfArray;
    cudaTextureDesc description{};
    description.addressMode[0] = cudaAddressModeClamp;
    description.addressMode[1] = cudaAddressModeClamp;
    description.filterMode = cudaFilterModePoint;
    description.readMode = cudaReadModeElementType;
    description.normalizedCoords = 1;
    NR_GPU_CHECK(cudaCreateTextureObject(
        &env.cdfTexture, &resource, &description, nullptr));
}

void Raytracer::updateMeshes()
{
    gpuScene.meshes = scene.getMeshAssets().data();
    gpuScene.meshCount = static_cast<uint32_t>(scene.getMeshAssets().size());
}

void Raytracer::updateTLAS()
{
    NR_GPU_CHECK(cudaStreamSynchronize(stream));
    auto& gpuInstances = scene.getGpuInstancesRef();
    tlas.build(optixCtx, stream, scene, gpuInstances);
    gpuScene.tlasHandle = tlas.getTraversable();
    gpuScene.instances = gpuInstances.data();
    gpuScene.instanceCount = static_cast<uint32_t>(gpuInstances.size());
    NR_GPU_CHECK(cudaStreamSynchronize(stream));
}

void Raytracer::updateLights()
{
    // Scene light arrays are managed allocations and may move when their size
    // changes. Refresh both pointers and counts before the next kernel launch.
    NR_GPU_CHECK(cudaStreamSynchronize(stream));
    gpuScene.pointLights = scene.getPointLights();
    gpuScene.spotLights = scene.getSpotLights();
    gpuScene.rectLights = scene.getRectLights();
    gpuScene.directionalLights = scene.getDirectionalLights();
    gpuScene.pointLightCount = scene.getPointLightCount();
    gpuScene.spotLightCount = scene.getSpotLightCount();
    gpuScene.rectLightCount = scene.getRectLightCount();
    gpuScene.directionalLightCount = scene.getDirectionalLightCount();
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

void Raytracer::launchShade(const KernelParams& params, const cudaStream_t stream) const
{
    constexpr uint32_t blockSize = 256;
    const dim3 grid((params.queues.capacity + blockSize - 1) / blockSize, 1, 1);
    void* args[] = { const_cast<KernelParams*>(&params) };
    NR_GPU_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(&shadeKernel), grid, blockSize, args, 0, stream));
}

void Raytracer::launchExtend(const KernelParams& params, const cudaStream_t stream) const
{
    void* deviceParams = nullptr;
    NR_GPU_CHECK(cudaMallocAsync(&deviceParams, sizeof(params), stream));
    NR_GPU_CHECK(cudaMemcpyAsync(deviceParams, &params, sizeof(params), cudaMemcpyHostToDevice, stream));
    optixLaunch(optixPipeline, stream,
        reinterpret_cast<CUdeviceptr>(deviceParams), sizeof(KernelParams),
        &optixExtendSbt, params.queues.capacity, 1, 1);
    NR_GPU_CHECK(cudaFreeAsync(deviceParams, stream));
}

void Raytracer::launchConnect(const KernelParams& params, const cudaStream_t stream) const
{
    void* deviceParams = nullptr;
    NR_GPU_CHECK(cudaMallocAsync(&deviceParams, sizeof(params), stream));
    NR_GPU_CHECK(cudaMemcpyAsync(deviceParams, &params, sizeof(params), cudaMemcpyHostToDevice, stream));
    optixLaunch(optixPipeline, stream,
        reinterpret_cast<CUdeviceptr>(deviceParams), sizeof(KernelParams),
        &optixConnectSbt, params.queues.capacity, 1, 1);
    NR_GPU_CHECK(cudaFreeAsync(deviceParams, stream));
}

void Raytracer::render(const PushData& pushData)
{
    CameraInstance* activeCamera = scene.getActiveCamera();
    if (!activeCamera)
        return;

    gpuScene.renderSettings = scene.getRenderSettings();
    const uint32_t buffer = nextBuffer;
    const uint64_t frameValue = ++submittedFrame;
    if (lastUseValue[buffer] != 0)
    {
        cudaExternalSemaphoreWaitParams waitParams{};
        waitParams.params.fence.value = lastUseValue[buffer];
        NR_GPU_CHECK(cudaWaitExternalSemaphoresAsync(&cudaBufferReleased, &waitParams, 1, stream));
    }

    if (m_eventsRecorded) {
        NR_GPU_CHECK(cudaEventSynchronize(m_stopEvent));
        NR_GPU_CHECK(cudaEventElapsedTime(&m_gpuTimeMs, m_startEvent, m_stopEvent));
    }

    if (pushData.frame == 0)
    {
        NR_GPU_CHECK(cudaMemsetAsync(accumulation, 0, sizeof(glm::vec4) * width * height, stream));
        NR_GPU_CHECK(cudaMemsetAsync(adaptiveState, 0, sizeof(glm::vec4) * width * height, stream));
    }

    KernelParams params{};
    params.scene = gpuScene;
    params.scene.renderSettings = scene.getRenderSettings();
    params.scene.camera = activeCamera->getCamera();
    params.queues = queues;
    params.output = {
        color[buffer].getSurface(),
        albedo[buffer].getSurface(),
        normal[buffer].getSurface(),
        cryptomatte[buffer].getSurface(),
        position[buffer].getSurface(),
        width, height};
    params.frame.frame = pushData.frame;
    params.frame.isMoving = pushData.isMoving;
    params.frame.width = width;
    params.frame.height = height;
    params.accumulation = accumulation;
    params.adaptiveState = adaptiveState;

    const RenderSettings& renderSettings = scene.getRenderSettings();
    const uint32_t samplesPerFrame = static_cast<uint32_t>(std::max(1, renderSettings.samples));
    const uint32_t maxShaderBounces = std::min(MaxBounces - 1,
        static_cast<uint32_t>(std::max({renderSettings.diffuseBounces,
                                        renderSettings.specularBounces,
                                        renderSettings.transmissionBounces, 1}) + 1));

    NR_GPU_CHECK(cudaEventRecord(m_startEvent, stream));

    for (uint32_t s = 0; s < samplesPerFrame; ++s)
    {
        NR_GPU_CHECK(cudaMemsetAsync(queues.rayCounts, 0, sizeof(uint32_t) * MaxBounces, stream));
        NR_GPU_CHECK(cudaMemsetAsync(queues.pathStates, 0, sizeof(PathState) * queues.capacity, stream));
        params.frame.sampleIndex = s;
        params.frame.totalAccumulated = static_cast<uint32_t>(pushData.frame) * samplesPerFrame + s;

        launchGenerate(params, stream);
        NR_GPU_CHECK(cudaGetLastError());
        for (uint32_t depth = 0; depth < maxShaderBounces; ++depth)
        {
            params.depth = depth;
            launchExtend(params, stream);
            launchShade(params, stream);
            launchConnect(params, stream);
        }
        launchFinalize(params, stream);
    }
    NR_GPU_CHECK(cudaPeekAtLastError());

    NR_GPU_CHECK(cudaEventRecord(m_stopEvent, stream));
    m_eventsRecorded = true;

    {
        cudaExternalSemaphoreSignalParams signalParams{};
        signalParams.params.fence.value = frameValue;
        NR_GPU_CHECK(cudaSignalExternalSemaphoresAsync(&cudaRenderReady, &signalParams, 1, stream));
    }
    lastLaunched = buffer;
    lastReadyValue = frameValue;
    lastUseValue[buffer] = frameValue;
    nextBuffer = 1 - buffer;
}

void Raytracer::debugSave(const std::string& path) const
{
    NR_GPU_CHECK(cudaStreamSynchronize(stream));

    // Read back ray count for depth 0 to check if any rays were generated
    uint32_t rayCount0 = 0;
    NR_GPU_CHECK(cudaMemcpy(&rayCount0, queues.rayCounts, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::cout << "[debug] rayCounts[0]=" << rayCount0
              << " meshCount=" << scene.getMeshAssets().size()
              << " instanceCount=" << scene.getGpuInstanceCount()
              << " traversable=" << tlas.getTraversable()
              << " size=" << width << "x" << height << "\n";

    const size_t pixelCount = static_cast<size_t>(width) * height;
    std::vector<glm::vec4> host(pixelCount);
    NR_GPU_CHECK(cudaMemcpy(host.data(), accumulation, pixelCount * sizeof(glm::vec4), cudaMemcpyDeviceToHost));

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
    std::cout << "[debug] R: max=" << maxR << " avg=" << sumR/n
              << "  G: max=" << maxG << " avg=" << sumG/n
              << "  B: max=" << maxB << " avg=" << sumB/n
              << "  A: min=" << minA << " max=" << maxA << "\n";
    // Print first 4 pixel values
    for (int i = 0; i < 4; ++i)
        printf("[debug] pixel[%d] r=%.4f g=%.4f b=%.4f a=%.4f\n", i,
            host[i].x, host[i].y, host[i].z, host[i].w);
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
        std::cout << "[debug] saved " << path << "\n";
    else
        std::cerr << "[debug] failed to write " << path << "\n";
}

FrameInfo Raytracer::getFrameInfo() const
{
    return {lastLaunched, lastReadyValue, renderReady.get(), bufferReleased.get()};
}
