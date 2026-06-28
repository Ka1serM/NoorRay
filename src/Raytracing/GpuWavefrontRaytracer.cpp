#include "Raytracing/GpuWavefrontRaytracer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "stb_image_write.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "Camera/CameraInstance.h"
#include "GPU/Checks.h"
#include "GPU/CudaDevice.h"
#include "GPU/Memory.h"
#include "Kernels/Shade.h"
#include "Kernels/OptixLaunchParams.h"
#include "NoorRayOptixIr.h"
#include "Mesh/MeshAsset.h"
#include "Scene/MeshInstance.h"
#include "Scene/Scene.h"
#include "Vulkan/Context.h"

namespace
{
template <typename T>
T* allocateDevice(const size_t count, const cudaStream_t stream)
{
    return static_cast<T*>(nr::gpu::mallocDevice(sizeof(T) * count, stream));
}

template <typename T>
void uploadVector(T* destination, const std::vector<T>& source, const cudaStream_t stream)
{
    if (!source.empty())
        NR_GPU_CHECK(cudaMemcpyAsync(destination, source.data(), source.size() * sizeof(T),
                                     cudaMemcpyHostToDevice, stream));
}

std::array<float, 12> toOptixTransform(const mat4& matrix)
{
    return {matrix[0][0], matrix[1][0], matrix[2][0], matrix[3][0],
            matrix[0][1], matrix[1][1], matrix[2][1], matrix[3][1],
            matrix[0][2], matrix[1][2], matrix[2][2], matrix[3][2]};
}

void copyTransform(float destination[12], const mat4& matrix)
{
    const auto source = toOptixTransform(matrix);
    std::copy(source.begin(), source.end(), destination);
}

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

extern NR_GPU_KERNEL void generateKernel(const KernelParams);
extern NR_GPU_KERNEL void finalizeKernel(const KernelParams);
extern NR_GPU_KERNEL void shadeKernel(const KernelParams);

GpuWavefrontRaytracer::GpuWavefrontRaytracer(
    Context& context,
    Scene& scene,
    const uint32_t width,
    const uint32_t height)
    : context(context), scene(scene)
{
    selectCudaDeviceForVulkan(context.getPhysicalDevice());
    int leastPriority = 0;
    int greatestPriority = 0;
    NR_GPU_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    NR_GPU_CHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatestPriority));
    CUcontext cudaContext{};
    if (cuCtxGetCurrent(&cudaContext) != CUDA_SUCCESS || cudaContext == nullptr)
        throw std::runtime_error("CUDA primary context is unavailable");

    NR_OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions ctxOptions{};
    ctxOptions.logCallbackFunction = optixLogCallback;
    ctxOptions.logCallbackLevel = 3;
    NR_OPTIX_CHECK(optixDeviceContextCreate(cudaContext, &ctxOptions, &optixCtx));

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

    interop = std::make_unique<ImageInterop>(context);
    NR_GPU_CHECK(cudaEventCreate(&m_startEvent));
    NR_GPU_CHECK(cudaEventCreate(&m_stopEvent));
    setup(width, height);
}

GpuWavefrontRaytracer::~GpuWavefrontRaytracer()
{
    if (stream != nullptr)
        cudaStreamSynchronize(stream);
    triangleBlas.destroy(stream);
    tlas.destroy(stream);
    freeSceneData();
    freeQueues();
    if (stream != nullptr)
        cudaStreamSynchronize(stream);
    cudaEventDestroy(m_startEvent);
    cudaEventDestroy(m_stopEvent);
    interop.reset();

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
    if (optixCtx != nullptr) { optixDeviceContextDestroy(optixCtx); optixCtx = nullptr; }
    optixExtendSbt = {};
    optixConnectSbt = {};

    if (stream != nullptr)
        cudaStreamDestroy(stream);
}

void GpuWavefrontRaytracer::setup(const uint32_t newWidth, const uint32_t newHeight)
{
    if (newWidth == 0 || newHeight == 0)
        throw std::runtime_error("GPU renderer dimensions must be non-zero");
    NR_GPU_CHECK(cudaStreamSynchronize(stream));
    width = newWidth;
    height = newHeight;
    interop->resize(width, height);
    freeQueues();
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

void GpuWavefrontRaytracer::resize(const uint32_t newWidth, const uint32_t newHeight)
{
    if (newWidth != width || newHeight != height)
        setup(newWidth, newHeight);
}

void GpuWavefrontRaytracer::allocateQueues()
{
    queues.capacity = width * height;
    queues.rayCounts = allocateDevice<uint32_t>(MaxBounces, stream);
    queues.pathStates = allocateDevice<PathState>(queues.capacity, stream);
    queues.primaryStates = allocateDevice<PrimaryState>(queues.capacity, stream);
    queues.rayQueues[0] = allocateDevice<PathRayWorkItem>(queues.capacity, stream);
    queues.rayQueues[1] = allocateDevice<PathRayWorkItem>(queues.capacity, stream);
    queues.hitQueue = allocateDevice<HitWorkItem>(queues.capacity, stream);
    queues.shadowQueue = allocateDevice<ShadowWorkItem>(queues.capacity, stream);
    accumulation = allocateDevice<glm::vec4>(static_cast<size_t>(width) * height, stream);
    adaptiveState = allocateDevice<glm::vec4>(static_cast<size_t>(width) * height, stream);
    NR_GPU_CHECK(cudaMemsetAsync(accumulation, 0, sizeof(glm::vec4) * width * height, stream));
    NR_GPU_CHECK(cudaMemsetAsync(adaptiveState, 0, sizeof(glm::vec4) * width * height, stream));
}

void GpuWavefrontRaytracer::freeQueues() noexcept
{
    nr::gpu::freeDevice(queues.rayCounts, stream);
    nr::gpu::freeDevice(queues.pathStates, stream);
    nr::gpu::freeDevice(queues.primaryStates, stream);
    nr::gpu::freeDevice(queues.rayQueues[0], stream);
    nr::gpu::freeDevice(queues.rayQueues[1], stream);
    nr::gpu::freeDevice(queues.hitQueue, stream);
    nr::gpu::freeDevice(queues.shadowQueue, stream);
    nr::gpu::freeDevice(accumulation, stream);
    nr::gpu::freeDevice(adaptiveState, stream);
    queues = {};
    accumulation = nullptr;
    adaptiveState = nullptr;
}

void GpuWavefrontRaytracer::freeSceneData() noexcept
{
    nr::gpu::freeDevice(deviceMeshes, stream);
    nr::gpu::freeDevice(deviceInstances, stream);
    nr::gpu::freeDevice(deviceTextureObjects, stream);
    for (const DeviceMeshAllocation& allocation : meshAllocations)
    {
        nr::gpu::freeDevice(allocation.vertices, stream);
        nr::gpu::freeDevice(allocation.indices, stream);
        nr::gpu::freeDevice(allocation.faces, stream);
        nr::gpu::freeDevice(allocation.materials, stream);
    }
    for (const cudaTextureObject_t texture : textureObjects)
        cudaDestroyTextureObject(texture);
    for (const cudaArray_t array : textureArrays)
        cudaFreeArray(array);
    deviceMeshes = nullptr;
    deviceInstances = nullptr;
    deviceTextureObjects = nullptr;
    meshAllocations.clear();
    textureObjects.clear();
    textureArrays.clear();
    gpuScene = {};
    gpuScene.renderSettings = &scene.getRenderSettings();
    gpuScene.environment = scene.getEnvironment().settings;
}

void GpuWavefrontRaytracer::updateTextures()
{
    NR_GPU_CHECK(cudaStreamSynchronize(stream));
    if (deviceTextureObjects != nullptr)
        nr::gpu::freeDevice(deviceTextureObjects, stream);
    for (const cudaTextureObject_t texture : textureObjects)
        NR_GPU_CHECK(cudaDestroyTextureObject(texture));
    for (const cudaArray_t array : textureArrays)
        NR_GPU_CHECK(cudaFreeArray(array));
    deviceTextureObjects = nullptr;
    textureObjects.clear();
    textureArrays.clear();

    for (const Texture& texture : scene.getTextures())
    {
        cudaArray_t array{};
        const cudaChannelFormatDesc format = cudaCreateChannelDesc<float4>();
        NR_GPU_CHECK(cudaMallocArray(&array, &format, texture.getWidth(), texture.getHeight()));
        NR_GPU_CHECK(cudaMemcpy2DToArrayAsync(
            array, 0, 0, texture.getPixels().data(), texture.getWidth() * sizeof(float4),
            texture.getWidth() * sizeof(float4), texture.getHeight(), cudaMemcpyHostToDevice, stream));
        cudaResourceDesc resource{};
        resource.resType = cudaResourceTypeArray;
        resource.res.array.array = array;
        cudaTextureDesc description{};
        description.addressMode[0] = cudaAddressModeWrap;
        description.addressMode[1] = cudaAddressModeWrap;
        description.filterMode = cudaFilterModeLinear;
        description.readMode = cudaReadModeElementType;
        description.normalizedCoords = 1;
        description.mipmapFilterMode = cudaFilterModeLinear;
        cudaTextureObject_t object{};
        NR_GPU_CHECK(cudaCreateTextureObject(&object, &resource, &description, nullptr));
        textureArrays.push_back(array);
        textureObjects.push_back(object);
    }
    if (!textureObjects.empty())
    {
        deviceTextureObjects = allocateDevice<cudaTextureObject_t>(textureObjects.size(), stream);
        uploadVector(deviceTextureObjects, textureObjects, stream);
    }
    gpuScene.textures = deviceTextureObjects;
    gpuScene.textureCount = static_cast<uint32_t>(textureObjects.size());
}

void GpuWavefrontRaytracer::updateMeshes()
{
    NR_GPU_CHECK(cudaStreamSynchronize(stream));
    triangleBlas.destroy(stream);
    tlas.destroy(stream);
    nr::gpu::freeDevice(deviceMeshes, stream);
    nr::gpu::freeDevice(deviceInstances, stream);
    for (const DeviceMeshAllocation& allocation : meshAllocations)
    {
        nr::gpu::freeDevice(allocation.vertices, stream);
        nr::gpu::freeDevice(allocation.indices, stream);
        nr::gpu::freeDevice(allocation.faces, stream);
        nr::gpu::freeDevice(allocation.materials, stream);
    }
    meshAllocations.clear();

    std::vector<GpuMesh> gpuMeshes;
    std::vector<AccelMeshInput> accelMeshes;
    for (const auto& asset : scene.getMeshAssets())
    {
        asset->updateMaterials();
        DeviceMeshAllocation allocation{};
        allocation.vertices = allocateDevice<Vertex>(asset->getVertices().size(), stream);
        allocation.indices = allocateDevice<uint32_t>(asset->getIndices().size(), stream);
        allocation.faces = allocateDevice<Face>(asset->getFaces().size(), stream);
        std::vector<GpuMaterial> materials;
        materials.reserve(asset->getMaterials().size());
        for (const Material& source : asset->getMaterials())
        {
            GpuMaterial material{};
            material.albedo = source.albedo;
            material.albedoIndex = source.albedoIndex;
            material.specular = source.specular;
            material.metallic = source.metallic;
            material.roughness = source.roughness;
            material.ior = source.ior;
            material.specularIndex = source.specularIndex;
            material.metallicIndex = source.metallicIndex;
            material.roughnessIndex = source.roughnessIndex;
            material.normalIndex = source.normalIndex;
            material.transmissionColor = source.transmissionColor;
            material.transmission = source.transmission;
            material.transmissionIndex = source.transmissionIndex;
            material.emission = source.emission;
            material.emissionIndex = source.emissionIndex;
            material.emissionStrength = source.emissionStrength;
            material.opacity = source.opacity;
            material.opacityIndex = source.opacityIndex;
            materials.push_back(material);
        }
        allocation.materials = allocateDevice<GpuMaterial>(materials.size(), stream);
        uploadVector(allocation.vertices, asset->getVertices(), stream);
        uploadVector(allocation.indices, asset->getIndices(), stream);
        uploadVector(allocation.faces, asset->getFaces(), stream);
        uploadVector(allocation.materials, materials, stream);
        gpuMeshes.push_back({allocation.vertices, allocation.indices, allocation.faces, allocation.materials,
            static_cast<uint32_t>(asset->getVertices().size()),
            static_cast<uint32_t>(asset->getIndices().size() / 3),
            static_cast<uint32_t>(materials.size()), 0});
        accelMeshes.push_back({reinterpret_cast<CUdeviceptr>(allocation.vertices),
            static_cast<uint32_t>(asset->getVertices().size()), sizeof(Vertex),
            reinterpret_cast<CUdeviceptr>(allocation.indices),
            static_cast<uint32_t>(asset->getIndices().size() / 3)});
        meshAllocations.push_back(allocation);
    }
    if (!gpuMeshes.empty())
    {
        deviceMeshes = allocateDevice<GpuMesh>(gpuMeshes.size(), stream);
        uploadVector(deviceMeshes, gpuMeshes, stream);
    }
    triangleBlas.build(optixCtx, stream, accelMeshes);
    gpuScene.meshes = deviceMeshes;
    gpuScene.meshCount = static_cast<uint32_t>(gpuMeshes.size());

    std::vector<GpuInstance> instances;
    const std::vector<AccelInstanceInput> accelInstances = buildInstanceInputs(triangleBlas.getHandles(), &instances);
    if (!instances.empty())
    {
        deviceInstances = allocateDevice<GpuInstance>(instances.size(), stream);
        uploadVector(deviceInstances, instances, stream);
    }
    tlas.build(optixCtx, stream, accelInstances);
    gpuScene.instances = deviceInstances;
    gpuScene.instanceCount = static_cast<uint32_t>(instances.size());
    gpuScene.renderSettings = &scene.getRenderSettings();
    gpuScene.environment = scene.getEnvironment().settings;
}

std::vector<AccelInstanceInput> GpuWavefrontRaytracer::buildInstanceInputs(
    const std::vector<OptixTraversableHandle>& meshHandles,
    std::vector<GpuInstance>* gpuInstances) const
{
    std::vector<AccelInstanceInput> result;
    const auto instances = scene.getMeshInstances();
    result.reserve(instances.size());
    if (gpuInstances != nullptr)
        gpuInstances->reserve(instances.size());
    for (uint32_t index = 0; index < instances.size(); ++index)
    {
        const MeshInstance& instance = *instances[index];
        const mat4 objectToWorld = instance.getWorldTransform().getMatrix();
        const mat4 worldToObject = inverse(objectToWorld);
        const uint32_t meshIndex = instance.getMeshAsset().getMeshIndex();
        result.push_back({toOptixTransform(objectToWorld), meshHandles[meshIndex], index});
        if (gpuInstances != nullptr)
        {
            GpuInstance gpuInstance{};
            copyTransform(gpuInstance.objectToWorld, objectToWorld);
            copyTransform(gpuInstance.worldToObject, worldToObject);
            const mat4 normal = transpose(worldToObject);
            gpuInstance.normalToWorld[0] = normal[0][0]; gpuInstance.normalToWorld[1] = normal[1][0]; gpuInstance.normalToWorld[2] = normal[2][0];
            gpuInstance.normalToWorld[3] = normal[0][1]; gpuInstance.normalToWorld[4] = normal[1][1]; gpuInstance.normalToWorld[5] = normal[2][1];
            gpuInstance.normalToWorld[6] = normal[0][2]; gpuInstance.normalToWorld[7] = normal[1][2]; gpuInstance.normalToWorld[8] = normal[2][2];
            gpuInstance.meshIndex = instance.getMeshAsset().getMeshIndex();
            gpuInstance.objectIndex = index;
            gpuInstances->push_back(gpuInstance);
        }
    }
    return result;
}

void GpuWavefrontRaytracer::updateTLAS()
{
    NR_GPU_CHECK(cudaStreamSynchronize(stream));
    std::vector<GpuInstance> instances;
    const auto accelInstances = buildInstanceInputs(triangleBlas.getHandles(), &instances);
    if (instances.size() != gpuScene.instanceCount)
    {
        updateMeshes();
        return;
    }
    uploadVector(deviceInstances, instances, stream);
    tlas.update(optixCtx, stream, accelInstances);
    NR_GPU_CHECK(cudaStreamSynchronize(stream));
}

void GpuWavefrontRaytracer::launchGenerate(const KernelParams& params, const cudaStream_t stream) const
{
    constexpr uint32_t blockSize = 256;
    const uint32_t count = params.frame.width * params.frame.height;
    const dim3 grid((count + blockSize - 1) / blockSize, 1, 1);
    void* args[] = { const_cast<KernelParams*>(&params) };
    NR_GPU_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(generateKernel), grid, blockSize, args, 0, stream));
}

void GpuWavefrontRaytracer::launchFinalize(const KernelParams& params, const cudaStream_t stream) const
{
    constexpr uint32_t blockSize = 256;
    const uint32_t count = params.frame.width * params.frame.height;
    const dim3 grid((count + blockSize - 1) / blockSize, 1, 1);
    void* args[] = { const_cast<KernelParams*>(&params) };
    NR_GPU_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(finalizeKernel), grid, blockSize, args, 0, stream));
}

void GpuWavefrontRaytracer::launchShade(const KernelParams& params, const cudaStream_t stream) const
{
    constexpr uint32_t blockSize = 256;
    const dim3 grid((params.queues.capacity + blockSize - 1) / blockSize, 1, 1);
    void* args[] = { const_cast<KernelParams*>(&params) };
    NR_GPU_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(shadeKernel), grid, blockSize, args, 0, stream));
}

void GpuWavefrontRaytracer::launchExtend(const WavefrontQueues& queues, const uint32_t depth,
                                         const uint32_t capacity, const cudaStream_t stream) const
{
    const OptixLaunchParams params{queues, tlas.getTraversable(), depth};
    void* deviceParams = nullptr;
    NR_GPU_CHECK(cudaMallocAsync(&deviceParams, sizeof(params), stream));
    NR_GPU_CHECK(cudaMemcpyAsync(deviceParams, &params, sizeof(params), cudaMemcpyHostToDevice, stream));
    optixLaunch(optixPipeline, stream,
        reinterpret_cast<CUdeviceptr>(deviceParams), sizeof(OptixLaunchParams),
        &optixExtendSbt, capacity, 1, 1);
    NR_GPU_CHECK(cudaFreeAsync(deviceParams, stream));
}

void GpuWavefrontRaytracer::launchConnect(const WavefrontQueues& queues, const uint32_t depth,
                                          const uint32_t capacity, const cudaStream_t stream) const
{
    const OptixLaunchParams params{queues, tlas.getTraversable(), depth};
    void* deviceParams = nullptr;
    NR_GPU_CHECK(cudaMallocAsync(&deviceParams, sizeof(params), stream));
    NR_GPU_CHECK(cudaMemcpyAsync(deviceParams, &params, sizeof(params), cudaMemcpyHostToDevice, stream));
    optixLaunch(optixPipeline, stream,
        reinterpret_cast<CUdeviceptr>(deviceParams), sizeof(OptixLaunchParams),
        &optixConnectSbt, capacity, 1, 1);
    NR_GPU_CHECK(cudaFreeAsync(deviceParams, stream));
}

void GpuWavefrontRaytracer::render(const PushData& pushData)
{
    const uint32_t buffer = nextBuffer;
    const uint64_t frameValue = ++submittedFrame;
    if (lastUseValue[buffer] != 0)
        interop->waitForRelease(stream, lastUseValue[buffer]);

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
    params.scene.camera = scene.getActiveCamera()->getCamera();
    params.queues = queues;
    params.output = interop->getSurfaces(buffer);
    params.frame.frame = pushData.frame;
    params.frame.isMoving = pushData.isMoving;
    params.frame.pixelSizePercent = pushData.pixelSizePercent;
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
        for (uint32_t depth = 0; depth < maxShaderBounces; ++depth)
        {
            params.depth = depth;
            launchExtend(queues, depth, queues.capacity, stream);
            launchShade(params, stream);
            launchConnect(queues, depth, queues.capacity, stream);
        }
        launchFinalize(params, stream);
    }
    NR_GPU_CHECK(cudaPeekAtLastError());

    NR_GPU_CHECK(cudaEventRecord(m_stopEvent, stream));
    m_eventsRecorded = true;

    interop->signalReady(stream, frameValue);
    lastLaunched = buffer;
    lastReadyValue = frameValue;
    lastUseValue[buffer] = frameValue;
    nextBuffer = 1 - buffer;
}

void GpuWavefrontRaytracer::debugSave(const std::string& path) const
{
    NR_GPU_CHECK(cudaStreamSynchronize(stream));

    // Read back ray count for depth 0 to check if any rays were generated
    uint32_t rayCount0 = 0;
    NR_GPU_CHECK(cudaMemcpy(&rayCount0, queues.rayCounts, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::cout << "[debug] rayCounts[0]=" << rayCount0
              << " meshCount=" << gpuScene.meshCount
              << " instanceCount=" << gpuScene.instanceCount
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

FrameInfo GpuWavefrontRaytracer::getFrameInfo() const
{
    return {lastLaunched, lastReadyValue,
            interop->getRenderReadySemaphore(), interop->getBufferReleasedSemaphore()};
}

Image& GpuWavefrontRaytracer::getOutputColor() { return interop->getImage(lastLaunched, Aov::Color); }
Image& GpuWavefrontRaytracer::getOutputAlbedo() { return interop->getImage(lastLaunched, Aov::Albedo); }
Image& GpuWavefrontRaytracer::getOutputNormal() { return interop->getImage(lastLaunched, Aov::Normal); }
Image& GpuWavefrontRaytracer::getOutputCrypto() { return interop->getImage(lastLaunched, Aov::Cryptomatte); }
Image& GpuWavefrontRaytracer::getOutputPosition() { return interop->getImage(lastLaunched, Aov::Position); }
Image& GpuWavefrontRaytracer::getOutputMaterial() { return interop->getImage(lastLaunched, Aov::Material); }
Image& GpuWavefrontRaytracer::getOutputImage(const uint32_t bufferIndex, const Aov aov)
{
    return interop->getImage(bufferIndex, aov);
}
