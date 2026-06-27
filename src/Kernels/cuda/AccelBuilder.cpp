#include "Kernels/cuda/AccelBuilder.h"

#include <algorithm>
#include <stdexcept>

#include <optix_stubs.h>

#include "GPU/Checks.h"

AccelBuilder::~AccelBuilder()
{
    destroy();
}

void AccelBuilder::initialize(const OptixDeviceContext newContext, const cudaStream_t newStream)
{
    context = newContext;
    stream = newStream;
}

void AccelBuilder::rebuild(
    const std::vector<AccelMeshInput>& meshes,
    const std::vector<AccelInstanceInput>& instances)
{
    destroy();
    if (context == nullptr || stream == nullptr)
        throw std::runtime_error("AccelBuilder is not initialized");

    blasBuffers.reserve(meshes.size());
    blasHandles.reserve(meshes.size());
    for (const AccelMeshInput& mesh : meshes)
    {
        OptixBuildInput buildInput{};
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        buildInput.triangleArray.vertexStrideInBytes = mesh.vertexStride;
        buildInput.triangleArray.numVertices = mesh.vertexCount;
        buildInput.triangleArray.vertexBuffers = &mesh.vertices;
        buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        buildInput.triangleArray.indexStrideInBytes = sizeof(uint32_t) * 3;
        buildInput.triangleArray.numIndexTriplets = mesh.triangleCount;
        buildInput.triangleArray.indexBuffer = mesh.indices;
        static constexpr unsigned int geometryFlags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
        buildInput.triangleArray.flags = &geometryFlags;
        buildInput.triangleArray.numSbtRecords = 1;

        OptixAccelBuildOptions options{};
        options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        options.operation = OPTIX_BUILD_OPERATION_BUILD;
        OptixAccelBufferSizes sizes{};
        NR_OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &options, &buildInput, 1, &sizes));

        void* scratch = nullptr;
        void* output = nullptr;
        NR_GPU_CHECK(cudaMallocAsync(&scratch, sizes.tempSizeInBytes, stream));
        NR_GPU_CHECK(cudaMallocAsync(&output, sizes.outputSizeInBytes, stream));
        OptixTraversableHandle handle{};
        NR_OPTIX_CHECK(optixAccelBuild(
            context,
            stream,
            &options,
            &buildInput,
            1,
            reinterpret_cast<CUdeviceptr>(scratch),
            sizes.tempSizeInBytes,
            reinterpret_cast<CUdeviceptr>(output),
            sizes.outputSizeInBytes,
            &handle,
            nullptr,
            0));
        NR_GPU_CHECK(cudaFreeAsync(scratch, stream));
        blasBuffers.push_back(reinterpret_cast<CUdeviceptr>(output));
        blasHandles.push_back(handle);
    }
    buildTlas(instances, OPTIX_BUILD_OPERATION_BUILD);
}

void AccelBuilder::updateInstances(const std::vector<AccelInstanceInput>& instances)
{
    if (instances.size() != instanceCount)
        buildTlas(instances, OPTIX_BUILD_OPERATION_BUILD);
    else
        buildTlas(instances, OPTIX_BUILD_OPERATION_UPDATE);
}

void AccelBuilder::buildTlas(
    const std::vector<AccelInstanceInput>& instances,
    const OptixBuildOperation operation)
{
    if (instances.empty())
    {
        tlasHandle = 0;
        return;
    }

    std::vector<OptixInstance> optixInstances(instances.size());
    for (size_t index = 0; index < instances.size(); ++index)
    {
        const AccelInstanceInput& source = instances[index];
        if (source.meshIndex >= blasHandles.size())
            throw std::runtime_error("OptiX instance references an invalid mesh index");
        OptixInstance& destination = optixInstances[index];
        std::copy(source.transform.begin(), source.transform.end(), destination.transform);
        destination.instanceId = source.instanceId;
        destination.sbtOffset = 0;
        destination.visibilityMask = 0xff;
        destination.flags = OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
        destination.traversableHandle = blasHandles[source.meshIndex];
    }

    const size_t instanceBytes = optixInstances.size() * sizeof(OptixInstance);
    if (instanceBuffer == 0 || instanceCount != instances.size())
    {
        if (instanceBuffer != 0)
            NR_GPU_CHECK(cudaFreeAsync(reinterpret_cast<void*>(instanceBuffer), stream));
        void* allocation = nullptr;
        NR_GPU_CHECK(cudaMallocAsync(&allocation, instanceBytes, stream));
        instanceBuffer = reinterpret_cast<CUdeviceptr>(allocation);
    }
    NR_GPU_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(instanceBuffer),
        optixInstances.data(),
        instanceBytes,
        cudaMemcpyHostToDevice,
        stream));

    OptixBuildInput buildInput{};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    buildInput.instanceArray.instances = instanceBuffer;
    buildInput.instanceArray.numInstances = static_cast<unsigned int>(instances.size());
    OptixAccelBuildOptions options{};
    options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    options.operation = operation;
    OptixAccelBufferSizes sizes{};
    NR_OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &options, &buildInput, 1, &sizes));

    if (operation == OPTIX_BUILD_OPERATION_BUILD && sizes.outputSizeInBytes > tlasBufferSize)
    {
        if (tlasBuffer != 0)
            NR_GPU_CHECK(cudaFreeAsync(reinterpret_cast<void*>(tlasBuffer), stream));
        void* allocation = nullptr;
        NR_GPU_CHECK(cudaMallocAsync(&allocation, sizes.outputSizeInBytes, stream));
        tlasBuffer = reinterpret_cast<CUdeviceptr>(allocation);
        tlasBufferSize = sizes.outputSizeInBytes;
    }

    const size_t scratchSize = operation == OPTIX_BUILD_OPERATION_UPDATE
        ? sizes.tempUpdateSizeInBytes
        : sizes.tempSizeInBytes;
    void* scratch = nullptr;
    NR_GPU_CHECK(cudaMallocAsync(&scratch, scratchSize, stream));
    NR_OPTIX_CHECK(optixAccelBuild(
        context,
        stream,
        &options,
        &buildInput,
        1,
        reinterpret_cast<CUdeviceptr>(scratch),
        scratchSize,
        tlasBuffer,
        tlasBufferSize,
        &tlasHandle,
        nullptr,
        0));
    NR_GPU_CHECK(cudaFreeAsync(scratch, stream));
    instanceCount = static_cast<uint32_t>(instances.size());
}

void AccelBuilder::destroy() noexcept
{
    if (stream != nullptr)
    {
        if (instanceBuffer != 0)
            cudaFreeAsync(reinterpret_cast<void*>(instanceBuffer), stream);
        if (tlasBuffer != 0)
            cudaFreeAsync(reinterpret_cast<void*>(tlasBuffer), stream);
        for (const CUdeviceptr buffer : blasBuffers)
            cudaFreeAsync(reinterpret_cast<void*>(buffer), stream);
    }
    instanceBuffer = 0;
    tlasBuffer = 0;
    tlasBufferSize = 0;
    instanceCount = 0;
    tlasHandle = 0;
    blasBuffers.clear();
    blasHandles.clear();
}
