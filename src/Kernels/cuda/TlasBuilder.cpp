#include "Kernels/cuda/TlasBuilder.h"

#include <algorithm>
#include <stdexcept>

#include <optix_stubs.h>

#include "GPU/Checks.h"

TlasBuilder::~TlasBuilder()
{
    // destroy() should be called explicitly with a valid stream
}

void TlasBuilder::build(
    const OptixDeviceContext context,
    const cudaStream_t stream,
    const std::vector<AccelInstanceInput>& instances)
{
    destroy(stream);
    buildInternal(context, stream, instances, OPTIX_BUILD_OPERATION_BUILD);
}

void TlasBuilder::update(
    const OptixDeviceContext context,
    const cudaStream_t stream,
    const std::vector<AccelInstanceInput>& instances)
{
    const OptixBuildOperation operation = instances.size() != instanceCount
        ? OPTIX_BUILD_OPERATION_BUILD
        : OPTIX_BUILD_OPERATION_UPDATE;
    buildInternal(context, stream, instances, operation);
}

void TlasBuilder::buildInternal(
    const OptixDeviceContext context,
    const cudaStream_t stream,
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
        OptixInstance& destination = optixInstances[index];
        std::copy(source.transform.begin(), source.transform.end(), destination.transform);
        destination.instanceId = source.instanceId;
        destination.sbtOffset = 0;
        destination.visibilityMask = 0xff;
        destination.flags = OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
        destination.traversableHandle = source.blasHandle;
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

void TlasBuilder::destroy(const cudaStream_t stream) noexcept
{
    if (instanceBuffer != 0)
        cudaFreeAsync(reinterpret_cast<void*>(instanceBuffer), stream);
    if (tlasBuffer != 0)
        cudaFreeAsync(reinterpret_cast<void*>(tlasBuffer), stream);
    instanceBuffer = 0;
    tlasBuffer = 0;
    tlasBufferSize = 0;
    instanceCount = 0;
    tlasHandle = 0;
}
