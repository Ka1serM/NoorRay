#include "Backend/OptiX/Acceleration/AnalyticLightBlas.h"

#include <cuda_runtime.h>
#include <optix_stubs.h>

#include "Backend/CUDA/Checks.h"

void AnalyticLightBlas::build(
    const OptixDeviceContext context,
    const cudaStream_t stream,
    const std::vector<OptixAabb>& aabbs)
{
    reset();
    if (aabbs.empty())
        return;

    nr::cuda::UniqueAsyncDeviceBuffer aabbBuffer(
        aabbs.size() * sizeof(OptixAabb), stream);
    NR_GPU_CHECK(cudaMemcpyAsync(aabbBuffer.get(), aabbs.data(),
        aabbs.size() * sizeof(OptixAabb), cudaMemcpyHostToDevice, stream));

    OptixBuildInput buildInput{};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    CUdeviceptr aabbPointer = aabbBuffer.devicePtr();
    buildInput.customPrimitiveArray.aabbBuffers = &aabbPointer;
    buildInput.customPrimitiveArray.numPrimitives =
        static_cast<uint32_t>(aabbs.size());
    static constexpr unsigned int geometryFlags[] = {
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT};
    buildInput.customPrimitiveArray.flags = geometryFlags;
    buildInput.customPrimitiveArray.numSbtRecords = 1;

    OptixAccelBuildOptions options{};
    options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    options.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes sizes{};
    NR_OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context, &options, &buildInput, 1, &sizes));

    nr::cuda::UniqueAsyncDeviceBuffer scratch(sizes.tempSizeInBytes, stream);
    buffer.allocate(sizes.outputSizeInBytes, stream);
    NR_OPTIX_CHECK(optixAccelBuild(
        context, stream, &options, &buildInput, 1,
        scratch.devicePtr(), sizes.tempSizeInBytes,
        buffer.devicePtr(), sizes.outputSizeInBytes,
        &handle, nullptr, 0));
    NR_GPU_CHECK(cudaStreamSynchronize(stream));
    primitiveCount_ = static_cast<uint32_t>(aabbs.size());
}

void AnalyticLightBlas::reset() noexcept
{
    buffer.reset();
    handle = {};
    primitiveCount_ = 0;
}
