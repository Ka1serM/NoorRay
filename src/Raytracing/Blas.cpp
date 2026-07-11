#include "Raytracing/Blas.h"

#include <utility>

#include <cuda_runtime.h>
#include <optix_stubs.h>

#include "CUDA/Checks.h"

Blas::Blas(Blas&& other) noexcept
    : handle(std::exchange(other.handle, {})),
      buffer(std::move(other.buffer))
{
}

Blas& Blas::operator=(Blas&& other) noexcept
{
    if (this != &other)
    {
        reset();
        handle = std::exchange(other.handle, {});
        buffer = std::move(other.buffer);
    }
    return *this;
}

void Blas::build(
    const OptixDeviceContext context,
    const cudaStream_t stream,
    const void* vertices,
    const uint32_t vertexCount,
    const uint32_t vertexStride,
    const uint32_t* indices,
    const uint32_t triangleCount)
{
    reset();

    const CUdeviceptr vertexBuffer = reinterpret_cast<CUdeviceptr>(vertices);
    OptixBuildInput buildInput{};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = vertexStride;
    buildInput.triangleArray.numVertices = vertexCount;
    buildInput.triangleArray.vertexBuffers = &vertexBuffer;
    buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.indexStrideInBytes = sizeof(uint32_t) * 3;
    buildInput.triangleArray.numIndexTriplets = triangleCount;
    buildInput.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(indices);
    static constexpr unsigned int geometryFlags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    buildInput.triangleArray.flags = &geometryFlags;
    buildInput.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions options{};
    options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    options.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes sizes{};
    NR_OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &options, &buildInput, 1, &sizes));

    nr::cuda::UniqueAsyncDeviceBuffer scratch(sizes.tempSizeInBytes, stream);
    buffer.allocate(sizes.outputSizeInBytes, stream);
    NR_OPTIX_CHECK(optixAccelBuild(
        context, stream, &options, &buildInput, 1,
        scratch.devicePtr(), sizes.tempSizeInBytes,
        buffer.devicePtr(), sizes.outputSizeInBytes,
        &handle, nullptr, 0));
}

void Blas::reset() noexcept
{
    buffer.reset();
    handle = {};
}
