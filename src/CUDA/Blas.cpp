#include "CUDA/Blas.h"

#include <utility>

#include <cuda_runtime.h>
#include <optix_stubs.h>

#include "CUDA/Checks.h"

Blas::~Blas() noexcept
{
    destroy();
}

Blas::Blas(Blas&& other) noexcept
    : handle(std::exchange(other.handle, {})),
      buffer(std::exchange(other.buffer, {}))
{
}

Blas& Blas::operator=(Blas&& other) noexcept
{
    if (this != &other)
    {
        destroy();
        handle = std::exchange(other.handle, {});
        buffer = std::exchange(other.buffer, {});
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
    if (buffer != 0)
        destroy(stream);

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

    void* scratch = nullptr;
    void* output = nullptr;
    NR_GPU_CHECK(cudaMallocAsync(&scratch, sizes.tempSizeInBytes, stream));
    NR_GPU_CHECK(cudaMallocAsync(&output, sizes.outputSizeInBytes, stream));
    NR_OPTIX_CHECK(optixAccelBuild(
        context, stream, &options, &buildInput, 1,
        reinterpret_cast<CUdeviceptr>(scratch), sizes.tempSizeInBytes,
        reinterpret_cast<CUdeviceptr>(output), sizes.outputSizeInBytes,
        &handle, nullptr, 0));
    NR_GPU_CHECK(cudaFreeAsync(scratch, stream));
    buffer = reinterpret_cast<CUdeviceptr>(output);
}

void Blas::destroy(const cudaStream_t stream) noexcept
{
    if (buffer != 0)
        cudaFreeAsync(reinterpret_cast<void*>(buffer), stream);
    buffer = 0;
    handle = {};
}

void Blas::destroy() noexcept
{
    if (buffer != 0)
        cudaFree(reinterpret_cast<void*>(buffer));
    buffer = 0;
    handle = {};
}
