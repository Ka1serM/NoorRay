#include "Backend/OptiX/Acceleration/Blas.h"

#include <utility>

#include <cuda_runtime.h>
#include <optix_stubs.h>

#include "Backend/CUDA/Checks.h"
namespace
{
// SVM selects the material from the face record in Geometry.h.  The OptiX
// hitgroup is therefore independent of material slots: every triangle uses
// the one shared mesh hitgroup record.
void setMaterialRouting(OptixBuildInput& buildInput)
{
    buildInput.triangleArray.numSbtRecords = 1;
}
}

Blas::Blas(Blas&& other) noexcept
    : handle(std::exchange(other.handle, {})),
      buffer(std::move(other.buffer)),
      vertexCount_(std::exchange(other.vertexCount_, {})),
      triangleCount_(std::exchange(other.triangleCount_, {})),
      vertexStride_(std::exchange(other.vertexStride_, {})),
      outputSizeInBytes_(std::exchange(other.outputSizeInBytes_, {}))
{
}

Blas& Blas::operator=(Blas&& other) noexcept
{
    if (this != &other)
    {
        reset();
        handle = std::exchange(other.handle, {});
        buffer = std::move(other.buffer);
        vertexCount_ = std::exchange(other.vertexCount_, {});
        triangleCount_ = std::exchange(other.triangleCount_, {});
        vertexStride_ = std::exchange(other.vertexStride_, {});
        outputSizeInBytes_ = std::exchange(other.outputSizeInBytes_, {});
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
    const uint32_t triangleCount,
    const int* materialIndices,
    const uint32_t materialIndexStride,
    const uint32_t materialSlotCount)
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
    static constexpr unsigned int geometryFlags[] = {OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT};
    buildInput.triangleArray.flags = geometryFlags;
    setMaterialRouting(buildInput);

    OptixAccelBuildOptions options{};
    options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
        | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
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

    vertexCount_ = vertexCount;
    triangleCount_ = triangleCount;
    vertexStride_ = vertexStride;
    outputSizeInBytes_ = sizes.outputSizeInBytes;
}

void Blas::refit(
    const OptixDeviceContext context,
    const cudaStream_t stream,
    const void* vertices,
    const uint32_t vertexCount,
    const uint32_t vertexStride,
    const uint32_t* indices,
    const uint32_t triangleCount,
    const int* materialIndices,
    const uint32_t materialIndexStride,
    const uint32_t materialSlotCount)
{
    if (handle == 0 || vertexCount != vertexCount_ || triangleCount != triangleCount_
        || vertexStride != vertexStride_)
    {
        build(context, stream, vertices, vertexCount, vertexStride, indices, triangleCount,
            materialIndices, materialIndexStride, materialSlotCount);
        return;
    }

    NR_GPU_CHECK(cudaStreamSynchronize(stream));

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
    static constexpr unsigned int geometryFlags[] = {OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT};
    buildInput.triangleArray.flags = geometryFlags;
    setMaterialRouting(buildInput);

    OptixAccelBuildOptions options{};
    options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
        | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    options.operation = OPTIX_BUILD_OPERATION_UPDATE;

    OptixAccelBufferSizes sizes{};
    NR_OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &options, &buildInput, 1, &sizes));

    nr::cuda::UniqueAsyncDeviceBuffer scratch(sizes.tempUpdateSizeInBytes, stream);
    NR_OPTIX_CHECK(optixAccelBuild(
        context, stream, &options, &buildInput, 1,
        scratch.devicePtr(), sizes.tempUpdateSizeInBytes,
        buffer.devicePtr(), outputSizeInBytes_,
        &handle, nullptr, 0));
}

void Blas::reset() noexcept
{
    buffer.reset();
    handle = {};
    vertexCount_ = {};
    triangleCount_ = {};
    vertexStride_ = {};
    outputSizeInBytes_ = {};
}
