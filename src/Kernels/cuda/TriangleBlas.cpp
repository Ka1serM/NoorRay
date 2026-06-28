#include "Kernels/cuda/TriangleBlas.h"

#include <stdexcept>

#include <optix_stubs.h>

#include "GPU/Checks.h"

TriangleBlas::~TriangleBlas()
{
    // destroy() should be called explicitly with a valid stream
}

void TriangleBlas::build(
    const OptixDeviceContext context,
    const cudaStream_t stream,
    const std::vector<AccelMeshInput>& meshes)
{
    destroy(stream);

    buffers.reserve(meshes.size());
    handles.reserve(meshes.size());
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
        buffers.push_back(reinterpret_cast<CUdeviceptr>(output));
        handles.push_back(handle);
    }
}

void TriangleBlas::destroy(const cudaStream_t stream) noexcept
{
    for (const CUdeviceptr buffer : buffers)
        cudaFreeAsync(reinterpret_cast<void*>(buffer), stream);
    buffers.clear();
    handles.clear();
}
