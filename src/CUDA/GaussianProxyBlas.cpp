#include "CUDA/GaussianProxyBlas.h"

#include <array>
#include <cmath>
#include <utility>

#include <cuda_runtime.h>
#include <optix_stubs.h>

#include <glm/mat3x3.hpp>
#include <glm/vec3.hpp>

#include "CUDA/Checks.h"
#include "Mesh/GaussianCutoff.h"

static constexpr float Phi = 1.618033988749895f; // (1 + sqrt(5)) / 2

// Unit icosahedron — 12 vertices.
static constexpr std::array<float, 36> IcosahedronVertices =
{{
    // Ring at y = -1
    -1.0f,  Phi,  0.0f,
     1.0f,  Phi,  0.0f,
    -1.0f, -Phi,  0.0f,
     1.0f, -Phi,  0.0f,
    // Ring at y = 0
     0.0f, -1.0f,  Phi,
     0.0f,  1.0f,  Phi,
     0.0f, -1.0f, -Phi,
     0.0f,  1.0f, -Phi,
    // Ring at z = 0
     Phi,  0.0f, -1.0f,
     Phi,  0.0f,  1.0f,
    -Phi,  0.0f, -1.0f,
    -Phi,  0.0f,  1.0f,
}};

// 20 triangular faces (each entry is vertex index * 3).
// Wound so that outward-facing normals are correct for a unit icosahedron.
static constexpr std::array<uint32_t, 60> IcosahedronIndices =
{{
     0,  1,  4,
     0,  4, 11,
     0, 11,  5,
     0,  5,  1,
     1,  5,  9,
     1,  9,  8,
     1,  8,  4,
     4,  8,  2,
     4,  2, 11,
    11,  2, 10,
    11, 10,  5,
     5, 10,  6,
     5,  6,  9,
     9,  6,  7,
     9,  7,  8,
     8,  7,  3,
     8,  3,  2,
     2,  3, 10,
    10,  3,  6,
     6,  3,  7,
}};

// Geometry flags: NO DISABLE_ANYHIT — the any-hit program must run for
// Gaussian proxy triangles. This is the one geometry-level difference from
// mesh BLASes.
static constexpr unsigned int GaussianGeometryFlags = 0u;

GaussianProxyBlas::~GaussianProxyBlas() noexcept
{
    destroy();
}

GaussianProxyBlas::GaussianProxyBlas(GaussianProxyBlas&& other) noexcept
    : handle(std::exchange(other.handle, {})),
      buffer(std::exchange(other.buffer, {}))
{
}

GaussianProxyBlas& GaussianProxyBlas::operator=(GaussianProxyBlas&& other) noexcept
{
    if (this != &other)
    {
        destroy();
        handle = std::exchange(other.handle, {});
        buffer = std::exchange(other.buffer, {});
    }
    return *this;
}

void GaussianProxyBlas::build(
    const OptixDeviceContext context,
    const cudaStream_t stream)
{
    if (buffer != 0)
        destroy(stream);

    // Scale the shared unit icosahedron up by GaussianCutoffSigma once, here,
    // so it tightly bounds a default (unit-sigma) Gaussian's cutoff region.
    // Each Gaussian's own anisotropic scale is applied on top of this via the
    // per-instance transform (hardware TRS) — see GaussianAsset.cpp — instead
    // of correcting for it in the hit shader.
    const glm::mat3 cutoffScale(GaussianCutoffSigma);
    std::array<float, 36> scaledVertices{};
    for (size_t i = 0; i < IcosahedronVertices.size(); i += 3)
    {
        const glm::vec3 v = cutoffScale * glm::vec3(
            IcosahedronVertices[i], IcosahedronVertices[i + 1], IcosahedronVertices[i + 2]);
        scaledVertices[i] = v.x;
        scaledVertices[i + 1] = v.y;
        scaledVertices[i + 2] = v.z;
    }

    // Upload vertex + index data to device.
    constexpr size_t vertexBytes = IcosahedronVertices.size() * sizeof(float);
    constexpr size_t indexBytes  = IcosahedronIndices.size() * sizeof(uint32_t);
    CUdeviceptr vertexBuffer = 0;
    CUdeviceptr indexBuffer  = 0;
    NR_GPU_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&vertexBuffer), vertexBytes, stream));
    NR_GPU_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&indexBuffer),  indexBytes, stream));
    NR_GPU_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(vertexBuffer),
        scaledVertices.data(), vertexBytes, cudaMemcpyHostToDevice, stream));
    NR_GPU_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(indexBuffer),
        IcosahedronIndices.data(), indexBytes, cudaMemcpyHostToDevice, stream));

    OptixBuildInput buildInput{};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = 3 * sizeof(float);
    buildInput.triangleArray.numVertices = 12;
    buildInput.triangleArray.vertexBuffers = &vertexBuffer;
    buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.indexStrideInBytes = 3 * sizeof(uint32_t);
    buildInput.triangleArray.numIndexTriplets = 20;
    buildInput.triangleArray.indexBuffer = indexBuffer;
    buildInput.triangleArray.flags = &GaussianGeometryFlags;
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

    // Free the temporary vertex/index buffers after the BLAS is built.
    NR_GPU_CHECK(cudaFreeAsync(reinterpret_cast<void*>(vertexBuffer), stream));
    NR_GPU_CHECK(cudaFreeAsync(reinterpret_cast<void*>(indexBuffer), stream));
}

void GaussianProxyBlas::destroy(const cudaStream_t stream) noexcept
{
    if (buffer != 0)
        cudaFreeAsync(reinterpret_cast<void*>(buffer), stream);
    buffer = 0;
    handle = {};
}

void GaussianProxyBlas::destroy() noexcept
{
    if (buffer != 0)
        cudaFree(reinterpret_cast<void*>(buffer));
    buffer = 0;
    handle = {};
}
