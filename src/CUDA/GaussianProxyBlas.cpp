#include "CUDA/GaussianProxyBlas.h"

#include <array>
#include <cmath>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <optix_stubs.h>

#include <glm/mat3x3.hpp>
#include <glm/vec3.hpp>

#include "CUDA/Checks.h"

// ── Icosahedron ────────────────────────────────────────────────────────────
static constexpr std::array<float, 36> IcosahedronVertices =
{{
    -1.0f,  1.618033988749895f,  0.0f,
     1.0f,  1.618033988749895f,  0.0f,
    -1.0f, -1.618033988749895f,  0.0f,
     1.0f, -1.618033988749895f,  0.0f,
     0.0f, -1.0f,  1.618033988749895f,
     0.0f,  1.0f,  1.618033988749895f,
     0.0f, -1.0f, -1.618033988749895f,
     0.0f,  1.0f, -1.618033988749895f,
     1.618033988749895f,  0.0f, -1.0f,
     1.618033988749895f,  0.0f,  1.0f,
    -1.618033988749895f,  0.0f, -1.0f,
    -1.618033988749895f,  0.0f,  1.0f,
}};

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

static constexpr size_t IcosahedronVertCount  = 12;
static constexpr size_t IcosahedronTriCount   = 20;

// ── Octahedron ─────────────────────────────────────────────────────────────
static constexpr std::array<float, 18> OctahedronVertices =
{{
     0.0f,  1.0f,  0.0f,
     0.0f, -1.0f,  0.0f,
     1.0f,  0.0f,  0.0f,
     0.0f,  0.0f,  1.0f,
    -1.0f,  0.0f,  0.0f,
     0.0f,  0.0f, -1.0f,
}};

static constexpr std::array<uint32_t, 24> OctahedronIndices =
{{
     0,  3,  2,
     0,  4,  3,
     0,  5,  4,
     0,  2,  5,
     1,  2,  3,
     1,  3,  4,
     1,  4,  5,
     1,  5,  2,
}};

static constexpr size_t OctahedronVertCount = 6;
static constexpr size_t OctahedronTriCount  = 8;

// ── Tetrahedron ────────────────────────────────────────────────────────────
static constexpr std::array<float, 12> TetrahedronVertices =
{{
     0.0f,                       0.0f,                       1.0f,
     0.9428090415820634f,        0.0f,                      -0.3333333333333333f,
    -0.4714045207910317f,        0.8164965809277260f,       -0.3333333333333333f,
    -0.4714045207910317f,       -0.8164965809277260f,       -0.3333333333333333f,
}};

static constexpr std::array<uint32_t, 12> TetrahedronIndices =
{{
     1,  2,  3,
     0,  3,  2,
     0,  1,  3,
     0,  2,  1,
}};

static constexpr size_t TetrahedronVertCount = 4;
static constexpr size_t TetrahedronTriCount  = 4;

// Circumradius / inradius ratio for each shape.
// scale = cutoffSigma * R_over_r so inradius = cutoffSigma for all shapes.
//   tetrahedron: R/r = 3
//   octahedron:  R/r = √3
//   icosahedron: R/r = 3·√(10+2√5) / (√3·(3+√5))
static constexpr float OctahedronRoverR    = std::sqrt(3.0f);
static constexpr float TetrahedronRoverR   = 3.0f;
static constexpr float IcosahedronRoverR   = 3.0f * std::sqrt(10.0f + 2.0f * std::sqrt(5.0f))
                                           / (std::sqrt(3.0f) * (3.0f + std::sqrt(5.0f)));

// Geometry flags: NO DISABLE_ANYHIT — the any-hit program must run for
// Gaussian proxy triangles.
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
    const cudaStream_t stream,
    const GaussianProxyType type,
    const float cutoffSigma)
{
    if (buffer != 0)
        destroy(stream);

    // Select vertex/index data and compute shape-specific scale so every
    // proxy has inradius = cutoffSigma — the same face-plane distance for
    // all shapes guarantees identical visual Gaussian coverage.
    const float* srcVertices = nullptr;
    const uint32_t* srcIndices = nullptr;
    size_t vertexCount = 0;
    size_t indexCount = 0;
    size_t triCount = 0;
    float scale = 0.0f;

    switch (type)
    {
    case GaussianProxyType::Icosahedron:
        srcVertices = IcosahedronVertices.data();
        srcIndices  = IcosahedronIndices.data();
        vertexCount = IcosahedronVertCount;
        indexCount  = IcosahedronIndices.size();
        triCount    = IcosahedronTriCount;
        scale = cutoffSigma * IcosahedronRoverR;
        break;
    case GaussianProxyType::Octahedron:
        srcVertices = OctahedronVertices.data();
        srcIndices  = OctahedronIndices.data();
        vertexCount = OctahedronVertCount;
        indexCount  = OctahedronIndices.size();
        triCount    = OctahedronTriCount;
        scale = cutoffSigma * OctahedronRoverR;
        break;
    case GaussianProxyType::Tetrahedron:
        srcVertices = TetrahedronVertices.data();
        srcIndices  = TetrahedronIndices.data();
        vertexCount = TetrahedronVertCount;
        indexCount  = TetrahedronIndices.size();
        triCount    = TetrahedronTriCount;
        scale = cutoffSigma * TetrahedronRoverR;
        break;
    }

    // Scale the unit proxy so inradius = cutoffSigma.
    const glm::mat3 cutoffScale(scale);
    std::vector<float> scaledVertices(vertexCount * 3);
    for (size_t i = 0; i < vertexCount * 3; i += 3)
    {
        const glm::vec3 v = cutoffScale * glm::vec3(
            srcVertices[i], srcVertices[i + 1], srcVertices[i + 2]);
        scaledVertices[i]     = v.x;
        scaledVertices[i + 1] = v.y;
        scaledVertices[i + 2] = v.z;
    }

    // Upload vertex + index data to device.
    const size_t vertexBytes = vertexCount * 3 * sizeof(float);
    const size_t indexBytes  = indexCount * sizeof(uint32_t);
    CUdeviceptr vertexBuffer = 0;
    CUdeviceptr indexBuffer  = 0;
    NR_GPU_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&vertexBuffer), vertexBytes, stream));
    NR_GPU_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&indexBuffer),  indexBytes, stream));
    NR_GPU_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(vertexBuffer),
        scaledVertices.data(), vertexBytes, cudaMemcpyHostToDevice, stream));
    NR_GPU_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(indexBuffer),
        srcIndices, indexBytes, cudaMemcpyHostToDevice, stream));

    OptixBuildInput buildInput{};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = 3 * sizeof(float);
    buildInput.triangleArray.numVertices = static_cast<uint32_t>(vertexCount);
    buildInput.triangleArray.vertexBuffers = &vertexBuffer;
    buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.indexStrideInBytes = 3 * sizeof(uint32_t);
    buildInput.triangleArray.numIndexTriplets = static_cast<uint32_t>(triCount);
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
