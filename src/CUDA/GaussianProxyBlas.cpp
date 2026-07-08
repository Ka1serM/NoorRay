#include "CUDA/GaussianProxyBlas.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <optix_stubs.h>

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include "CUDA/Checks.h"

namespace
{
struct ProxyMesh
{
    std::vector<glm::vec3> vertices;
    std::vector<uint32_t> indices;
    float inradius{};
};

void normalizeVertices(ProxyMesh& mesh)
{
    glm::vec3 center(0.0f);
    for (const glm::vec3 vertex : mesh.vertices)
        center += vertex;
    center /= static_cast<float>(mesh.vertices.size());

    float radius = 0.0f;
    for (glm::vec3& vertex : mesh.vertices)
    {
        vertex -= center;
        radius = std::max(radius, glm::length(vertex));
    }
    if (!(radius > 0.0f))
        throw std::runtime_error("Gaussian proxy has zero radius");
    for (glm::vec3& vertex : mesh.vertices)
        vertex /= radius;
}

void generateConvexHullFaces(ProxyMesh& mesh)
{
    constexpr float epsilon = 1e-5f;
    const uint32_t count = static_cast<uint32_t>(mesh.vertices.size());
    for (uint32_t i = 0; i < count; ++i)
    for (uint32_t j = i + 1; j < count; ++j)
    for (uint32_t k = j + 1; k < count; ++k)
    {
        const glm::vec3 a = mesh.vertices[i];
        const glm::vec3 normal = glm::cross(
            mesh.vertices[j] - a, mesh.vertices[k] - a);
        if (glm::dot(normal, normal) <= epsilon * epsilon)
            continue;

        bool positive = false;
        bool negative = false;
        for (uint32_t vertex = 0; vertex < count; ++vertex)
        {
            if (vertex == i || vertex == j || vertex == k)
                continue;
            const float side = glm::dot(normal, mesh.vertices[vertex] - a);
            positive |= side > epsilon;
            negative |= side < -epsilon;
        }
        if ((positive && negative) || (!positive && !negative))
            continue;

        uint32_t b = j;
        uint32_t c = k;
        if (glm::dot(normal, a) < 0.0f)
            std::swap(b, c);
        mesh.indices.insert(mesh.indices.end(), {i, b, c});
    }
}

void validateMesh(ProxyMesh& mesh, const size_t expectedVertices, const size_t expectedFaces)
{
    constexpr float epsilon = 1e-4f;
    if (mesh.vertices.size() != expectedVertices || mesh.indices.size() != expectedFaces * 3)
        throw std::runtime_error("Generated Gaussian proxy has an unexpected topology");

    float radius = 0.0f;
    float minInradius = std::numeric_limits<float>::max();
    float maxInradius = 0.0f;
    float minEdgeLength = std::numeric_limits<float>::max();
    float maxEdgeLength = 0.0f;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> edgeUse;
    for (const glm::vec3 vertex : mesh.vertices)
        radius = std::max(radius, glm::length(vertex));

    for (size_t i = 0; i < mesh.indices.size(); i += 3)
    {
        const uint32_t ia = mesh.indices[i];
        const uint32_t ib = mesh.indices[i + 1];
        const uint32_t ic = mesh.indices[i + 2];
        const glm::vec3 a = mesh.vertices[ia];
        const glm::vec3 normal = glm::cross(
            mesh.vertices[ib] - a, mesh.vertices[ic] - a);
        const float normalLength = glm::length(normal);
        const float distance = normalLength > 0.0f
            ? glm::dot(normal, a) / normalLength : 0.0f;
        if (!(distance > 0.0f))
            throw std::runtime_error("Generated Gaussian proxy has an invalid face");
        minInradius = std::min(minInradius, distance);
        maxInradius = std::max(maxInradius, distance);
        for (const auto edge : {std::pair{ia, ib}, std::pair{ib, ic}, std::pair{ic, ia}})
            ++edgeUse[std::minmax(edge.first, edge.second)];
    }

    if (std::abs(radius - 1.0f) > epsilon)
        throw std::runtime_error("Generated Gaussian proxy is not unit normalized");
    if (maxInradius - minInradius > epsilon)
        throw std::runtime_error("Generated Gaussian proxy faces are not uniformly tangent");
    for (const auto& [edge, uses] : edgeUse)
    {
        if (uses != 2)
            throw std::runtime_error("Generated Gaussian proxy is not a closed manifold");
        const float edgeLength = glm::length(
            mesh.vertices[edge.first] - mesh.vertices[edge.second]);
        minEdgeLength = std::min(minEdgeLength, edgeLength);
        maxEdgeLength = std::max(maxEdgeLength, edgeLength);
    }
    if (maxEdgeLength - minEdgeLength > epsilon)
        throw std::runtime_error("Generated Gaussian proxy does not have regular edges");
    mesh.inradius = minInradius;
}

ProxyMesh generateProxy(const GaussianProxyType type)
{
    ProxyMesh mesh;
    size_t expectedVertices = 0;
    size_t expectedFaces = 0;
    switch (type)
    {
    case GaussianProxyType::Icosahedron:
    {
        constexpr float phi = std::numbers::phi_v<float>;
        for (const float a : {-1.0f, 1.0f})
        for (const float b : {-phi, phi})
        {
            mesh.vertices.emplace_back(0.0f, a, b);
            mesh.vertices.emplace_back(a, b, 0.0f);
            mesh.vertices.emplace_back(b, 0.0f, a);
        }
        expectedVertices = 12;
        expectedFaces = 20;
        break;
    }
    case GaussianProxyType::Octahedron:
        for (uint32_t axis = 0; axis < 3; ++axis)
        for (const float sign : {-1.0f, 1.0f})
        {
            glm::vec3 vertex(0.0f);
            vertex[axis] = sign;
            mesh.vertices.push_back(vertex);
        }
        expectedVertices = 6;
        expectedFaces = 8;
        break;
    case GaussianProxyType::TriangularBipyramid:
    {
        const float baseRadius = 1.0f / std::sqrt(2.0f);
        for (uint32_t i = 0; i < 3; ++i)
        {
            const float angle = 2.0f * std::numbers::pi_v<float>
                              * static_cast<float>(i) / 3.0f;
            mesh.vertices.emplace_back(
                baseRadius * std::cos(angle), baseRadius * std::sin(angle), 0.0f);
        }
        mesh.vertices.emplace_back(0.0f, 0.0f, 1.0f);
        mesh.vertices.emplace_back(0.0f, 0.0f, -1.0f);
        expectedVertices = 5;
        expectedFaces = 6;
        break;
    }
    }
    normalizeVertices(mesh);
    generateConvexHullFaces(mesh);
    validateMesh(mesh, expectedVertices, expectedFaces);
    return mesh;
}

const std::array<ProxyMesh, 3>& generatedProxies()
{
    // Constructing the table validates every supported shape, not only the
    // currently selected one. Any bad topology fails before OptiX sees it.
    static const std::array<ProxyMesh, 3> meshes = {
        generateProxy(GaussianProxyType::Icosahedron),
        generateProxy(GaussianProxyType::Octahedron),
        generateProxy(GaussianProxyType::TriangularBipyramid),
    };
    return meshes;
}
}

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

    const ProxyMesh& proxy = generatedProxies().at(static_cast<size_t>(type));
    const size_t vertexCount = proxy.vertices.size();
    const size_t indexCount = proxy.indices.size();
    const size_t triCount = indexCount / 3;
    const float cutoffScale = cutoffSigma / proxy.inradius;
    std::vector<float> scaledVertices(vertexCount * 3);
    for (size_t i = 0; i < vertexCount; ++i)
    {
        const glm::vec3 vertex = proxy.vertices[i] * cutoffScale;
        scaledVertices[i * 3] = vertex.x;
        scaledVertices[i * 3 + 1] = vertex.y;
        scaledVertices[i * 3 + 2] = vertex.z;
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
        proxy.indices.data(), indexBytes, cudaMemcpyHostToDevice, stream));

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
