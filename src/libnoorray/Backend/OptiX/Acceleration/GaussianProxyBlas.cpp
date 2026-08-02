#include "Backend/OptiX/Acceleration/GaussianProxyBlas.h"

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

#include "Backend/CUDA/Checks.h"

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

void validateMesh(ProxyMesh& mesh, const size_t expectedVertices,
    const size_t expectedFaces, const bool requireRegularFaces = true)
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
    if (requireRegularFaces && maxInradius - minInradius > epsilon)
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
    if (requireRegularFaces && maxEdgeLength - minEdgeLength > epsilon)
        throw std::runtime_error("Generated Gaussian proxy does not have regular edges");
    mesh.inradius = minInradius;
}

ProxyMesh generateIcosahedronProxy()
{
    ProxyMesh mesh;
    constexpr float phi = std::numbers::phi_v<float>;
    for (const float a : {-1.0f, 1.0f})
    for (const float b : {-phi, phi})
    {
        mesh.vertices.emplace_back(0.0f, a, b);
        mesh.vertices.emplace_back(a, b, 0.0f);
        mesh.vertices.emplace_back(b, 0.0f, a);
    }

    normalizeVertices(mesh);
    generateConvexHullFaces(mesh);
    validateMesh(mesh, 12, 20);
    return mesh;
}

ProxyMesh subdivideIcosphere(ProxyMesh mesh)
{
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> midpoints;
    const auto midpoint = [&](const uint32_t a, const uint32_t b) {
        const auto edge = std::minmax(a, b);
        if (const auto found = midpoints.find(edge); found != midpoints.end())
            return found->second;
        const uint32_t index = static_cast<uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back(glm::normalize(mesh.vertices[a] + mesh.vertices[b]));
        midpoints.emplace(edge, index);
        return index;
    };

    std::vector<uint32_t> refinedIndices;
    refinedIndices.reserve(mesh.indices.size() * 4);
    for (size_t index = 0; index < mesh.indices.size(); index += 3)
    {
        const uint32_t a = mesh.indices[index];
        const uint32_t b = mesh.indices[index + 1];
        const uint32_t c = mesh.indices[index + 2];
        const uint32_t ab = midpoint(a, b);
        const uint32_t bc = midpoint(b, c);
        const uint32_t ca = midpoint(c, a);
        refinedIndices.insert(refinedIndices.end(), {
            a, ab, ca,
            b, bc, ab,
            c, ca, bc,
            ab, bc, ca,
        });
    }
    mesh.indices = std::move(refinedIndices);
    return mesh;
}

ProxyMesh generateIcosphereProxy()
{
    ProxyMesh mesh = subdivideIcosphere(generateIcosahedronProxy());

    // Minimax orientation for the frequency-2 vertex set. It reduces the
    // object-space BLAS AABB half-extent from 1 to 0.94502682 on every axis.
    for (glm::vec3& vertex : mesh.vertices)
    {
        vertex = {
            0.74293414f * vertex.x - 0.32699283f * vertex.y
                + 0.58405870f * vertex.z,
           -0.25706586f * vertex.x - 0.94502682f * vertex.y
                - 0.20209268f * vertex.z,
            0.61803399f * vertex.x - 0.78615138f * vertex.z,
        };
    }
    normalizeVertices(mesh);
    validateMesh(mesh, 42, 80, false);
    return mesh;
}

ProxyMesh generateIcosphereLevel2Proxy()
{
    ProxyMesh mesh = subdivideIcosphere(generateIcosphereProxy());
    validateMesh(mesh, 162, 320, false);
    return mesh;
}

ProxyMesh generateOctahedronProxy()
{
    ProxyMesh mesh;
    for (uint32_t axis = 0; axis < 3; ++axis)
    for (const float sign : {-1.0f, 1.0f})
    {
        glm::vec3 vertex(0.0f);
        vertex[axis] = sign;
        mesh.vertices.push_back(vertex);
    }
    normalizeVertices(mesh);
    generateConvexHullFaces(mesh);
    validateMesh(mesh, 6, 8);
    return mesh;
}

const ProxyMesh& generatedProxy(const GaussianProxyType type)
{
    static const std::array meshes{
        generateIcosphereProxy(),
        generateOctahedronProxy(),
        generateIcosahedronProxy(),
        generateIcosphereLevel2Proxy(),
    };
    return meshes.at(static_cast<size_t>(type));
}
}

// Geometry flags: NO DISABLE_ANYHIT — the any-hit program must run for
// Gaussian proxy triangles.
static constexpr unsigned int GaussianGeometryFlags = 0u;

GaussianProxyBlas::~GaussianProxyBlas() noexcept
{
    reset();
}

GaussianProxyBlas::GaussianProxyBlas(GaussianProxyBlas&& other) noexcept
    : handle(std::exchange(other.handle, {})),
      buffer(std::move(other.buffer))
{
}

GaussianProxyBlas& GaussianProxyBlas::operator=(GaussianProxyBlas&& other) noexcept
{
    if (this != &other)
    {
        reset();
        handle = std::exchange(other.handle, {});
        buffer = std::move(other.buffer);
    }
    return *this;
}

void GaussianProxyBlas::build(
    const OptixDeviceContext context,
    const cudaStream_t stream,
    const GaussianProxyType type,
    const float cutoffSigma)
{
    reset();

    OptixBuildInput buildInput{};
    nr::cuda::UniqueAsyncDeviceBuffer vertexBuffer;
    nr::cuda::UniqueAsyncDeviceBuffer indexBuffer;

    const ProxyMesh& proxy = generatedProxy(type);
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
    vertexBuffer.allocate(vertexBytes, stream);
    indexBuffer.allocate(indexBytes, stream);
    NR_GPU_CHECK(cudaMemcpyAsync(vertexBuffer.get(),
        scaledVertices.data(), vertexBytes, cudaMemcpyHostToDevice, stream));
    NR_GPU_CHECK(cudaMemcpyAsync(indexBuffer.get(),
        proxy.indices.data(), indexBytes, cudaMemcpyHostToDevice, stream));

    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = 3 * sizeof(float);
    buildInput.triangleArray.numVertices = static_cast<uint32_t>(vertexCount);
    CUdeviceptr vertexBufferPtr = vertexBuffer.devicePtr();
    buildInput.triangleArray.vertexBuffers = &vertexBufferPtr;
    buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.indexStrideInBytes = 3 * sizeof(uint32_t);
    buildInput.triangleArray.numIndexTriplets = static_cast<uint32_t>(triCount);
    buildInput.triangleArray.indexBuffer = indexBuffer.devicePtr();
    buildInput.triangleArray.flags = &GaussianGeometryFlags;
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

void GaussianProxyBlas::reset() noexcept
{
    buffer.reset();
    handle = {};
}
