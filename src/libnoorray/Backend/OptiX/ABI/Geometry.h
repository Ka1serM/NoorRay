#pragma once

#include <cstdint>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include <glm/geometric.hpp>
#include <glm/vec4.hpp>

#include "Backend/CUDA/Annotations.h"
#include "Geometry/Mesh/VertexColor.h"
#include "Backend/OptiX/ABI/Types.h"

#if defined(NR_GPU_CODE)
#include "Backend/OptiX/ABI/SceneData.h"
#endif

// Forward declarations.
struct Material;
struct GpuSceneData;
struct RayHit;

// A reconstructed mesh hit.  This is deliberately data-only: geometry owns
// reconstruction, Material owns shading, and the path kernels own transport.
struct Surface
{
    glm::vec3 position{};
    glm::vec3 normal{};
    glm::vec3 geometricNormal{};
    glm::vec3 tangent{};
    float tangentSign{1.0f};
    glm::vec2 uv{};
    glm::vec4 color{1.0f};
    const Material* material{};
    uint32_t instanceIndex{};
    uint32_t primitiveIndex{};
    float barycentricU{};
    float barycentricV{};

#if defined(NR_GPU_CODE)
    NR_GPU static Surface fromHit(const GpuSceneData& scene, const RayHit& hit);
#endif
};

#if defined(NR_GPU_CODE)

NR_GPU inline Surface Surface::fromHit(
    const GpuSceneData& scene, const RayHit& hit)
{
    Surface surface{};
    const GpuInstance instance = scene.instances[hit.instanceIndex];
    const MeshAsset& mesh = scene.meshes[instance.meshIndex];
    const int materialSlot = mesh.getFaces()[hit.primitiveIndex].materialIndex;
    surface.material = &scene.materials[mesh.getMaterialIds()[materialSlot]];
    const auto& indices = mesh.getIndices();
    const auto& vertices = mesh.getVertices();
    const uint32_t i0 = indices[hit.primitiveIndex * 3];
    const uint32_t i1 = indices[hit.primitiveIndex * 3 + 1];
    const uint32_t i2 = indices[hit.primitiveIndex * 3 + 2];
    const Vertex a = vertices[i0];
    const Vertex b = vertices[i1];
    const Vertex c = vertices[i2];
    const float w = 1.0f - hit.u - hit.v;
    surface.color = nr::vertex_color::unpackLinear(a.color) * w
        + nr::vertex_color::unpackLinear(b.color) * hit.u
        + nr::vertex_color::unpackLinear(c.color) * hit.v;
    const glm::vec3 objectNormal = a.normal * w + b.normal * hit.u + c.normal * hit.v;
    const glm::vec3 worldA = glm::vec3(instance.objectToWorld * glm::vec4(a.position, 1.0f));
    const glm::vec3 worldB = glm::vec3(instance.objectToWorld * glm::vec4(b.position, 1.0f));
    const glm::vec3 worldC = glm::vec3(instance.objectToWorld * glm::vec4(c.position, 1.0f));
    surface.position = worldA * w + worldB * hit.u + worldC * hit.v;
    surface.normal = nr::safeNormalize(
        instance.normalToWorld * objectNormal, glm::vec3(0.0f, 1.0f, 0.0f));
    surface.geometricNormal = nr::safeNormalize(
        glm::cross(worldB - worldA, worldC - worldA), surface.normal);
    glm::vec3 worldTangent = glm::vec3(instance.objectToWorld * glm::vec4(
        a.tangent * w + b.tangent * hit.u + c.tangent * hit.v, 0.0f));
    worldTangent -= surface.normal * glm::dot(surface.normal, worldTangent);
    if (glm::dot(worldTangent, worldTangent) <= 1e-12f)
    {
        const glm::vec3 axis = fabsf(surface.normal.x) < 0.9f
            ? glm::vec3(1.0f, 0.0f, 0.0f)
            : glm::vec3(0.0f, 1.0f, 0.0f);
        worldTangent = glm::cross(axis, surface.normal);
    }
    surface.tangent = nr::safeNormalize(worldTangent,
        glm::vec3(1.0f, 0.0f, 0.0f));
    const float vertexSign =
        a.tangentSign * w + b.tangentSign * hit.u + c.tangentSign * hit.v;
    const glm::vec3 transformX = glm::vec3(instance.objectToWorld[0]);
    const glm::vec3 transformY = glm::vec3(instance.objectToWorld[1]);
    const glm::vec3 transformZ = glm::vec3(instance.objectToWorld[2]);
    const float transformSign =
        glm::dot(glm::cross(transformX, transformY), transformZ) < 0.0f
        ? -1.0f : 1.0f;
    surface.tangentSign = (vertexSign < 0.0f ? -1.0f : 1.0f) * transformSign;
    surface.uv = glm::vec2(a.uv.x * w + b.uv.x * hit.u + c.uv.x * hit.v,
                       a.uv.y * w + b.uv.y * hit.u + c.uv.y * hit.v);
    surface.instanceIndex = hit.instanceIndex;
    surface.primitiveIndex = hit.primitiveIndex;
    surface.barycentricU = hit.u;
    surface.barycentricV = hit.v;
    return surface;
}

#endif
