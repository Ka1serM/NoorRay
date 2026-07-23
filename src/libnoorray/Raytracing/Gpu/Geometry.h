#pragma once

#include <cstdint>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include <glm/geometric.hpp>
#include <glm/vec4.hpp>

#include "CUDA/Annotations.h"
#include "Raytracing/Gpu/Types.h"

#if defined(NR_GPU_CODE)
#include "Raytracing/Gpu/SceneData.h"
#endif

// Forward declarations.
struct Material;
struct GpuSceneData;
struct RayHit;

// A reconstructed mesh hit.  This is deliberately data-only: geometry owns
// reconstruction, Material owns shading, and PathIntegrator owns transport.
struct Surface
{
    glm::vec3 position{};
    glm::vec3 normal{};
    glm::vec3 geometricNormal{};
    glm::vec3 tangent{};
    glm::vec2 uv{};
    const Material* material{};
    uint32_t instanceIndex{};
    uint32_t primitiveIndex{};
    float barycentricU{};
    float barycentricV{};

#if defined(NR_GPU_CODE)
    NR_GPU static Surface fromHit(const GpuSceneData& scene, const RayHit& hit);
#endif
};

// The shadow walk only needs these two fields, so it never reconstructs a
// full Surface.
struct ShadowSurface
{
    glm::vec2 uv{};
    const Material* material{};

#if defined(NR_GPU_CODE)
    NR_GPU static ShadowSurface fromHit(
        const GpuSceneData& scene, const RayHit& hit);
#endif
};

#if defined(NR_GPU_CODE)

NR_GPU inline ShadowSurface ShadowSurface::fromHit(
    const GpuSceneData& scene, const RayHit& hit)
{
    ShadowSurface surface{};
    const GpuInstance instance = scene.instances[hit.instanceIndex];
    const MeshAsset& mesh = scene.meshes[instance.meshIndex];
    const int materialSlot = mesh.getFaces()[hit.primitiveIndex].materialIndex;
    surface.material = &scene.materials[mesh.getMaterialIds()[materialSlot]];

    // Constant opacity/transmission materials do not consume UVs in the
    // shadow walk. Avoid three vertex loads and the interpolation for the
    // overwhelmingly common untextured case.
    if (surface.material->opacityIndex < 0
        && surface.material->transmissionIndex < 0)
        return surface;

    const auto& indices = mesh.getIndices();
    const auto& vertices = mesh.getVertices();
    const uint32_t i0 = indices[hit.primitiveIndex * 3];
    const uint32_t i1 = indices[hit.primitiveIndex * 3 + 1];
    const uint32_t i2 = indices[hit.primitiveIndex * 3 + 2];
    const float w = 1.0f - hit.u - hit.v;
    surface.uv = vertices[i0].uv * w + vertices[i1].uv * hit.u
        + vertices[i2].uv * hit.v;
    return surface;
}

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
    const glm::vec3 objectNormal = a.normal * w + b.normal * hit.u + c.normal * hit.v;
    const glm::vec3 worldA = glm::vec3(instance.objectToWorld * glm::vec4(a.position, 1.0f));
    const glm::vec3 worldB = glm::vec3(instance.objectToWorld * glm::vec4(b.position, 1.0f));
    const glm::vec3 worldC = glm::vec3(instance.objectToWorld * glm::vec4(c.position, 1.0f));
    surface.position = worldA * w + worldB * hit.u + worldC * hit.v;
    surface.normal = glm::normalize(instance.normalToWorld * objectNormal);
    surface.geometricNormal = glm::normalize(glm::cross(worldB - worldA, worldC - worldA));
    // Tangent reconstruction includes an interpolation, matrix transform and
    // normalization, but only normal-mapped materials use it.
    if (surface.material->normalIndex >= 0)
        surface.tangent = glm::normalize(glm::vec3(instance.objectToWorld * glm::vec4(
            a.tangent * w + b.tangent * hit.u + c.tangent * hit.v, 0.0f)));
    surface.uv = glm::vec2(a.uv.x * w + b.uv.x * hit.u + c.uv.x * hit.v,
                       a.uv.y * w + b.uv.y * hit.u + c.uv.y * hit.v);
    surface.instanceIndex = hit.instanceIndex;
    surface.primitiveIndex = hit.primitiveIndex;
    surface.barycentricU = hit.u;
    surface.barycentricV = hit.v;
    return surface;
}

#endif
