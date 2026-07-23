#pragma once

#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "Raytracing/Gpu/SceneData.h"

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

#if defined(NR_GPU_CODE)
    NR_GPU static glm::vec3 positionFromHit(
        const GpuSceneData& scene, const RayHit& hit);
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
// Shadow transparency only needs UVs and the material. Avoid reconstructing
// and transforming the full shading frame for every transparent blocker.
NR_GPU inline ShadowSurface ShadowSurface::fromHit(
    const GpuSceneData& scene, const RayHit& hit)
{
    ShadowSurface surface{};
    const GpuInstance instance = scene.instances[hit.instanceIndex];
    const MeshAsset& mesh = scene.meshes[instance.meshIndex];
    const auto& indices = mesh.getIndices();
    const auto& vertices = mesh.getVertices();
    const uint32_t i0 = indices[hit.primitiveIndex * 3];
    const uint32_t i1 = indices[hit.primitiveIndex * 3 + 1];
    const uint32_t i2 = indices[hit.primitiveIndex * 3 + 2];
    const float w = 1.0f - hit.u - hit.v;
    surface.uv = vertices[i0].uv * w + vertices[i1].uv * hit.u
        + vertices[i2].uv * hit.v;
    const int materialIndex = mesh.getFaces()[hit.primitiveIndex].materialIndex;
    surface.material = &mesh.getMaterials()[materialIndex];
    return surface;
}

NR_GPU inline glm::vec3 Surface::positionFromHit(
    const GpuSceneData& scene, const RayHit& hit)
{
    const GpuInstance instance = scene.instances[hit.instanceIndex];
    const MeshAsset& mesh = scene.meshes[instance.meshIndex];
    const auto& indices = mesh.getIndices();
    const auto& vertices = mesh.getVertices();
    const glm::vec3 a = vertices[indices[hit.primitiveIndex * 3]].position;
    const glm::vec3 b = vertices[indices[hit.primitiveIndex * 3 + 1]].position;
    const glm::vec3 c = vertices[indices[hit.primitiveIndex * 3 + 2]].position;
    const glm::vec3 objectPosition = a * (1.0f - hit.u - hit.v) + b * hit.u + c * hit.v;
    return glm::vec3(instance.objectToWorld * glm::vec4(objectPosition, 1.0f));
}

NR_GPU inline Surface Surface::fromHit(
    const GpuSceneData& scene, const RayHit& hit)
{
    Surface surface{};
    const GpuInstance instance = scene.instances[hit.instanceIndex];
    const MeshAsset& mesh = scene.meshes[instance.meshIndex];
    const auto& indices = mesh.getIndices();
    const auto& vertices = mesh.getVertices();
    const uint32_t i0 = indices[hit.primitiveIndex * 3];
    const uint32_t i1 = indices[hit.primitiveIndex * 3 + 1];
    const uint32_t i2 = indices[hit.primitiveIndex * 3 + 2];
    const Vertex a = vertices[i0];
    const Vertex b = vertices[i1];
    const Vertex c = vertices[i2];
    const float w = 1.0f - hit.u - hit.v;
    const glm::vec3 objectPosition = a.position * w + b.position * hit.u + c.position * hit.v;
    const glm::vec3 objectNormal = a.normal * w + b.normal * hit.u + c.normal * hit.v;
    const glm::vec3 worldA = glm::vec3(instance.objectToWorld * glm::vec4(a.position, 1.0f));
    const glm::vec3 worldB = glm::vec3(instance.objectToWorld * glm::vec4(b.position, 1.0f));
    const glm::vec3 worldC = glm::vec3(instance.objectToWorld * glm::vec4(c.position, 1.0f));
    surface.position = worldA * w + worldB * hit.u + worldC * hit.v;
    surface.normal = glm::normalize(instance.normalToWorld * objectNormal);
    surface.geometricNormal = glm::normalize(glm::cross(worldB - worldA, worldC - worldA));
    surface.tangent = glm::normalize(glm::vec3(instance.objectToWorld * glm::vec4(
        a.tangent * w + b.tangent * hit.u + c.tangent * hit.v, 0.0f)));
    surface.uv = glm::vec2(a.uv.x * w + b.uv.x * hit.u + c.uv.x * hit.v,
                       a.uv.y * w + b.uv.y * hit.u + c.uv.y * hit.v);
    const int materialIndex = mesh.getFaces()[hit.primitiveIndex].materialIndex;
    surface.material = &mesh.getMaterials()[materialIndex];
    return surface;
}

NR_GPU inline Bsdf Material::makeBsdf(
    const GpuSceneData& scene,
    const Surface& surface,
    const Ray& incident,
    const glm::vec3& geometricNormal,
    const SampledWavelengths& wavelengths) const
{
    glm::vec3 shadingNormal = shadingNormalAt(
        scene.textures, surface.uv, surface.tangent, surface.normal);
    if (glm::dot(shadingNormal, incident.direction) > 0.0f)
        shadingNormal = -shadingNormal;
    const glm::vec3 viewDirection = -incident.direction;
    shadingNormal = Bsdf::clampShadingNormal(
        geometricNormal, shadingNormal, viewDirection);
    return makeBsdf(scene.textures, surface.uv, viewDirection,
        surface.geometricNormal, shadingNormal, wavelengths,
        scene.spectrumTableScale, scene.spectrumTableCoeffs, scene.openPbrLuts);
}
#endif
