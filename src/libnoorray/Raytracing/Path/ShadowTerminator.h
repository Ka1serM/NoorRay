#pragma once

#include <cmath>

#include <glm/geometric.hpp>
#include <glm/mat3x3.hpp>
#include <glm/vec3.hpp>

#include "CUDA/Annotations.h"

#if defined(NR_GPU_CODE)
#include "Raytracing/Gpu/SceneData.h"
#endif

namespace nr
{

// Blender Cycles uses 0.1 by default: roughly the last 10-20% of grazing
// shadow rays receive the geometry offset, with a smooth transition.
inline constexpr float ShadowTerminatorGeometryOffset = 0.1f;

struct ShadowTerminatorTriangle
{
    glm::vec3 vertices[3]{};
    glm::vec3 normals[3]{};
    glm::mat3 objectToWorld{1.0f};
};

// Adapted from Blender Cycles' shadow_ray_smooth_surface_offset() and
// shadow_ray_offset(), Copyright Blender Foundation, Apache-2.0. It builds a
// parabolic approximation to the smooth surface represented by a flat
// triangle and its vertex normals, then lifts a grazing shadow-ray origin onto
// that approximation.
NR_CPU_GPU inline glm::vec3 shadowRaySmoothSurfaceOffset(
    const ShadowTerminatorTriangle& triangle,
    const float barycentricU,
    const float barycentricV,
    const glm::vec3 geometricNormal)
{
    const glm::vec3* V = triangle.vertices;
    const glm::vec3* N = triangle.normals;
    const float u = 1.0f - barycentricU - barycentricV;
    const float v = barycentricU;
    const float w = barycentricV;
    const glm::vec3 P = V[0] * u + V[1] * v + V[2] * w;
    glm::vec3 n = N[0] * u + N[1] * v + N[2] * w;

    const float a = glm::dot(N[2] - N[0], V[0] - V[2]);
    const float b = glm::dot(N[2] - N[1], V[1] - V[2]);
    const float c = glm::dot(N[1] - N[0], V[1] - V[0]);
    float h = a * u * (u - 1.0f) + (a + b + c) * u * v
        + b * v * (v - 1.0f);

    // Cycles computes the envelope in object space, then transforms the
    // displacement as a direction. This also preserves its non-uniform-scale
    // behavior.
    n = triangle.objectToWorld * n;

    if (glm::dot(n, geometricNormal) > 0.0f)
    {
        float h0 = fmaxf(fmaxf(
            glm::dot(V[1] - V[0], N[0]), glm::dot(V[2] - V[0], N[0])), 0.0f);
        float h1 = fmaxf(fmaxf(
            glm::dot(V[0] - V[1], N[1]), glm::dot(V[2] - V[1], N[1])), 0.0f);
        float h2 = fmaxf(fmaxf(
            glm::dot(V[0] - V[2], N[2]), glm::dot(V[1] - V[2], N[2])), 0.0f);
        h0 = fmaxf(glm::dot(V[0] - P, N[0]) + h0, 0.0f);
        h1 = fmaxf(glm::dot(V[1] - P, N[1]) + h1, 0.0f);
        h2 = fmaxf(glm::dot(V[2] - P, N[2]) + h2, 0.0f);
        h = fmaxf(fminf(fminf(h0, h1), h2), h * 0.5f);
    }
    else
    {
        float h0 = fmaxf(fmaxf(
            glm::dot(V[0] - V[1], N[0]), glm::dot(V[0] - V[2], N[0])), 0.0f);
        float h1 = fmaxf(fmaxf(
            glm::dot(V[1] - V[0], N[1]), glm::dot(V[1] - V[2], N[1])), 0.0f);
        float h2 = fmaxf(fmaxf(
            glm::dot(V[2] - V[0], N[2]), glm::dot(V[2] - V[1], N[2])), 0.0f);
        h0 = fmaxf(glm::dot(P - V[0], N[0]) + h0, 0.0f);
        h1 = fmaxf(glm::dot(P - V[1], N[1]) + h1, 0.0f);
        h2 = fmaxf(glm::dot(P - V[2], N[2]) + h2, 0.0f);
        h = fminf(-fminf(fminf(h0, h1), h2), h * 0.5f);
    }

    return n * h;
}

NR_CPU_GPU inline float shadowTerminatorOffsetAmount(
    const glm::vec3 shadingNormal,
    const glm::vec3 geometricNormal,
    const glm::vec3 lightDirection,
    const float offsetCutoff = ShadowTerminatorGeometryOffset)
{
    if (offsetCutoff <= 0.0f)
        return 0.0f;

    float normalLight = glm::dot(shadingNormal, lightDirection);
    const bool transmission = normalLight < 0.0f;
    normalLight = fabsf(normalLight);

    const glm::vec3 outgoingGeometricNormal = transmission
        ? -geometricNormal : geometricNormal;
    const float geometricNormalLight = glm::dot(
        outgoingGeometricNormal, lightDirection);
    return normalLight < offsetCutoff
        ? fminf(fmaxf(2.0f - (geometricNormalLight + normalLight)
            / offsetCutoff, 0.0f), 1.0f)
        : fminf(fmaxf(1.0f - geometricNormalLight / offsetCutoff, 0.0f), 1.0f);
}

NR_CPU_GPU inline glm::vec3 shadowTerminatorOffset(
    const ShadowTerminatorTriangle& triangle,
    const float barycentricU,
    const float barycentricV,
    const glm::vec3 shadingNormal,
    const glm::vec3 geometricNormal,
    const glm::vec3 lightDirection,
    const float offsetCutoff = ShadowTerminatorGeometryOffset)
{
    const float offsetAmount = shadowTerminatorOffsetAmount(
        shadingNormal, geometricNormal, lightDirection, offsetCutoff);
    if (offsetAmount <= 0.0f)
        return glm::vec3(0.0f);

    const glm::vec3 outgoingGeometricNormal = glm::dot(shadingNormal, lightDirection) < 0.0f
        ? -geometricNormal : geometricNormal;
    return shadowRaySmoothSurfaceOffset(
        triangle, barycentricU, barycentricV, outgoingGeometricNormal) * offsetAmount;
}

#if defined(NR_GPU_CODE)
NR_GPU inline glm::vec3 shadowTerminatorOffset(
    const GpuSceneData& scene,
    const uint32_t instanceIndex,
    const uint32_t primitiveIndex,
    const float barycentricU,
    const float barycentricV,
    const glm::vec3& shadingNormal,
    const glm::vec3& geometricNormal,
    const glm::vec3& direction)
{
    const float offsetAmount = shadowTerminatorOffsetAmount(
        shadingNormal, geometricNormal, direction);
    if (offsetAmount <= 0.0f)
        return glm::vec3(0.0f);

    const GpuInstance instance = scene.instances[instanceIndex];
    const MeshAsset& mesh = scene.meshes[instance.meshIndex];
    const auto& indices = mesh.getIndices();
    const auto& vertices = mesh.getVertices();
    const Vertex a = vertices[indices[primitiveIndex * 3]];
    const Vertex b = vertices[indices[primitiveIndex * 3 + 1]];
    const Vertex c = vertices[indices[primitiveIndex * 3 + 2]];
    const ShadowTerminatorTriangle triangle{
        {a.position, b.position, c.position},
        {a.normal, b.normal, c.normal},
        glm::mat3(instance.objectToWorld)};
    const glm::vec3 outgoingGeometricNormal = glm::dot(shadingNormal, direction) < 0.0f
        ? -geometricNormal : geometricNormal;
    return shadowRaySmoothSurfaceOffset(
        triangle, barycentricU, barycentricV, outgoingGeometricNormal) * offsetAmount;
}
#endif

}
