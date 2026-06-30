#pragma once

#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "Raytracing/SceneData.h"

struct SurfaceData
{
    glm::vec3 position{};
    glm::vec3 normal{};
    glm::vec3 geometricNormal{};
    glm::vec3 tangent{};
    glm::vec2 uv{};
    uint32_t objectIndex{InvalidIndex};
    const Material* material{};
};

NR_GPU inline SurfaceData loadSurface(
    const GpuSceneData scene,
    const uint32_t instanceIndex,
    const uint32_t primitiveIndex,
    const float u,
    const float v)
{
    SurfaceData surface{};
    const GpuInstance instance = scene.instances[instanceIndex];
    const MeshAsset& mesh = scene.meshes[instance.meshIndex];
    const auto& indices = mesh.getIndices();
    const auto& vertices = mesh.getVertices();
    const uint32_t i0 = indices[primitiveIndex * 3];
    const uint32_t i1 = indices[primitiveIndex * 3 + 1];
    const uint32_t i2 = indices[primitiveIndex * 3 + 2];
    const Vertex a = vertices[i0];
    const Vertex b = vertices[i1];
    const Vertex c = vertices[i2];
    const float w = 1.0f - u - v;
    const glm::vec3 objectPosition = a.position * w + b.position * u + c.position * v;
    const glm::vec3 objectNormal = a.normal * w + b.normal * u + c.normal * v;
    const glm::vec3 worldA = glm::vec3(instance.objectToWorld * glm::vec4(a.position, 1.0f));
    const glm::vec3 worldB = glm::vec3(instance.objectToWorld * glm::vec4(b.position, 1.0f));
    const glm::vec3 worldC = glm::vec3(instance.objectToWorld * glm::vec4(c.position, 1.0f));
    surface.position = worldA * w + worldB * u + worldC * v;
    surface.normal = glm::normalize(instance.normalToWorld * objectNormal);
    surface.geometricNormal = glm::normalize(glm::cross(worldB - worldA, worldC - worldA));
    surface.tangent = glm::normalize(glm::vec3(instance.objectToWorld * glm::vec4(
        a.tangent * w + b.tangent * u + c.tangent * v, 0.0f)));
    surface.uv = glm::vec2(a.uv.x * w + b.uv.x * u + c.uv.x * v,
                       a.uv.y * w + b.uv.y * u + c.uv.y * v);
    surface.objectIndex = instance.objectIndex;
    const int materialIndex = mesh.getFaces()[primitiveIndex].materialIndex;
    surface.material = &mesh.getMaterials()[materialIndex];
    return surface;
}
