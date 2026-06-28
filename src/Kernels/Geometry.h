#pragma once

#include "Kernels/SceneData.h"

struct SurfaceData
{
    glm::vec3 position{};
    glm::vec3 normal{};
    glm::vec3 geometricNormal{};
    glm::vec3 tangent{};
    glm::vec2 uv{};
    uint32_t objectIndex{InvalidIndex};
    const GpuMaterial* material{};
};

NR_GPU inline glm::vec3 transformPoint(const float matrix[12], const glm::vec3 p)
{
    return glm::vec3(
        matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z + matrix[3],
        matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z + matrix[7],
        matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z + matrix[11]);
}

NR_GPU inline glm::vec3 transformVector(const float matrix[12], const glm::vec3 v)
{
    return glm::vec3(
        matrix[0] * v.x + matrix[1] * v.y + matrix[2] * v.z,
        matrix[4] * v.x + matrix[5] * v.y + matrix[6] * v.z,
        matrix[8] * v.x + matrix[9] * v.y + matrix[10] * v.z);
}

NR_GPU inline glm::vec3 transformNormal(const float matrix[9], const glm::vec3 n)
{
    return glm::vec3(
        matrix[0] * n.x + matrix[1] * n.y + matrix[2] * n.z,
        matrix[3] * n.x + matrix[4] * n.y + matrix[5] * n.z,
        matrix[6] * n.x + matrix[7] * n.y + matrix[8] * n.z);
}

NR_GPU inline SurfaceData loadSurface(
    const GpuSceneData scene,
    const uint32_t instanceIndex,
    const uint32_t primitiveIndex,
    const float u,
    const float v)
{
    SurfaceData surface{};
    if (instanceIndex >= scene.instanceCount)
        return surface;
    const GpuInstance instance = scene.instances[instanceIndex];
    if (instance.meshIndex >= scene.meshCount)
        return surface;
    const GpuMesh mesh = scene.meshes[instance.meshIndex];
    if (primitiveIndex >= mesh.triangleCount)
        return surface;
    const uint32_t i0 = mesh.indices[primitiveIndex * 3];
    const uint32_t i1 = mesh.indices[primitiveIndex * 3 + 1];
    const uint32_t i2 = mesh.indices[primitiveIndex * 3 + 2];
    const Vertex a = mesh.vertices[i0];
    const Vertex b = mesh.vertices[i1];
    const Vertex c = mesh.vertices[i2];
    const float w = 1.0f - u - v;
    const glm::vec3 objectPosition = a.position * w + b.position * u + c.position * v;
    const glm::vec3 objectNormal = a.normal * w + b.normal * u + c.normal * v;
    surface.position = transformPoint(instance.objectToWorld, objectPosition);
    surface.normal = transformNormal(instance.normalToWorld, objectNormal);
    surface.geometricNormal = glm::normalize(transformVector(
        instance.objectToWorld, glm::cross(b.position - a.position, c.position - a.position)));
    surface.tangent = glm::normalize(transformVector(instance.objectToWorld,
        a.tangent * w + b.tangent * u + c.tangent * v));
    surface.uv = glm::vec2(a.uv.x * w + b.uv.x * u + c.uv.x * v,
                       a.uv.y * w + b.uv.y * u + c.uv.y * v);
    surface.objectIndex = instance.objectIndex;
    const int materialIndex = mesh.faces[primitiveIndex].materialIndex;
    if (materialIndex >= 0 && static_cast<uint32_t>(materialIndex) < mesh.materialCount)
        surface.material = &mesh.materials[materialIndex];
    return surface;
}
