#pragma once

#include "Kernels/Math.h"
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
    const glm::vec3 objectPosition = add(add(multiply(a.position, w), multiply(b.position, u)), multiply(c.position, v));
    const glm::vec3 objectNormal = add(add(multiply(a.normal, w), multiply(b.normal, u)), multiply(c.normal, v));
    surface.position = transformPoint(instance.objectToWorld, objectPosition);
    surface.normal = transformNormal(instance.normalToWorld, objectNormal);
    surface.geometricNormal = normalize3(transformVector(
        instance.objectToWorld, cross3(subtract(b.position, a.position), subtract(c.position, a.position))));
    surface.tangent = normalize3(transformVector(instance.objectToWorld,
        add(add(multiply(a.tangent, w), multiply(b.tangent, u)), multiply(c.tangent, v))));
    surface.uv = glm::vec2(a.uv.x * w + b.uv.x * u + c.uv.x * v,
                       a.uv.y * w + b.uv.y * u + c.uv.y * v);
    surface.objectIndex = instance.objectIndex;
    const int materialIndex = mesh.faces[primitiveIndex].materialIndex;
    if (materialIndex >= 0 && static_cast<uint32_t>(materialIndex) < mesh.materialCount)
        surface.material = &mesh.materials[materialIndex];
    return surface;
}
