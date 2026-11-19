#include "NoorRay/Import/Primitives.h"

#include <cmath>

#include "NoorRay/Import/MeshBuilder.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Import {

namespace {

void addQuad(MeshData& data, const Core::float3 center, const Core::float3 right, const Core::float3 up) {
    const auto first = static_cast<std::uint32_t>(data.positions.size());
    const Core::float3 corners[] = {center - right - up, center + right - up, center + right + up, center - right + up};
    const Core::float2 uvs[] = {{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}};
    for (int corner = 0; corner < 4; ++corner) {
        data.positions.push_back(corners[corner]);
        data.uvs[0].push_back(uvs[corner]);
    }
    for (const std::uint32_t index : {0u, 1u, 2u, 0u, 2u, 3u})
        data.indices.push_back(first + index);
}

MeshData makeCube() {
    MeshData data;
    data.name = "Cube";
    const Core::float3 axes[] = {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};
    for (int axis = 0; axis < 3; ++axis)
        for (const float side : {1.0f, -1.0f}) {
            const Core::float3 normal = axes[axis] * side;
            const Core::float3 right = axes[(axis + 1) % 3] * side;
            const Core::float3 up = axes[(axis + 2) % 3];
            addQuad(data, normal * 0.5f, right * 0.5f, up * 0.5f);
        }
    return data;
}

MeshData makePlane() {
    MeshData data;
    data.name = "Plane";
    addQuad(data, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f});
    return data;
}

MeshData makeSphere() {
    constexpr std::uint32_t rings = 32;
    constexpr std::uint32_t segments = 64;
    MeshData data;
    data.name = "Sphere";
    for (std::uint32_t ring = 0; ring <= rings; ++ring) {
        const float v = static_cast<float>(ring) / rings;
        const float theta = v * Core::pi;
        for (std::uint32_t segment = 0; segment <= segments; ++segment) {
            const float u = static_cast<float>(segment) / segments;
            const float phi = u * 2.0f * Core::pi;
            const Core::float3 normal{std::sin(theta) * std::sin(phi), std::cos(theta), std::sin(theta) * std::cos(phi)};
            data.positions.push_back(normal * 0.5f);
            data.normals.push_back(normal);
            data.uvs[0].push_back({u, 1.0f - v});
        }
    }
    for (std::uint32_t ring = 0; ring < rings; ++ring)
        for (std::uint32_t segment = 0; segment < segments; ++segment) {
            const std::uint32_t a = ring * (segments + 1) + segment;
            const std::uint32_t b = a + segments + 1;
            for (const std::uint32_t index : {a, b, a + 1, a + 1, b, b + 1})
                data.indices.push_back(index);
        }
    return data;
}

MeshData makeMesh(const Primitive primitive) {
    switch (primitive) {
    case Primitive::Cube:
        return makeCube();
    case Primitive::Plane:
        return makePlane();
    case Primitive::Sphere:
        return makeSphere();
    }
    return makeCube();
}

}

Scene::InstanceHandle Primitives::add(Scene::Scene& scene, const Primitive primitive) {
    MeshData data = makeMesh(primitive);
    Scene::Material material;
    material.name = data.name + " Material";
    const Scene::MaterialHandle materialHandle = scene.addMaterial(std::move(material));
    data.appendSubset(materialHandle, 0, static_cast<std::uint32_t>(data.indices.size()));
    std::string name = data.name;
    Scene::Instance instance;
    instance.mesh = scene.addMesh(MeshBuilder::build(std::move(data)));
    instance.flags = Scene::InstanceFlags::DoubleSided;
    const Scene::InstanceHandle added = scene.addInstance(instance);
    scene.addNode({std::move(name), {}, {}, added});
    return added;
}

}
