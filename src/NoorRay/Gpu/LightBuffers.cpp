#include "NoorRay/Gpu/LightBuffers.h"

#include <cmath>

#include "NoorRay/Gpu/AliasTable.h"
#include "NoorRay/Gpu/BufferUpload.h"

namespace NoorRay::Gpu {

namespace {

Core::float4 toFloat4(const Core::float3 value, const float w) {
    return {value.x, value.y, value.z, w};
}

}

Shaders::LightRecord LightBuffers::makeRecord(const Scene::Light& light, float& weight, const float sceneRadius) {
    const Core::float3 xAxis = light.transform.getColumn(0);
    const Core::float3 yAxis = light.transform.getColumn(1);
    const Core::float3 zAxis = light.transform.getColumn(2);
    const Core::float3 direction = Core::normalize(-zAxis);
    const float scaleX = Core::length(xAxis);
    const float scaleY = Core::length(yAxis);
    const float luminance = std::max(Core::luminance(light.color), 0.0f);
    Shaders::LightRecord record{};
    record.position = toFloat4(light.transform.getTranslation(), light.radius);
    record.direction = toFloat4(direction, -1.0f);
    record.meta = {Shaders::lightTypePoint, Shaders::invalidIndex, Shaders::invalidIndex, 0};
    const float pointIntensity = light.power / (4.0f * Core::pi);
    switch (light.type) {
    case Scene::LightType::Point:
    case Scene::LightType::Sphere: {
        const float radius = light.radius * std::max({scaleX, scaleY, Core::length(zAxis)});
        record.position.w = radius;
        if (radius > 0.0f) {
            record.meta.x = Shaders::lightTypeSphere;
            record.radiance = toFloat4(light.color * (light.power / (4.0f * Core::pi * Core::pi * radius * radius)), 0.0f);
        } else
            record.radiance = toFloat4(light.color * pointIntensity, 0.0f);
        weight = light.power * luminance;
        break;
    }
    case Scene::LightType::Spot: {
        const float cosOuter = std::cos(0.5f * light.spotAngle);
        record.meta.x = Shaders::lightTypeSpot;
        record.direction.w = cosOuter;
        record.tangent.w = std::max((1.0f - cosOuter) * light.spotBlend, 1e-6f);
        record.position.w = light.radius;
        record.radiance = toFloat4(light.color * pointIntensity, 0.0f);
        weight = light.power * luminance;
        break;
    }
    case Scene::LightType::Rect:
    case Scene::LightType::Disk: {
        const float halfWidth = 0.5f * light.width * scaleX;
        const float halfHeight = 0.5f * light.height * scaleY;
        const bool disk = light.type == Scene::LightType::Disk;
        const float area = disk ? Core::pi * halfWidth * halfHeight : 4.0f * halfWidth * halfHeight;
        record.meta.x = disk ? Shaders::lightTypeDisk : Shaders::lightTypeRect;
        record.tangent = toFloat4(Core::normalize(xAxis) * halfWidth, 0.0f);
        record.bitangent = toFloat4(Core::normalize(yAxis) * halfHeight, 0.0f);
        record.radiance = toFloat4(area > 0.0f ? light.color * (light.power / (area * Core::pi)) : Core::float3{}, area);
        weight = area > 0.0f ? light.power * luminance : 0.0f;
        break;
    }
    case Scene::LightType::Directional: {
        record.meta.x = Shaders::lightTypeDirectional;
        record.direction.w = std::cos(0.5f * std::max(light.angle, 0.0f));
        record.radiance = toFloat4(light.color * light.power, 0.0f);
        weight = light.power * luminance * Core::pi * sceneRadius * sceneRadius;
        break;
    }
    }
    return record;
}

std::expected<void, Core::Error> LightBuffers::update(NoorRHI::Device& device, const Scene::Scene& scene, const std::span<const MaterialInfo> materials,
                                                      const float sceneRadius) {
    std::vector<Shaders::LightRecord> values;
    std::vector<float> weights;
    for (const Scene::Light& light : scene.getLights().getValues()) {
        float weight = 0.0f;
        values.push_back(makeRecord(light, weight, sceneRadius));
        weights.push_back(weight);
    }
    const std::span<const Scene::Instance> instances = scene.getInstances().getValues();
    instanceLights.assign(instances.size(), Shaders::invalidIndex);
    for (std::uint32_t instanceIndex = 0; instanceIndex < instances.size(); ++instanceIndex) {
        const Scene::Instance& instance = instances[instanceIndex];
        const Scene::Mesh* mesh = scene.find(instance.mesh);
        if (!mesh || Scene::hasFlag(instance.flags, Scene::InstanceFlags::Hidden))
            continue;
        std::vector<Core::float3> emission(mesh->getTriangleCount());
        bool emissive = false;
        const std::uint32_t override = scene.getMaterials().denseIndex(instance.material);
        for (const Scene::MeshSubset& subset : mesh->subsets) {
            const std::uint32_t material = override != Shaders::invalidIndex ? override : scene.getMaterials().denseIndex(subset.material);
            if (material >= materials.size() || (materials[material].flags & Shaders::materialFlagEmissive) == 0)
                continue;
            const std::uint32_t first = subset.firstIndex / 3;
            const std::uint32_t last = std::min<std::uint32_t>(first + subset.indexCount / 3, mesh->getTriangleCount());
            for (std::uint32_t triangle = first; triangle < last; ++triangle)
                emission[triangle] = materials[material].emission;
            emissive = emissive || Core::maxComponent(materials[material].emission) > 0.0f;
        }
        if (!emissive)
            continue;
        instanceLights[instanceIndex] = static_cast<std::uint32_t>(values.size());
        for (std::uint32_t triangle = 0; triangle < mesh->getTriangleCount(); ++triangle) {
            const Core::float3 p0 = instance.transform.transformPoint(mesh->positions[mesh->indices[triangle * 3]]);
            const Core::float3 p1 = instance.transform.transformPoint(mesh->positions[mesh->indices[triangle * 3 + 1]]);
            const Core::float3 p2 = instance.transform.transformPoint(mesh->positions[mesh->indices[triangle * 3 + 2]]);
            const float area = 0.5f * Core::length(Core::cross(p1 - p0, p2 - p0));
            Shaders::LightRecord record{};
            record.position = toFloat4(p0, 0.0f);
            record.tangent = toFloat4(p1 - p0, 0.0f);
            record.bitangent = toFloat4(p2 - p0, 0.0f);
            record.radiance = toFloat4(emission[triangle], area);
            record.meta = {Shaders::lightTypeTriangle, instanceIndex, triangle, 0};
            values.push_back(record);
            weights.push_back(area * Core::pi * std::max(Core::luminance(emission[triangle]), 0.0f));
        }
    }
    count = static_cast<std::uint32_t>(values.size());
    totalPower = 0.0f;
    for (const float weight : weights)
        totalPower += weight;
    if (auto uploaded = uploadValues<Shaders::LightRecord>(device, records, values); !uploaded)
        return uploaded;
    const std::vector<Shaders::AliasEntry> table = buildAliasTable(weights);
    return uploadValues<Shaders::AliasEntry>(device, alias, table);
}

}
