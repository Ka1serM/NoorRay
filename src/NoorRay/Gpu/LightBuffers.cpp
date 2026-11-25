#include "NoorRay/Gpu/LightBuffers.h"

#include <algorithm>
#include <bit>
#include <cmath>

#include "NoorRay/Gpu/BufferUpload.h"

namespace NoorRay::Gpu {

namespace {

Core::float4 toFloat4(const Core::float3 value, const float w) {
    return {value.x, value.y, value.z, w};
}

std::uint32_t swapFaces(const std::uint32_t faces) {
    return (faces & Shaders::lightFlagNoFront ? Shaders::lightFlagNoBack : 0u) | (faces & Shaders::lightFlagNoBack ? Shaders::lightFlagNoFront : 0u);
}

}

Shaders::LightRecord LightBuffers::makeRecord(const Scene::Light& light, const LightBake& bake, LightBounds& bounds) {
    const Core::float3 xAxis = light.transform.getColumn(0);
    const Core::float3 yAxis = light.transform.getColumn(1);
    const Core::float3 zAxis = light.transform.getColumn(2);
    const Core::float3 direction = Core::normalize(-zAxis);
    const float scaleX = Core::length(xAxis);
    const float scaleY = Core::length(yAxis);
    //! A light texture multiplies the radiance on the device; a constant light shader folds into the colour here.
    const Core::float3 color = bake.texels ? light.color : light.color * bake.average;
    const float luminance = std::max(Core::luminance(light.color * bake.average), 0.0f);
    Shaders::LightRecord record{};
    record.position = toFloat4(light.transform.getTranslation(), light.radius);
    record.direction = toFloat4(direction, -1.0f);
    record.tangent = toFloat4(Core::normalize(xAxis), 0.0f);
    record.bitangent = toFloat4(Core::normalize(yAxis), 0.0f);
    record.meta = {Shaders::lightTypePoint, bake.texels.low, bake.texels.high, bake.texels ? bake.resolution : 0};
    const std::uint32_t flags = (bake.directional ? Shaders::lightFlagLeavingTexture : 0u) | light.falloffPower << Shaders::lightFalloffShift
                                | (light.castsShadow ? 0u : Shaders::lightFlagNoShadow)
                                | std::min(light.maxBounces, Shaders::lightBouncesUnlimited) << Shaders::lightBouncesShift
                                | light.hidden << Shaders::lightHiddenShift | (light.visibleToCamera ? Shaders::lightFlagCamera : 0u);
    record.linking = {static_cast<std::uint32_t>(light.lightMask), static_cast<std::uint32_t>(light.lightMask >> 32), flags,
                      std::bit_cast<std::uint32_t>(light.falloffSmooth)};
    const float pointIntensity = light.power / (4.0f * Core::pi);
    const Core::float3 position = light.transform.getTranslation();
    bounds = {};
    bounds.sets = light.lightMask;
    bounds.bounds.grow(position);
    //! A light that emits all around has a full cone of normals and no emission beyond it.
    bounds.cosThetaO = -1.0f;
    bounds.cosThetaE = 1.0f;
    bounds.power = pointIntensity * luminance;
    bounds.mean = position;
    bounds.flux = 4.0f * Core::pi * bounds.power;
    switch (light.type) {
    case Scene::LightType::Point:
    case Scene::LightType::Sphere: {
        const float radius = light.radius * std::max({scaleX, scaleY, Core::length(zAxis)});
        record.position.w = radius;
        if (radius > 0.0f) {
            record.meta.x = Shaders::lightTypeSphere;
            record.radiance = toFloat4(color * (light.power / (4.0f * Core::pi * Core::pi * radius * radius)), 0.0f);
        } else
            record.radiance = toFloat4(color * pointIntensity, 0.0f);
        bounds.bounds.grow(position - Core::float3{radius, radius, radius});
        bounds.bounds.grow(position + Core::float3{radius, radius, radius});
        bounds.variance = radius * radius;
        break;
    }
    case Scene::LightType::Spot: {
        const float cosOuter = std::cos(0.5f * light.spotAngle);
        record.meta.x = Shaders::lightTypeSpot;
        record.direction.w = cosOuter;
        record.tangent.w = std::max((1.0f - cosOuter) * light.spotBlend, 1e-6f);
        record.bitangent.w = 0.5f / std::tan(0.5f * std::clamp(light.spotAngle, 1e-3f, 3.14f));
        record.position.w = light.radius;
        record.radiance = toFloat4(color * pointIntensity, 0.0f);
        //! The disk of a spot with a radius turns to the receiver, so its cone is seen from anywhere in its bounds.
        bounds.bounds.grow(position - Core::float3{light.radius, light.radius, light.radius});
        bounds.bounds.grow(position + Core::float3{light.radius, light.radius, light.radius});
        bounds.axis = direction;
        bounds.cosThetaO = cosOuter;
        //! The mean direction of a uniform cone of emission.
        bounds.emission = direction * (0.5f * (1.0f + cosOuter));
        bounds.flux = 2.0f * Core::pi * (1.0f - cosOuter) * bounds.power;
        bounds.variance = light.radius * light.radius;
        break;
    }
    case Scene::LightType::Rect:
    case Scene::LightType::Disk: {
        const float halfWidth = 0.5f * light.width * scaleX;
        const float halfHeight = 0.5f * light.height * scaleY;
        const bool disk = light.type == Scene::LightType::Disk;
        //! Cycles' area_light_spread_attenuation, whose normalization keeps the power of the light.
        const float halfSpread = 0.5f * std::clamp(light.spread, 1e-2f, Core::pi);
        const bool spreads = halfSpread < 0.5f * Core::pi - 1e-4f;
        const float tangentSpread = spreads ? std::tan(halfSpread) : 0.0f;
        const float spreadScale = spreads ? tangentSpread / (tangentSpread - halfSpread) : 1.0f;
        record.direction.w = tangentSpread;
        const float area = disk ? Core::pi * halfWidth * halfHeight : 4.0f * halfWidth * halfHeight;
        record.meta.x = disk ? Shaders::lightTypeDisk : Shaders::lightTypeRect;
        record.tangent = toFloat4(Core::normalize(xAxis) * halfWidth, 0.0f);
        record.bitangent = toFloat4(Core::normalize(yAxis) * halfHeight, 0.0f);
        record.radiance = toFloat4(area > 0.0f ? color * (light.power * spreadScale / (area * Core::pi)) : Core::float3{}, area);
        for (const float x : {-1.0f, 1.0f})
            for (const float y : {-1.0f, 1.0f})
                bounds.bounds.grow(position + Core::normalize(xAxis) * (x * halfWidth) + Core::normalize(yAxis) * (y * halfHeight));
        bounds.axis = direction;
        bounds.cosThetaO = 1.0f;
        bounds.cosThetaE = 0.0f;
        bounds.planar = true;
        bounds.planeOffset = Core::dot(direction, position);
        bounds.power = area > 0.0f ? light.power / Core::pi * luminance : 0.0f;
        //! Tokuyoshi et al. fit the vMF of a one-sided area light to Lambert's cosine with a mean direction of half the normal.
        bounds.emission = direction * (0.5f + 0.5f * std::max(1.0f - 2.0f * halfSpread / Core::pi, 0.0f));
        bounds.flux = Core::pi * bounds.power;
        bounds.variance = disk ? (halfWidth * halfWidth + halfHeight * halfHeight) / 4.0f : (halfWidth * halfWidth + halfHeight * halfHeight) / 3.0f;
        break;
    }
    case Scene::LightType::Directional: {
        record.meta.x = Shaders::lightTypeDirectional;
        record.direction.w = std::cos(0.5f * std::max(light.angle, 0.0f));
        record.radiance = toFloat4(color * light.power, 0.0f);
        bounds.distant = true;
        bounds.axis = -direction;
        bounds.cosThetaO = record.direction.w;
        bounds.power = light.power * luminance;
        break;
    }
    }
    return record;
}

std::expected<void, Core::Error> LightBuffers::update(NoorRHI::Device& device, const Scene::Scene& scene, const std::span<const MaterialInfo> materials,
                                                      const std::span<const LightBake> bakes) {
    std::vector<Shaders::LightRecord> values;
    std::vector<LightBounds> lightBounds;
    const std::span<const Scene::Light> sceneLights = scene.getLights().getValues();
    for (std::size_t index = 0; index < sceneLights.size(); ++index) {
        LightBounds bounds;
        values.push_back(makeRecord(sceneLights[index], index < bakes.size() ? bakes[index] : LightBake{}, bounds));
        lightBounds.push_back(bounds);
    }
    const std::span<const Scene::Instance> instances = scene.getInstances().getValues();
    std::vector<std::pair<std::uint32_t, std::uint32_t>> emitters;
    const std::uint32_t previousCount = count;
    const std::vector<std::uint32_t> previousInstanceLights = std::move(instanceLights);
    instanceLights.assign(instances.size(), Shaders::invalidIndex);
    for (std::uint32_t instanceIndex = 0; instanceIndex < instances.size(); ++instanceIndex) {
        const Scene::Instance& instance = instances[instanceIndex];
        const Scene::Mesh* mesh = scene.find(instance.mesh);
        if (!mesh || Scene::hasFlag(instance.flags, Scene::InstanceFlags::Hidden))
            continue;
        std::vector<Core::float3> emission;
        std::vector<std::uint32_t> programs;
        std::vector<std::uint32_t> faces;
        const std::uint32_t override = scene.getMaterials().denseIndex(instance.material);
        for (const Scene::MeshSubset& subset : mesh->subsets) {
            const std::uint32_t material = override != Shaders::invalidIndex ? override : scene.getMaterials().denseIndex(subset.material);
            if (material >= materials.size() || (materials[material].flags & Shaders::materialFlagEmissive) == 0 || !materials[material].sampled
                || Core::maxComponent(materials[material].emission) <= 0.0f)
                continue;
            if (emission.empty()) {
                emission.resize(mesh->getTriangleCount());
                programs.assign(mesh->getTriangleCount(), Shaders::invalidIndex);
                faces.assign(mesh->getTriangleCount(), 0);
            }
            const std::uint32_t first = subset.firstIndex / 3;
            const std::uint32_t last = std::min<std::uint32_t>(first + subset.indexCount / 3, mesh->getTriangleCount());
            for (std::uint32_t triangle = first; triangle < last; ++triangle) {
                emission[triangle] = materials[material].emission;
                programs[triangle] = materials[material].emitterProgram;
                faces[triangle] = materials[material].skippedFaces;
            }
        }
        if (emission.empty())
            continue;
        const auto hide = [&](const Scene::InstanceFlags flag, const std::uint32_t bit) { return Scene::hasFlag(instance.flags, flag) ? bit : 0u; };
        const std::uint32_t hidden = (hide(Scene::InstanceFlags::NoDiffuseRays, Shaders::lightHiddenDiffuse)
                                      | hide(Scene::InstanceFlags::NoGlossyRays, Shaders::lightHiddenGlossy)
                                      | hide(Scene::InstanceFlags::NoTransmissionRays, Shaders::lightHiddenTransmission)
                                      | hide(Scene::InstanceFlags::NoVolumeScatterRays, Shaders::lightHiddenVolumeScatter)) << Shaders::lightHiddenShift;
        instanceLights[instanceIndex] = static_cast<std::uint32_t>(values.size());
        //! A mirroring transform turns the winding, so the face that cross(tangent, bitangent) points to is the back of the mesh.
        const bool mirrored = instance.transform.determinant() < 0.0f;
        for (std::uint32_t triangle = 0; triangle < mesh->getTriangleCount(); ++triangle) {
            const Core::float3 p0 = instance.transform.transformPoint(mesh->positions[mesh->indices[triangle * 3]]);
            const Core::float3 p1 = instance.transform.transformPoint(mesh->positions[mesh->indices[triangle * 3 + 1]]);
            const Core::float3 p2 = instance.transform.transformPoint(mesh->positions[mesh->indices[triangle * 3 + 2]]);
            const float area = 0.5f * Core::length(Core::cross(p1 - p0, p2 - p0));
            Shaders::LightRecord record{};
            record.position = toFloat4(p0, emission[triangle].x);
            record.tangent = toFloat4(p1 - p0, emission[triangle].y);
            record.bitangent = toFloat4(p2 - p0, emission[triangle].z);
            record.radiance = toFloat4(emission[triangle], area);
            record.meta = {Shaders::lightTypeTriangle, instanceIndex, triangle, 0};
            if (programs[triangle] != Shaders::invalidIndex)
                emitters.emplace_back(programs[triangle], static_cast<std::uint32_t>(values.size()));
            const std::uint32_t skipped = mirrored ? swapFaces(faces[triangle]) : faces[triangle];
            record.linking = {static_cast<std::uint32_t>(instance.lightMask), static_cast<std::uint32_t>(instance.lightMask >> 32),
                              skipped | Shaders::lightBouncesUnlimited << Shaders::lightBouncesShift | hidden, 0};
            values.push_back(record);
            LightBounds bounds;
            bounds.bounds.grow(p0);
            bounds.bounds.grow(p1);
            bounds.bounds.grow(p2);
            //! The front face is the side that cross(tangent, bitangent) points to; a triangle that emits on one side has that side's normal.
            const Core::float3 normal = Core::normalize(Core::cross(p1 - p0, p2 - p0));
            bounds.axis = (skipped & Shaders::lightFlagNoFront) != 0 ? -normal : normal;
            bounds.cosThetaO = 1.0f;
            bounds.cosThetaE = 0.0f;
            bounds.twoSided = (skipped & (Shaders::lightFlagNoFront | Shaders::lightFlagNoBack)) == 0;
            bounds.planar = !bounds.twoSided;
            bounds.planeOffset = Core::dot(bounds.axis, p0);
            const bool emits = (skipped & (Shaders::lightFlagNoFront | Shaders::lightFlagNoBack)) != (Shaders::lightFlagNoFront | Shaders::lightFlagNoBack);
            bounds.power = emits && Core::dot(normal, normal) > 0.0f ? area * std::max(Core::luminance(emission[triangle]), 0.0f) : 0.0f;
            bounds.mean = (p0 + p1 + p2) / 3.0f;
            const Core::float3 e1 = p1 - p0;
            const Core::float3 e2 = p2 - p0;
            bounds.variance = (Core::dot(e1, e1) + Core::dot(e2, e2) - Core::dot(e1, e2)) / 18.0f;
            bounds.emission = bounds.twoSided ? Core::float3{} : bounds.axis * 0.5f;
            bounds.flux = (bounds.twoSided ? 2.0f : 1.0f) * Core::pi * bounds.power;
            bounds.sets = instance.lightMask;
            lightBounds.push_back(bounds);
        }
    }
    count = static_cast<std::uint32_t>(values.size());
    visibleToCamera = std::ranges::any_of(sceneLights, &Scene::Light::visibleToCamera);
    if (count != previousCount || instanceLights != previousInstanceLights)
        ++layout;
    if (auto uploaded = uploadValues<Shaders::LightRecord>(device, records, values); !uploaded)
        return uploaded;
    std::ranges::sort(emitters);
    std::vector<std::uint32_t> emitterValues;
    std::vector<NoorRHI::uint2> rangeValues(emitters.empty() ? 0 : emitters.back().first + 1);
    emitterPrograms.clear();
    for (const auto& [program, light] : emitters) {
        if (emitterPrograms.empty() || emitterPrograms.back().first != program) {
            emitterPrograms.emplace_back(program, 0);
            rangeValues[program].x = static_cast<std::uint32_t>(emitterValues.size());
        }
        ++emitterPrograms.back().second;
        ++rangeValues[program].y;
        emitterValues.push_back(light);
    }
    if (auto uploaded = uploadValues<std::uint32_t>(device, emitterLights, emitterValues); !uploaded)
        return uploaded;
    if (auto uploaded = uploadValues<NoorRHI::uint2>(device, emitterRanges, rangeValues); !uploaded)
        return uploaded;
    std::uint32_t sets = 1;
    for (const Scene::Instance& instance : instances)
        sets = std::max(sets, std::min(instance.lightSet, Shaders::lightSetCount - 1) + 1);
    for (const Scene::Volume& volume : scene.getVolumes().getValues())
        sets = std::max(sets, std::min(volume.lightSet, Shaders::lightSetCount - 1) + 1);
    std::expected<LightTrees, Core::Error> built = buildLightTrees(lightBounds, sets);
    if (!built)
        return std::unexpected(std::move(built.error()));
    ++trees;
    if (auto uploaded = uploadValues<Shaders::LightTreeNode>(device, nodes, built->nodes); !uploaded)
        return uploaded;
    if (auto uploaded = uploadValues<Shaders::LightTreeSet>(device, treeSets, built->sets); !uploaded)
        return uploaded;
    return uploadValues<NoorRHI::uint2>(device, trails, built->trails);
}
}
