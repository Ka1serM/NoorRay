#include "NoorRay/Gpu/GpuScene.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "NoorRay/Gpu/BufferUpload.h"

namespace NoorRay::Gpu {

namespace {

using Scene::Change;

constexpr Change allChanges = Scene::lightingChanges | Change::Camera;

void writeRows(NoorRHI::float4 (&rows)[3], const Core::Transform& transform) {
    for (int row = 0; row < 3; ++row)
        rows[row] = {transform.rows[row][0], transform.rows[row][1], transform.rows[row][2], transform.rows[row][3]};
}

}

std::expected<void, Core::Error> GpuScene::open(NoorRHI::Device& openedDevice) {
    device = &openedDevice;
    if (auto result = textures.open(openedDevice); !result)
        return result;
    if (auto result = spectral.open(openedDevice); !result)
        return result;
    if (auto result = energy.open(openedDevice); !result)
        return result;
    if (auto result = Core::check(record.open(openedDevice, 1)); !result)
        return result;
    first = true;
    return {};
}

std::expected<void, Core::Error> GpuScene::uploadInstances(const Scene::Scene& scene) {
    const std::span<const Scene::Instance> sceneInstances = scene.getInstances().getValues();
    const std::span<const std::uint32_t> instanceLights = lights.getInstanceLights();
    std::vector<Shaders::InstanceData> values(sceneInstances.size());
    for (std::size_t i = 0; i < sceneInstances.size(); ++i) {
        const Scene::Instance& instance = sceneInstances[i];
        Shaders::InstanceData& data = values[i];
        writeRows(data.objectToWorld, instance.transform);
        writeRows(data.worldToObject, instance.transform.inverse());
        writeRows(data.previousObjectToWorld, instance.previousTransform);
        std::uint32_t flags = Scene::hasFlag(instance.flags, Scene::InstanceFlags::DoubleSided) ? Shaders::instanceFlagDoubleSided : 0u;
        const std::uint32_t light = i < instanceLights.size() ? instanceLights[i] : Shaders::invalidIndex;
        if (light != Shaders::invalidIndex)
            flags |= Shaders::instanceFlagEmissive;
        data.meta = {scene.getMeshes().denseIndex(instance.mesh), flags, scene.getMaterials().denseIndex(instance.material), light};
    }
    if (auto uploaded = uploadValues<Shaders::InstanceData>(*device, instances, values); !uploaded)
        return uploaded;
    return uploadRasterDraws(scene);
}

std::expected<void, Core::Error> GpuScene::uploadRasterDraws(const Scene::Scene& scene) {
    const std::span<const Scene::Instance> sceneInstances = scene.getInstances().getValues();
    const std::span<const Scene::Mesh> sceneMeshes = scene.getMeshes().getValues();
    rasterDraws.clear();
    std::vector<NoorRHI::uint2> records;
    std::uint32_t firstTriangle = 0;
    for (std::uint32_t index = 0; index < sceneInstances.size(); ++index) {
        const Scene::Instance& instance = sceneInstances[index];
        if (Scene::hasFlag(instance.flags, Scene::InstanceFlags::Hidden) || Scene::hasFlag(instance.flags, Scene::InstanceFlags::NoCameraRays))
            continue;
        const std::uint32_t mesh = scene.getMeshes().denseIndex(instance.mesh);
        const std::uint32_t triangles = mesh < sceneMeshes.size() ? sceneMeshes[mesh].getTriangleCount() : 0;
        if (triangles == 0)
            continue;
        rasterDraws.push_back({index, firstTriangle, triangles * 3});
        records.push_back({firstTriangle, index});
        firstTriangle += triangles;
    }
    return uploadValues<NoorRHI::uint2>(*device, rasterDrawRecords, records);
}

std::expected<void, Core::Error> GpuScene::uploadRecord() {
    return Core::check(record.upload(std::span(&hostRecord, 1)));
}

std::expected<void, Core::Error> GpuScene::update(Scene::Scene& scene) {
    if (!device)
        return Core::fail(Core::ErrorCode::InvalidState, "the GPU scene is not open");
    const Scene::SceneChange change = scene.takeChange();
    const Change changes = first ? allChanges : change.getChanges();
    lastChange = changes;
    if (changes == Change::None)
        return {};
    const auto has = [&](const Change mask) { return (changes & mask) != Change::None; };
    if (has(Change::Textures))
        if (auto result = textures.update(*device, scene); !result)
            return result;
    if (has(Change::MaterialSet | Change::MaterialValues | Change::Textures))
        if (auto result = materials.update(*device, scene, textures); !result)
            return result;
    std::vector<std::uint32_t> materialFlags;
    for (const MaterialInfo& info : materials.getInfos())
        materialFlags.push_back(info.flags);
    Scene::SceneChange meshChange = change;
    if (first)
        meshChange.mark(Change::MeshTopology);
    if (auto result = meshes.update(*device, scene, meshChange, materialFlags); !result)
        return result;
    if (has(Scene::geometryChanges))
        bounds = scene.computeBounds();
    const float sceneRadius = bounds.isEmpty() ? 1.0f : std::max(0.5f * Core::length(bounds.extent()), 1e-3f);
    const bool lightsChanged = has(Change::Lights | Change::InstanceTransforms | Change::InstanceSet | Change::MeshPositions | Change::MeshTopology
                                   | Change::MaterialSet | Change::MaterialValues | Change::Textures);
    if (lightsChanged)
        if (auto result = lights.update(*device, scene, materials.getInfos(), sceneRadius); !result)
            return result;
    if (lightsChanged || has(Change::InstanceTransforms | Change::InstanceSet | Change::MeshTopology))
        if (auto result = uploadInstances(scene); !result)
            return result;
    if (has(Change::GaussianSet | Change::GaussianValues))
        if (auto result = gaussians.update(*device, scene); !result)
            return result;
    if (has(Change::Volumes))
        if (auto result = volumes.update(*device, scene); !result)
            return result;
    if (has(Change::Environment | Change::Textures))
        if (auto result = environment.update(*device, scene, textures); !result)
            return result;
    if (has(Change::InstanceTransforms))
        accelerationStructures.markInstancesMoved();
    if (has(Change::InstanceSet | Change::MeshTopology | Change::MeshPositions | Change::MaterialSet))
        accelerationStructures.markInstanceSetChanged();
    if (has(Change::GaussianSet | Change::GaussianValues))
        accelerationStructures.markGaussiansChanged();

    content.hasGaussians = gaussians.getCount() > 0;
    content.hasVolumes = volumes.getCount() > 0;
    content.hasDispersion = (materials.getFlags() & Shaders::materialFlagDispersive) != 0;
    content.hasAlphaTest = (materials.getFlags() & Shaders::materialFlagAlphaTest) != 0;
    content.hasEmissiveMeshes = std::ranges::any_of(lights.getInstanceLights(), [](const std::uint32_t light) { return light != Shaders::invalidIndex; });

    hostRecord.meshes = meshes.getRecords();
    hostRecord.instances = instances.getPtr();
    hostRecord.materials = materials.getRecords();
    hostRecord.lights = lights.getRecords();
    hostRecord.lightAlias = lights.getAlias();
    hostRecord.gaussianClouds = gaussians.getRecords();
    hostRecord.volumes = volumes.getRecords();
    hostRecord.spectral = spectral.getEntries();
    hostRecord.conductors = spectral.getConductors();
    hostRecord.energyTables = energy.getValues();
    const bool stale = accelerationStructures.isStale();
    hostRecord.tlas = stale ? NoorRHI::AccelerationStructureHandle{} : accelerationStructures.getTlas();
    hostRecord.gaussianTlas = stale ? NoorRHI::AccelerationStructureHandle{} : accelerationStructures.getGaussianTlas();
    hostRecord.counts = {static_cast<std::uint32_t>(scene.getInstances().size()), lights.getCount(), gaussians.getCount(), volumes.getCount()};
    environment.fill(hostRecord);
    const float environmentPower = environment.getAverageLuminance() * Core::pi * Core::pi * sceneRadius * sceneRadius;
    const float lightPower = lights.getTotalPower();
    float environmentProbability = 0.0f;
    if (environmentPower > 0.0f)
        environmentProbability = lightPower > 0.0f ? std::clamp(environmentPower / (environmentPower + lightPower), 0.05f, 0.95f) : 1.0f;
    hostRecord.environmentParams.y = environmentProbability;
    hostRecord.environmentParams.z = lightPower;
    hostRecord.environmentParams.w = sceneRadius;
    const Core::float3 center = bounds.isEmpty() ? Core::float3{} : bounds.center();
    hostRecord.sceneCenter = {center.x, center.y, center.z, hostRecord.sceneCenter.w};
    first = false;
    return uploadRecord();
}

std::expected<void, Core::Error> GpuScene::buildAccelerationStructures(const Scene::Scene& scene) {
    if (!accelerationStructures.isStale())
        return {};
    if (auto result = accelerationStructures.build(*device, scene, meshes); !result)
        return result;
    hostRecord.tlas = accelerationStructures.getTlas();
    hostRecord.gaussianTlas = accelerationStructures.getGaussianTlas();
    return uploadRecord();
}

GpuSceneStats GpuScene::getStats() const {
    GpuSceneStats stats;
    stats.meshes = meshes.getCount();
    stats.instances = hostRecord.counts.x;
    stats.materials = materials.getCount();
    stats.lights = lights.getCount();
    stats.gaussians = gaussians.getGaussianCount();
    stats.meshBytes = meshes.getByteSize();
    stats.textureBytes = textures.getByteSize();
    return stats;
}

}
