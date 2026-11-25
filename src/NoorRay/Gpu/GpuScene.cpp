#include "NoorRay/Gpu/GpuScene.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <utility>
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
    std::size_t attributeCount = 0;
    for (const Scene::Instance& instance : sceneInstances)
        attributeCount = std::max(attributeCount, instance.attributes.size());
    std::vector<Core::float4> attributes(sceneInstances.size() * attributeCount);
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
        if (Scene::hasFlag(instance.flags, Scene::InstanceFlags::Holdout))
            flags |= Shaders::instanceFlagHoldout;
        flags |= std::min(instance.lightSet, Shaders::lightSetCount - 1) << Shaders::lightSetShift;
        data.meta = {scene.getMeshes().denseIndex(instance.mesh), flags, scene.getMaterials().denseIndex(instance.material), light};
        data.info = {Core::packHalf2(instance.color.x, instance.color.y), Core::packHalf2(instance.color.z, instance.color.w),
                     std::min(instance.passIndex, 0xffffu) | static_cast<std::uint32_t>(std::lround(instance.random * 65535.0f)) << 16,
                     static_cast<std::uint32_t>(i * attributeCount)};
        std::ranges::copy(instance.attributes, attributes.begin() + std::ptrdiff_t(i * attributeCount));
    }
    if (auto uploaded = uploadValues<Shaders::InstanceData>(*device, instances, values); !uploaded)
        return uploaded;
    if (attributes.empty())
        instanceAttributes = {};
    else if (auto uploaded = uploadValues<Core::float4>(*device, instanceAttributes, attributes); !uploaded)
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

void GpuScene::setExactVolumeDensity(const bool exact) {
    if (exact == volumes.hasExactDensity())
        return;
    volumes.setExactDensity(exact);
    volumeModeChanged = true;
}

std::expected<void, Core::Error> GpuScene::setLightTree(const bool enabled) {
    if (lightTree == enabled)
        return {};
    lightTree = enabled;
    fillFlags();
    return uploadRecord();
}

void GpuScene::fillFlags() {
    const std::uint32_t flags = (lightTree ? 0u : Shaders::sceneFlagByPower) | (lights.areVisibleToCamera() ? Shaders::sceneFlagCameraLights : 0u)
                                | environment.getHidden() << Shaders::sceneHiddenShift;
    hostRecord.environmentParams.z = std::bit_cast<float>(flags);
}

std::expected<void, Core::Error> GpuScene::update(Scene::Scene& scene) {
    if (!device)
        return Core::fail(Core::ErrorCode::InvalidState, "the GPU scene is not open");
    const Scene::SceneChange change = scene.takeChange();
    Change changes = first ? allChanges : change.getChanges();
    std::expected<bool, Core::Error> averaged = bakes.collectAverages();
    if (!averaged)
        return std::unexpected(std::move(averaged.error()));
    //! A light mean that the device sent back late changes only the light sampling weights.
    if (*averaged)
        changes |= Change::Lights;
    if (std::exchange(volumeModeChanged, false))
        changes |= Change::Volumes;
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
    std::expected<bool, Core::Error> displaced = displacement.update(*device, scene, change, first, textures, meshes);
    if (!displaced)
        return std::unexpected(std::move(displaced.error()));
    if (first || has(Change::MeshTopology | Change::MeshPositions | Change::MeshReplaced | Change::MaterialSet | Change::InstanceSet) || *displaced)
        if (auto result = meshes.uploadRecords(*device, scene); !result)
            return result;
    //! A displaced mesh moves on the device alone, for the renderer and the acceleration structures as for a position upload.
    if (*displaced)
        changes |= Change::MeshPositions;
    lastChange = changes;
    if (has(Scene::geometryChanges))
        bounds = scene.computeBounds();
    const float sceneRadius = bounds.isEmpty() ? 1.0f : std::max(0.5f * Core::length(bounds.extent()), 1e-3f);
    const bool lightsChanged = has(Change::Lights | Change::InstanceTransforms | Change::InstanceSet | Change::MeshPositions | Change::MeshTopology
                                   | Change::MeshReplaced | Change::MaterialSet | Change::MaterialValues | Change::Textures | Change::Volumes);
    if (has(Change::Lights | Change::Textures))
        if (auto result = bakes.updateLights(*device, scene, textures); !result)
            return result;
    if (lightsChanged)
        if (auto result = lights.update(*device, scene, materials.getInfos(), bakes.getLights()); !result)
            return result;
    if (lightsChanged || has(Change::InstanceTransforms | Change::InstanceSet | Change::MeshTopology | Change::MeshReplaced))
        if (auto result = uploadInstances(scene); !result)
            return result;
    if (has(Change::GaussianSet | Change::GaussianValues))
        if (auto result = gaussians.update(*device, scene); !result)
            return result;
    if (has(Change::Volumes))
        if (auto result = volumes.update(*device, scene, bakes, textures); !result)
            return result;
    if (has(Change::Environment | Change::Textures))
        if (auto result = environment.update(*device, scene, textures, bakes); !result)
            return result;
    if (has(Change::InstanceTransforms))
        accelerationStructures.markInstancesMoved();
    if (has(Change::InstanceSet | Change::MeshTopology | Change::MeshPositions | Change::MeshReplaced | Change::MaterialSet))
        accelerationStructures.markInstanceSetChanged();
    if (has(Change::GaussianSet | Change::GaussianValues))
        gaussianAccelerationStructure.markChanged();

    content.hasGaussians = GaussianAccelerationStructure::isAvailable() && gaussians.getGaussianCount() > 0;
    content.hasVolumes = volumes.getCount() > 0;
    content.hasDispersion = (materials.getFlags() & Shaders::materialFlagDispersive) != 0;
    content.hasAlphaTest = (materials.getFlags() & Shaders::materialFlagAlphaTest) != 0;
    content.hasSubsurface = (materials.getFlags() & Shaders::materialFlagSubsurface) != 0;
    content.hasEmissiveMeshes = std::ranges::any_of(lights.getInstanceLights(), [](const std::uint32_t light) { return light != Shaders::invalidIndex; });

    hostRecord.meshes = meshes.getRecords();
    hostRecord.instances = instances.getPtr();
    hostRecord.instanceAttributes = instanceAttributes.getPtr();
    hostRecord.materials = materials.getRecords();
    hostRecord.graphParameters = materials.getGraphParameters();
    hostRecord.lights = lights.getRecords();
    hostRecord.lightNodes = lights.getNodes();
    hostRecord.lightSets = lights.getTreeSets();
    hostRecord.lightTrails = lights.getTrails();
    hostRecord.emitterLights = lights.getEmitterLights();
    hostRecord.emitterRanges = lights.getEmitterRanges();
    hostRecord.gaussianClouds = gaussians.getRecords();
    hostRecord.volumes = volumes.getRecords();
    hostRecord.spectral = spectral.getEntries();
    hostRecord.conductors = spectral.getConductors();
    hostRecord.energyTables = energy.getValues();
    hostRecord.tlas = accelerationStructures.isStale() ? NoorRHI::AccelerationStructureHandle{} : accelerationStructures.getTlas();
    hostRecord.gaussianTlas = gaussianAccelerationStructure.isStale() ? NoorRHI::AccelerationStructureHandle{} : gaussianAccelerationStructure.getTlas();
    hostRecord.counts = {static_cast<std::uint32_t>(scene.getInstances().size()), lights.getCount(), gaussians.getCount(), volumes.getCount()};
    environment.fill(hostRecord);
    //! The light tree weighs the environment by the irradiance of its average radiance on a surface.
    hostRecord.environmentParams.y = scene.getEnvironment().sampled ? environment.getAverageLuminance() : 0.0f;
    hostRecord.environmentParams.w = sceneRadius;
    fillFlags();
    const Core::float3 center = bounds.isEmpty() ? Core::float3{} : bounds.center();
    hostRecord.sceneCenter = {center.x, center.y, center.z, hostRecord.sceneCenter.w};
    first = false;
    return uploadRecord();
}

std::expected<void, Core::Error> GpuScene::buildAccelerationStructures(const Scene::Scene& scene, const bool tracesGaussians) {
    const bool tracedGaussians = tracesGaussians && content.hasGaussians;
    if (!accelerationStructures.isStale() && !gaussianAccelerationStructure.needsBuild(tracedGaussians))
        return {};
    if (auto result = accelerationStructures.build(*device, scene, meshes); !result)
        return result;
    if (auto result = gaussianAccelerationStructure.build(*device, scene, tracedGaussians); !result)
        return result;
    hostRecord.tlas = accelerationStructures.getTlas();
    hostRecord.gaussianTlas = gaussianAccelerationStructure.getTlas();
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
