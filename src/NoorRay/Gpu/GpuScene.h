#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "NoorRHI/Resources/Buffer.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Gpu/AccelerationStructures.h"
#include "NoorRay/Gpu/EnergyTables.h"
#include "NoorRay/Gpu/EnvironmentBuffers.h"
#include "NoorRay/Gpu/GaussianBuffers.h"
#include "NoorRay/Gpu/LightBuffers.h"
#include "NoorRay/Gpu/MaterialTable.h"
#include "NoorRay/Gpu/MeshBuffers.h"
#include "NoorRay/Gpu/TextureTable.h"
#include "NoorRay/Gpu/VolumeBuffers.h"
#include "NoorRay/Scene/Scene.h"
#include "NoorRay/Spectral/SpectralTables.h"

namespace NoorRay::Gpu {

//! Scene content that selects shader variants.
struct SceneContent {
    bool hasGaussians = false;
    bool hasVolumes = false;
    bool hasDispersion = false;
    bool hasAlphaTest = false;
    bool hasEmissiveMeshes = false;
    bool operator==(const SceneContent&) const = default;
};

struct RasterDraw {
    std::uint32_t instance = 0;
    std::uint32_t firstTriangle = 0;
    std::uint32_t vertexCount = 0;
};

struct GpuSceneStats {
    std::uint32_t meshes = 0;
    std::uint32_t instances = 0;
    std::uint32_t materials = 0;
    std::uint32_t lights = 0;
    std::uint64_t gaussians = 0;
    std::size_t meshBytes = 0;
    std::size_t textureBytes = 0;
};

//! All GPU data of one scene. update applies the scene's changes with the cheapest upload; the acceleration structures wait for buildAccelerationStructures.
class GpuScene {
public:
    [[nodiscard]] std::expected<void, Core::Error> open(NoorRHI::Device& device);
    [[nodiscard]] std::expected<void, Core::Error> update(Scene::Scene& scene);
    [[nodiscard]] std::expected<void, Core::Error> buildAccelerationStructures(const Scene::Scene& scene);

    NoorRHI::Ptr<Shaders::SceneRecord> getRecord() const { return record.getPtr(); }
    const Shaders::SceneRecord& getHostRecord() const { return hostRecord; }
    //! The changes that the last update applied, so that the renderer can restart accumulation or validate reservoirs.
    Scene::Change getLastChange() const { return lastChange; }
    const SceneContent& getContent() const { return content; }
    GpuSceneStats getStats() const;
    Core::Bounds getBounds() const { return bounds; }

    const MaterialTable& getMaterials() const { return materials; }
    const MeshBuffers& getMeshes() const { return meshes; }
    std::span<const RasterDraw> getRasterDraws() const { return rasterDraws; }
    NoorRHI::Ptr<NoorRHI::uint2> getRasterDrawRecords() const { return rasterDrawRecords.getPtr(); }
    const TextureTable& getTextures() const { return textures; }
    std::uint32_t getGaussianCount() const { return static_cast<std::uint32_t>(gaussians.getGaussianCount()); }
    NoorRHI::Device& getDevice() const { return *device; }

private:
    [[nodiscard]] std::expected<void, Core::Error> uploadInstances(const Scene::Scene& scene);
    [[nodiscard]] std::expected<void, Core::Error> uploadRasterDraws(const Scene::Scene& scene);
    [[nodiscard]] std::expected<void, Core::Error> uploadRecord();

    NoorRHI::Device* device = nullptr;
    TextureTable textures;
    MaterialTable materials;
    MeshBuffers meshes;
    LightBuffers lights;
    EnvironmentBuffers environment;
    GaussianBuffers gaussians;
    VolumeBuffers volumes;
    AccelerationStructures accelerationStructures;
    Spectral::SpectralTables spectral;
    EnergyTables energy;
    NoorRHI::Buffer<Shaders::InstanceData> instances;
    std::vector<RasterDraw> rasterDraws;
    NoorRHI::Buffer<NoorRHI::uint2> rasterDrawRecords;
    NoorRHI::Buffer<Shaders::SceneRecord> record;
    Shaders::SceneRecord hostRecord{};
    SceneContent content;
    Core::Bounds bounds;
    Scene::Change lastChange = Scene::Change::None;
    bool first = true;
};

}
