#include "NoorRay/Gpu/VolumeBuffers.h"

#include <algorithm>
#include <bit>
#include <span>
#include <vector>

#include "NoorRay/Core/Hash.h"
#include "NoorRay/Core/Log.h"
#include "NoorRay/Gpu/BufferUpload.h"
#include "NoorRay/Materials/Graph.h"

namespace NoorRay::Gpu {

namespace {

//! A density graph bakes to a sparse grid with this many times the voxels per side of the volume resolution, which fits noise finer than the dense grid did.
constexpr std::uint32_t graphGridDetail = 2;

Core::uint3 graphGridCells(const Core::uint3 resolution) {
    const auto cells = [](const std::uint32_t side) {
        return std::max((side * graphGridDetail + Shaders::volumeBrickSize - 1) / Shaders::volumeBrickSize, 1u);
    };
    return {cells(resolution.x), cells(resolution.y), cells(resolution.z)};
}

}

namespace {

std::vector<std::uint32_t> packPairs(const std::span<const std::uint16_t> values) {
    std::vector<std::uint32_t> packed((values.size() + 1) / 2, 0);
    for (std::size_t i = 0; i < values.size(); ++i)
        packed[i / 2] |= std::uint32_t{values[i]} << (16 * (i % 2));
    return packed;
}

std::expected<void, Core::Error> uploadEmissionAndMask(NoorRHI::Device& device, const Scene::Volume& volume, VolumeGrid& grid) {
    if (!volume.emissionGrid.empty())
        if (auto uploaded = uploadValues<std::uint32_t>(device, grid.emission, volume.emissionGrid, false); !uploaded)
            return uploaded;
    if (volume.mask.empty())
        return {};
    return uploadValues<std::uint32_t>(device, grid.mask, packPairs(volume.mask), false);
}

//! A volume with an emission of its own has some collisions across its bounds, at which the paths gather it.
float emissionMajorant(const Scene::Volume& volume) {
    constexpr float collisions = 8.0f;
    const Core::float3 rate = volume.emissionRate;
    if (rate.x + rate.y + rate.z <= 0.0f && volume.emissionGrid.empty())
        return 0.0f;
    const Core::float3 diagonal = volume.transform.transformVector(volume.bounds.max - volume.bounds.min);
    return collisions / std::max(Core::length(diagonal), 1e-6f);
}

}

std::expected<const VolumeGrid*, Core::Error> VolumeBuffers::getGrid(NoorRHI::Device& device, const Scene::Volume& volume, GraphBakes& bakes,
                                                                     const TextureTable& textures) {
    std::uint64_t hash = Core::hashValue(volume.resolution, Core::hashSeed);
    if (volume.densityGraph) {
        hash = Core::hashValue(volume.bounds, Core::combineHash(hash, hashGraphValues(*volume.densityGraph, volume.densityParameters, volume.densityTextures)));
        if (readsPlacement(*volume.densityGraph))
            hash = Core::hashValue(volume.transform, hash);
    } else
        hash = Core::hashBytes(std::as_bytes(std::span(volume.density)), hash);
    hash = Core::hashBytes(std::as_bytes(std::span(volume.emissionGrid)), hash);
    hash = Core::hashBytes(std::as_bytes(std::span(volume.mask)), hash);
    if (const auto found = grids.find(hash); found != grids.end()) {
        found->second.used = true;
        return &found->second.device;
    }
    Grid& grid = grids[hash];
    grid.used = true;
    if (auto uploaded = uploadEmissionAndMask(device, volume, grid.device); !uploaded) {
        grids.erase(hash);
        return std::unexpected(std::move(uploaded.error()));
    }
    if (volume.densityGraph) {
        if (auto baked = bakes.bakeVolume(device, *volume.densityGraph, volume.densityParameters, volume.densityTextures, textures, volume.transform,
                                          volume.bounds, graphGridCells(volume.resolution), grid.device);
            !baked) {
            grids.erase(hash);
            return std::unexpected(std::move(baked.error()));
        }
        return &grid.device;
    }
    //! A file grid comes from the host once, so its maximum is found there.
    const std::vector<std::uint32_t> packed = packPairs(volume.density);
    float maximum = 0.0f;
    for (const std::uint16_t value : volume.density)
        maximum = std::max(maximum, Core::fromHalf(Core::half{value}));
    const NoorRHI::Atomic<std::uint32_t> bits{std::bit_cast<std::uint32_t>(maximum)};
    auto uploaded = uploadValues<std::uint32_t>(device, grid.device.density, packed, false);
    if (uploaded)
        uploaded = uploadValues<NoorRHI::Atomic<std::uint32_t>>(device, grid.device.maximum, std::span(&bits, 1), false);
    if (!uploaded) {
        grids.erase(hash);
        return std::unexpected(std::move(uploaded.error()));
    }
    return &grid.device;
}

void VolumeBuffers::setExactDensity(const bool exact) {
    exactDensity = exact;
}

std::expected<void, Core::Error> VolumeBuffers::updateGraphCode(const std::span<const Scene::Volume> volumes) {
    const auto graphVolume =
        std::ranges::find_if(volumes, [](const Scene::Volume& volume) { return volume.kind == Scene::VolumeKind::Grid && volume.densityGraph; });
    if (!exactDensity || graphVolume == volumes.end()) {
        codeGraph.reset();
        graphCode.reset();
        return {};
    }
    if (codeGraph && codeGraph->hasSameProgram(*graphVolume->densityGraph))
        return {};
    std::expected<Materials::GraphCode, Core::Error> code = Materials::writeGraphCode(*graphVolume->densityGraph);
    if (!code)
        return std::unexpected(std::move(code.error()));
    codeGraph = graphVolume->densityGraph;
    graphCode = std::make_shared<const Materials::GraphCode>(std::move(*code));
    return {};
}

std::expected<void, Core::Error> VolumeBuffers::update(NoorRHI::Device& device, const Scene::Scene& scene, GraphBakes& bakes, const TextureTable& textures) {
    const std::span<const Scene::Volume> volumes = scene.getVolumes().getValues();
    if (auto updated = updateGraphCode(volumes); !updated)
        return updated;
    for (auto& [hash, grid] : grids)
        grid.used = false;
    count = static_cast<std::uint32_t>(volumes.size());
    std::vector<Shaders::VolumeRecord> values(volumes.size());
    for (std::size_t v = 0; v < volumes.size(); ++v) {
        const Scene::Volume& volume = volumes[v];
        Shaders::VolumeRecord& record = values[v];
        const Core::Transform inverse = volume.transform.inverse();
        for (int row = 0; row < 3; ++row)
            record.worldToObject[row] = {inverse.rows[row][0], inverse.rows[row][1], inverse.rows[row][2], inverse.rows[row][3]};
        record.boundsMin = {volume.bounds.min.x, volume.bounds.min.y, volume.bounds.min.z, volume.densityScale};
        record.boundsMax = {volume.bounds.max.x, volume.bounds.max.y, volume.bounds.max.z, volume.anisotropy};
        record.absorption = {volume.absorption.x, volume.absorption.y, volume.absorption.z, volume.emissionRate.x};
        record.scattering = {volume.scattering.x, volume.scattering.y, volume.scattering.z, volume.emissionRate.y};
        record.emission = {volume.emission.x, volume.emission.y, volume.emission.z, volume.emissionRate.z};
        record.maskResolution.w = std::bit_cast<std::uint32_t>(emissionMajorant(volume));
        record.resolution = {volume.resolution.x, volume.resolution.y, volume.resolution.z, 0};
        record.kind = volume.kind == Scene::VolumeKind::Grid ? Shaders::volumeKindGrid : Shaders::volumeKindHomogeneous;
        record.draineAlpha = volume.draineAlpha;
        const std::uint32_t lightSet = std::min(volume.lightSet, Shaders::lightSetCount - 1) << Shaders::lightSetShift;
        record.kind |= lightSet;
        const std::size_t expected = std::size_t{volume.resolution.x} * volume.resolution.y * volume.resolution.z;
        const bool grid = volume.kind == Scene::VolumeKind::Grid && expected > 0 && (volume.densityGraph || volume.density.size() == expected);
        if (!grid) {
            record.kind = Shaders::volumeKindHomogeneous | lightSet;
            continue;
        }
        std::expected<const VolumeGrid*, Core::Error> found = getGrid(device, volume, bakes, textures);
        if (!found)
            return std::unexpected(std::move(found.error()));
        record.density = (*found)->density.getPtr();
        record.maximum = (*found)->maximum.getPtr();
        if (!volume.emissionGrid.empty()) {
            record.emissionGrid = (*found)->emission.getPtr();
            record.flags.x = 1;
        }
        if (!volume.mask.empty()) {
            record.mask = (*found)->mask.getPtr();
            record.maskResolution = {volume.maskResolution.x, volume.maskResolution.y, volume.maskResolution.z, record.maskResolution.w};
        }
        if (!volume.densityGraph)
            continue;
        const Core::uint3 cells = graphGridCells(volume.resolution);
        record.resolution = {cells.x, cells.y, cells.z, 0};
        record.cells = (*found)->cells.getPtr();
        record.bricks = (*found)->bricks.getPtr();
        record.kind = Shaders::volumeKindSparse | lightSet;
        if (!codeGraph)
            continue;
        //! The medium kernels hold one density program, so the volumes of other programs keep their sparse grid.
        if (!codeGraph->hasSameProgram(*volume.densityGraph)) {
            Core::Log::warning("the volume '{}' has a second density graph program, so it uses its sparse grid", volume.name);
            continue;
        }
        record.kind = Shaders::volumeKindGraph | lightSet;
        record.graphParameters = (*found)->inputs.parameters.getPtr();
        record.graphTextures = (*found)->inputs.textures.getPtr();
    }
    std::erase_if(grids, [](const auto& item) { return !item.second.used; });
    return uploadValues<Shaders::VolumeRecord>(device, records, values);
}
}
