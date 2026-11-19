#include "NoorRay/Gpu/VolumeBuffers.h"

#include <algorithm>

#include "NoorRay/Gpu/BufferUpload.h"

namespace NoorRay::Gpu {

std::expected<void, Core::Error> VolumeBuffers::update(NoorRHI::Device& device, const Scene::Scene& scene) {
    const std::span<const Scene::Volume> volumes = scene.getVolumes().getValues();
    count = static_cast<std::uint32_t>(volumes.size());
    densities.clear();
    std::vector<Shaders::VolumeRecord> values(volumes.size());
    for (std::size_t v = 0; v < volumes.size(); ++v) {
        const Scene::Volume& volume = volumes[v];
        Shaders::VolumeRecord& record = values[v];
        const Core::Transform inverse = volume.transform.inverse();
        for (int row = 0; row < 3; ++row)
            record.worldToObject[row] = {inverse.rows[row][0], inverse.rows[row][1], inverse.rows[row][2], inverse.rows[row][3]};
        record.boundsMin = {volume.bounds.min.x, volume.bounds.min.y, volume.bounds.min.z, volume.densityScale};
        record.boundsMax = {volume.bounds.max.x, volume.bounds.max.y, volume.bounds.max.z, volume.anisotropy};
        record.absorption = {volume.absorption.x, volume.absorption.y, volume.absorption.z, 1.0f};
        record.scattering = {volume.scattering.x, volume.scattering.y, volume.scattering.z, 0.0f};
        record.emission = {volume.emission.x, volume.emission.y, volume.emission.z, 0.0f};
        record.resolution = {volume.resolution.x, volume.resolution.y, volume.resolution.z, 0};
        record.kind = volume.kind == Scene::VolumeKind::Grid ? Shaders::volumeKindGrid : Shaders::volumeKindHomogeneous;
        record.unused = 0;
        const std::size_t expected = std::size_t{volume.resolution.x} * volume.resolution.y * volume.resolution.z;
        if (volume.kind != Scene::VolumeKind::Grid || expected == 0 || volume.density.size() != expected) {
            record.kind = Shaders::volumeKindHomogeneous;
            continue;
        }
        std::vector<std::uint32_t> packed((expected + 1) / 2, 0);
        float maxDensity = 0.0f;
        for (std::size_t i = 0; i < expected; ++i) {
            packed[i / 2] |= std::uint32_t{volume.density[i]} << (16 * (i % 2));
            maxDensity = std::max(maxDensity, Core::fromHalf(Core::half{volume.density[i]}));
        }
        record.absorption.w = maxDensity;
        auto& buffer = densities.emplace_back(std::make_unique<NoorRHI::Buffer<std::uint32_t>>());
        if (auto result = uploadValues<std::uint32_t>(device, *buffer, packed, false); !result)
            return result;
        record.density = buffer->getPtr();
    }
    return uploadValues<Shaders::VolumeRecord>(device, records, values);
}

}
