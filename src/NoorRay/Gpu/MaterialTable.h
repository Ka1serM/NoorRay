#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "NoorRHI/Resources/Buffer.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Scene/Scene.h"
#include "NoorRay/Shaders/Common/GpuRecords.h"

namespace NoorRay::Gpu {

class TextureTable;

//! emission is the emitted radiance factor (colour times strength) for light sampling weights; an emission texture counts as one.
struct MaterialInfo {
    std::uint32_t flags = 0;
    Core::float3 emission;
};

//! The GPU records of the scene materials in dense order, and the texture records of their set slots.
class MaterialTable {
public:
    [[nodiscard]] std::expected<void, Core::Error> update(NoorRHI::Device& device, const Scene::Scene& scene, const TextureTable& textures);

    NoorRHI::Ptr<Shaders::MaterialRecord> getRecords() const { return records.getPtr(); }
    std::uint32_t getCount() const { return static_cast<std::uint32_t>(infos.size()); }
    std::span<const MaterialInfo> getInfos() const { return infos; }
    std::uint32_t getFlags() const { return flagUnion; }

private:
    std::vector<MaterialInfo> infos;
    std::uint32_t flagUnion = 0;
    NoorRHI::Buffer<Shaders::MaterialRecord> records;
    NoorRHI::Buffer<Shaders::MaterialTextureRecord> textureRecords;
};

}
