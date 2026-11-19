#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <vector>

#include "NoorRHI/Resources/Buffer.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Scene/Scene.h"
#include "NoorRay/Shaders/Common/GpuRecords.h"

namespace NoorRay::Gpu {

class VolumeBuffers {
public:
    [[nodiscard]] std::expected<void, Core::Error> update(NoorRHI::Device& device, const Scene::Scene& scene);

    NoorRHI::Ptr<Shaders::VolumeRecord> getRecords() const { return records.getPtr(); }
    std::uint32_t getCount() const { return count; }

private:
    std::vector<std::unique_ptr<NoorRHI::Buffer<std::uint32_t>>> densities;
    NoorRHI::Buffer<Shaders::VolumeRecord> records;
    std::uint32_t count = 0;
};

}
