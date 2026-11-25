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

struct GaussianCloudGpu {
    NoorRHI::Buffer<NoorRHI::float4> positionOpacity;
    NoorRHI::Buffer<NoorRHI::float4> rotations;
    NoorRHI::Buffer<NoorRHI::float4> scales;
    NoorRHI::Buffer<std::uint32_t> sh;
};

class GaussianBuffers {
public:
    [[nodiscard]] std::expected<void, Core::Error> update(NoorRHI::Device& device, const Scene::Scene& scene);

    NoorRHI::Ptr<Shaders::GaussianCloudRecord> getRecords() const { return records.getPtr(); }
    std::uint32_t getCount() const { return static_cast<std::uint32_t>(clouds.size()); }
    GaussianCloudGpu& getCloud(const std::uint32_t index) { return *clouds[index]; }
    std::uint64_t getGaussianCount() const { return gaussianCount; }

private:
    std::vector<std::unique_ptr<GaussianCloudGpu>> clouds;
    NoorRHI::Buffer<Shaders::GaussianCloudRecord> records;
    std::uint64_t gaussianCount = 0;
};

}
