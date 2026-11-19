#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <vector>

#include "NoorRHI/Pipelines/RayTracing.h"
#include "NoorRHI/Resources/Buffer.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Scene/Scene.h"
#include "NoorRay/Shaders/Common/GpuRecords.h"

namespace NoorRay::Gpu {

//! Encloses the unit sphere; outward faces wind counterclockwise.
struct GaussianProxy {
    std::vector<Core::float3> vertices;
    std::vector<std::uint32_t> indices;
    std::uint32_t getTriangleCount() const { return static_cast<std::uint32_t>(indices.size() / 3); }
};

struct GaussianCloudGpu {
    NoorRHI::Buffer<NoorRHI::float4> positionOpacity;
    NoorRHI::Buffer<NoorRHI::float4> rotations;
    NoorRHI::Buffer<NoorRHI::float4> scales;
    NoorRHI::Buffer<std::uint32_t> sh;
};

class GaussianBuffers {
public:
    [[nodiscard]] std::expected<void, Core::Error> update(NoorRHI::Device& device, const Scene::Scene& scene);

    static const GaussianProxy& getProxy(const NoorRHI::Device& device);
    //! The opacity cutoff 1/255 ends a Gaussian at sqrt(2 ln(255 opacity)) standard deviations.
    static void writeInstances(const Scene::GaussianCloud& cloud, std::uint32_t cloudIndex, std::uint32_t first, std::span<NoorRHI::InstanceRecord> records,
                               std::uint64_t proxyAddress);

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
