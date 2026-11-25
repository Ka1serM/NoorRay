#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "DiffRenderer/Training/Shaders/GaussianInstances.h"
#include "NoorRHI/NoorRHI.h"

namespace DiffRenderer::Core {

//! Encloses the unit sphere with 80 triangles; outward faces wind counterclockwise.
class GaussianProxyMesh {
public:
    GaussianProxyMesh();

    std::span<const NoorRHI::float3> getVertices() const { return vertices; }
    std::span<const std::uint32_t> getIndices() const { return indices; }

private:
    void subdivide();
    float getInradius() const;

    std::vector<NoorRHI::float3> vertices;
    std::vector<std::uint32_t> indices;
};

//! The acceleration structure of the Gaussians: one instance of an 80-triangle icosphere around the cutoff ellipsoid of each Gaussian.
class GaussianProxies {
public:
    [[nodiscard]] std::expected<void, NoorRHI::Error> open(NoorRHI::Device& device, std::uint32_t capacity);
    //! Unit quaternions (x, y, z, w), linear scales and activated opacities.
    [[nodiscard]] std::expected<void, NoorRHI::Error> update(NoorRHI::Device& device, std::span<const float> means, std::span<const float> quaternionsXyzw,
                                                             std::span<const float> scales, std::span<const float> opacities, std::uint32_t count);
    //! Writes the instances from the raw parameters on the GPU; count is at most the capacity.
    void recordInstances(NoorRHI::CommandList& list, const Instances::Arguments& parameters) const;
    //! Builds over the first count instances; the list with recordInstances must be submitted before.
    [[nodiscard]] std::expected<void, NoorRHI::Error> rebuild(std::uint32_t count);
    //! Refits to the moved instances of the last build, which costs less than rebuild and traces slower.
    [[nodiscard]] std::expected<void, NoorRHI::Error> refit();

    NoorRHI::AccelerationStructureHandle getTlas() const { return tlas.getHandle(); }
    std::uint32_t getCapacity() const { return capacity; }

private:
    NoorRHI::Device* device = nullptr;
    NoorRHI::ComputePipeline writeInstances;
    NoorRHI::Buffer<NoorRHI::float3> vertices;
    NoorRHI::Buffer<std::uint32_t> indices;
    NoorRHI::Buffer<NoorRHI::InstanceRecord> records;
    NoorRHI::AccelerationStructure blas;
    NoorRHI::AccelerationStructure tlas;
    std::vector<NoorRHI::InstanceRecord> hostRecords;
    std::uint32_t capacity = 0;
    std::uint32_t built = 0;
};

}
