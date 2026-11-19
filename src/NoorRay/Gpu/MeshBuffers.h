#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <memory>
#include <unordered_map>
#include <vector>

#include "NoorRHI/Pipelines/RayTracing.h"
#include "NoorRHI/Resources/Buffer.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Scene/Scene.h"
#include "NoorRay/Shaders/Common/GpuRecords.h"

namespace NoorRay::Gpu {

//! The GPU streams of one mesh and its BLAS, which is built on first use. blasStale asks for a build, blasRefit for a refit after a position upload.
struct MeshGpu {
    NoorRHI::Buffer<NoorRHI::float3> positions;
    NoorRHI::Buffer<std::uint32_t> indices;
    NoorRHI::Buffer<std::uint32_t> normals;
    NoorRHI::Buffer<std::uint32_t> tangents;
    std::array<NoorRHI::Buffer<std::uint32_t>, Scene::maxUvSets> uvs;
    NoorRHI::Buffer<std::uint32_t> colors;
    NoorRHI::Buffer<std::uint32_t> triangleMaterials;
    NoorRHI::AccelerationStructure blas;
    bool blasStale = true;
    bool blasRefit = false;
    bool opaque = true;
};

//! Meshes stay by handle, so that adding a mesh uploads only that mesh. Records follow the dense scene order.
class MeshBuffers {
public:
    [[nodiscard]] std::expected<void, Core::Error> update(NoorRHI::Device& device, const Scene::Scene& scene, const Scene::SceneChange& change,
                                                          std::span<const std::uint32_t> materialFlags);

    NoorRHI::Ptr<Shaders::MeshRecord> getRecords() const { return records.getPtr(); }
    std::uint32_t getCount() const { return static_cast<std::uint32_t>(dense.size()); }
    MeshGpu& getMesh(const std::uint32_t denseIndex) { return *dense[denseIndex]; }
    const MeshGpu& getMesh(const std::uint32_t denseIndex) const { return *dense[denseIndex]; }
    std::size_t getByteSize() const { return byteSize; }

private:
    [[nodiscard]] static std::expected<void, Core::Error> upload(NoorRHI::Device& device, const Scene::Mesh& mesh, MeshGpu& gpu);
    [[nodiscard]] static std::expected<void, Core::Error> uploadMaterials(NoorRHI::Device& device, const Scene::Scene& scene, const Scene::Mesh& mesh,
                                                                          std::span<const std::uint32_t> materialFlags, bool alphaTestedOverride, MeshGpu& gpu);

    std::unordered_map<std::uint64_t, std::unique_ptr<MeshGpu>> meshes;
    std::vector<MeshGpu*> dense;
    NoorRHI::Buffer<Shaders::MeshRecord> records;
    std::size_t byteSize = 0;
};

}
