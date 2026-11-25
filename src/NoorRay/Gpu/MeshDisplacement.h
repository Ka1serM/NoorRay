#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <unordered_map>
#include <vector>

#include "NoorRHI/Pipelines/Pipeline.h"
#include "NoorRHI/Resources/Buffer.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Scene/Scene.h"
#include "NoorRay/Shaders/Materials/DisplacementRecords.h"

namespace NoorRay::Materials {
class Graph;
class Program;
}

namespace NoorRay::Gpu {

class MeshBuffers;
class TextureTable;
struct MeshGpu;

//! Displaces the meshes whose materials have a displacement program or graph on the GPU, over the streams of MeshBuffers, from a GPU copy of the rest mesh.
class MeshDisplacement {
public:
    //! Runs after MeshBuffers::update; true when a mesh moved, so that its BLAS is refit.
    [[nodiscard]] std::expected<bool, Core::Error> update(NoorRHI::Device& device, const Scene::Scene& scene, const Scene::SceneChange& change, bool everything,
                                                          const TextureTable& textures, MeshBuffers& meshes);
    std::size_t getCount() const { return entries.size(); }

private:
    struct Kernel {
        std::array<std::vector<std::byte>, 4> code;
        NoorRHI::Shader shader;
        NoorRHI::ComputePipeline pipeline;
    };

    //! The points that the graph of one material moves.
    struct GraphPass {
        Scene::MaterialHandle material;
        const NoorRHI::ComputePipeline* pipeline = nullptr;
        NoorRHI::Buffer<NoorRHI::uint2> points;
        NoorRHI::Buffer<float> normals;
        NoorRHI::Buffer<NoorRHI::float4> parameters;
        NoorRHI::Buffer<Shaders::MaterialTextureRecord> textures;
        NoorRHI::Buffer<Shaders::DisplacementGraphRecord> record;
        std::uint32_t pointCount = 0;
    };

    struct Entry {
        Scene::MaterialHandle material;
        const void* program = nullptr;
        std::vector<Core::float4> registers;
        NoorRHI::Buffer<NoorRHI::uint2> instructions;
        NoorRHI::Buffer<NoorRHI::float4> registerValues;
        NoorRHI::Buffer<NoorRHI::float4> attributes;
        NoorRHI::Buffer<float> restPositions;
        NoorRHI::Buffer<float> restNormals;
        NoorRHI::Buffer<float> faceNormals;
        NoorRHI::Buffer<float> vertexNormals;
        NoorRHI::Buffer<std::uint32_t> cornerVertices;
        NoorRHI::Buffer<std::uint32_t> faceOffsets;
        NoorRHI::Buffer<std::uint32_t> cornerOffsets;
        NoorRHI::Buffer<NoorRHI::uint2> corners;
        NoorRHI::Buffer<std::uint32_t> restTangents;
        NoorRHI::Buffer<Shaders::DisplacementRecord> record;
        Shaders::DisplacementRecord values{};
        std::vector<GraphPass> graphs;
        std::uint64_t graphKey = 0;
    };

    [[nodiscard]] static std::expected<void, Core::Error> prepare(NoorRHI::Device& device, const Scene::Mesh& mesh, const Materials::Program* program,
                                                                  Entry& entry);
    //! The points of each displacing material of the mesh: a vertex goes with the first triangle that has it and a displacing material.
    [[nodiscard]] static std::expected<void, Core::Error> prepareGraphs(NoorRHI::Device& device, const Scene::Scene& scene, const Scene::Mesh& mesh,
                                                                        Entry& entry);
    [[nodiscard]] std::expected<void, Core::Error> updateGraphs(NoorRHI::Device& device, const Scene::Scene& scene, const Scene::Mesh& mesh,
                                                                const Core::Transform& transform, const TextureTable& textures, MeshGpu& gpu, Entry& entry);
    [[nodiscard]] std::expected<const NoorRHI::ComputePipeline*, Core::Error> getKernel(NoorRHI::Device& device, const Materials::Graph& graph);
    [[nodiscard]] std::expected<void, Core::Error> openPipelines(NoorRHI::Device& device);

    std::unordered_map<std::uint64_t, Entry> entries;
    std::unordered_map<std::uint64_t, std::unique_ptr<Kernel>> kernels;
    std::array<NoorRHI::ComputePipeline, 4> pipelines;
    bool opened = false;
    std::vector<std::uint64_t> warned;
};

}
