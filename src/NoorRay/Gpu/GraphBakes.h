#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

#include "NoorRHI/Pipelines/Pipeline.h"
#include "NoorRHI/Resources/Buffer.h"
#include "NoorRHI/Resources/Readback.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Scene/Scene.h"
#include "NoorRay/Shaders/Materials/GraphBakeRecords.h"

namespace NoorRay::Materials {
class Graph;
}

namespace NoorRay::Gpu {

class TextureTable;

//! The light shader of one light: a half4 texture over the light UV, or none when constant; average scales the light colour for the sampling weight.
struct LightBake {
    NoorRHI::Ptr<std::uint32_t> texels;
    std::uint32_t resolution = 0;
    Core::float3 average{1.0f, 1.0f, 1.0f};
    //! The texture is over the leaving direction, for a shader that reads only the direction, such as an IES profile.
    bool directional = false;
};

//! The device buffers one bake reads besides its output.
struct BakeInputs {
    NoorRHI::Buffer<NoorRHI::float4> parameters;
    NoorRHI::Buffer<Shaders::MaterialTextureRecord> textures;
    NoorRHI::Buffer<Shaders::GraphBakeRecord> record;
};

//! A device density grid: a file grid as half pairs, x fastest, or a sparse graph grid of cells and bricks; maximum holds the bits of the largest density.
struct VolumeGrid {
    NoorRHI::Buffer<std::uint32_t> density;
    NoorRHI::Buffer<NoorRHI::uint2> cells;
    NoorRHI::Buffer<std::uint32_t> bricks;
    NoorRHI::Buffer<std::uint32_t> emission;
    NoorRHI::Buffer<std::uint32_t> mask;
    //! The bits of the largest density, the count of bricks and the next free brick slot.
    NoorRHI::Buffer<NoorRHI::Atomic<std::uint32_t>> maximum;
    NoorRHI::Readback<NoorRHI::Atomic<std::uint32_t>> readback;
    BakeInputs inputs;
};

//! True when graph reads a value that depends on where its object is, so that a move of the object bakes again.
bool readsPlacement(const Materials::Graph& graph);
//! True when the emission of graph reads the incoming direction and no input that varies over a light surface.
bool readsOnlyDirection(const Materials::Graph& graph);
//! A hash of the program, parameter values and textures of a graph.
std::uint64_t hashGraphValues(const Materials::Graph& graph, std::span<const Core::float4> parameters, std::span<const Scene::MaterialTexture> textures);

//! Bakes the light and volume node graphs on the device: one kernel per graph, which runs the graph at the texel or voxel centres.
class GraphBakes {
public:
    //! A light bake runs again only when its graph, values or (for world space inputs) placement change; its mean comes back later, except at the first bake.
    [[nodiscard]] std::expected<void, Core::Error> updateLights(NoorRHI::Device& device, const Scene::Scene& scene, const TextureTable& textures);
    //! Takes the means that the device has sent back since the last call; true when one changes the light weights.
    [[nodiscard]] std::expected<bool, Core::Error> collectAverages();
    std::span<const LightBake> getLights() const { return lights; }

    //! Bakes a volume graph density into a sparse grid of cells (bricks of volumeBrickSize voxels) in the local bounds; it waits once for the brick count.
    [[nodiscard]] std::expected<void, Core::Error> bakeVolume(NoorRHI::Device& device, const Materials::Graph& graph, std::span<const Core::float4> parameters,
                                                              std::span<const Scene::MaterialTexture> graphTextures, const TextureTable& textures,
                                                              const Core::Transform& transform, const Core::Bounds& bounds, Core::uint3 cells,
                                                              VolumeGrid& grid);

    //! The radiance of a world graph over an equirect map of width by width / 2 texels as RGB9E5, in the axes of the environment without rotation.
    [[nodiscard]] std::expected<std::vector<std::uint32_t>, Core::Error> bakeWorld(NoorRHI::Device& device, const Scene::Environment& environment,
                                                                                  const TextureTable& textures, std::uint32_t kind, std::uint32_t width);

private:
    struct Kernel {
        std::array<std::vector<std::byte>, 4> code;
        NoorRHI::Shader shader;
        NoorRHI::ComputePipeline pipeline;
    };

    struct Entry {
        std::uint64_t key = 0;
        NoorRHI::Buffer<std::uint32_t> output;
        BakeInputs inputs;
        LightBake bake;
        NoorRHI::Buffer<NoorRHI::float4> average;
        NoorRHI::Readback<NoorRHI::float4> readback;
        std::uint32_t light = 0;
        bool pending = false;
        bool used = false;
    };

    [[nodiscard]] std::expected<const NoorRHI::ComputePipeline*, Core::Error> getKernel(NoorRHI::Device& device, const Materials::Graph& graph);
    //! Queues threads of the kernel of graph over the record, whose output the caller sets; later work on the queue sees the result.
    [[nodiscard]] std::expected<void, Core::Error> bake(NoorRHI::Device& device, const Materials::Graph& graph, std::span<const Core::float4> parameters,
                                                        std::span<const Scene::MaterialTexture> graphTextures, const TextureTable& textures,
                                                        Shaders::GraphBakeRecord record, std::uint32_t threads, BakeInputs& inputs);

    //! Queues the mean of the texture of entry and its copy to the host; waits for it when the light has no mean yet.
    [[nodiscard]] std::expected<void, Core::Error> average(NoorRHI::Device& device, std::uint32_t kind, std::uint32_t resolution, bool wait, Entry& entry);

    std::unordered_map<std::uint64_t, std::unique_ptr<Kernel>> kernels;
    NoorRHI::ComputePipeline averagePipeline;
    //! The light bakes by light key; a bake without a light at the next update goes.
    std::unordered_map<std::uint64_t, Entry> entries;
    std::vector<LightBake> lights;
};

}
