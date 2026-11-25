#include "NoorRay/Gpu/GraphBakes.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <utility>

#include "NoorRHI/Commands/CommandList.h"
#include "NoorRHI/Device/Device.h"
#include "NoorRay/Core/Hash.h"
#include "NoorRay/Core/Log.h"
#include "NoorRay/Gpu/BufferUpload.h"
#include "NoorRay/Gpu/MaterialTable.h"
#include "NoorRay/Gpu/TextureTable.h"
#include "NoorRay/Materials/Graph.h"
#include "NoorRay/Materials/GraphShader.h"
#include "NoorRay/Shaders/Common/GpuRecords.h"
#include "NoorRayShadersVariants.h"

namespace NoorRay::Gpu {

namespace {

//! The texels per side of a light texture: at least enough for a gobo of fine noise, and as many as its largest image, so no image detail is lost.
constexpr std::uint32_t minLightResolution = 512;
constexpr std::uint32_t maxLightResolution = 4096;

std::uint32_t lightResolution(const Scene::Scene& scene, const std::span<const Scene::MaterialTexture> graphTextures) {
    std::uint32_t resolution = minLightResolution;
    for (const Scene::MaterialTexture& item : graphTextures)
        if (const Scene::Texture* texture = scene.find(item.texture))
            resolution = std::max({resolution, texture->width, texture->height});
    return std::min(resolution, maxLightResolution);
}

std::uint64_t getKey(const Scene::LightHandle handle) {
    return std::uint64_t{handle.generation} << 32 | handle.index;
}

std::uint32_t bakeKind(const Scene::LightType type, const bool directional) {
    switch (type) {
    case Scene::LightType::Spot:
        return Shaders::graphBakeSpot;
    case Scene::LightType::Point:
    case Scene::LightType::Sphere:
        return Shaders::graphBakeSphere;
    default:
        return directional ? Shaders::graphBakeAreaDirection : Shaders::graphBakeArea;
    }
}

Shaders::GraphBakeRecord makeRecord(const Core::Transform& transform, const std::uint32_t kind) {
    Shaders::GraphBakeRecord record{};
    record.kind = kind;
    for (std::size_t row = 0; row < 3; ++row)
        record.objectToWorld[row] = {transform.rows[row][0], transform.rows[row][1], transform.rows[row][2], transform.rows[row][3]};
    return record;
}

NoorRHI::DispatchSize getDispatch(const std::uint32_t threads) {
    const std::uint32_t groups = std::max<std::uint32_t>((threads + Shaders::graphBakeGroupSize - 1) / Shaders::graphBakeGroupSize, 1);
    const std::uint32_t rows = (groups + Shaders::graphBakeRowGroups - 1) / Shaders::graphBakeRowGroups;
    return {std::min(groups, Shaders::graphBakeRowGroups) * Shaders::graphBakeGroupSize, rows, 1};
}

}

bool readsPlacement(const Materials::Graph& graph) {
    return std::ranges::any_of(graph.getInstructions(), [](const Materials::GraphInstruction& instruction) {
        if (instruction.op != Materials::GraphOp::Input)
            return false;
        const auto input = static_cast<Materials::GraphInput>(instruction.index);
        return input == Materials::GraphInput::Position || input == Materials::GraphInput::Normal || input == Materials::GraphInput::GeometricNormal
               || input == Materials::GraphInput::Tangent || input == Materials::GraphInput::Incoming || input == Materials::GraphInput::ObjectLocation;
    });
}

bool readsOnlyDirection(const Materials::Graph& graph) {
    const std::span<const Materials::GraphInstruction> instructions = graph.getInstructions();
    std::vector<bool> used(instructions.size());
    used[graph.getOutput(Materials::GraphOutput::Emission)] = true;
    bool incoming = false;
    for (std::size_t index = instructions.size(); index-- > 0;) {
        const Materials::GraphInstruction& instruction = instructions[index];
        if (!used[index])
            continue;
        for (const std::uint32_t operand : graph.operands(instruction))
            used[operand] = true;
        if (instruction.op != Materials::GraphOp::Input)
            continue;
        switch (static_cast<Materials::GraphInput>(instruction.index)) {
        case Materials::GraphInput::Incoming:
            incoming = true;
            break;
        case Materials::GraphInput::Position:
        case Materials::GraphInput::Normal:
        case Materials::GraphInput::GeometricNormal:
        case Materials::GraphInput::Uv0:
        case Materials::GraphInput::Uv1:
        case Materials::GraphInput::Uv2:
        case Materials::GraphInput::Uv3:
        case Materials::GraphInput::ObjectPosition:
        case Materials::GraphInput::Generated:
            return false;
        default:
            break;
        }
    }
    return incoming;
}

std::uint64_t hashGraphValues(const Materials::Graph& graph, const std::span<const Core::float4> parameters,
                              const std::span<const Scene::MaterialTexture> textures) {
    std::uint64_t hash = Core::combineHash(Core::hashSeed, graph.getHash());
    hash = Core::hashBytes(std::as_bytes(parameters), hash);
    for (const Scene::MaterialTexture& texture : textures) {
        hash = Core::combineHash(hash, std::uint64_t{texture.texture.generation} << 32 | texture.texture.index);
        hash = Core::combineHash(hash, std::uint64_t{std::to_underlying(texture.wrap)} << 8 | std::to_underlying(texture.filter));
        hash = Core::hashValue(texture.transform, hash);
    }
    return hash;
}

std::expected<const NoorRHI::ComputePipeline*, Core::Error> GraphBakes::getKernel(NoorRHI::Device& device, const Materials::Graph& graph) {
    std::expected<Materials::GraphCode, Core::Error> code = Materials::writeGraphCode(graph);
    if (!code)
        return std::unexpected(std::move(code.error()));
    std::unique_ptr<Kernel>& kernel = kernels[code->key];
    if (kernel)
        return &kernel->pipeline;
    auto created = std::make_unique<Kernel>();
    if (auto spliced = Materials::spliceGraphShader(*Shaders::bakeGraph[0], *code, created->code, created->shader); !spliced) {
        kernels.erase(code->key);
        return std::unexpected(std::move(spliced.error()));
    }
    if (auto opened = Core::check(created->pipeline.open(device, created->shader)); !opened) {
        kernels.erase(code->key);
        return std::unexpected(std::move(opened.error()));
    }
    kernel = std::move(created);
    return &kernel->pipeline;
}

std::expected<void, Core::Error> GraphBakes::bake(NoorRHI::Device& device, const Materials::Graph& graph, const std::span<const Core::float4> parameters,
                                                  const std::span<const Scene::MaterialTexture> graphTextures, const TextureTable& textures,
                                                  Shaders::GraphBakeRecord record, const std::uint32_t threads, BakeInputs& inputs) {
    if (parameters.size() != graph.getParameterSlots() || graphTextures.size() != graph.getTextureCount())
        return Core::fail(Core::ErrorCode::InvalidArgument,
                          std::format("a light or volume graph has {} parameters and {} textures, but reads {} and {}", parameters.size(), graphTextures.size(),
                                      graph.getParameterSlots(), graph.getTextureCount()));
    std::expected<const NoorRHI::ComputePipeline*, Core::Error> pipeline = getKernel(device, graph);
    if (!pipeline)
        return std::unexpected(std::move(pipeline.error()));
    std::vector<Shaders::MaterialTextureRecord> textureRecords;
    for (const Scene::MaterialTexture& texture : graphTextures)
        textureRecords.push_back(makeTextureRecord(texture, textures));
    if (auto result = uploadValues<NoorRHI::float4>(device, inputs.parameters, parameters, false); !result)
        return result;
    if (auto result = uploadValues<Shaders::MaterialTextureRecord>(device, inputs.textures, textureRecords, false); !result)
        return result;
    record.parameters = inputs.parameters.getPtr();
    record.textures = inputs.textures.getPtr();
    if (auto result = uploadValues<Shaders::GraphBakeRecord>(device, inputs.record, std::span(&record, 1), false); !result)
        return result;
    NoorRHI::CommandList list(device);
    list.pushLabel("GraphBake");
    list.dispatch(**pipeline, getDispatch(threads), Shaders::GraphBakeArguments{inputs.record.getPtr()});
    list.popLabel();
    if (auto token = Core::check(device.submit(std::move(list))); !token)
        return std::unexpected(std::move(token.error()));
    return {};
}

std::expected<void, Core::Error> GraphBakes::average(NoorRHI::Device& device, const std::uint32_t kind, const std::uint32_t resolution, const bool wait,
                                                     Entry& entry) {
    if (!averagePipeline)
        if (auto opened = Core::check(averagePipeline.open(device, *Shaders::averageLightBake[0])); !opened)
            return opened;
    if (!entry.average)
        if (auto opened = Core::check(entry.average.open(device, 1)); !opened)
            return opened;
    if (!entry.readback)
        if (auto opened = Core::check(entry.readback.open(device, 1)); !opened)
            return opened;
    NoorRHI::CommandList list(device);
    list.pushLabel("LightAverage");
    list.dispatch(averagePipeline, NoorRHI::DispatchSize{Shaders::lightAverageGroupSize, 1, 1},
                  Shaders::LightAverageArguments{entry.output.getPtr(), entry.average.getPtr(), resolution, kind});
    list.download(entry.average, entry.readback);
    list.popLabel();
    auto token = Core::check(device.submit(std::move(list)));
    if (!token)
        return std::unexpected(std::move(token.error()));
    entry.pending = true;
    if (!wait)
        return {};
    if (auto waited = Core::check(device.wait(*token)); !waited)
        return waited;
    return collectAverages().transform([](bool) {});
}

std::expected<bool, Core::Error> GraphBakes::collectAverages() {
    bool changed = false;
    for (auto& [key, entry] : entries) {
        if (!entry.pending || !entry.readback.isReady())
            continue;
        auto values = Core::check(entry.readback.read());
        if (!values)
            return std::unexpected(std::move(values.error()));
        entry.pending = false;
        const NoorRHI::float4 mean = (*values)[0];
        const Core::float3 average{mean.x, mean.y, mean.z};
        changed = changed || average != entry.bake.average;
        entry.bake.average = average;
        if (entry.light < lights.size())
            lights[entry.light].average = average;
    }
    return changed;
}

std::expected<void, Core::Error> GraphBakes::updateLights(NoorRHI::Device& device, const Scene::Scene& scene, const TextureTable& textures) {
    const Core::SlotMap<Scene::Light>& sceneLights = scene.getLights();
    lights.assign(sceneLights.size(), LightBake{});
    for (auto& [key, entry] : entries)
        entry.used = false;
    for (std::uint32_t index = 0; index < sceneLights.size(); ++index) {
        const Scene::Light& light = sceneLights.getValues()[index];
        if (!light.graph)
            continue;
        const Materials::Graph& graph = *light.graph;
        if (const float* constant = graph.constantOf(graph.getOutput(Materials::GraphOutput::Emission))) {
            lights[index].average = Core::float3{constant[0], constant[1], constant[2]} * std::max(constant[3], 0.0f);
            continue;
        }
        const bool directional = readsOnlyDirection(graph);
        const std::uint32_t kind = bakeKind(light.type, directional);
        Shaders::GraphBakeRecord record = makeRecord(light.transform, kind);
        record.shape = {0.5f * light.width, 0.5f * light.height, light.radius, 0.5f / std::tan(0.5f * std::clamp(light.spotAngle, 1e-3f, 3.14f))};
        const std::uint32_t resolution = lightResolution(scene, light.graphTextures);
        record.resolution = {resolution, resolution, 1, 0};
        std::uint64_t hash = Core::combineHash(Core::combineHash(hashGraphValues(graph, light.graphParameters, light.graphTextures), kind), resolution);
        hash = Core::hashValue(record.shape, hash);
        if (readsPlacement(graph))
            hash = Core::hashValue(light.transform, hash);
        Entry& entry = entries[getKey(sceneLights.keyAt(index))];
        entry.used = true;
        entry.light = index;
        if (entry.key != hash || !entry.output) {
            const std::size_t texels = std::size_t{resolution} * resolution;
            if (!entry.output || entry.output.getSize() != texels)
                if (auto opened = Core::check(entry.output.open(device, texels)); !opened)
                    return opened;
            record.output = entry.output.getPtr();
            if (auto baked =
                    bake(device, graph, light.graphParameters, light.graphTextures, textures, record, static_cast<std::uint32_t>(texels), entry.inputs);
                !baked)
                return baked;
            //! A mean from an earlier bake still gives unbiased light sampling, since the sampling and its pdf read the same table.
            const bool first = entry.key == 0;
            if (first || !entry.pending)
                if (auto averaged = average(device, kind, resolution, first, entry); !averaged)
                    return averaged;
            entry.key = hash;
            entry.bake.resolution = resolution;
            entry.bake.texels = entry.output.getPtr();
            entry.bake.directional = directional;
            if (light.type == Scene::LightType::Directional) {
                Core::Log::warning("light '{}': a sun shader bakes to its mean colour", light.name);
                entry.bake.texels = {};
                entry.bake.resolution = 0;
            }
        }
        lights[index] = entry.bake;
    }
    std::erase_if(entries, [](const auto& item) { return !item.second.used; });
    return {};
}

std::expected<std::vector<std::uint32_t>, Core::Error> GraphBakes::bakeWorld(NoorRHI::Device& device, const Scene::Environment& environment,
                                                                             const TextureTable& textures, const std::uint32_t kind,
                                                                             const std::uint32_t width) {
    //! The object space of the bake is the z-up space of Blender.
    Core::Transform toWorld;
    toWorld.rows[1][1] = 0.0f;
    toWorld.rows[1][2] = 1.0f;
    toWorld.rows[2][1] = -1.0f;
    toWorld.rows[2][2] = 0.0f;
    Shaders::GraphBakeRecord record = makeRecord(toWorld, kind);
    record.resolution = {width, width / 2, 1, 0};
    const std::size_t texels = std::size_t{width} * (width / 2);
    NoorRHI::Buffer<std::uint32_t> output;
    if (auto opened = Core::check(output.open(device, texels)); !opened)
        return std::unexpected(std::move(opened.error()));
    NoorRHI::Readback<std::uint32_t> readback;
    if (auto opened = Core::check(readback.open(device, texels)); !opened)
        return std::unexpected(std::move(opened.error()));
    record.output = output.getPtr();
    BakeInputs inputs;
    if (auto baked = bake(device, *environment.graph, environment.graphParameters, environment.graphTextures, textures, record,
                          static_cast<std::uint32_t>(texels), inputs);
        !baked)
        return std::unexpected(std::move(baked.error()));
    NoorRHI::CommandList list(device);
    list.download(output, readback);
    auto token = Core::check(device.submit(std::move(list)));
    if (!token)
        return std::unexpected(std::move(token.error()));
    if (auto waited = Core::check(device.wait(*token)); !waited)
        return std::unexpected(std::move(waited.error()));
    auto values = Core::check(readback.read());
    if (!values)
        return std::unexpected(std::move(values.error()));
    return std::vector<std::uint32_t>(values->begin(), values->end());
}

std::expected<void, Core::Error> GraphBakes::bakeVolume(NoorRHI::Device& device, const Materials::Graph& graph, const std::span<const Core::float4> parameters,
                                                        const std::span<const Scene::MaterialTexture> graphTextures, const TextureTable& textures,
                                                        const Core::Transform& transform, const Core::Bounds& bounds, const Core::uint3 cells,
                                                        VolumeGrid& grid) {
    Shaders::GraphBakeRecord record = makeRecord(transform, Shaders::graphBakeVolumeCells);
    record.shape = {bounds.min.x, bounds.min.y, bounds.min.z, 0.0f};
    record.boundsMax = {bounds.max.x, bounds.max.y, bounds.max.z, 0.0f};
    //! The sample count comes from the record, so that the driver does not unroll the loop over a brick with the graph inlined, which never finishes compiling.
    record.resolution = {cells.x, cells.y, cells.z, Shaders::volumeBrickSamples * Shaders::volumeBrickSamples * Shaders::volumeBrickSamples};
    const std::size_t count = std::size_t{cells.x} * cells.y * cells.z;
    if (auto opened = Core::check(grid.cells.open(device, count)); !opened)
        return opened;
    if (auto opened = openZeroed(device, grid.maximum, 3); !opened)
        return opened;
    if (!grid.readback)
        if (auto opened = Core::check(grid.readback.open(device, 3)); !opened)
            return opened;
    record.output = grid.cells.getPtr().as<std::uint32_t>();
    record.maximum = grid.maximum.getPtr();
    if (auto baked = bake(device, graph, parameters, graphTextures, textures, record, static_cast<std::uint32_t>(count), grid.inputs); !baked)
        return baked;
    NoorRHI::CommandList list(device);
    list.download(grid.maximum, grid.readback);
    auto token = Core::check(device.submit(std::move(list)));
    if (!token)
        return std::unexpected(std::move(token.error()));
    if (auto waited = Core::check(device.wait(*token)); !waited)
        return waited;
    auto counts = Core::check(grid.readback.read());
    if (!counts)
        return std::unexpected(std::move(counts.error()));
    const std::size_t bricks = (*counts)[1].value;
    Core::Log::info("baked a sparse volume grid of {}x{}x{} cells: {} bricks ({:.1f} MB)", cells.x, cells.y, cells.z, bricks,
                    static_cast<double>(bricks * Shaders::volumeBrickWords * 4 + count * 8) / (1 << 20));
    if (bricks == 0)
        return {};
    if (auto opened = Core::check(grid.bricks.open(device, bricks * Shaders::volumeBrickWords)); !opened)
        return opened;
    record.kind = Shaders::graphBakeVolumeBricks;
    record.bricks = grid.bricks.getPtr();
    return bake(device, graph, parameters, graphTextures, textures, record, static_cast<std::uint32_t>(count), grid.inputs);
}

}
