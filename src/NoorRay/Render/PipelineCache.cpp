#include "NoorRay/Render/PipelineCache.h"

#include <algorithm>
#include <format>
#include <string_view>
#include <utility>

#include "NoorRay/Core/Hash.h"
#include "NoorRay/Core/Log.h"
#include "NoorRay/Materials/Graph.h"
#include "NoorRay/Materials/GraphShader.h"
#include "NoorRayShadersVariants.h"

namespace NoorRay::Render {

namespace {

//! The number of graph sets whose material kernels stay, so that a switch between the scenes of the sessions does not compile again.
constexpr std::size_t keptSets = 4;

//! A density graph of zero outputs for the medium kernels without exact density: NVIDIA stalls compiling the placeholder graph, which calls every helper.
std::expected<Materials::GraphCode, Core::Error> writeEmptyGraphCode() {
    Materials::GraphBuilder builder;
    const std::uint32_t zero = builder.constant(Core::float4{0.0f, 0.0f, 0.0f, 0.0f});
    for (std::size_t output = 0; output < Materials::graphOutputCount; ++output)
        builder.setOutput(static_cast<Materials::GraphOutput>(output), zero);
    Materials::Graph graph;
    if (auto opened = graph.open(builder.take(), "the empty volume graph"); !opened)
        return std::unexpected(std::move(opened.error()));
    return Materials::writeGraphCode(graph);
}

void markReady(bool& ready, const std::chrono::steady_clock::time_point start, const std::string_view what) {
    if (std::exchange(ready, true))
        return;
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
    Core::Log::info("compiled {} in {} ms", what, elapsed.count());
}

}

std::expected<void, Core::Error> PipelineCache::setGraphCodes(NoorRHI::Device& device, const std::span<const std::shared_ptr<const Materials::GraphCode>> codes,
                                                              const std::span<const std::shared_ptr<const Materials::GraphCode>> alphaCodes) {
    if (!recentSets.empty() && std::ranges::equal(codes, graphCodes) && std::ranges::equal(alphaCodes, alphaGraphCodes))
        return {};
    graphCodes.assign(codes.begin(), codes.end());
    alphaGraphCodes.assign(alphaCodes.begin(), alphaCodes.end());
    GraphSet set{Core::hashSeed, {}, Core::hashSeed};
    for (const std::shared_ptr<const Materials::GraphCode>& code : graphCodes) {
        set.key = Core::combineHash(set.key, code->key);
        set.graphs.push_back(code->key);
        if (auto opened = openMaterial(device, *code); !opened)
            return std::unexpected(std::move(opened.error()));
    }
    for (const std::shared_ptr<const Materials::GraphCode>& code : alphaGraphCodes)
        set.alphaKey = Core::combineHash(set.alphaKey, code->key);
    set.key = Core::combineHash(set.key, set.alphaKey);
    if (auto opened = openAlphaDonors(); !opened)
        return opened;
    std::erase_if(recentSets, [&](const GraphSet& other) { return other.key == set.key; });
    recentSets.push_back(std::move(set));
    dropUnusedSets();
    return {};
}

std::expected<void, Core::Error> PipelineCache::openAlphaDonors() {
    for (const std::shared_ptr<const Materials::GraphCode>& code : alphaGraphCodes) {
        std::unique_ptr<AlphaDonor>& donor = alphaDonors[code->key];
        if (donor)
            continue;
        donor = std::make_unique<AlphaDonor>();
        if (auto spliced = Materials::spliceGraphShader(*Shaders::alphaGraph[0], *code, donor->code, donor->shader, Materials::alphaGraphFunctionName);
            !spliced) {
            alphaDonors.erase(code->key);
            return spliced;
        }
    }
    return {};
}

void PipelineCache::dropUnusedSets() {
    while (recentSets.size() > keptSets)
        recentSets.erase(recentSets.begin());
    const auto unused = [&](const auto& item) {
        return std::ranges::none_of(recentSets, [&](const GraphSet& set) { return std::ranges::contains(set.graphs, item.first); });
    };
    std::erase_if(materials, unused);
    std::erase_if(graphKernels, [&](const auto& item) {
        const std::uint64_t key = item.second->alphaKey;
        return key != 0 && std::ranges::none_of(recentSets, [&](const GraphSet& set) { return set.alphaKey == key; });
    });
    std::erase_if(alphaDonors,
                  [&](const auto& item) { return !std::ranges::contains(alphaGraphCodes, item.first, [](const auto& code) { return code->key; }); });
}

void PipelineCache::setVolumeGraphCode(std::shared_ptr<const Materials::GraphCode> code) {
    volumeCode = std::move(code);
    if (!volumeCode || (!recentVolumeCodes.empty() && recentVolumeCodes.back() == volumeCode->key))
        return;
    std::erase(recentVolumeCodes, volumeCode->key);
    recentVolumeCodes.push_back(volumeCode->key);
    while (recentVolumeCodes.size() > keptSets)
        recentVolumeCodes.erase(recentVolumeCodes.begin());
    std::erase_if(graphKernels, [&](const auto& item) {
        const std::uint64_t key = item.second->volumeKey;
        return key != 0 && key != emptyVolumeCode.key && !std::ranges::contains(recentVolumeCodes, key);
    });
}

std::expected<PipelineCache::GraphCalls, Core::Error> PipelineCache::getGraphCalls(const NoorRHI::Shader& shader) {
    if (const auto found = graphCallers.find(&shader); found != graphCallers.end())
        return found->second;
    std::expected<bool, Core::Error> volume = Materials::hasGraphFunction(shader.code[0]);
    if (!volume)
        return std::unexpected(std::move(volume.error()));
    std::expected<bool, Core::Error> alpha = Materials::hasGraphFunction(shader.code[0], Materials::alphaGraphSwitchName);
    if (!alpha)
        return std::unexpected(std::move(alpha.error()));
    const GraphCalls calls{*volume, *alpha};
    graphCallers.emplace(&shader, calls);
    return calls;
}

std::expected<PipelineCache::Compute*, Core::Error> PipelineCache::openGraphs(NoorRHI::Device& device, const NoorRHI::Shader& shader, const GraphCalls calls,
                                                                              const bool async) {
    if (emptyVolumeCode.code.empty()) {
        std::expected<Materials::GraphCode, Core::Error> empty = writeEmptyGraphCode();
        if (!empty)
            return std::unexpected(std::move(empty.error()));
        emptyVolumeCode = std::move(*empty);
    }
    if (recentSets.empty())
        return Core::fail(Core::ErrorCode::InvalidState, std::format("{} runs graphs, but the pipeline cache has no graph codes", shader.entryPoint));
    const Materials::GraphCode& volume = volumeCode ? *volumeCode : emptyVolumeCode;
    const std::uint64_t volumeKey = calls.volume ? volume.key : 0;
    const std::uint64_t alphaKey = calls.alpha ? recentSets.back().alphaKey : 0;
    const std::uint64_t key =
        Core::combineHash(Core::combineHash(Core::hashValue(reinterpret_cast<std::uintptr_t>(&shader), Core::hashSeed), volumeKey), alphaKey);
    std::unique_ptr<Compute>& entry = graphKernels[key];
    if (entry)
        return entry.get();
    entry = std::make_unique<Compute>();
    entry->start = std::chrono::steady_clock::now();
    entry->volumeKey = volumeKey;
    entry->alphaKey = alphaKey;
    std::array<std::vector<std::byte>, 4> spliced;
    NoorRHI::Shader base = shader;
    if (calls.volume) {
        if (auto done = Materials::spliceGraphShader(shader, volume, spliced, base); !done) {
            graphKernels.erase(key);
            return std::unexpected(std::move(done.error()));
        }
    }
    if (!calls.alpha) {
        entry->code = std::move(spliced);
        entry->shader = base;
    }
    std::vector<const NoorRHI::Shader*> donors;
    for (const std::shared_ptr<const Materials::GraphCode>& code : alphaGraphCodes)
        donors.push_back(&alphaDonors.at(code->key)->shader);
    //! The volume graph goes in first: the link copies a graph function of each alpha graph, which the splice would take for the volume graph.
    if (calls.alpha) {
        if (auto done = Materials::linkAlphaGraphShader(base, donors, entry->code, entry->shader); !done) {
            graphKernels.erase(key);
            return std::unexpected(std::move(done.error()));
        }
    }
    Core::Log::info("compiling {} with the volume graph {:016x} and {} alpha graphs {:016x}{}", shader.entryPoint, volumeKey, donors.size(), alphaKey,
                    async ? " in the background" : "");
    if (auto opened = async ? entry->pipeline.openAsync(device, entry->shader) : entry->pipeline.open(device, entry->shader); !opened) {
        graphKernels.erase(key);
        Core::Log::error("compiling {} with the volume graph {:016x} and the alpha graphs {:016x} failed: {}", shader.entryPoint, volumeKey, alphaKey,
                         opened.error().message);
        return Core::fail(std::move(opened.error()));
    }
    return entry.get();
}

std::expected<PipelineCache::Compute*, Core::Error> PipelineCache::openCompute(NoorRHI::Device& device, const NoorRHI::Shader& shader, const bool async) {
    std::expected<GraphCalls, Core::Error> calls = getGraphCalls(shader);
    if (!calls)
        return std::unexpected(std::move(calls.error()));
    if (calls->volume || calls->alpha)
        return openGraphs(device, shader, *calls, async);
    std::unique_ptr<Compute>& entry = computes[&shader];
    if (entry)
        return entry.get();
    entry = std::make_unique<Compute>();
    entry->start = std::chrono::steady_clock::now();
    Core::Log::info("compiling {}{}", shader.entryPoint, async ? " in the background" : "");
    if (auto opened = async ? entry->pipeline.openAsync(device, shader) : entry->pipeline.open(device, shader); !opened) {
        computes.erase(&shader);
        Core::Log::error("compiling {} failed: {}", shader.entryPoint, opened.error().message);
        return Core::fail(std::move(opened.error()));
    }
    return entry.get();
}

std::expected<PipelineCache::Material*, Core::Error> PipelineCache::openMaterial(NoorRHI::Device& device, const Materials::GraphCode& code) {
    std::unique_ptr<Material>& entry = materials[code.key];
    if (entry)
        return entry.get();
    entry = std::make_unique<Material>();
    entry->start = std::chrono::steady_clock::now();
    if (auto spliced = Materials::spliceGraphShader(*Shaders::evaluateMaterials[0], code, entry->code, entry->shader); !spliced) {
        materials.erase(code.key);
        return std::unexpected(std::move(spliced.error()));
    }
    Core::Log::info("compiling the material kernel of the graph {:016x} in the background", code.key);
    if (auto opened = entry->pipeline.openAsync(device, entry->shader); !opened) {
        materials.erase(code.key);
        Core::Log::error("compiling the material kernel of the graph {:016x} failed: {}", code.key, opened.error().message);
        return Core::fail(std::move(opened.error()));
    }
    return entry.get();
}

std::expected<const NoorRHI::ComputePipeline*, Core::Error> PipelineCache::getMaterial(NoorRHI::Device& device, const std::uint32_t program) {
    if (program == 0)
        return get(device, *Shaders::evaluateFixedMaterials[0]);
    if (program > graphCodes.size())
        return Core::fail(Core::ErrorCode::InvalidArgument, std::format("the material program {} has no graph", program));
    const Materials::GraphCode& code = *graphCodes[program - 1];
    std::expected<Material*, Core::Error> entry = openMaterial(device, code);
    if (!entry)
        return std::unexpected(std::move(entry.error()));
    if (auto waited = (*entry)->pipeline.wait(); !waited) {
        Core::Log::error("compiling the material kernel of the graph {:016x} failed: {}", code.key, waited.error().message);
        return Core::fail(std::move(waited.error()));
    }
    markReady((*entry)->ready, (*entry)->start, std::format("the material kernel of the graph {:016x}", code.key));
    return &(*entry)->pipeline;
}

std::expected<const NoorRHI::ComputePipeline*, Core::Error> PipelineCache::get(NoorRHI::Device& device, const NoorRHI::Shader& shader) {
    std::expected<Compute*, Core::Error> entry = openCompute(device, shader, false);
    if (!entry)
        return std::unexpected(std::move(entry.error()));
    if (auto waited = (*entry)->pipeline.wait(); !waited) {
        Core::Log::error("compiling {} failed: {}", shader.entryPoint, waited.error().message);
        return Core::fail(std::move(waited.error()));
    }
    markReady((*entry)->ready, (*entry)->start, shader.entryPoint);
    return &(*entry)->pipeline;
}

std::expected<bool, Core::Error> PipelineCache::isReady(NoorRHI::Device& device, const NoorRHI::Shader& shader) {
    std::expected<Compute*, Core::Error> entry = openCompute(device, shader, true);
    if (!entry)
        return std::unexpected(std::move(entry.error()));
    if (!(*entry)->pipeline.isReady())
        return false;
    return get(device, shader).transform([](const auto*) { return true; });
}

}
