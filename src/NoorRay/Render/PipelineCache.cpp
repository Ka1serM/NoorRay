#include "NoorRay/Render/PipelineCache.h"

#include "NoorRay/Core/Log.h"

namespace NoorRay::Render {

std::expected<PipelineCache::Entry*, Core::Error> PipelineCache::open(NoorRHI::Device& device, const NoorRHI::Shader& shader, const bool async) {
    std::unique_ptr<Entry>& entry = pipelines[&shader];
    if (entry)
        return entry.get();
    entry = std::make_unique<Entry>();
    entry->start = std::chrono::steady_clock::now();
    Core::Log::info("compiling {}{}", shader.entryPoint, async ? " in the background" : "");
    if (auto opened = async ? entry->pipeline.openAsync(device, shader) : entry->pipeline.open(device, shader); !opened) {
        pipelines.erase(&shader);
        Core::Log::error("compiling {} failed: {}", shader.entryPoint, opened.error().message);
        return Core::fail(std::move(opened.error()));
    }
    return entry.get();
}

void PipelineCache::markReady(Entry& entry, const NoorRHI::Shader& shader) {
    if (entry.ready)
        return;
    entry.ready = true;
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - entry.start);
    Core::Log::info("compiled {} in {} ms", shader.entryPoint, elapsed.count());
}

std::expected<const NoorRHI::ComputePipeline*, Core::Error> PipelineCache::get(NoorRHI::Device& device, const NoorRHI::Shader& shader) {
    std::expected<Entry*, Core::Error> entry = open(device, shader, false);
    if (!entry)
        return std::unexpected(std::move(entry.error()));
    if (auto waited = (*entry)->pipeline.wait(); !waited) {
        Core::Log::error("compiling {} failed: {}", shader.entryPoint, waited.error().message);
        return Core::fail(std::move(waited.error()));
    }
    markReady(**entry, shader);
    return &(*entry)->pipeline;
}

std::expected<bool, Core::Error> PipelineCache::isReady(NoorRHI::Device& device, const NoorRHI::Shader& shader) {
    std::expected<Entry*, Core::Error> entry = open(device, shader, true);
    if (!entry)
        return std::unexpected(std::move(entry.error()));
    if (!(*entry)->pipeline.isReady())
        return false;
    if (auto waited = (*entry)->pipeline.wait(); !waited) {
        Core::Log::error("compiling {} failed: {}", shader.entryPoint, waited.error().message);
        return Core::fail(std::move(waited.error()));
    }
    markReady(**entry, shader);
    return true;
}

}
