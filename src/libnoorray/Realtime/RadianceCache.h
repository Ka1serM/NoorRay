#pragma once

#include <cstdint>

#include <noorrhi/noorrhi.hpp>

#include "Realtime/FrameContext.h"
#include "Shared/RealtimeArgs.h"

// NVIDIA's SHaRC world-space radiance cache (external/SHARC). Each frame a
// sparse update pass traces paths over the view and feeds every vertex into
// the cache, and a resolve pass folds them into the radiance the image pass
// queries. Off, both passes are skipped and the image pass sees a zero
// capacity, so indirect paths trace to their end.
class RadianceCache
{
public:
    // `stages` holds the realtime miss and hit stages the update paths trace
    // through; the cache supplies its own ray generation.
    RadianceCache(noorrhi::Device& device, noorrhi::RayTracingPipelineDesc stages);
    // Rebuilds the ray-tracing pipeline around new miss, hit and callable stages.
    void setTraceStages(noorrhi::RayTracingPipelineDesc stages);

    void setMode(RadianceCacheMode mode) { mode_ = mode; }
    RadianceCacheMode mode() const { return mode_; }

    nr::graphics::RadianceCacheArgs args(const FrameContext& frame) const;
    // Update, then resolve. Leaves the resolved cache visible to ray tracing.
    // `args` is what `root` points at.
    void record(const nr::graphics::RealtimeArgs& args, nr::graphics::RealtimeRoot root) const;

private:
    noorrhi::Device& device_;
    RadianceCacheMode mode_{RenderSettings{}.radianceCacheMode};
    noorrhi::RayTracingPipeline updatePipeline_;
    noorrhi::ComputePipeline resolvePipeline_;
    // Hash keys, per-frame accumulation and resolved radiance. Both radiance
    // records are 16 bytes; their layout lives in SharcTypes.h.
    noorrhi::Buffer<std::uint64_t> hashEntries_;
    noorrhi::Buffer<std::uint32_t> accumulation_;
    noorrhi::Buffer<std::uint32_t> resolved_;
};
