#include "Realtime/RadianceCache.h"

#include <algorithm>
#include <span>
#include <vector>

#include "Realtime/ShaderLoading.h"

namespace
{
alignas(uint32_t) constexpr unsigned char updateSpv[] = {
    #embed "RealtimeRaytracer/RealtimeSharcUpdate.spv"
};
alignas(uint32_t) constexpr unsigned char resolveSpv[] = {
    #embed "RealtimeRaytracer/SharcResolve.spv"
};

// NVIDIA's default capacity; memory is 40 bytes per entry.
constexpr uint32_t Capacity = 1u << 22;
constexpr uint32_t RecordWords = 4u;
constexpr uint32_t ResolveGroupSize = 256u;
// The update pass spreads one path per UpdateStride^2 of the output, so the
// cache costs a fixed fraction of a full-size frame.
constexpr uint32_t UpdateStride = 4u;
// ...but never fewer paths than this, whatever the viewport measures. The
// cache is world-space: its voxels are sized from camera distance, so a small
// window asks the same number of voxels to be filled as a maximized one and
// only gives the pass a fraction of the rays to fill them with. Below roughly
// this budget the per-voxel variance stops averaging out and the cache reads
// as blocky lighting. 320x180 is a 720p frame's worth of update paths - about
// 58k - which costs little next to a primary pass but is what the grid needs
// to converge at all. Raise it if blocks survive in a small viewport; lower it
// if the cache costs too much at one.
constexpr uint32_t MinUpdateGridWidth = 320u;
constexpr uint32_t MinUpdateGridHeight = 180u;
// Voxels span about camera distance / scene scale, independent of units.
constexpr float SceneScale = 50.0f;
constexpr float RadianceScale = 1.0e3f;
constexpr uint32_t AccumulationFrames = 32u;
constexpr uint32_t StaleFrames = 64u;

template<class T>
noorrhi::Buffer<T> zeroed(noorrhi::Device& device, const std::size_t count)
{
    noorrhi::Buffer<T> buffer = device.buffer<T>(count);
    buffer.upload(std::span<const T>(std::vector<T>(count)));
    return buffer;
}
}

RadianceCache::RadianceCache(noorrhi::Device& device, noorrhi::RayTracingPipelineDesc stages)
    : device_(device)
    , resolvePipeline_(device.compute(loadShader(device, resolveSpv)))
    // SHaRC requires every buffer to start zeroed.
    , hashEntries_(zeroed<std::uint64_t>(device, Capacity))
    , accumulation_(zeroed<std::uint32_t>(device, std::size_t(Capacity) * RecordWords))
    , resolved_(zeroed<std::uint32_t>(device, std::size_t(Capacity) * RecordWords))
{
    setTraceStages(std::move(stages));
}

void RadianceCache::setTraceStages(noorrhi::RayTracingPipelineDesc stages)
{
    stages.raygen = loadShader(device_, updateSpv);
    updatePipeline_ = device_.ray_tracing(stages);
}

nr::graphics::RadianceCacheArgs RadianceCache::args(const FrameContext& frame) const
{
    nr::graphics::RadianceCacheArgs args{};
    if (mode_ == RadianceCacheMode::Off)
        return args;
    args.hashEntries = hashEntries_.ptr().address;
    args.accumulation = accumulation_.ptr().address;
    args.resolved = resolved_.ptr().address;
    args.sceneScale = SceneScale;
    args.radianceScale = RadianceScale;
    args.capacity = Capacity;
    args.accumulationFrames = AccumulationFrames;
    args.staleFrames = StaleFrames;
    // The budget scales with the view the voxels cover, not with the
    // resolution that view happens to be traced at.
    args.updateGridWidth = std::max(divideRoundingUp(frame.output.width, UpdateStride),
        MinUpdateGridWidth);
    args.updateGridHeight = std::max(divideRoundingUp(frame.output.height, UpdateStride),
        MinUpdateGridHeight);
    return args;
}

void RadianceCache::record(const nr::graphics::RealtimeArgs& args,
    const nr::graphics::RealtimeRoot root) const
{
    if (mode_ == RadianceCacheMode::Off)
        return;
    updatePipeline_.trace({args.radianceCache.updateGridWidth,
        args.radianceCache.updateGridHeight, 1}, root);
    device_.barrier(noorrhi::Stage::RayTracing, noorrhi::Stage::Compute);
    resolvePipeline_.launch({divideRoundingUp(Capacity, ResolveGroupSize), 1, 1}, root);
    device_.barrier(noorrhi::Stage::Compute, noorrhi::Stage::RayTracing);
}
