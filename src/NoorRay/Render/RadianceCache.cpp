#include "NoorRay/Render/RadianceCache.h"

#include <utility>

#include "NoorRay/Core/Log.h"

namespace NoorRay::Render {

namespace {

//! SHaRC's baseline: 2^22 entries of 40 bytes.
constexpr std::uint32_t capacity = 1u << 22;
//! The first bounce depth that may end in the cache, counting the primary vertex as 1: the vertex after the first indirect bounce, so two BSDF bounces blur its
//! voxels.
constexpr std::uint32_t endDepth = 3;
//! One update path per 5x5 pixels, about 4% of the pixels, as SHaRC proposes.
constexpr std::uint32_t updateStride = 5;
//! Voxels per metre at the first grid level; SHaRC's sample uses 50.
constexpr float sceneScale = 50.0f;
constexpr float radianceScale = 1000.0f;
constexpr std::uint32_t accumulationFrames = 32;

}

std::expected<void, Core::Error> RadianceCache::reserve(NoorRHI::Device& device, const std::uint32_t slots, const std::uint64_t layout) {
    if (hashes.getSize() != capacity) {
        if (auto opened = Core::check(hashes.open(device, capacity)); !opened)
            return opened;
        if (auto opened = Core::check(accumulation.open(device, capacity)); !opened)
            return opened;
        if (auto opened = Core::check(resolved.open(device, capacity)); !opened)
            return opened;
        clears = true;
        Core::Log::info("SHaRC radiance cache: {} entries, {} MB", capacity,
                        (hashes.getSize() * sizeof(NoorRHI::uint2) + (accumulation.getSize() + resolved.getSize()) * sizeof(NoorRHI::uint4)) >> 20);
    }
    if (std::exchange(lightLayout, layout) != layout)
        clears = true;
    if (states.getSize() != slots)
        return Core::check(states.open(device, slots));
    return {};
}

Shaders::RadianceCacheRecord RadianceCache::makeRecord(const NoorRHI::float3 camera) const {
    return {
        .hashes = hashes.getPtr(),
        .accumulation = accumulation.getPtr(),
        .resolved = resolved.getPtr(),
        .states = states.getPtr(),
        .camera = camera,
        .sceneScale = sceneScale,
        .previousCamera = previousCamera,
        .capacity = capacity,
        .depth = endDepth,
        .updateStride = updateStride,
        .radianceScale = radianceScale,
        .accumulationFrames = accumulationFrames,
        .frameIndex = frameIndex,
        .unused0 = 0,
        .unused1 = 0,
        .unused2 = 0,
    };
}

void RadianceCache::end(const NoorRHI::float3 camera) {
    previousCamera = camera;
    ++frameIndex;
    clears = false;
}

}
