#pragma once

#include <cstdint>
#include <expected>

#include "NoorRHI/Device/Device.h"
#include "NoorRHI/Resources/Buffer.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Shaders/Wavefront/WavefrontRecords.h"

namespace NoorRay::Render {

//! The SHaRC world-space radiance cache that the wavefront path tracer and ReSTIR PT share; each path pool slot has an update state.
class RadianceCache {
public:
    //! Opens the buffers once and the states for the slots; a new light layout clears the cache on the next frame.
    [[nodiscard]] std::expected<void, Core::Error> reserve(NoorRHI::Device& device, std::uint32_t slots, std::uint64_t lightLayout);
    //! Whether this frame must clear the cache before its paths use it: the buffers are new, or the lights changed.
    bool clearsCache() const { return clears; }
    Shaders::RadianceCacheRecord makeRecord(NoorRHI::float3 camera) const;
    //! The next frame resolves against the camera of this one.
    void end(NoorRHI::float3 camera);

private:
    NoorRHI::Buffer<NoorRHI::uint2> hashes;
    NoorRHI::Buffer<NoorRHI::uint4> accumulation;
    NoorRHI::Buffer<NoorRHI::uint4> resolved;
    NoorRHI::Buffer<Shaders::RadianceCacheState> states;
    NoorRHI::float3 previousCamera;
    std::uint64_t lightLayout = 0;
    std::uint32_t frameIndex = 0;
    bool clears = false;
};

}
