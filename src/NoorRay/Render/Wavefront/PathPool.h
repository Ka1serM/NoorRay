#pragma once

#include <cstdint>
#include <expected>

#include "NoorRHI/Resources/Buffer.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Shaders/Wavefront/WavefrontRecords.h"

namespace NoorRay::Render {

//! The SoA state of every path slot. The capacity does not depend on the resolution.
class PathPool {
public:
    [[nodiscard]] std::expected<void, Core::Error> open(NoorRHI::Device& device, std::uint32_t capacity);

    const Shaders::PathPoolRecord& getRecord() const { return record; }
    std::uint32_t getCapacity() const { return record.capacity; }

private:
    NoorRHI::Buffer<NoorRHI::float4> origin;
    NoorRHI::Buffer<NoorRHI::float4> direction;
    NoorRHI::Buffer<NoorRHI::float4> throughput;
    NoorRHI::Buffer<NoorRHI::float4> radiance;
    NoorRHI::Buffer<NoorRHI::uint4> state;
    NoorRHI::Buffer<NoorRHI::uint4> hit;
    NoorRHI::Buffer<std::uint32_t> hitInstance;
    NoorRHI::Buffer<NoorRHI::float4> shadowOrigin;
    NoorRHI::Buffer<NoorRHI::float4> shadowDirection;
    NoorRHI::Buffer<NoorRHI::float4> shadowRadiance;
    NoorRHI::Buffer<std::uint32_t> pathBin;
    NoorRHI::Buffer<Shaders::PackedClosure> closures;
    NoorRHI::Buffer<NoorRHI::uint2> lightPath;
    NoorRHI::Buffer<float> minRayPdf;
    NoorRHI::Buffer<std::uint32_t> primaryDirection;
    NoorRHI::Buffer<NoorRHI::uint4> receiver;
    NoorRHI::Buffer<std::uint32_t> shadowEntry;
    Shaders::PathPoolRecord record{};
};

}
