#pragma once

#include <cstdint>
#include <expected>

#include "NoorRHI/Resources/Buffer.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Shaders/Wavefront/WavefrontRecords.h"

namespace NoorRay::Render {

//! The film of one resolution: the per-frame sums, the accumulation, the displayed image and the primary AOVs.
class Targets {
public:
    [[nodiscard]] std::expected<void, Core::Error> open(NoorRHI::Device& device, std::uint32_t width, std::uint32_t height);

    //! accumulate is false on the frame that restarts the accumulation.
    Shaders::FilmRecord getRecord(bool accumulate) const;
    std::uint32_t getWidth() const { return width; }
    std::uint32_t getHeight() const { return height; }
    std::uint32_t getPixelCount() const { return width * height; }
    const NoorRHI::Buffer<NoorRHI::float4>& getOutput() const { return output; }
    const NoorRHI::Buffer<NoorRHI::float4>& getAlbedo() const { return albedo; }
    const NoorRHI::Buffer<NoorRHI::float4>& getNormalDepth() const { return normalDepth; }
    const NoorRHI::Buffer<std::uint32_t>& getObjectIds() const { return objectIds; }

private:
    NoorRHI::Buffer<NoorRHI::Atomic<float>> frame;
    NoorRHI::Buffer<NoorRHI::float4> accumulated;
    NoorRHI::Buffer<NoorRHI::float4> output;
    NoorRHI::Buffer<NoorRHI::float4> albedo;
    NoorRHI::Buffer<NoorRHI::float4> normalDepth;
    NoorRHI::Buffer<std::uint32_t> objectIds;
    NoorRHI::Buffer<NoorRHI::uint4> pixels;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

}
