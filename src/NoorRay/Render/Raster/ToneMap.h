#pragma once

#include <cstdint>
#include <expected>

#include "NoorRHI/Commands/CommandList.h"
#include "NoorRHI/Pipelines/Pipeline.h"
#include "NoorRHI/Resources/Buffer.h"
#include "NoorRHI/Resources/Image.h"
#include "NoorRay/Core/Error.h"

namespace NoorRay::Render {

enum class ToneMapping : std::uint32_t { Linear, Neutral, Agx };

struct DisplaySettings {
    ToneMapping toneMapping = ToneMapping::Agx;
    float exposure = 0.0f;
    bool operator==(const DisplaySettings&) const = default;
};

//! Maps the linear film, with the Gaussian splats over it when given, to the sRGB display image; exposure is in stops.
class ToneMap {
public:
    static constexpr NoorRHI::ImageFormat displayFormat = NoorRHI::ImageFormat::Rgba8Unorm;

    [[nodiscard]] std::expected<void, Core::Error> record(NoorRHI::Device& device, NoorRHI::CommandList& list, const NoorRHI::Buffer<NoorRHI::float4>& film,
                                                          const NoorRHI::Image& display, const DisplaySettings& settings,
                                                          const NoorRHI::Image* splats = nullptr);

private:
    NoorRHI::GraphicsPipeline pipeline;
};

}
