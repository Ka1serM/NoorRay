#include "NoorRay/Render/Raster/ToneMap.h"

#include "NoorRay/Shaders/Display/DisplayRecords.h"
#include "NoorRayShadersVariants.h"

namespace NoorRay::Render {

static_assert(static_cast<std::uint32_t>(ToneMapping::Linear) == Shaders::toneMapLinear);
static_assert(static_cast<std::uint32_t>(ToneMapping::Neutral) == Shaders::toneMapNeutral);
static_assert(static_cast<std::uint32_t>(ToneMapping::Agx) == Shaders::toneMapAgx);

std::expected<void, Core::Error> ToneMap::record(NoorRHI::Device& device, NoorRHI::CommandList& list, const NoorRHI::Buffer<NoorRHI::float4>& film,
                                                 const NoorRHI::Image& display, const DisplaySettings& settings, const NoorRHI::Image* splats) {
    if (film.getSize() < std::size_t{display.getWidth()} * display.getHeight())
        return Core::fail(Core::ErrorCode::InvalidArgument, "the film is smaller than the display image");
    if (!pipeline) {
        const NoorRHI::GraphicsPipelineDesc desc{
            .vertex = *Shaders::toneMapVertex[0],
            .fragment = *Shaders::toneMapFragment[0],
            .state = {.cull = NoorRHI::CullMode::None},
            .colorFormat = displayFormat,
        };
        if (auto opened = Core::check(pipeline.open(device, desc)); !opened)
            return opened;
    }
    const Shaders::DisplayArguments arguments{
        .source = film.getPtr(),
        .width = display.getWidth(),
        .height = display.getHeight(),
        .exposure = settings.exposure,
        .toneMap = static_cast<std::uint32_t>(settings.toneMapping),
        .splats = splats ? splats->getSampledHandle() : NoorRHI::TextureHandle{},
        .splatted = splats ? 1u : 0u,
    };
    list.pushLabel("ToneMap");
    list.beginRender({.color = &display, .clear = false});
    list.draw(pipeline, 3, 1, arguments);
    list.endRender();
    list.popLabel();
    return {};
}

}
