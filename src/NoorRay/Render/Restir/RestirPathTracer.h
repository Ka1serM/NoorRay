#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <optional>

#include "NoorRHI/Commands/CommandList.h"
#include "NoorRHI/Resources/Buffer.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Render/PipelineCache.h"
#include "NoorRay/Render/VariantKey.h"
#include "NoorRay/Shaders/Restir/RestirRecords.h"
#include "Rtxdi/PT/ReSTIRPT.h"

namespace NoorRay::Render {

struct RestirInputs {
    NoorRHI::Ptr<Shaders::WavefrontFrame> wavefront;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t maxDepth = 1;
    std::uint32_t seed = 0;
    std::uint32_t gridThreads = 0;
};

//! ReSTIR PT through RTXDI: one traced primary surface per pixel, then initial, temporal and spatial path resampling and the final shading into the film.
class RestirPathTracer {
public:
    //! Compiles the ReSTIR kernels in the background one after another, so that only one compilation holds memory; true once all are compiled.
    [[nodiscard]] static std::expected<bool, Core::Error> isReady(NoorRHI::Device& device, PipelineCache& pipelines, VariantKey key);
    [[nodiscard]] std::expected<void, Core::Error> record(NoorRHI::Device& device, NoorRHI::CommandList& list, PipelineCache& pipelines, VariantKey key,
                                                          const RestirInputs& inputs);

private:
    [[nodiscard]] std::expected<void, Core::Error> reserve(NoorRHI::Device& device, std::uint32_t width, std::uint32_t height);
    Shaders::RestirFrame makeFrame(const RestirInputs& inputs);

    std::optional<rtxdi::ReSTIRPTContext> context;
    NoorRHI::Buffer<RTXDI_PackedPTReservoir> reservoirs;
    NoorRHI::Buffer<NoorRHI::float2> neighborOffsets;
    std::array<NoorRHI::Buffer<Shaders::RestirSurfaceRecord>, 2> surfaces;
    NoorRHI::Buffer<Shaders::RestirFrame> frameBuffer;
    std::uint32_t frameIndex = 0;
    bool history = false;
};

}
