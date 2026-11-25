#pragma once

#include <cstdint>
#include <expected>

#include "NoorRHI/Commands/CommandList.h"
#include "NoorRHI/Resources/Buffer.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Render/PipelineCache.h"
#include "NoorRay/Render/VariantKey.h"
#include "NoorRay/Shaders/Restir/RestirRecords.h"

namespace NoorRay::Render {

struct RestirDirectInputs {
    NoorRHI::Ptr<Shaders::WavefrontFrame> wavefront;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint64_t lightLayout = 0;
    //! False resamples only the neighbors of this frame.
    bool temporal = true;
};

//! ReSTIR DI through RTXDI: the wavefront paths hand the light sample of their primary vertex to it, and it resamples and shades that light after they end.
class RestirDirectLighting {
public:
    //! Compiles the kernels of ReSTIR DI in the background; true once all are compiled.
    [[nodiscard]] static std::expected<bool, Core::Error> isReady(NoorRHI::Device& device, PipelineCache& pipelines, VariantKey key);
    //! Opens the buffers for the inputs and uploads the frame record.
    [[nodiscard]] std::expected<NoorRHI::Ptr<Shaders::RestirFrame>, Core::Error> begin(NoorRHI::Device& device, NoorRHI::CommandList& list,
                                                                                       const RestirDirectInputs& inputs);
    //! The G-buffer of this frame that Shade writes; valid after begin.
    Shaders::DirectLightingRecord getRecord() const;
    //! The next frame reuses this one.
    void end();
    //! A frame without ReSTIR DI breaks the history, so the next one starts over.
    void resetHistory() { history = false; }

private:
    [[nodiscard]] std::expected<void, Core::Error> reserve(NoorRHI::Device& device, std::uint32_t width, std::uint32_t height);
    Shaders::RestirFrame makeFrame(const RestirDirectInputs& inputs) const;

    NoorRHI::Buffer<RTXDI_PackedDIReservoir> reservoirs;
    NoorRHI::Buffer<NoorRHI::float2> neighborOffsets;
    NoorRHI::Buffer<Shaders::RestirSurfaceRecord> surfaces;
    NoorRHI::Buffer<Shaders::PackedClosure> closures;
    NoorRHI::Buffer<Shaders::DirectVertex> vertices;
    NoorRHI::Buffer<Shaders::RestirFrame> frameBuffer;
    RTXDI_ReservoirBufferParameters reservoirBuffer{};
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t frameIndex = 0;
    bool history = false;
    std::uint64_t lightLayout = 0;
};

}
