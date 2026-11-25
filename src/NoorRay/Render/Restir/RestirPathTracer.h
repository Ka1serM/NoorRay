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
    //! The slots of the wavefront path pool; the replays of one resampling sweep use each slot at most once.
    std::uint32_t poolCapacity = 0;
    std::uint64_t lightLayout = 0;
    //! False resamples only the neighbors of this frame.
    bool temporal = true;
    //! The initial paths fork at volume scatters, which needs a fork record per pixel.
    bool forks = false;
    //! The shared SHaRC cache; null buffers when it is off.
    Shaders::RadianceCacheRecord cache{};
};

//! The buffers and parameters of ReSTIR PT through RTXDI. The wavefront path tracer traces its initial paths and its random replays in the path pool.
class RestirPathTracer {
public:
    //! Compiles the ReSTIR kernels in the background one after another, so that only one compilation holds memory; true once all are compiled.
    [[nodiscard]] static std::expected<bool, Core::Error> isReady(NoorRHI::Device& device, PipelineCache& pipelines, VariantKey key);
    //! Opens the buffers for the inputs and uploads the frame record.
    [[nodiscard]] std::expected<NoorRHI::Ptr<Shaders::RestirFrame>, Core::Error> begin(NoorRHI::Device& device, NoorRHI::CommandList& list,
                                                                                       const RestirInputs& inputs);
    //! The next frame reuses this one.
    void end();

    static std::uint32_t getPathDepth(std::uint32_t maxDepth);
    //! The most bounces of a replay: the hybrid shift replays a path only up to the vertex before its reconnection vertex.
    std::uint32_t getReplayDepth(std::uint32_t maxDepth) const;
    //! The forward shifts per pixel of a resampling mode, which is also the most inverse shifts.
    std::uint32_t getForwardShifts(std::uint32_t mode) const;

private:
    [[nodiscard]] std::expected<void, Core::Error> reserve(NoorRHI::Device& device, std::uint32_t width, std::uint32_t height, std::uint32_t poolCapacity);
    Shaders::RestirFrame makeFrame(const RestirInputs& inputs);

    std::optional<rtxdi::ReSTIRPTContext> context;
    NoorRHI::Buffer<RTXDI_PackedPTReservoir> reservoirs;
    NoorRHI::Buffer<NoorRHI::float2> neighborOffsets;
    NoorRHI::Buffer<Shaders::RestirSurfaceRecord> surfaces;
    NoorRHI::Buffer<Shaders::PackedClosure> closures;
    NoorRHI::Buffer<Shaders::RestirPathRecord> paths;
    NoorRHI::Buffer<Shaders::RestirReplayRecord> replays;
    NoorRHI::Buffer<std::uint32_t> replayStates;
    NoorRHI::Buffer<std::uint32_t> pending;
    NoorRHI::Buffer<Shaders::RestirForkRecord> forks;
    NoorRHI::Buffer<NoorRHI::Atomic<std::uint32_t>> counters;
    NoorRHI::Buffer<Shaders::RestirFrame> frameBuffer;
    std::uint32_t poolCapacity = 0;
    std::uint32_t pendingCapacity = 0;
    std::uint32_t frameIndex = 0;
    bool history = false;
    std::uint64_t lightLayout = 0;
};

}
