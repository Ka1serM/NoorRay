#include "NoorRay/Render/Restir/RestirPathTracer.h"

#include <algorithm>
#include <cstring>
#include <span>
#include <vector>

#include "NoorRay/Core/Log.h"
#include "NoorRay/Gpu/BufferUpload.h"
#include "NoorRayShadersVariants.h"
#include "Rtxdi/RtxdiUtils.h"

namespace NoorRay::Render {

namespace {

constexpr std::uint32_t neighborOffsetCount = 8192;
//! RTXDI keeps a third reservoir slot only for its decorrelation pass, which NoorRay does not run.
constexpr std::uint32_t reservoirSlots = 2;
constexpr std::uint32_t maxPathDepth = 3;
constexpr std::uint32_t disocclusionBoostSamples = 2;

const std::array<const KernelVariants*, 5> restirKernels{&Shaders::restirPrimary, &Shaders::restirInitial, &Shaders::restirTemporal, &Shaders::restirSpatial,
                                                         &Shaders::restirShade};

std::vector<NoorRHI::float2> makeNeighborOffsets() {
    std::vector<std::uint8_t> packed(neighborOffsetCount * 2);
    rtxdi::FillNeighborOffsetBuffer(packed.data(), neighborOffsetCount);
    std::vector<NoorRHI::float2> offsets(neighborOffsetCount);
    //! RTXDI fills the offsets for an R8G8_SNORM buffer.
    for (std::uint32_t index = 0; index < neighborOffsetCount; ++index)
        offsets[index] = {static_cast<float>(static_cast<std::int8_t>(packed[index * 2])) / 127.0f,
                          static_cast<float>(static_cast<std::int8_t>(packed[index * 2 + 1])) / 127.0f};
    return offsets;
}

void clearReconnectionModePadding(RTXDI_PTReconnectionParameters& reconnection) {
    std::memset(reinterpret_cast<std::byte*>(&reconnection.reconnectionMode) + sizeof(reconnection.reconnectionMode), 0,
                offsetof(RTXDI_PTReconnectionParameters, pad1) - offsetof(RTXDI_PTReconnectionParameters, reconnectionMode)
                    - sizeof(reconnection.reconnectionMode));
}

}

std::expected<void, Core::Error> RestirPathTracer::reserve(NoorRHI::Device& device, const std::uint32_t width, const std::uint32_t height) {
    if (context && context->GetStaticParams().RenderWidth == width && context->GetStaticParams().RenderHeight == height)
        return {};
    if (!frameBuffer) {
        if (auto opened = Core::check(frameBuffer.open(device, 1)); !opened)
            return opened;
        const std::vector<NoorRHI::float2> offsets = makeNeighborOffsets();
        if (auto opened = Gpu::uploadValues(device, neighborOffsets, std::span<const NoorRHI::float2>(offsets), false); !opened)
            return opened;
    }
    context.emplace(rtxdi::ReSTIRPTStaticParameters{.RenderWidth = width, .RenderHeight = height});
    context->SetResamplingMode(rtxdi::ReSTIRPT_ResamplingMode::TemporalAndSpatial);
    RTXDI_BoilingFilterParameters boiling = context->GetBoilingFilterParameters();
    boiling.enableBoilingFilter = 0;
    context->SetBoilingFilterParameters(boiling);
    RTXDI_PTSpatialResamplingParameters spatial = context->GetSpatialResamplingParameters();
    spatial.numDisocclusionBoostSamples = disocclusionBoostSamples;
    context->SetSpatialResamplingParameters(spatial);
    RTXDI_PTReconnectionParameters reconnection = context->GetReconnectionParameters();
    //! RTXDI's footprint mode evaluates the BSDF pdf at each reconnection; the fixed threshold reads only the roughness.
    reconnection.reconnectionMode = RTXDI_PTReconnectionMode::FixedThreshold;
    context->SetReconnectionParameters(reconnection);
    history = false;
    const std::size_t reservoirCount = std::size_t{reservoirSlots} * context->GetReservoirBufferParameters().reservoirArrayPitch;
    if (auto opened = Core::check(reservoirs.open(device, reservoirCount)); !opened)
        return opened;
    for (NoorRHI::Buffer<Shaders::RestirSurfaceRecord>& buffer : surfaces)
        if (auto opened = Core::check(buffer.open(device, std::size_t{width} * height)); !opened)
            return opened;
    Core::Log::info("ReSTIR PT buffers for {}x{}: {} MB", width, height,
                    (reservoirs.getSize() * sizeof(RTXDI_PackedPTReservoir) + 2 * surfaces[0].getSize() * sizeof(Shaders::RestirSurfaceRecord)) >> 20);
    return {};
}

Shaders::RestirFrame RestirPathTracer::makeFrame(const RestirInputs& inputs) {
    const std::uint32_t depth = std::min(inputs.maxDepth, maxPathDepth);
    RTXDI_PTInitialSamplingParameters initial = context->GetInitialSamplingParameters();
    initial.maxBounceDepth = depth;
    context->SetInitialSamplingParameters(initial);
    RTXDI_PTHybridShiftPerFrameParameters hybrid = context->GetHybridShiftParameters();
    hybrid.maxBounceDepth = depth;
    context->SetHybridShiftParameters(hybrid);
    context->SetFrameIndex(frameIndex);
    const std::uint32_t parity = frameIndex & 1u;
    Shaders::RestirFrame frame{
        .wavefront = inputs.wavefront,
        .reservoirs = reservoirs.getPtr(),
        .neighborOffsets = neighborOffsets.getPtr(),
        .surfaces = surfaces[parity].getPtr(),
        .previousSurfaces = history ? surfaces[parity ^ 1u].getPtr() : NoorRHI::Ptr<Shaders::RestirSurfaceRecord>{},
        .width = inputs.width,
        .height = inputs.height,
        .seed = inputs.seed,
        .gridThreads = inputs.gridThreads,
        .runtime = {.neighborOffsetMask = neighborOffsetCount - 1, .activeCheckerboardField = 0, .frameIndex = frameIndex, .pad2 = 0},
        .pt =
            {
                .reservoirBuffer = context->GetReservoirBufferParameters(),
                .bufferIndices = context->GetBufferIndices(),
                .initialSampling = context->GetInitialSamplingParameters(),
                .decorrelation = context->GetDecorrelationParameters(),
                .reconnection = context->GetReconnectionParameters(),
                .temporalResampling = context->GetTemporalResamplingParameters(),
                .hybridShift = context->GetHybridShiftParameters(),
                .boilingFilter = context->GetBoilingFilterParameters(),
                .spatialResampling = context->GetSpatialResamplingParameters(),
            },
    };
    clearReconnectionModePadding(frame.pt.reconnection);
    return frame;
}

std::expected<bool, Core::Error> RestirPathTracer::isReady(NoorRHI::Device& device, PipelineCache& pipelines, const VariantKey key) {
    for (const KernelVariants* variants : restirKernels) {
        std::expected<bool, Core::Error> compiled = pipelines.isReady(device, key.select(*variants));
        if (!compiled || !*compiled)
            return compiled;
    }
    return true;
}

std::expected<void, Core::Error> RestirPathTracer::record(NoorRHI::Device& device, NoorRHI::CommandList& list, PipelineCache& pipelines, const VariantKey key,
                                                          const RestirInputs& inputs) {
    if (auto reserved = reserve(device, inputs.width, inputs.height); !reserved)
        return reserved;
    const Shaders::RestirFrame frame = makeFrame(inputs);
    list.upload(frameBuffer, std::span<const Shaders::RestirFrame>(&frame, 1));
    const Shaders::RestirArguments arguments{frameBuffer.getPtr()};
    for (const KernelVariants* variants : restirKernels) {
        std::expected<const NoorRHI::ComputePipeline*, Core::Error> pipeline = pipelines.get(device, key.select(*variants));
        if (!pipeline)
            return std::unexpected(std::move(pipeline.error()));
        list.dispatch(**pipeline, NoorRHI::DispatchSize{inputs.gridThreads, 1, 1}, arguments);
    }
    ++frameIndex;
    history = true;
    return {};
}

}
