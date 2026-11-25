#include "NoorRay/Render/Restir/RestirPathTracer.h"

#include <algorithm>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

#include "NoorRay/Core/Log.h"
#include "NoorRay/Gpu/BufferUpload.h"
#include "NoorRay/Render/Restir/NeighborOffsets.h"
#include "NoorRayShadersVariants.h"

namespace NoorRay::Render {

namespace {

//! RTXDI keeps a third reservoir slot only for its decorrelation pass, which NoorRay does not run.
constexpr std::uint32_t reservoirSlots = 2;
//! The replay flags keep the bounce count in 4 bits.
constexpr std::uint32_t maxPathDepth = 15;
constexpr std::uint32_t disocclusionBoostSamples = 2;
//! Without temporal reuse each pixel takes two neighbors; more cost much time and remove little noise.
constexpr std::uint32_t spatialOnlySamples = 2;

const std::array<const KernelVariants*, 7> restirKernels{&Shaders::restirExtend, &Shaders::restirBounce, &Shaders::restirResample,      &Shaders::restirReplay,
                                                         &Shaders::restirShade,  &Shaders::restirFork,   &Shaders::radianceCacheResolve};

void clearReconnectionModePadding(RTXDI_PTReconnectionParameters& reconnection) {
    std::memset(reinterpret_cast<std::byte*>(&reconnection.reconnectionMode) + sizeof(reconnection.reconnectionMode), 0,
                offsetof(RTXDI_PTReconnectionParameters, pad1) - offsetof(RTXDI_PTReconnectionParameters, reconnectionMode)
                    - sizeof(reconnection.reconnectionMode));
}

template <typename T>
std::size_t byteSize(const NoorRHI::Buffer<T>& buffer) {
    return buffer.getSize() * sizeof(T);
}

}

std::expected<void, Core::Error> RestirPathTracer::reserve(NoorRHI::Device& device, const std::uint32_t width, const std::uint32_t height,
                                                           const std::uint32_t capacity) {
    const bool sized = context && context->GetStaticParams().RenderWidth == width && context->GetStaticParams().RenderHeight == height;
    if (sized && capacity == poolCapacity)
        return {};
    if (!frameBuffer) {
        if (auto opened = Core::check(frameBuffer.open(device, 1)); !opened)
            return opened;
        if (auto opened = Core::check(counters.open(device, Shaders::restirStages)); !opened)
            return opened;
        const std::vector<NoorRHI::float2> offsets = makeNeighborOffsets();
        if (auto opened = Gpu::uploadValues(device, neighborOffsets, std::span<const NoorRHI::float2>(offsets), false); !opened)
            return opened;
    }
    const std::size_t pixels = std::size_t{width} * height;
    if (!sized) {
        context.emplace(rtxdi::ReSTIRPTStaticParameters{.RenderWidth = width, .RenderHeight = height});
        RTXDI_BoilingFilterParameters boiling = context->GetBoilingFilterParameters();
        boiling.enableBoilingFilter = 0;
        context->SetBoilingFilterParameters(boiling);
        RTXDI_PTReconnectionParameters reconnection = context->GetReconnectionParameters();
        //! RTXDI's footprint mode evaluates the BSDF pdf at each reconnection; the fixed threshold reads only the roughness.
        reconnection.reconnectionMode = RTXDI_PTReconnectionMode::FixedThreshold;
        //! Reconnecting at roughness 0.2 doubles the error of glossy surfaces; 0.3 also fixes 0.25 but costs 1.1 ms per frame, 0.2 costs 0.2 ms.
        reconnection.roughnessThreshold = 0.2f;
        context->SetReconnectionParameters(reconnection);
        RTXDI_PTTemporalResamplingParameters temporal = context->GetTemporalResamplingParameters();
        //! Rejecting old samples drops the bright ones that live longest and biases the image; the history cap already limits the life of a sample.
        temporal.enableAgeBasedRejection = 0;
        context->SetTemporalResamplingParameters(temporal);
        history = false;
        const std::size_t reservoirCount = std::size_t{reservoirSlots} * context->GetReservoirBufferParameters().reservoirArrayPitch;
        if (auto opened = Core::check(reservoirs.open(device, reservoirCount)); !opened)
            return opened;
        if (auto opened = Core::check(surfaces.open(device, 2 * pixels)); !opened)
            return opened;
        if (auto opened = Core::check(closures.open(device, 2 * pixels)); !opened)
            return opened;
    }
    poolCapacity = capacity;
    pendingCapacity = static_cast<std::uint32_t>(std::min<std::size_t>(capacity, pixels));
    if (auto opened = Core::check(paths.open(device, capacity)); !opened)
        return opened;
    if (auto opened = Core::check(replays.open(device, 2 * std::size_t{capacity})); !opened)
        return opened;
    if (auto opened = Core::check(replayStates.open(device, 2 * std::size_t{capacity})); !opened)
        return opened;
    if (auto opened = Core::check(pending.open(device, std::size_t{Shaders::restirStages} * pendingCapacity)); !opened)
        return opened;
    const std::size_t bytes =
        byteSize(reservoirs) + byteSize(surfaces) + byteSize(closures) + byteSize(paths) + byteSize(replays) + byteSize(replayStates) + byteSize(pending);
    Core::Log::info("ReSTIR PT buffers for {}x{} with {} path slots: {} MB", width, height, capacity, bytes >> 20);
    return {};
}

std::uint32_t RestirPathTracer::getPathDepth(const std::uint32_t maxDepth) {
    return std::clamp(maxDepth, 1u, maxPathDepth);
}

std::uint32_t RestirPathTracer::getReplayDepth(const std::uint32_t maxDepth) const {
    return std::min(getPathDepth(maxDepth), std::max(context->GetInitialSamplingParameters().maxRcVertexLength, 2u) - 1);
}

std::uint32_t RestirPathTracer::getForwardShifts(const std::uint32_t mode) const {
    if (mode == Shaders::restirModeTemporal)
        return 1;
    const RTXDI_PTSpatialResamplingParameters spatial = context->GetSpatialResamplingParameters();
    return std::max({spatial.numSpatialSamples, spatial.numDisocclusionBoostSamples, 1u});
}

Shaders::RestirFrame RestirPathTracer::makeFrame(const RestirInputs& inputs) {
    const std::uint32_t depth = getPathDepth(inputs.maxDepth);
    RTXDI_PTInitialSamplingParameters initial = context->GetInitialSamplingParameters();
    initial.maxBounceDepth = depth;
    context->SetInitialSamplingParameters(initial);
    RTXDI_PTHybridShiftPerFrameParameters hybrid = context->GetHybridShiftParameters();
    hybrid.maxBounceDepth = depth;
    context->SetHybridShiftParameters(hybrid);
    context->SetResamplingMode(inputs.temporal ? rtxdi::ReSTIRPT_ResamplingMode::TemporalAndSpatial : rtxdi::ReSTIRPT_ResamplingMode::Spatial);
    RTXDI_PTSpatialResamplingParameters spatial = context->GetSpatialResamplingParameters();
    spatial.numSpatialSamples = inputs.temporal ? 1u : spatialOnlySamples;
    spatial.numDisocclusionBoostSamples = inputs.temporal ? disocclusionBoostSamples : spatialOnlySamples;
    context->SetSpatialResamplingParameters(spatial);
    context->SetFrameIndex(frameIndex);
    Shaders::RestirFrame frame{
        .wavefront = inputs.wavefront,
        .reservoirs = reservoirs.getPtr(),
        .neighborOffsets = neighborOffsets.getPtr(),
        .surfaces = surfaces.getPtr(),
        .closures = closures.getPtr(),
        .paths = paths.getPtr(),
        .replays = replays.getPtr(),
        .replayStates = replayStates.getPtr(),
        .pending = pending.getPtr(),
        .counters = counters.getPtr(),
        .pendingCapacity = pendingCapacity,
        .width = inputs.width,
        .height = inputs.height,
        .current = frameIndex & 1u,
        .history = history ? 1u : 0u,
        .unused = 0,
        .forks = inputs.forks ? forks.getPtr() : NoorRHI::Ptr<Shaders::RestirForkRecord>{},
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
    frame.cache = inputs.cache;
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

std::expected<NoorRHI::Ptr<Shaders::RestirFrame>, Core::Error> RestirPathTracer::begin(NoorRHI::Device& device, NoorRHI::CommandList& list,
                                                                                       const RestirInputs& inputs) {
    if (inputs.poolCapacity == 0)
        return Core::fail(Core::ErrorCode::InvalidArgument, "ReSTIR PT needs path pool slots");
    if (auto reserved = reserve(device, inputs.width, inputs.height, inputs.poolCapacity); !reserved)
        return std::unexpected(std::move(reserved.error()));
    //! The reservoirs of the previous frame name lights by index; after the light layout changes they would name other lights or none.
    if (std::exchange(lightLayout, inputs.lightLayout) != inputs.lightLayout)
        history = false;
    const std::size_t pixels = std::size_t{inputs.width} * inputs.height;
    if (inputs.forks && forks.getSize() != pixels)
        if (auto opened = Core::check(forks.open(device, pixels)); !opened)
            return std::unexpected(std::move(opened.error()));
    const Shaders::RestirFrame frame = makeFrame(inputs);
    list.upload(frameBuffer, std::span<const Shaders::RestirFrame>(&frame, 1));
    return frameBuffer.getPtr();
}

void RestirPathTracer::end() {
    ++frameIndex;
    history = true;
}
}
