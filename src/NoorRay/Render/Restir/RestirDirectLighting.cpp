#include "NoorRay/Render/Restir/RestirDirectLighting.h"

#include <array>
#include <span>
#include <utility>
#include <vector>

#include "NoorRay/Core/Log.h"
#include "NoorRay/Gpu/BufferUpload.h"
#include "NoorRay/Render/Restir/NeighborOffsets.h"
#include "NoorRayShadersVariants.h"
#include "Rtxdi/DI/ReSTIRDI.h"
#include "Rtxdi/RtxdiUtils.h"

namespace NoorRay::Render {

namespace {

//! The temporal pass writes slot 1 from the slot 0 of the last frame; the spatial pass reads slot 1 and writes slot 0 for the next frame.
constexpr std::uint32_t reservoirSlots = 2;
constexpr std::uint32_t finalSlot = 0;
constexpr std::uint32_t temporalSlot = 1;
//! The light tree already picks good lights: each further candidate costs a tree walk and barely lowers the noise.
constexpr std::uint32_t initialCandidates = 1;
constexpr std::uint32_t disocclusionBoostSamples = 4;
//! Without temporal reuse each pixel takes more neighbors, as its own reservoir saw only the initial candidates.
constexpr std::uint32_t spatialOnlySamples = 4;

const std::array<const KernelVariants*, 2> directKernels{&Shaders::restirDirect, &Shaders::shadeSurfaceDirect};

}

std::expected<bool, Core::Error> RestirDirectLighting::isReady(NoorRHI::Device& device, PipelineCache& pipelines, const VariantKey key) {
    for (const KernelVariants* variants : directKernels) {
        std::expected<bool, Core::Error> compiled = pipelines.isReady(device, key.select(*variants));
        if (!compiled || !*compiled)
            return compiled;
    }
    return true;
}

std::expected<void, Core::Error> RestirDirectLighting::reserve(NoorRHI::Device& device, const std::uint32_t newWidth, const std::uint32_t newHeight) {
    if (!frameBuffer) {
        if (auto opened = Core::check(frameBuffer.open(device, 1)); !opened)
            return opened;
        const std::vector<NoorRHI::float2> offsets = makeNeighborOffsets();
        if (auto opened = Gpu::uploadValues(device, neighborOffsets, std::span<const NoorRHI::float2>(offsets), false); !opened)
            return opened;
    }
    if (newWidth == width && newHeight == height)
        return {};
    const std::size_t pixels = std::size_t{newWidth} * newHeight;
    reservoirBuffer = rtxdi::CalculateReservoirBufferParameters(newWidth, newHeight, rtxdi::CheckerboardMode::Off);
    if (auto opened = Core::check(reservoirs.open(device, std::size_t{reservoirSlots} * reservoirBuffer.reservoirArrayPitch)); !opened)
        return opened;
    if (auto opened = Core::check(surfaces.open(device, 2 * pixels)); !opened)
        return opened;
    if (auto opened = Core::check(closures.open(device, 2 * pixels)); !opened)
        return opened;
    //! A zero stamp marks a vertex that no path wrote, which the first frame must not read.
    if (auto opened = Gpu::openZeroed(device, vertices, pixels); !opened)
        return opened;
    width = newWidth;
    height = newHeight;
    history = false;
    Core::Log::info("ReSTIR DI buffers for {}x{}: {} MB", width, height,
                    (reservoirs.getSize() * sizeof(RTXDI_PackedDIReservoir) + surfaces.getSize() * sizeof(Shaders::RestirSurfaceRecord)
                     + closures.getSize() * sizeof(Shaders::PackedClosure) + vertices.getSize() * sizeof(Shaders::DirectVertex))
                        >> 20);
    return {};
}

Shaders::RestirFrame RestirDirectLighting::makeFrame(const RestirDirectInputs& inputs) const {
    const bool temporal = inputs.temporal && history;
    RTXDI_DITemporalResamplingParameters temporalResampling = rtxdi::GetDefaultReSTIRDITemporalResamplingParams();
    temporalResampling.biasCorrectionMode = ReSTIRDI_TemporalBiasCorrectionMode::Raytraced;
    //! Permutation sampling trades bias for less correlation, and the denoiser handles the correlation.
    temporalResampling.enablePermutationSampling = 0;
    RTXDI_DISpatialResamplingParameters spatialResampling = rtxdi::GetDefaultReSTIRDISpatialResamplingParams();
    spatialResampling.biasCorrectionMode = ReSTIRDI_SpatialBiasCorrectionMode::Raytraced;
    spatialResampling.numSamples = inputs.temporal ? 1u : spatialOnlySamples;
    spatialResampling.numDisocclusionBoostSamples = inputs.temporal ? disocclusionBoostSamples : spatialOnlySamples;
    spatialResampling.targetHistoryLength = inputs.temporal ? temporalResampling.maxHistoryLength : 0u;
    //! The naive sample discount skips neighbors of small M in the selection but still counts them in the ray traced normalization, which darkens young
    //! history.
    spatialResampling.discountNaiveSamples = 0;
    RTXDI_DIBufferIndices bufferIndices{};
    bufferIndices.initialSamplingOutputBufferIndex = temporalSlot;
    bufferIndices.temporalResamplingInputBufferIndex = finalSlot;
    bufferIndices.temporalResamplingOutputBufferIndex = temporalSlot;
    bufferIndices.spatialResamplingInputBufferIndex = temporalSlot;
    bufferIndices.spatialResamplingOutputBufferIndex = finalSlot;
    bufferIndices.shadingInputBufferIndex = finalSlot;
    Shaders::RestirFrame frame{};
    frame.wavefront = inputs.wavefront;
    frame.neighborOffsets = neighborOffsets.getPtr();
    frame.surfaces = surfaces.getPtr();
    frame.closures = closures.getPtr();
    frame.width = inputs.width;
    frame.height = inputs.height;
    frame.current = frameIndex & 1u;
    frame.history = temporal ? 1u : 0u;
    frame.runtime = {.neighborOffsetMask = neighborOffsetCount - 1, .activeCheckerboardField = 0, .frameIndex = frameIndex, .pad2 = 0};
    frame.lightReservoirs = reservoirs.getPtr();
    frame.vertices = vertices.getPtr();
    frame.direct = {
        .reservoirBuffer = reservoirBuffer,
        .bufferIndices = bufferIndices,
        .temporalResampling = temporalResampling,
        .spatialResampling = spatialResampling,
        .temporal = temporal ? 1u : 0u,
        .candidates = initialCandidates,
        .unused0 = 0,
        .unused1 = 0,
    };
    return frame;
}

Shaders::DirectLightingRecord RestirDirectLighting::getRecord() const {
    return {
        .surfaces = surfaces.getPtr(),
        .closures = closures.getPtr(),
        .vertices = vertices.getPtr(),
        .base = (frameIndex & 1u) * width * height,
        .frame = frameIndex + 1,
    };
}

std::expected<NoorRHI::Ptr<Shaders::RestirFrame>, Core::Error> RestirDirectLighting::begin(NoorRHI::Device& device, NoorRHI::CommandList& list,
                                                                                           const RestirDirectInputs& inputs) {
    if (auto reserved = reserve(device, inputs.width, inputs.height); !reserved)
        return std::unexpected(std::move(reserved.error()));
    //! The reservoirs of the previous frame name lights by index; after the light layout changes they would name other lights or none.
    if (std::exchange(lightLayout, inputs.lightLayout) != inputs.lightLayout)
        history = false;
    const Shaders::RestirFrame frame = makeFrame(inputs);
    list.upload(frameBuffer, std::span<const Shaders::RestirFrame>(&frame, 1));
    return frameBuffer.getPtr();
}

void RestirDirectLighting::end() {
    ++frameIndex;
    history = true;
}

}
