#include "Realtime/Restir.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <Rtxdi/ImportanceSamplingContext.h>
#include <Rtxdi/LightSampling/RISBufferSegmentAllocator.h>
#include <Rtxdi/RtxdiUtils.h>

#include "Realtime/ShaderLoading.h"

namespace
{
alignas(uint32_t) constexpr unsigned char presampleLightsSpv[] = {
    #embed "RealtimeRaytracer/RtxdiPresampleLights.spv"
};
alignas(uint32_t) constexpr unsigned char presampleReGIRSpv[] = {
    #embed "RealtimeRaytracer/RtxdiPresampleReGIR.spv"
};
alignas(uint32_t) constexpr unsigned char diInitialSpv[] = {
    #embed "RealtimeRaytracer/RtxdiDIInitial.spv"
};
alignas(uint32_t) constexpr unsigned char diTemporalSpv[] = {
    #embed "RealtimeRaytracer/RtxdiDITemporal.spv"
};
alignas(uint32_t) constexpr unsigned char diBoilingSpv[] = {
    #embed "RealtimeRaytracer/RtxdiDIBoiling.spv"
};
alignas(uint32_t) constexpr unsigned char diSpatialSpv[] = {
    #embed "RealtimeRaytracer/RtxdiDISpatial.spv"
};
alignas(uint32_t) constexpr unsigned char diShadeSpv[] = {
    #embed "RealtimeRaytracer/RtxdiDIShade.spv"
};
alignas(uint32_t) constexpr unsigned char giTemporalSpv[] = {
    #embed "RealtimeRaytracer/RtxdiGITemporal.spv"
};
alignas(uint32_t) constexpr unsigned char giBoilingSpv[] = {
    #embed "RealtimeRaytracer/RtxdiGIBoiling.spv"
};
alignas(uint32_t) constexpr unsigned char giSpatialSpv[] = {
    #embed "RealtimeRaytracer/RtxdiGISpatial.spv"
};
alignas(uint32_t) constexpr unsigned char giShadeSpv[] = {
    #embed "RealtimeRaytracer/RtxdiGIShade.spv"
};

// Group sizes, matching RtxdiPasses.slang.
constexpr uint32_t PresampleGroupSize = 256u;
constexpr uint32_t BoilingGroupSize = 16u;
// Screen-space offsets spatial resampling draws its neighbours from.
constexpr uint32_t NeighborOffsetCount = 8192u;
// Candidate lights per vertex in world-space (ReGIR) sampling beyond the
// G-buffer; primary surfaces use RTXDI's default of eight too.
constexpr uint32_t SecondaryLocalLightSamples = 8u;
// ReGIR's grid spans the lit region in this many cells per axis.
constexpr float ReGIRCellsAcrossLights = 16.0f;
}

Restir::Restir(noorrhi::Device& device, noorrhi::RayTracingPipelineDesc stages)
    : device_(device)
    , presampleLightsPipeline_(device.compute(loadShader(device, presampleLightsSpv)))
    , presampleReGIRPipeline_(device.compute(loadShader(device, presampleReGIRSpv)))
    , diBoilingPipeline_(device.compute(loadShader(device, diBoilingSpv)))
    , giBoilingPipeline_(device.compute(loadShader(device, giBoilingSpv)))
{
    setTraceStages(std::move(stages));

    // RTXDI packs its offsets as RG8_SNORM texels; the shaders read floats.
    std::vector<std::uint8_t> packedOffsets(NeighborOffsetCount * 2u);
    rtxdi::FillNeighborOffsetBuffer(packedOffsets.data(), NeighborOffsetCount);
    std::vector<float> offsets(packedOffsets.size());
    for (std::size_t i = 0; i < offsets.size(); ++i)
        offsets[i] = std::max(static_cast<float>(static_cast<std::int8_t>(packedOffsets[i]))
            / 127.0f, -1.0f);
    neighborOffsets_ = device.buffer<float>(offsets.size());
    neighborOffsets_.upload(std::span<const float>(offsets));
    // An empty light table until the first light upload.
    lightAlias_ = device.buffer<std::uint32_t>(4u);
    lightAlias_.upload(std::span<const std::uint32_t>(std::vector<std::uint32_t>(4u)));
}

void Restir::setTraceStages(noorrhi::RayTracingPipelineDesc stages)
{
    const auto rayTracing = [&](const auto& raygen) {
        stages.raygen = loadShader(device_, raygen);
        return device_.ray_tracing(stages);
    };
    diInitialPipeline_ = rayTracing(diInitialSpv);
    diTemporalPipeline_ = rayTracing(diTemporalSpv);
    diSpatialPipeline_ = rayTracing(diSpatialSpv);
    diShadePipeline_ = rayTracing(diShadeSpv);
    giTemporalPipeline_ = rayTracing(giTemporalSpv);
    giSpatialPipeline_ = rayTracing(giSpatialSpv);
    giShadePipeline_ = rayTracing(giShadeSpv);
}

Restir::~Restir() = default;

void Restir::resize(const Extent render)
{
    rtxdi::ImportanceSamplingContext_StaticParameters parameters;
    parameters.renderWidth = render.width;
    parameters.renderHeight = render.height;
    parameters.NeighborOffsetCount = NeighborOffsetCount;
    parameters.regirStaticParams.mode = rtxdi::ReGIRMode::Grid;
    context_ = std::make_unique<rtxdi::ImportanceSamplingContext>(parameters);
    rtxdi::ReSTIRDIContext& di = context_->GetReSTIRDIContext();
    rtxdi::ReSTIRGIContext& gi = context_->GetReSTIRGIContext();
    di.SetResamplingMode(rtxdi::ReSTIRDI_ResamplingMode::TemporalAndSpatial);
    gi.SetResamplingMode(rtxdi::ReSTIRGI_ResamplingMode::TemporalAndSpatial);

    // Direct light at primary surfaces: eight ReGIR-driven candidates and one
    // directional light per pixel, the chosen one shadow-tested. The
    // environment is sampled outside RTXDI, and BRDF samples cannot find
    // analytic light shapes, so neither candidate kind is drawn.
    auto initial = di.GetInitialSamplingParameters();
    initial.localLightSamplingMode = ReSTIRDI_LocalLightSamplingMode::ReGIR_RIS;
    initial.numEnvironmentSamples = 0;
    initial.environmentMapImportanceSampling = 0;
    initial.numBrdfSamples = 0;
    di.SetInitialSamplingParameters(initial);
    // RTXDI's default Basic bias correction leaves visibility out of the
    // resampling MIS weights, which biases reuse across shadow boundaries;
    // the viewport accumulates these frames, so that bias never averages out.
    auto diTemporal = di.GetTemporalResamplingParameters();
    diTemporal.biasCorrectionMode = ReSTIRDI_TemporalBiasCorrectionMode::Raytraced;
    di.SetTemporalResamplingParameters(diTemporal);
    auto diSpatial = di.GetSpatialResamplingParameters();
    diSpatial.biasCorrectionMode = ReSTIRDI_SpatialBiasCorrectionMode::Raytraced;
    di.SetSpatialResamplingParameters(diSpatial);
    auto giTemporal = gi.GetTemporalResamplingParameters();
    giTemporal.biasCorrectionMode = RTXDI_GIBiasCorrectionMode::Raytraced;
    gi.SetTemporalResamplingParameters(giTemporal);
    auto giSpatial = gi.GetSpatialResamplingParameters();
    giSpatial.biasCorrectionMode = RTXDI_GIBiasCorrectionMode::Raytraced;
    gi.SetSpatialResamplingParameters(giSpatial);
    // ReSTIR GI's final pass skips its MIS against the initial sample: that
    // step exists for glossy lobes, and only the diffuse lobe is resampled.
    auto finalShading = gi.GetFinalShadingParameters();
    finalShading.enableFinalMIS = 0;
    gi.SetFinalShadingParameters(finalShading);

    constexpr std::size_t diReservoirWords = sizeof(RTXDI_PackedDIReservoir) / 4u;
    constexpr std::size_t giReservoirWords = sizeof(RTXDI_PackedGIReservoir) / 4u;
    constexpr std::size_t surfaceWords = sizeof(nr::graphics::RealtimeSurface) / 4u;
    diReservoirs_ = device_.buffer<std::uint32_t>(diReservoirWords
        * di.GetReservoirBufferParameters().reservoirArrayPitch
        * rtxdi::c_NumReSTIRDIReservoirBuffers);
    giReservoirs_ = device_.buffer<std::uint32_t>(giReservoirWords
        * gi.GetReservoirBufferParameters().reservoirArrayPitch
        * rtxdi::c_NumReSTIRGIReservoirBuffers);
    risBuffer_ = device_.buffer<std::uint32_t>(2u * std::max(
        context_->GetRISBufferSegmentAllocator().GetTotalSizeInElements(), 1u));
    for (auto& surfaces : surfaces_)
        surfaces = device_.buffer<std::uint32_t>(surfaceWords
            * std::max<std::size_t>(std::size_t(render.width) * render.height, 1u));
    frameIndex_ = 0;
    surfaceParity_ = 0;
    historyValid_ = false;
}

void Restir::uploadLights(const std::span<const nr::graphics::PointLight> points,
    const std::span<const nr::graphics::SpotLight> spots,
    const std::span<const nr::graphics::RectLight> rects,
    const std::span<const nr::graphics::DirectionalLight> directionals)
{
    // The alias table draws local lights by the same power-based selection
    // weight the other samplers use.
    std::vector<float> weights;
    glm::vec3 lower{std::numeric_limits<float>::max()};
    glm::vec3 upper{std::numeric_limits<float>::lowest()};
    const auto addLight = [&](const float weight, const glm::vec3& position) {
        weights.push_back(std::isfinite(weight) ? std::max(weight, 0.0f) : 0.0f);
        lower = glm::min(lower, position);
        upper = glm::max(upper, position);
    };
    for (const auto& light : points)
        addLight(light.selectionWeight, light.position);
    for (const auto& light : spots)
        addLight(light.selectionWeight, light.position);
    for (const auto& light : rects)
        addLight(light.selectionWeight, light.position);
    const auto count = static_cast<uint32_t>(weights.size());

    // Vose's alias method: each column keeps its own light with
    // `probability` and otherwise yields `alias`.
    double total = 0.0;
    for (const float weight : weights)
        total += weight;
    std::vector<nr::graphics::RealtimeLightAlias> table(std::max(count, 1u));
    if (count != 0) {
        std::vector<double> scaled(count);
        std::vector<uint32_t> small;
        std::vector<uint32_t> large;
        for (uint32_t i = 0; i < count; ++i) {
            const double pdf = total > 0.0 ? weights[i] / total : 1.0 / count;
            table[i].pdf = static_cast<float>(pdf);
            table[i].alias = i;
            scaled[i] = pdf * count;
            (scaled[i] < 1.0 ? small : large).push_back(i);
        }
        while (!small.empty() && !large.empty()) {
            const uint32_t below = small.back();
            small.pop_back();
            const uint32_t above = large.back();
            table[below].probability = static_cast<float>(scaled[below]);
            table[below].alias = above;
            scaled[above] -= 1.0 - scaled[below];
            if (scaled[above] < 1.0) {
                large.pop_back();
                small.push_back(above);
            }
        }
        for (const uint32_t i : large)
            table[i].probability = 1.0f;
        for (const uint32_t i : small)
            table[i].probability = 1.0f;
    }
    static_assert(sizeof(nr::graphics::RealtimeLightAlias) == 16);
    const auto* words = reinterpret_cast<const std::uint32_t*>(table.data());
    const std::span<const std::uint32_t> upload(words, table.size() * 4u);
    if (lightAlias_.size() != upload.size())
        lightAlias_ = device_.buffer<std::uint32_t>(upload.size());
    lightAlias_.upload(upload);

    localLightCount_ = count;
    infiniteLightCount_ = static_cast<uint32_t>(directionals.size());
    lightExtent_ = count != 0 ? std::max(glm::length(upper - lower), 1.0e-3f) : 1.0f;
    historyValid_ = false;
}

void Restir::prepare(const FrameContext& frame, nr::graphics::RealtimeArgs& args)
{
    using namespace nr::graphics;
    rtxdi::ReSTIRDIContext& di = context_->GetReSTIRDIContext();
    rtxdi::ReSTIRGIContext& gi = context_->GetReSTIRGIContext();
    rtxdi::ReGIRContext& regir = context_->GetReGIRContext();
    di.SetFrameIndex(frameIndex_);
    gi.SetFrameIndex(frameIndex_);
    ++frameIndex_;

    RTXDI_LightBufferParameters lightBuffer{};
    lightBuffer.localLightBufferRegion.firstLightIndex = 0;
    lightBuffer.localLightBufferRegion.numLights = localLightCount_;
    lightBuffer.infiniteLightBufferRegion.firstLightIndex = localLightCount_;
    lightBuffer.infiniteLightBufferRegion.numLights = infiniteLightCount_;
    lightBuffer.environmentLightParams.lightPresent = 0;
    context_->SetLightBufferParams(lightBuffer);

    // ReGIR's grid follows the camera, with cells sized to the lit region.
    const float3 camera = args.view.cameraPosition;
    rtxdi::ReGIRDynamicParameters regirParameters = regir.GetReGIRDynamicParameters();
    regirParameters.center = {camera.x, camera.y, camera.z};
    regirParameters.regirCellSize = lightExtent_ / ReGIRCellsAcrossLights;
    regir.SetDynamicParameters(regirParameters);

    RealtimeLighting& lighting = args.lighting;
    lighting = {};
    lighting.risBuffer = risBuffer_.ptr().address;
    lighting.diReservoirs = diReservoirs_.ptr().address;
    lighting.giReservoirs = giReservoirs_.ptr().address;
    lighting.neighborOffsets = neighborOffsets_.ptr().address;
    lighting.surfaces = surfaces_[surfaceParity_].ptr().address;
    lighting.previousSurfaces = surfaces_[surfaceParity_ ^ 1u].ptr().address;
    lighting.localLightAlias = lightAlias_.ptr().address;

    lighting.lightBufferParams = context_->GetLightBufferParameters();
    lighting.localLightsRISBufferSegmentParams =
        context_->GetLocalLightRISBufferSegmentParams();
    lighting.environmentLightRISBufferSegmentParams =
        context_->GetEnvironmentLightRISBufferSegmentParams();
    lighting.runtimeParams = di.GetRuntimeParams();

    lighting.restirDI.reservoirBufferParams = di.GetReservoirBufferParameters();
    lighting.restirDI.bufferIndices = di.GetBufferIndices();
    lighting.restirDI.initialSamplingParams = di.GetInitialSamplingParameters();
    lighting.restirDI.temporalResamplingParams = di.GetTemporalResamplingParameters();
    lighting.restirDI.boilingFilterParams = di.GetBoilingFilterParameters();
    lighting.restirDI.spatialResamplingParams = di.GetSpatialResamplingParameters();
    lighting.restirDI.spatioTemporalResamplingParams =
        di.GetSpatioTemporalResamplingParameters();
    lighting.restirDI.shadingParams = di.GetShadingParameters();

    lighting.restirGI.reservoirBufferParams = gi.GetReservoirBufferParameters();
    lighting.restirGI.bufferIndices = gi.GetBufferIndices();
    lighting.restirGI.temporalResamplingParams = gi.GetTemporalResamplingParameters();
    lighting.restirGI.boilingFilterParams = gi.GetBoilingFilterParameters();
    lighting.restirGI.spatialResamplingParams = gi.GetSpatialResamplingParameters();
    lighting.restirGI.spatioTemporalResamplingParams =
        gi.GetSpatioTemporalResamplingParameters();
    lighting.restirGI.finalShadingParams = gi.GetFinalShadingParameters();

    // As RTXDI's sample fills ReGIR_Parameters from the context.
    const rtxdi::ReGIRStaticParameters regirStatic = regir.GetReGIRStaticParameters();
    const rtxdi::ReGIROnionCalculatedParameters onion =
        regir.GetReGIROnionCalculatedParameters();
    ReGIR_Parameters& regirConstants = lighting.regir;
    regirConstants.gridParams.cellsX = regirStatic.gridParameters.gridSize.x;
    regirConstants.gridParams.cellsY = regirStatic.gridParameters.gridSize.y;
    regirConstants.gridParams.cellsZ = regirStatic.gridParameters.gridSize.z;
    regirConstants.commonParams.numRegirBuildSamples = regirParameters.regirNumBuildSamples;
    regirConstants.commonParams.risBufferOffset = regir.GetReGIRCellOffset();
    regirConstants.commonParams.lightsPerCell = regirStatic.lightsPerCell;
    regirConstants.commonParams.centerX = regirParameters.center.x;
    regirConstants.commonParams.centerY = regirParameters.center.y;
    regirConstants.commonParams.centerZ = regirParameters.center.z;
    // The onion works with radii; the cell size is a diameter.
    regirConstants.commonParams.cellSize = regirStatic.mode == rtxdi::ReGIRMode::Onion
        ? regirParameters.regirCellSize * 0.5f : regirParameters.regirCellSize;
    regirConstants.commonParams.localLightSamplingFallbackMode =
        static_cast<uint32_t>(regirParameters.fallbackSamplingMode);
    regirConstants.commonParams.localLightPresamplingMode =
        static_cast<uint32_t>(regirParameters.presamplingMode);
    regirConstants.commonParams.samplingJitter =
        std::max(0.0f, regirParameters.regirSamplingJitter * 2.0f);
    regirConstants.onionParams.cubicRootFactor = onion.regirOnionCubicRootFactor;
    regirConstants.onionParams.linearFactor = onion.regirOnionLinearFactor;
    regirConstants.onionParams.numLayerGroups =
        static_cast<uint32_t>(std::min<std::size_t>(onion.regirOnionLayers.size(),
            RTXDI_ONION_MAX_LAYER_GROUPS));
    for (uint32_t group = 0; group < regirConstants.onionParams.numLayerGroups; ++group) {
        regirConstants.onionParams.layers[group] = onion.regirOnionLayers[group];
        regirConstants.onionParams.layers[group].innerRadius *=
            regirConstants.commonParams.cellSize;
        regirConstants.onionParams.layers[group].outerRadius *=
            regirConstants.commonParams.cellSize;
    }
    for (std::size_t ring = 0; ring < std::min<std::size_t>(onion.regirOnionRings.size(),
             RTXDI_ONION_MAX_RINGS); ++ring)
        regirConstants.onionParams.rings[ring] = onion.regirOnionRings[ring];

    // Path vertices beyond the G-buffer: the same candidates, no visibility
    // test (the caller traces one shadow ray for the chosen light).
    lighting.secondaryInitialSamplingParams = lighting.restirDI.initialSamplingParams;
    lighting.secondaryInitialSamplingParams.numLocalLightSamples = SecondaryLocalLightSamples;
    lighting.secondaryInitialSamplingParams.enableInitialVisibility = 0;

    lighting.enabled = mode_ != RealtimeLightingMode::SingleSample ? 1u : 0u;
    lighting.mode = mode_ == RealtimeLightingMode::ReSTIRGI
        ? RealtimeLightingReSTIRGI : RealtimeLightingReSTIRDI;
    // Temporal reuse needs last frame's G-buffer and light indices to still
    // describe last frame.
    lighting.previousSurfacesValid =
        historyValid_ && !frame.resetHistory && historyMode_ == mode_ ? 1u : 0u;
    lighting.reservoirBlockRowPitch =
        lighting.restirDI.reservoirBufferParams.reservoirBlockRowPitch;

    // This frame's G-buffer is next frame's previous one.
    surfaceParity_ ^= 1u;
    historyValid_ = true;
    historyMode_ = mode_;
}

void Restir::presample(const nr::graphics::RealtimeArgs& args,
    const nr::graphics::RealtimeRoot root) const
{
    if (args.lighting.enabled == 0 || localLightCount_ == 0)
        return;
    const RTXDI_RISBufferSegmentParameters& tiles =
        args.lighting.localLightsRISBufferSegmentParams;
    presampleLightsPipeline_.launch({divideRoundingUp(tiles.tileSize, PresampleGroupSize),
        tiles.tileCount, 1}, root);
    device_.barrier(noorrhi::Stage::Compute, noorrhi::Stage::Compute);
    presampleReGIRPipeline_.launch({divideRoundingUp(
        context_->GetReGIRContext().GetReGIRLightSlotCount(), PresampleGroupSize), 1, 1}, root);
    device_.barrier(noorrhi::Stage::Compute, noorrhi::Stage::RayTracing);
}

void Restir::resample(const nr::graphics::RealtimeArgs& args,
    const nr::graphics::RealtimeRoot root, const Extent render) const
{
    if (args.lighting.enabled == 0)
        return;
    const noorrhi::DispatchSize pixels{render.width, render.height, 1};
    const noorrhi::DispatchSize tiles{divideRoundingUp(render.width, BoilingGroupSize),
        divideRoundingUp(render.height, BoilingGroupSize), 1};
    // Every pass reads what the previous one wrote: the image pass's G-buffer
    // and GI sample, then each resampling stage's reservoirs.
    const auto trace = [&](const noorrhi::RayTracingPipeline& pipeline) {
        pipeline.trace(pixels, root);
        device_.barrier(noorrhi::Stage::RayTracing, noorrhi::Stage::RayTracing);
    };
    const auto boil = [&](const noorrhi::ComputePipeline& pipeline, const bool enabled) {
        if (!enabled)
            return;
        device_.barrier(noorrhi::Stage::RayTracing, noorrhi::Stage::Compute);
        pipeline.launch(tiles, root);
        device_.barrier(noorrhi::Stage::Compute, noorrhi::Stage::RayTracing);
    };
    trace(diInitialPipeline_);
    trace(diTemporalPipeline_);
    boil(diBoilingPipeline_, args.lighting.restirDI.boilingFilterParams.enableBoilingFilter != 0);
    trace(diSpatialPipeline_);
    trace(diShadePipeline_);
    if (args.lighting.mode != nr::graphics::RealtimeLightingReSTIRGI)
        return;
    trace(giTemporalPipeline_);
    boil(giBoilingPipeline_, args.lighting.restirGI.boilingFilterParams.enableBoilingFilter != 0);
    trace(giSpatialPipeline_);
    trace(giShadePipeline_);
}
