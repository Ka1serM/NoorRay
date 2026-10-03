#include "Realtime/Restir.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <vector>

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <Rtxdi/ImportanceSamplingContext.h>
#include <Rtxdi/PT/ReSTIRPT.h>
#include <Rtxdi/LightSampling/RISBufferSegmentAllocator.h>
#include <Rtxdi/RtxdiUtils.h>

#include "Realtime/ShaderLoading.h"

namespace
{
constexpr const char* presampleLightsSpv = "RealtimeRaytracer/RtxdiPresampleLights.spv";
constexpr const char* presampleReGIRSpv = "RealtimeRaytracer/RtxdiPresampleReGIR.spv";
constexpr const char* presampleEnvironmentSpv = "RealtimeRaytracer/RtxdiPresampleEnvironment.spv";
constexpr const char* initialSpv = "RealtimeRaytracer/RtxdiInitial.spv";
constexpr const char* diTemporalSpv = "RealtimeRaytracer/RtxdiDITemporal.spv";
constexpr const char* diBoilingSpv = "RealtimeRaytracer/RtxdiDIBoiling.spv";
constexpr const char* diSpatialSpv = "RealtimeRaytracer/RtxdiDISpatial.spv";
constexpr const char* ptTemporalSpv = "RealtimeRaytracer/RtxdiPTTemporal.spv";
constexpr const char* ptSpatialSpv = "RealtimeRaytracer/RtxdiPTSpatial.spv";
constexpr const char* ptBoilingSpv = "RealtimeRaytracer/RtxdiPTBoiling.spv";
constexpr const char* shadeSpv = "RealtimeRaytracer/RtxdiShade.spv";


// Group sizes, matching RtxdiPasses.slang.
constexpr uint32_t PresampleGroupSize = 256u;
constexpr uint32_t BoilingGroupSize = 16u;
constexpr uint32_t ScreenSpaceGroupSize = 8u;
// Screen-space offsets spatial resampling draws its neighbours from.
constexpr uint32_t NeighborOffsetCount = 8192u;
// Candidate lights per vertex in world-space (ReGIR) sampling beyond the
// G-buffer, as RTXDI's sample takes for its path tracer's NEE; primary
// surfaces keep RTXDI's default of eight.
constexpr uint32_t SecondaryLocalLightSamples = 4u;
// ReGIR's grid spans the lit region in this many cells per axis.
constexpr float ReGIRCellsAcrossLights = 16.0f;

// Zero for an absent buffer.
std::uint64_t address(const noorrhi::Buffer<std::uint32_t>& buffer)
{
    return buffer ? buffer.ptr().address : 0u;
}
}

Restir::Restir(noorrhi::Device& device)
    : device_(device)
    , presampleLightsPipeline_(device.compute(loadShader(device, presampleLightsSpv)))
    , presampleReGIRPipeline_(device.compute(loadShader(device, presampleReGIRSpv)))
    , presampleEnvironmentPipeline_(device.compute(loadShader(device, presampleEnvironmentSpv)))
    , initialRaygen_(loadShader(device, initialSpv))
    , diTemporalPipeline_(device.compute(loadShader(device, diTemporalSpv)))
    , diBoilingPipeline_(device.compute(loadShader(device, diBoilingSpv)))
    , diSpatialPipeline_(device.compute(loadShader(device, diSpatialSpv)))
    , ptTemporalRaygen_(loadShader(device, ptTemporalSpv))
    , ptBoilingPipeline_(device.compute(loadShader(device, ptBoilingSpv)))
    , ptSpatialRaygen_(loadShader(device, ptSpatialSpv))
    , shadeRaygen_(loadShader(device, shadeSpv))
{
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
    constexpr std::size_t aliasWords = sizeof(nr::graphics::RealtimeLightAlias) / 4u;
    lightAlias_ = device.buffer<std::uint32_t>(aliasWords);
    lightAlias_.upload(std::span<const std::uint32_t>(std::vector<std::uint32_t>(aliasWords)));
}

Restir::~Restir() = default;

void Restir::resize(const Extent lighting, const bool layers)
{
    // RTXDI derives the reservoirs' block-linear pitch from these, so frames
    // of any smaller rectangle address them alike.
    rtxdi::ImportanceSamplingContext_StaticParameters parameters;
    parameters.renderWidth = lighting.width;
    parameters.renderHeight = lighting.height;
    parameters.NeighborOffsetCount = NeighborOffsetCount;
    parameters.regirStaticParams.mode = rtxdi::ReGIRMode::Grid;
    context_ = std::make_unique<rtxdi::ImportanceSamplingContext>(parameters);
    rtxdi::ReSTIRPTStaticParameters ptParameters;
    ptParameters.RenderWidth = lighting.width;
    ptParameters.RenderHeight = lighting.height;
    ptContext_ = std::make_unique<rtxdi::ReSTIRPTContext>(ptParameters);
    ptContext_->SetResamplingMode(rtxdi::ReSTIRPT_ResamplingMode::TemporalAndSpatial);
    auto ptDecorrelation = ptContext_->GetDecorrelationParameters();
    ptDecorrelation.decorrelationMode = RTXDI_PTDecorrelationMode::None;
    ptDecorrelation.decorrelationFactor = 0.0f;
    ptContext_->SetDecorrelationParameters(ptDecorrelation);
    rtxdi::ReSTIRDIContext& di = context_->GetReSTIRDIContext();
    di.SetResamplingMode(rtxdi::ReSTIRDI_ResamplingMode::TemporalAndSpatial);

    // All direct light at primary surfaces: eight ReGIR-driven candidates, one
    // directional light, one environment direction and one BRDF direction per
    // pixel, the chosen one shadow-tested. BRDF rays can only find the
    // environment, which RTXDI then weighs against its light samples, so
    // glossy surfaces still see sharp reflections of it.
    auto initial = di.GetInitialSamplingParameters();
    initial.localLightSamplingMode = ReSTIRDI_LocalLightSamplingMode::ReGIR_RIS;
    di.SetInitialSamplingParameters(initial);
    // Prefer speed and stable interactive output over correcting the estimator's
    // bias. Ray-traced correction adds work to temporal and spatial reuse and
    // can introduce extra visibility noise around shadow boundaries.
    auto diTemporal = di.GetTemporalResamplingParameters();
    diTemporal.biasCorrectionMode = ReSTIRDI_TemporalBiasCorrectionMode::Off;
    di.SetTemporalResamplingParameters(diTemporal);
    auto diSpatial = di.GetSpatialResamplingParameters();
    diSpatial.biasCorrectionMode = ReSTIRDI_SpatialBiasCorrectionMode::Off;
    di.SetSpatialResamplingParameters(diSpatial);

    risBuffer_ = device_.buffer<std::uint32_t>(2u * std::max(
        context_->GetRISBufferSegmentAllocator().GetTotalSizeInElements(), 1u));
    const std::size_t pixels =
        std::max<std::size_t>(std::size_t(lighting.width) * lighting.height, 1u);
    opaque_ = surfaceSet(pixels);
    layers_.reset();
    if (layers)
        layers_ = surfaceSet(pixels);
    frameIndex_ = 0;
    surfaceParity_ = 0;
    historyValid_ = false;
}

Restir::SurfaceSet Restir::surfaceSet(const std::size_t pixels) const
{
    constexpr std::size_t diReservoirWords = sizeof(RTXDI_PackedDIReservoir) / 4u;
    constexpr std::size_t ptReservoirWords = sizeof(RTXDI_PackedPTReservoir) / 4u;
    constexpr std::size_t surfaceWords = sizeof(nr::graphics::RealtimeSurface) / 4u;
    SurfaceSet set;
    set.diReservoirs = device_.buffer<std::uint32_t>(diReservoirWords
        * context_->GetReSTIRDIContext().GetReservoirBufferParameters().reservoirArrayPitch
        * rtxdi::c_NumReSTIRDIReservoirBuffers);
    set.ptReservoirs = device_.buffer<std::uint32_t>(ptReservoirWords
        * ptContext_->GetReservoirBufferParameters().reservoirArrayPitch
        * rtxdi::c_NumReSTIRPTReservoirBuffers);
    for (auto& surfaces : set.surfaces)
        surfaces = device_.buffer<std::uint32_t>(surfaceWords * pixels);
    return set;
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
    const auto* words = reinterpret_cast<const std::uint32_t*>(table.data());
    const std::span<const std::uint32_t> upload(words,
        table.size() * sizeof(nr::graphics::RealtimeLightAlias) / 4u);
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
    rtxdi::ReGIRContext& regir = context_->GetReGIRContext();
    di.SetFrameIndex(frameIndex_);
    ptContext_->SetFrameIndex(frameIndex_);
    auto ptInitial = ptContext_->GetInitialSamplingParameters();
    // RTXDI limits the path vertex index of the light: the primary surface is
    // vertex 1, direct light vertex 2, so N indirect bounces reach vertex N + 2.
    // RTXDI packs the selected path length and reconnection length into bytes.
    const uint32_t maxLightVertexIndex = std::clamp(args.frame.maxBounces + 2u, 3u, 253u);
    ptInitial.maxBounceDepth = maxLightVertexIndex;
    ptInitial.maxRcVertexLength = maxLightVertexIndex + 2u;
    ptContext_->SetInitialSamplingParameters(ptInitial);
    auto ptHybrid = ptContext_->GetHybridShiftParameters();
    ptHybrid.maxBounceDepth = maxLightVertexIndex;
    ptHybrid.maxRcVertexLength = ptInitial.maxRcVertexLength;
    ptContext_->SetHybridShiftParameters(ptHybrid);
    ++frameIndex_;

    RTXDI_LightBufferParameters lightBuffer{};
    lightBuffer.localLightBufferRegion.firstLightIndex = 0;
    lightBuffer.localLightBufferRegion.numLights = localLightCount_;
    lightBuffer.infiniteLightBufferRegion.firstLightIndex = localLightCount_;
    lightBuffer.infiniteLightBufferRegion.numLights = infiniteLightCount_;
    lightBuffer.environmentLightParams.lightPresent = args.frame.environment != 0 ? 1u : 0u;
    lightBuffer.environmentLightParams.lightIndex = localLightCount_ + infiniteLightCount_;
    context_->SetLightBufferParams(lightBuffer);

    // ReGIR's grid follows the camera, with cells sized to the lit region.
    const float3 camera = args.view.cameraPosition;
    rtxdi::ReGIRDynamicParameters regirParameters = regir.GetReGIRDynamicParameters();
    regirParameters.center = {camera.x, camera.y, camera.z};
    regirParameters.regirCellSize = lightExtent_ / ReGIRCellsAcrossLights;
    regir.SetDynamicParameters(regirParameters);

    // Last frame's surfaces are this frame's previous ones.
    surfaceParity_ ^= 1u;
    RealtimeLighting& lighting = args.lighting;
    lighting = {};
    lighting.risBuffer = risBuffer_.ptr().address;
    lighting.diReservoirs = opaque_.diReservoirs.ptr().address;
    lighting.ptReservoirs = address(opaque_.ptReservoirs);
    lighting.neighborOffsets = neighborOffsets_.ptr().address;
    lighting.surfaces = opaque_.surfaces[surfaceParity_].ptr().address;
    lighting.previousSurfaces = opaque_.surfaces[surfaceParity_ ^ 1u].ptr().address;
    lighting.layerSurfaces = layers_ ? layers_->surfaces[surfaceParity_].ptr().address : 0;
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

    lighting.restirPT.reservoirBuffer = ptContext_->GetReservoirBufferParameters();
    lighting.restirPT.bufferIndices = ptContext_->GetBufferIndices();
    lighting.restirPT.initialSampling = ptContext_->GetInitialSamplingParameters();
    lighting.restirPT.decorrelation = ptContext_->GetDecorrelationParameters();
    lighting.restirPT.reconnection = ptContext_->GetReconnectionParameters();
    lighting.restirPT.temporalResampling = ptContext_->GetTemporalResamplingParameters();
    lighting.restirPT.hybridShift = ptContext_->GetHybridShiftParameters();
    lighting.restirPT.boilingFilter = ptContext_->GetBoilingFilterParameters();
    lighting.restirPT.spatialResampling = ptContext_->GetSpatialResamplingParameters();

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

    // Surfaces beyond the G-buffer: the same light candidates without BRDF
    // rays, which the path tracer traces itself, and no visibility test (the
    // caller traces one shadow ray for the chosen light).
    lighting.secondaryInitialSamplingParams = lighting.restirDI.initialSamplingParams;
    lighting.secondaryInitialSamplingParams.numLocalLightSamples = SecondaryLocalLightSamples;
    lighting.secondaryInitialSamplingParams.numBrdfSamples = 0;
    lighting.secondaryInitialSamplingParams.enableInitialVisibility = 0;

    // Temporal reuse needs last frame's G-buffer and light indices to still
    // describe last frame.
    // The path-tracing reservoirs are not written while there are no indirect
    // bounces, so the first frame with indirect light has no history either.
    const bool indirect = args.frame.maxBounces > 0u;
    lighting.previousSurfacesValid =
        historyValid_ && !frame.resetHistory && (previousFrameIndirect_ || !indirect) ? 1u : 0u;
    previousFrameIndirect_ = indirect;
    lighting.reservoirBlockRowPitch =
        lighting.restirDI.reservoirBufferParams.reservoirBlockRowPitch;
    historyValid_ = true;
}

nr::graphics::RealtimeLighting Restir::layerLighting(
    const nr::graphics::RealtimeLighting& lighting) const
{
    const SurfaceSet& layers = layers_.value();
    nr::graphics::RealtimeLighting result = lighting;
    result.diReservoirs = layers.diReservoirs.ptr().address;
    result.ptReservoirs = address(layers.ptReservoirs);
    result.surfaces = layers.surfaces[surfaceParity_].ptr().address;
    result.previousSurfaces = layers.surfaces[surfaceParity_ ^ 1u].ptr().address;
    return result;
}

void Restir::presample(const nr::graphics::RealtimeArgs& args,
    const nr::graphics::RealtimeRoot root) const
{
    // The environment and local-light tiles are disjoint segments of the RIS
    // buffer, so both fill at once; only ReGIR reads the local-light tiles.
    const bool environment =
        args.lighting.lightBufferParams.environmentLightParams.lightPresent != 0;
    if (!environment && localLightCount_ == 0)
        return;
    if (environment) {
        const RTXDI_RISBufferSegmentParameters& environmentTiles =
            args.lighting.environmentLightRISBufferSegmentParams;
        device_.label("Presample Environment", [&] {
            presampleEnvironmentPipeline_.launch({divideRoundingUp(environmentTiles.tileSize,
                PresampleGroupSize), environmentTiles.tileCount, 1}, root);
        });
    }
    if (localLightCount_ != 0) {
        const RTXDI_RISBufferSegmentParameters& tiles =
            args.lighting.localLightsRISBufferSegmentParams;
        device_.label("Presample Local Lights", [&] {
            presampleLightsPipeline_.launch({divideRoundingUp(tiles.tileSize, PresampleGroupSize),
                tiles.tileCount, 1}, root);
        });
        device_.barrier(noorrhi::Stage::Compute, noorrhi::Stage::Compute);
        device_.label("Presample ReGIR", [&] {
            presampleReGIRPipeline_.launch({divideRoundingUp(
                context_->GetReGIRContext().GetReGIRLightSlotCount(), PresampleGroupSize), 1, 1},
                root);
        });
    }
    device_.barrier(noorrhi::Stage::Compute, noorrhi::Stage::RayTracing);
}

std::vector<noorrhi::Shader> Restir::raygens() const
{
    return {initialRaygen_, ptTemporalRaygen_, ptSpatialRaygen_, shadeRaygen_};
}

void Restir::resample(const nr::graphics::RealtimeArgs& args,
    const std::span<const nr::graphics::RealtimeRoot> surfaceSets, const Extent lighting,
    const noorrhi::RayTracingPipeline& tracePipeline) const
{
    using noorrhi::Stage;
    const bool diBoiling = args.lighting.restirDI.boilingFilterParams.enableBoilingFilter != 0;
    const bool indirect = args.frame.maxBounces > 0u;
    const bool ptBoiling = indirect && args.lighting.restirPT.boilingFilter.enableBoilingFilter != 0;
    const noorrhi::DispatchSize pixels{divideRoundingUp(lighting.width, ScreenSpaceGroupSize),
        divideRoundingUp(lighting.height, ScreenSpaceGroupSize), 1};
    const noorrhi::DispatchSize tiles{divideRoundingUp(lighting.width, BoilingGroupSize),
        divideRoundingUp(lighting.height, BoilingGroupSize), 1};
    const auto trace = [&](const std::string_view name, const noorrhi::Shader& raygen) {
        device_.label(name, [&] {
            for (const nr::graphics::RealtimeRoot root : surfaceSets)
                tracePipeline.trace(raygen, {lighting.width, lighting.height, 1}, root);
        });
    };
    const auto launch = [&](const std::string_view name, const noorrhi::ComputePipeline& pipeline,
        const noorrhi::DispatchSize size) {
        device_.label(name, [&] {
            for (const nr::graphics::RealtimeRoot root : surfaceSets)
                pipeline.launch(size, root);
        });
    };
    // The DI chain runs as compute and the PT chain as ray tracing; each
    // step's passes run together and read what the steps before them wrote,
    // starting with the surfaces the lighting pass recorded.
    const auto barrier = [&](const std::initializer_list<Stage> sources,
        const std::initializer_list<Stage> destinations) {
        for (const Stage source : sources)
            for (const Stage destination : destinations)
                device_.barrier(source, destination);
    };

    barrier({Stage::RayTracing}, {Stage::RayTracing});
    trace("Initial Samples", initialRaygen_);
    barrier({Stage::RayTracing}, {Stage::Compute, Stage::RayTracing});
    launch("DI Temporal", diTemporalPipeline_, pixels);
    if (indirect)
        trace("PT Temporal", ptTemporalRaygen_);
    if (diBoiling || ptBoiling) {
        barrier({Stage::Compute, Stage::RayTracing}, {Stage::Compute});
        if (diBoiling)
            launch("DI Boiling Filter", diBoilingPipeline_, tiles);
        if (ptBoiling)
            launch("PT Boiling Filter", ptBoilingPipeline_, tiles);
        barrier({Stage::Compute}, {Stage::Compute, Stage::RayTracing});
    } else {
        barrier({Stage::Compute, Stage::RayTracing}, {Stage::Compute, Stage::RayTracing});
    }
    launch("DI Spatial", diSpatialPipeline_, pixels);
    if (indirect)
        trace("PT Spatial", ptSpatialRaygen_);
    barrier({Stage::Compute, Stage::RayTracing}, {Stage::RayTracing});
    trace("Shade", shadeRaygen_);
}
