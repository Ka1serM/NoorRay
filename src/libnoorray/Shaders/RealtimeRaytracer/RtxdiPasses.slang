// ReSTIR DI and PT passes of the realtime renderer, one entry point
// per build: the command line defines exactly one RTXDI_PASS_* macro.
//
// As in RTXDI's sample, resampling tests RTXDI's conservative visibility with
// inline ray queries (RtxdiBridge.slang), so the DI passes are compute
// shaders. The PT resampling passes replay paths through the materials' hit
// shaders and the shading pass traces final visibility, so those are
// ray-generation stages of the realtime pipeline.
//
// Frame order (RealtimeRaytracer::renderImpl):
//   PRESAMPLE_ENVIRONMENT
//                     environment directions -> RIS tiles (compute)
//   PRESAMPLE_LIGHTS  local lights -> RIS tiles, by power (compute)
//   PRESAMPLE_REGIR   RIS tiles -> ReGIR world-space cells (compute)
//   ... the lighting pass writes the G-buffer and records the surfaces
//   (RealtimeRaytracer.slang) ...
//   INITIAL           draw DI's candidates and PT's initial paths
//   DI_TEMPORAL       reuse from the previous frame's reservoirs (compute)
//   DI_BOILING        drop reservoirs far heavier than their neighbours (compute)
//   DI_SPATIAL        reuse from screen-space neighbours (compute)
//   PT_TEMPORAL       temporal hybrid shift and reconnection in RTXDI
//   PT_BOILING        remove heavy temporal outliers (compute)
//   PT_SPATIAL        spatial hybrid shift and reconnection in RTXDI
//   SHADE             shade the chosen light and path, add to the primary
//                     pass's radiance and pack NRD's inputs
//
// The DI and PT chains touch disjoint reservoirs between barriers.
//
// Everything from INITIAL on runs once per surface set: the first opaque
// surfaces and the nearest fractional layers. Each launch's RealtimeArgs
// point `lighting` at the set's surfaces and reservoirs and the shaded lobes
// at the set's NRD inputs.

#ifndef RTXDI_PASS_PRESAMPLE_LIGHTS
#define RTXDI_PASS_PRESAMPLE_LIGHTS 0
#endif
#ifndef RTXDI_PASS_PRESAMPLE_REGIR
#define RTXDI_PASS_PRESAMPLE_REGIR 0
#endif
#ifndef RTXDI_PASS_PRESAMPLE_ENVIRONMENT
#define RTXDI_PASS_PRESAMPLE_ENVIRONMENT 0
#endif
#ifndef RTXDI_PASS_INITIAL
#define RTXDI_PASS_INITIAL 0
#endif
#ifndef RTXDI_PASS_DI_TEMPORAL
#define RTXDI_PASS_DI_TEMPORAL 0
#endif
#ifndef RTXDI_PASS_DI_BOILING
#define RTXDI_PASS_DI_BOILING 0
#endif
#ifndef RTXDI_PASS_DI_SPATIAL
#define RTXDI_PASS_DI_SPATIAL 0
#endif
#ifndef RTXDI_PASS_PT_TEMPORAL
#define RTXDI_PASS_PT_TEMPORAL 0
#endif
#ifndef RTXDI_PASS_PT_SPATIAL
#define RTXDI_PASS_PT_SPATIAL 0
#endif
#ifndef RTXDI_PASS_PT_BOILING
#define RTXDI_PASS_PT_BOILING 0
#endif
#ifndef RTXDI_PASS_SHADE
#define RTXDI_PASS_SHADE 0
#endif

#include "RtxdiBridge.slang"
#if RTXDI_PASS_INITIAL || RTXDI_PASS_PT_TEMPORAL || RTXDI_PASS_PT_SPATIAL
#include "RtxdiPTPathTrace.slang"
#elif RTXDI_PASS_SHADE || RTXDI_PASS_PT_BOILING
#include "RtxdiPTBridge.slang"
#include "Rtxdi/PT/Reservoir.hlsli"
#endif
#include "NrdSignal.slang"

#include "Rtxdi/DI/ReservoirStorage.hlsli"

#if RTXDI_PASS_PRESAMPLE_LIGHTS || RTXDI_PASS_PRESAMPLE_REGIR
#include "Rtxdi/LightSampling/PresamplingFunctions.hlsli"
#endif
#if RTXDI_PASS_INITIAL
#include "Rtxdi/PT/InitialSampling.hlsli"
#endif
#if RTXDI_PASS_DI_TEMPORAL
#include "Rtxdi/DI/TemporalResampling.hlsli"
#endif
#if RTXDI_PASS_DI_SPATIAL
#include "Rtxdi/DI/SpatialResampling.hlsli"
#endif
#if RTXDI_PASS_DI_BOILING
#define RTXDI_ENABLE_BOILING_FILTER
#define RTXDI_BOILING_FILTER_GROUP_SIZE 16
#include "Rtxdi/DI/BoilingFilter.hlsli"
#endif
#if RTXDI_PASS_PT_TEMPORAL
#include "Rtxdi/PT/TemporalResampling.hlsli"
#endif
#if RTXDI_PASS_PT_SPATIAL
#include "Rtxdi/PT/SpatialResampling.hlsli"
#endif
#if RTXDI_PASS_PT_BOILING
#define RTXDI_ENABLE_BOILING_FILTER
#define RTXDI_BOILING_FILTER_GROUP_SIZE 16
#include "Rtxdi/PT/BoilingFilter.hlsli"
#endif

static const uint RtxdiPresampleGroupSize = 256u;
static const uint RtxdiBoilingGroupSize = 16u;
// Screen-space compute passes, as RTXDI_SCREEN_SPACE_GROUP_SIZE in its sample.
static const uint RtxdiScreenSpaceGroupSize = 8u;

RealtimeLighting* rtxdiLighting()
{
    return &realtimeArgs()->lighting;
}

// Screen-space motion of a pixel's primary surface, in lighting pixels, with
// the change in linear depth in z, as RTXDI's temporal passes want it. Each
// frame's position is in its own lighting rectangle, which a viewport resize
// changes between them.
float3 rtxdiMotion(Frame* frame, RAB_Surface surface)
{
    const float4 current = transformToClip(realtimeArgs()->view.worldToClip, surface.worldPos);
    const float4 previous = transformToClip(realtimeArgs()->view.previousWorldToClip, surface.worldPos);
    if (current.w <= 1.0e-6 || previous.w <= 1.0e-6)
        return float3(0.0);
    const float2 currentUv = float2(0.5, 0.5) + float2(0.5, -0.5) * current.xy / current.w;
    const float2 previousUv = float2(0.5, 0.5) + float2(0.5, -0.5) * previous.xy / previous.w;
    return float3(previousUv * float2(realtimeLightingSize(true))
        - currentUv * float2(realtimeLightingSize(false)), previous.w - current.w);
}

#if RTXDI_PASS_PRESAMPLE_LIGHTS

// Fills one RIS tile entry with a local light drawn by power, through the
// host-built alias table.
[shader("compute")]
[numthreads(RtxdiPresampleGroupSize, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    RealtimeLighting* lighting = rtxdiLighting();
    const RTXDI_RISBufferSegmentParameters segment = lighting.localLightsRISBufferSegmentParams;
    if (threadId.x >= segment.tileSize || threadId.y >= segment.tileCount)
        return;
    const RTXDI_LightBufferRegion region = lighting.lightBufferParams.localLightBufferRegion;
    const uint risBufferPtr = segment.bufferOffset + threadId.y * segment.tileSize + threadId.x;
    if (region.numLights == 0u)
    {
        lighting.risBuffer[risBufferPtr] = uint2(0u, 0u);
        return;
    }
    RTXDI_RandomSamplerState rng = RTXDI_InitRandomSampler(threadId.xy,
        lighting.runtimeParams.frameIndex, 0u);
    const float u = RTXDI_GetNextRandom(rng) * float(region.numLights);
    const uint column = min(uint(u), region.numLights - 1u);
    const RealtimeLightAlias entry = lighting.localLightAlias[column];
    const uint light = (u - float(column)) < entry.probability ? column : entry.alias;
    const float pdf = lighting.localLightAlias[light].pdf;
    lighting.risBuffer[risBufferPtr] = uint2(region.firstLightIndex + light,
        asuint(pdf > 0.0 ? 1.0 / pdf : 0.0));
}

#elif RTXDI_PASS_PRESAMPLE_ENVIRONMENT

// Fills one RIS tile entry with an environment direction drawn from the
// environment's own importance distribution, stored as the uv parameters
// RAB_SamplePolymorphicLight() takes and their inverse density.
[shader("compute")]
[numthreads(RtxdiPresampleGroupSize, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    RealtimeLighting* lighting = rtxdiLighting();
    const RTXDI_RISBufferSegmentParameters segment =
        lighting.environmentLightRISBufferSegmentParams;
    if (threadId.x >= segment.tileSize || threadId.y >= segment.tileCount)
        return;
    const uint entry = threadId.y * segment.tileSize + threadId.x;
    Rng rng;
    rng.state = pcgHash(hashCombine32(lighting.runtimeParams.frameIndex, entry) ^ 0x454E5649u);
    const Environment environment = realtimeArgs()->frame.environment[0];
    const EnvironmentSample sample = sampleEnvironmentDirection(environment, rng);
    const float2 uv = environmentUv(environment, sample.direction);
    const float uvPdf = sample.pdf / environmentUvPerSolidAngle(environment, uv);
    const uint packedUv = uint(saturate(uv.x) * 0xffff) | (uint(saturate(uv.y) * 0xffff) << 16);
    lighting.risBuffer[segment.bufferOffset + entry] = uint2(packedUv,
        asuint(uvPdf > 0.0 ? 1.0 / uvPdf : 0.0));
}

#elif RTXDI_PASS_PRESAMPLE_REGIR

[shader("compute")]
[numthreads(RtxdiPresampleGroupSize, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    RealtimeLighting* lighting = rtxdiLighting();
    const uint slot = threadId.x;
    RTXDI_RandomSamplerState rng = RTXDI_InitRandomSampler(
        uint2(slot & 0xfffu, slot >> 12u), lighting.runtimeParams.frameIndex, 1u);
    RTXDI_RandomSamplerState coherentRng = RTXDI_InitRandomSampler(
        uint2(slot >> 8u, 0u), lighting.runtimeParams.frameIndex, 1u);
    RTXDI_PresampleLocalLightsForReGIR(rng, coherentRng, slot,
        lighting.lightBufferParams.localLightBufferRegion,
        lighting.localLightsRISBufferSegmentParams, lighting.regir);
}

#elif RTXDI_PASS_INITIAL

// The initial samples of the surface the lighting pass recorded: ReSTIR DI's
// candidates (ReGIR-driven local lights, the directional and environment
// lights and BRDF rays, the chosen one shadow-tested) and ReSTIR PT's
// multi-bounce path, traced through RTXDI's context.
[shader("raygeneration")]
void main()
{
    const uint2 pixel = DispatchRaysIndex().xy;
    if (!realtimePixelInView(int2(pixel)))
        return;
    RealtimeArgs* args = realtimeArgs();
    RealtimeLighting* lighting = rtxdiLighting();
    const RAB_Surface surface = RAB_GetGBufferSurface(int2(pixel), false);
    const bool valid = RAB_IsSurfaceValid(surface);

    RTXDI_DIReservoir light = RTXDI_EmptyDIReservoir();
    if (valid)
    {
        RTXDI_RandomSamplerState rng = RTXDI_InitRandomSampler(pixel,
            lighting.runtimeParams.frameIndex, RTXDI_DI_GENERATE_INITIAL_SAMPLES_RANDOM_SEED);
        RTXDI_RandomSamplerState tileRng = RTXDI_InitRandomSampler(
            pixel / RTXDI_TILE_SIZE_IN_PIXELS, lighting.runtimeParams.frameIndex,
            RTXDI_DI_GENERATE_INITIAL_SAMPLES_RANDOM_SEED);
        RAB_LightSample lightSample;
        light = RTXDI_SampleLightsForSurface(rng, tileRng, surface,
            lighting.restirDI.initialSamplingParams, lighting.lightBufferParams,
            lighting.localLightsRISBufferSegmentParams,
            lighting.environmentLightRISBufferSegmentParams, lighting.regir, lightSample);
    }
    RTXDI_StoreDIReservoir(light, lighting.restirDI.reservoirBufferParams, pixel,
        lighting.restirDI.bufferIndices.initialSamplingOutputBufferIndex);

    RTXDI_PTReservoir path = RTXDI_EmptyPTReservoir();
    if (valid)
    {
        RTXDI_PathTracerRandomContext random = RTXDI_InitializePathTracerRandomContext(
            pixel, lighting.runtimeParams.frameIndex,
            RTXDI_PT_GENERATE_INITIAL_SAMPLES_RANDOM_SEED,
            RTXDI_PT_GENERATE_INITIAL_SAMPLES_REPLAY_RANDOM_SEED);
        RTXDI_PTInitialSamplingRuntimeParameters runtime =
            RTXDI_EmptyPTInitialSamplingRuntimeParameters();
        runtime.cameraPos = args.view.cameraPosition;
        runtime.prevCameraPos = args.view.previousCameraPosition;
        runtime.prevPrevCameraPos = args.view.previousPreviousCameraPosition;
        RAB_PathTracerUserData userData = RAB_EmptyPathTracerUserData();
        RAB_PathTracerUserDataSetPathType(userData, RTXDI_PTPathTraceInvocationType_Initial);
        path = GenerateInitialSamples(lighting.restirPT.initialSampling, runtime,
            lighting.restirPT.reconnection, random, surface, userData);
    }
    RTXDI_StorePTReservoir(path, lighting.restirPT.reservoirBuffer, pixel,
        lighting.restirPT.bufferIndices.initialPathTracerOutputBufferIndex);
}

#elif RTXDI_PASS_DI_TEMPORAL

[shader("compute")]
[numthreads(RtxdiScreenSpaceGroupSize, RtxdiScreenSpaceGroupSize, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    Frame* frame = &realtimeArgs()->frame;
    RealtimeLighting* lighting = rtxdiLighting();
    const uint2 pixel = threadId.xy;
    if (!realtimePixelInView(int2(pixel)))
        return;
    RTXDI_RandomSamplerState rng = RTXDI_InitRandomSampler(pixel,
        lighting.runtimeParams.frameIndex, RTXDI_DI_TEMPORAL_RESAMPLING_RANDOM_SEED);
    const RAB_Surface surface = RAB_GetGBufferSurface(int2(pixel), false);
    const RTXDI_Parameters restir = lighting.restirDI;
    RTXDI_DIReservoir result = RTXDI_EmptyDIReservoir();
    if (RAB_IsSurfaceValid(surface))
    {
        const RTXDI_DIReservoir current = RTXDI_LoadDIReservoir(
            restir.reservoirBufferParams, pixel,
            restir.bufferIndices.initialSamplingOutputBufferIndex);
        int2 temporalPixel;
        RAB_LightSample selectedSample = RAB_EmptyLightSample();
        result = RTXDI_DITemporalResampling(pixel, surface, current, rng,
            lighting.runtimeParams, restir.reservoirBufferParams,
            rtxdiMotion(frame, surface),
            restir.bufferIndices.temporalResamplingInputBufferIndex,
            restir.temporalResamplingParams, temporalPixel, selectedSample);
    }
    RTXDI_StoreDIReservoir(result, restir.reservoirBufferParams, pixel,
        restir.bufferIndices.temporalResamplingOutputBufferIndex);
}

#elif RTXDI_PASS_DI_BOILING

// RTXDI's boiling filter needs group-shared memory, which ray-generation
// stages lack, so it runs as its own pass over the temporal output: within
// each 16x16 tile, reservoirs far heavier than the tile's average are the
// rare lucky samples that would otherwise spread through spatial reuse as
// flickering blotches, and are dropped.
[shader("compute")]
[numthreads(RtxdiBoilingGroupSize, RtxdiBoilingGroupSize, 1)]
void main(uint3 threadId : SV_DispatchThreadID, uint3 localId : SV_GroupThreadID)
{
    Frame* frame = &realtimeArgs()->frame;
    RealtimeLighting* lighting = rtxdiLighting();
    const uint2 pixel = threadId.xy;
    // Every thread joins the group reduction; outside threads carry nothing.
    const bool inView = realtimePixelInView(int2(pixel));
    const RTXDI_Parameters restir = lighting.restirDI;
    RTXDI_DIReservoir reservoir = inView
        ? RTXDI_LoadDIReservoir(restir.reservoirBufferParams, pixel,
            restir.bufferIndices.temporalResamplingOutputBufferIndex)
        : RTXDI_EmptyDIReservoir();
    RTXDI_BoilingFilter(localId.xy, restir.boilingFilterParams.boilingFilterStrength,
        reservoir);
    if (inView)
        RTXDI_StoreDIReservoir(reservoir, restir.reservoirBufferParams, pixel,
            restir.bufferIndices.temporalResamplingOutputBufferIndex);
}

#elif RTXDI_PASS_DI_SPATIAL

[shader("compute")]
[numthreads(RtxdiScreenSpaceGroupSize, RtxdiScreenSpaceGroupSize, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    Frame* frame = &realtimeArgs()->frame;
    RealtimeLighting* lighting = rtxdiLighting();
    const uint2 pixel = threadId.xy;
    if (!realtimePixelInView(int2(pixel)))
        return;
    RTXDI_RandomSamplerState rng = RTXDI_InitRandomSampler(pixel,
        lighting.runtimeParams.frameIndex, RTXDI_DI_SPATIAL_RESAMPLING_RANDOM_SEED);
    const RAB_Surface surface = RAB_GetGBufferSurface(int2(pixel), false);
    const RTXDI_Parameters restir = lighting.restirDI;
    RTXDI_DIReservoir result = RTXDI_EmptyDIReservoir();
    if (RAB_IsSurfaceValid(surface))
    {
        const RTXDI_DIReservoir center = RTXDI_LoadDIReservoir(
            restir.reservoirBufferParams, pixel,
            restir.bufferIndices.spatialResamplingInputBufferIndex);
        RAB_LightSample selectedSample = RAB_EmptyLightSample();
        result = RTXDI_DISpatialResampling(pixel, surface, center, rng,
            lighting.runtimeParams, restir.reservoirBufferParams,
            restir.bufferIndices.spatialResamplingInputBufferIndex,
            restir.spatialResamplingParams, selectedSample);
    }
    RTXDI_StoreDIReservoir(result, restir.reservoirBufferParams, pixel,
        restir.bufferIndices.spatialResamplingOutputBufferIndex);
}

#elif RTXDI_PASS_PT_TEMPORAL

[shader("raygeneration")]
void main()
{
    const uint2 pixel = DispatchRaysIndex().xy;
    if (!realtimePixelInView(int2(pixel)))
        return;
    RealtimeLighting* lighting = rtxdiLighting();
    const RTXDI_PTParameters pt = lighting.restirPT;
    RTXDI_RandomSamplerState rng = RTXDI_InitRandomSampler(pixel,
        lighting.runtimeParams.frameIndex, RTXDI_PT_TEMPORAL_RESAMPLING_RANDOM_SEED);
    const RAB_Surface surface = RAB_GetGBufferSurface(int2(pixel), false);
    RTXDI_PTTemporalResamplingRuntimeParameters runtime =
        RTXDI_EmptyPTTemporalResamplingRuntimeParameters();
    runtime.pixelPosition = pixel;
    runtime.reservoirPosition = pixel;
    runtime.motionVector = rtxdiMotion(&realtimeArgs()->frame, surface);
    runtime.cameraPos = realtimeArgs()->view.cameraPosition;
    runtime.prevCameraPos = realtimeArgs()->view.previousCameraPosition;
    runtime.prevPrevCameraPos = realtimeArgs()->view.previousPreviousCameraPosition;
    RAB_PathTracerUserData userData = RAB_EmptyPathTracerUserData();
    bool selectedPrevious = false;
    const RTXDI_PTReservoir result = RTXDI_PTTemporalResampling(
        pt.temporalResampling, runtime, pt.hybridShift, pt.reconnection,
        lighting.runtimeParams, pt.reservoirBuffer, rng, pt.bufferIndices,
        selectedPrevious, userData);
    RTXDI_StorePTReservoir(result, pt.reservoirBuffer, pixel,
        pt.bufferIndices.temporalResamplingOutputBufferIndex);
}

#elif RTXDI_PASS_PT_BOILING

[shader("compute")]
[numthreads(RtxdiBoilingGroupSize, RtxdiBoilingGroupSize, 1)]
void main(uint3 threadId : SV_DispatchThreadID, uint3 localId : SV_GroupThreadID)
{
    RealtimeLighting* lighting = rtxdiLighting();
    const uint2 pixel = threadId.xy;
    const RTXDI_PTParameters pt = lighting.restirPT;
    const bool inView = realtimePixelInView(int2(pixel));
    RTXDI_PTReservoir reservoir = inView
        ? RTXDI_LoadPTReservoir(pt.reservoirBuffer, pixel,
            pt.bufferIndices.temporalResamplingOutputBufferIndex)
        : RTXDI_EmptyPTReservoir();
    RTXDI_PTBoilingFilter(localId.xy, pt.boilingFilter.boilingFilterStrength, reservoir);
    if (inView)
        RTXDI_StorePTReservoir(reservoir, pt.reservoirBuffer, pixel,
            pt.bufferIndices.temporalResamplingOutputBufferIndex);
}

#elif RTXDI_PASS_PT_SPATIAL

[shader("raygeneration")]
void main()
{
    const uint2 pixel = DispatchRaysIndex().xy;
    if (!realtimePixelInView(int2(pixel)))
        return;
    RealtimeLighting* lighting = rtxdiLighting();
    const RTXDI_PTParameters pt = lighting.restirPT;
    RTXDI_RandomSamplerState rng = RTXDI_InitRandomSampler(pixel,
        lighting.runtimeParams.frameIndex, RTXDI_PT_SPATIAL_RESAMPLING_RANDOM_SEED);
    RTXDI_PTSpatialResamplingRuntimeParameters runtime =
        RTXDI_EmptyPTSpatialResamplingRuntimeParameters();
    runtime.pixelPosition = pixel;
    runtime.reservoirPosition = pixel;
    runtime.cameraPos = realtimeArgs()->view.cameraPosition;
    runtime.prevCameraPos = realtimeArgs()->view.previousCameraPosition;
    runtime.prevPrevCameraPos = realtimeArgs()->view.previousPreviousCameraPosition;
    runtime.viewportSize = realtimeLightingSize();
    RAB_PathTracerUserData userData = RAB_EmptyPathTracerUserData();
    bool resampled = false;
    const RTXDI_PTReservoir result = RTXDI_PTSpatialResampling(runtime,
        pt.spatialResampling, pt.hybridShift, pt.reconnection, pt.reservoirBuffer,
        pt.bufferIndices, lighting.runtimeParams, rng, resampled, userData);
    RTXDI_StorePTReservoir(result, pt.reservoirBuffer, pixel,
        pt.bufferIndices.spatialResamplingOutputBufferIndex);
}

#elif RTXDI_PASS_SHADE

// Shades each pixel's resampled light and path, demodulates them by the
// material factors and packs the denoiser's inputs.
// Demodulates the lobe by its material factor and packs it for the denoiser.
void storeLobe(RealtimeLobe lobe, uint signalTarget, uint directionTarget, float3 factor,
    uint2 pixel, RAB_Surface surface, float roughness)
{
    RWTexture2D<float4> signal = ResourceDescriptorHeap[signalTarget];
    const bool valid = RAB_IsSurfaceValid(surface);
    float4 sh1 = float4(0.0);
    signal[pixel] = valid ? packNrdSignal(lobe.radiance / factor, lobe.hitDistance(),
        surface.viewDepth, roughness, lobe.direction, lobe.radiance, sh1) : float4(0.0);
    // The direction image exists only in SH form, so only it fetches it.
    if (realtimeArgs()->denoiser.sphericalHarmonics != 0u)
    {
        RWTexture2D<float4> direction = ResourceDescriptorHeap[directionTarget];
        direction[pixel] = sh1;
    }
}

[shader("raygeneration")]
void main()
{
    const uint3 threadId = DispatchRaysIndex();
    Frame* frame = &realtimeArgs()->frame;
    RealtimeLighting* lighting = rtxdiLighting();
    const uint2 pixel = threadId.xy;
    if (!realtimePixelInView(int2(pixel)))
        return;
    const RAB_Surface surface = RAB_GetGBufferSurface(int2(pixel), false);
    const RenderTargetHandles targets = realtimeArgs()->targets;
    RealtimeLobe diffuse = {};
    RealtimeLobe specular = {};
    const RealtimeBsdf bsdf = realtimeSurfaceBsdf(surface);

    const RTXDI_Parameters restir = lighting.restirDI;
    RTXDI_DIReservoir reservoir = RTXDI_LoadDIReservoir(restir.reservoirBufferParams,
        pixel, restir.bufferIndices.shadingInputBufferIndex);
    if (RAB_IsSurfaceValid(surface) && RTXDI_IsValidDIReservoir(reservoir))
    {
        const RAB_LightSample lightSample = RAB_SamplePolymorphicLight(
            RAB_LoadLightInfo(RTXDI_GetDIReservoirLightIndex(reservoir), false),
            surface, RTXDI_GetDIReservoirSampleUV(reservoir));
        float3 direction;
        float distance;
        RAB_GetLightDirDistance(surface, lightSample, direction, distance);
        const float visibility = realtimeLightTransmittance(surface, lightSample);
        // Remembered so temporal reuse can skip re-testing a light it saw.
        RTXDI_StoreVisibilityInDIReservoir(reservoir, float3(visibility),
            restir.temporalResamplingParams.enableVisibilityShortcut != 0u);
        RTXDI_StoreDIReservoir(reservoir, restir.reservoirBufferParams, pixel,
            restir.bufferIndices.shadingInputBufferIndex);
        if (visibility > 0.0)
        {
            const float3 incoming = lightSample.radiance
                * (RTXDI_GetDIReservoirInvPdf(reservoir) / lightSample.solidAnglePdf)
                * visibility;
            // The light was picked for the whole BSDF, not the specular lobe:
            // its distance would break NRD's reflection reprojection, so only
            // diffuse takes it.
            diffuse.addDirect(bsdf.evaluateDiffuse(direction) * (incoming * lightSample.diffuseScale),
                direction, min(distance, 1.0e7));
            specular.add(bsdf.evaluateSpecular(direction) * (incoming * lightSample.specularScale),
                direction);
        }
    }

    if (RAB_IsSurfaceValid(surface))
    {
        const RTXDI_PTParameters pt = lighting.restirPT;
        const RTXDI_PTReservoir reservoir = RTXDI_LoadPTReservoir(
            pt.reservoirBuffer, pixel, pt.bufferIndices.finalShadingInputBufferIndex);
        if (RTXDI_IsValidPTReservoir(reservoir) && reservoir.weightSum > 0.0
            && realtimeIsFinite(reservoir.targetFunction))
        {
            const float3 delta = reservoir.translatedWorldPosition - surface.worldPos;
            float3 direction = dot(delta, delta) > 1.0e-12
                ? normalize(delta) : surface.normal;
            float distance = length(delta);
            if (reservoir.rcVertexLength > 2u)
            {
                RTXDI_RandomSamplerState replay = RTXDI_GetRngForShading(reservoir);
                RAB_SurfaceImportanceSampleBrdf(surface, replay, direction);
                RayDesc firstRay;
                firstRay.Origin = surface.worldPos + surface.geoNormal
                    * (dot(surface.geoNormal, direction) >= 0.0 ? 1.0e-2 : -1.0e-2);
                firstRay.Direction = direction;
                firstRay.TMin = 1.0e-3;
                firstRay.TMax = 1.0e30;
                // Only the hit distance is wanted, for NRD: opaque geometry
                // suffices and no material runs.
                RayQuery<RAY_FLAG_CULL_NON_OPAQUE> query;
                query.TraceRayInline(RaytracingAccelerationStructure(frame.topLevelAS),
                    RAY_FLAG_NONE, RaytracingMaskIndirect, firstRay);
                query.Proceed();
                distance = query.CommittedStatus() == COMMITTED_TRIANGLE_HIT
                    ? query.CommittedRayT() : NRD_INF;
            }
            const float3 diffuseBsdf = bsdf.evaluateDiffuse(direction);
            const float3 specularBsdf = bsdf.evaluateSpecular(direction);
            const float3 total = diffuseBsdf + specularBsdf;
            const float3 contribution = reservoir.targetFunction * reservoir.weightSum;
            const float3 diffusePart = contribution * diffuseBsdf / max(total, float3(1.0e-8));
            const float3 specularPart = contribution * specularBsdf / max(total, float3(1.0e-8));
            // The path's first bounce belongs to the lobe that picks it with
            // the lobe's share of the BSDF; the other lobe's hit distance
            // stays missing for NRD to reconstruct.
            RTXDI_RandomSamplerState rng = RTXDI_InitRandomSampler(pixel,
                lighting.runtimeParams.frameIndex, RTXDI_PT_FINAL_SHADING_RANDOM_SEED);
            const bool specularLobe = RTXDI_GetNextRandom(rng) * realtimeLuminance(total)
                < realtimeLuminance(specularBsdf);
            diffuse.addIndirect(diffusePart, direction, distance, !specularLobe);
            specular.addIndirect(specularPart, direction, distance, specularLobe);
        }
    }
    // The lighting pass's own material factors: the render-resolution ones
    // belong to different samples when lighting runs below that resolution.
    float3 diffuseFactor;
    float3 specularFactor;
    NRD_MaterialFactors(bsdf.normal, bsdf.view, bsdf.diffuse, bsdf.specularF0, bsdf.roughness,
        diffuseFactor, specularFactor);
    storeLobe(diffuse, targets.diffuse, targets.diffuseSh1, diffuseFactor, pixel, surface, 1.0);
    storeLobe(specular, targets.specular, targets.specularSh1, specularFactor, pixel, surface,
        bsdf.roughness);
}

#endif
