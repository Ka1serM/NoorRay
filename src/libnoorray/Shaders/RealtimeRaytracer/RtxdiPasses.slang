// ReSTIR DI and ReSTIR GI passes of the realtime renderer, one entry point
// per build: the command line defines exactly one RTXDI_PASS_* macro.
//
// Frame order (RealtimeRaytracer::renderImpl):
//   PRESAMPLE_LIGHTS  local lights -> RIS tiles, by power (compute)
//   PRESAMPLE_REGIR   RIS tiles -> ReGIR world-space cells (compute)
//   ... SHaRC update/resolve, then the primary pass writes the G-buffer ...
//   DI_INITIAL        candidates per pixel from ReGIR, one visibility ray
//   DI_TEMPORAL       reuse from the previous frame's reservoirs
//   DI_BOILING        drop reservoirs far heavier than their neighbours (compute)
//   DI_SPATIAL        reuse from screen-space neighbours
//   DI_SHADE          shade the chosen light, add it to the primary pass's
//                     indirect radiance and pack NRD's inputs (DI mode)
//   GI_TEMPORAL, GI_BOILING, GI_SPATIAL, GI_SHADE
//                     the same for the diffuse indirect sample the primary
//                     pass produced (GI mode); GI_SHADE packs instead.

#ifndef RTXDI_PASS_PRESAMPLE_LIGHTS
#define RTXDI_PASS_PRESAMPLE_LIGHTS 0
#endif
#ifndef RTXDI_PASS_PRESAMPLE_REGIR
#define RTXDI_PASS_PRESAMPLE_REGIR 0
#endif
#ifndef RTXDI_PASS_DI_INITIAL
#define RTXDI_PASS_DI_INITIAL 0
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
#ifndef RTXDI_PASS_DI_SHADE
#define RTXDI_PASS_DI_SHADE 0
#endif
#ifndef RTXDI_PASS_GI_TEMPORAL
#define RTXDI_PASS_GI_TEMPORAL 0
#endif
#ifndef RTXDI_PASS_GI_BOILING
#define RTXDI_PASS_GI_BOILING 0
#endif
#ifndef RTXDI_PASS_GI_SPATIAL
#define RTXDI_PASS_GI_SPATIAL 0
#endif
#ifndef RTXDI_PASS_GI_SHADE
#define RTXDI_PASS_GI_SHADE 0
#endif

#include "RtxdiBridge.slang"
#include "NrdSignal.slang"

#include "Rtxdi/DI/ReservoirStorage.hlsli"

#if RTXDI_PASS_PRESAMPLE_LIGHTS || RTXDI_PASS_PRESAMPLE_REGIR
#include "Rtxdi/LightSampling/PresamplingFunctions.hlsli"
#endif
#if RTXDI_PASS_DI_INITIAL
#include "Rtxdi/DI/InitialSampling.hlsli"
#endif
#if RTXDI_PASS_DI_TEMPORAL
#include "Rtxdi/DI/TemporalResampling.hlsli"
#endif
#if RTXDI_PASS_DI_SPATIAL
#include "Rtxdi/DI/SpatialResampling.hlsli"
#endif
#if RTXDI_PASS_DI_BOILING || RTXDI_PASS_GI_BOILING
#define RTXDI_ENABLE_BOILING_FILTER
#define RTXDI_BOILING_FILTER_GROUP_SIZE 16
#include "Rtxdi/DI/BoilingFilter.hlsli"
#include "Rtxdi/GI/BoilingFilter.hlsli"
#endif
#if RTXDI_PASS_GI_TEMPORAL
#include "Rtxdi/GI/TemporalResampling.hlsli"
#endif
#if RTXDI_PASS_GI_SPATIAL
#include "Rtxdi/GI/SpatialResampling.hlsli"
#endif
#if RTXDI_PASS_GI_SHADE
#include "Rtxdi/GI/Reservoir.hlsli"
#endif

static const uint RtxdiPresampleGroupSize = 256u;
static const uint RtxdiBoilingGroupSize = 16u;

RealtimeLighting* rtxdiLighting()
{
    return &realtimeArgs()->lighting;
}

// Screen-space motion of a pixel's primary surface, in pixels, with the
// change in linear depth in z, as RTXDI's temporal passes want it.
float3 rtxdiMotion(Frame* frame, RAB_Surface surface)
{
    const float4 current = transformToClip(realtimeArgs()->view.worldToClip, surface.worldPos);
    const float4 previous = transformToClip(realtimeArgs()->view.previousWorldToClip, surface.worldPos);
    if (current.w <= 1.0e-6 || previous.w <= 1.0e-6)
        return float3(0.0);
    const float2 currentUv = float2(0.5, 0.5) + float2(0.5, -0.5) * current.xy / current.w;
    const float2 previousUv = float2(0.5, 0.5) + float2(0.5, -0.5) * previous.xy / previous.w;
    return float3((previousUv - currentUv) * float2(frame.width, frame.height),
        previous.w - current.w);
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

#elif RTXDI_PASS_DI_INITIAL

[shader("raygeneration")]
void main()
{
    Frame* frame = &realtimeArgs()->frame;
    RealtimeLighting* lighting = rtxdiLighting();
    const uint2 pixel = DispatchRaysIndex().xy;
    if (pixel.x >= frame.width || pixel.y >= frame.height)
        return;
    RTXDI_RandomSamplerState rng = RTXDI_InitRandomSampler(pixel,
        lighting.runtimeParams.frameIndex, RTXDI_DI_GENERATE_INITIAL_SAMPLES_RANDOM_SEED);
    RTXDI_RandomSamplerState tileRng = RTXDI_InitRandomSampler(
        pixel / RTXDI_TILE_SIZE_IN_PIXELS, lighting.runtimeParams.frameIndex,
        RTXDI_DI_GENERATE_INITIAL_SAMPLES_RANDOM_SEED);
    const RAB_Surface surface = RAB_GetGBufferSurface(int2(pixel), false);
    RTXDI_DIReservoir reservoir = RTXDI_EmptyDIReservoir();
    if (RAB_IsSurfaceValid(surface))
    {
        RAB_LightSample lightSample;
        reservoir = RTXDI_SampleLightsForSurface(rng, tileRng, surface,
            lighting.restirDI.initialSamplingParams, lighting.lightBufferParams,
            lighting.localLightsRISBufferSegmentParams,
            lighting.environmentLightRISBufferSegmentParams, lighting.regir,
            lightSample);
    }
    RTXDI_StoreDIReservoir(reservoir, lighting.restirDI.reservoirBufferParams, pixel,
        lighting.restirDI.bufferIndices.initialSamplingOutputBufferIndex);
}

#elif RTXDI_PASS_DI_TEMPORAL

[shader("raygeneration")]
void main()
{
    Frame* frame = &realtimeArgs()->frame;
    RealtimeLighting* lighting = rtxdiLighting();
    const uint2 pixel = DispatchRaysIndex().xy;
    if (pixel.x >= frame.width || pixel.y >= frame.height)
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

#elif RTXDI_PASS_DI_BOILING || RTXDI_PASS_GI_BOILING

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
    const bool inView = pixel.x < frame.width && pixel.y < frame.height;
#if RTXDI_PASS_DI_BOILING
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
#else
    const RTXDI_GIParameters restir = lighting.restirGI;
    RTXDI_GIReservoir reservoir = inView
        ? RTXDI_LoadGIReservoir(restir.reservoirBufferParams, pixel,
            restir.bufferIndices.temporalResamplingOutputBufferIndex)
        : RTXDI_EmptyGIReservoir();
    RTXDI_GIBoilingFilter(localId.xy, restir.boilingFilterParams.boilingFilterStrength,
        reservoir);
    if (inView)
        RTXDI_StoreGIReservoir(reservoir, restir.reservoirBufferParams, pixel,
            restir.bufferIndices.temporalResamplingOutputBufferIndex);
#endif
}

#elif RTXDI_PASS_DI_SPATIAL

[shader("raygeneration")]
void main()
{
    Frame* frame = &realtimeArgs()->frame;
    RealtimeLighting* lighting = rtxdiLighting();
    const uint2 pixel = DispatchRaysIndex().xy;
    if (pixel.x >= frame.width || pixel.y >= frame.height)
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

#elif RTXDI_PASS_GI_TEMPORAL

[shader("raygeneration")]
void main()
{
    Frame* frame = &realtimeArgs()->frame;
    RealtimeLighting* lighting = rtxdiLighting();
    const uint2 pixel = DispatchRaysIndex().xy;
    if (pixel.x >= frame.width || pixel.y >= frame.height)
        return;
    RTXDI_RandomSamplerState rng = RTXDI_InitRandomSampler(pixel,
        lighting.runtimeParams.frameIndex, RTXDI_GI_TEMPORAL_RESAMPLING_RANDOM_SEED);
    const RAB_Surface surface = RAB_GetGBufferSurface(int2(pixel), false);
    const RTXDI_GIParameters restir = lighting.restirGI;
    RTXDI_GIReservoir reservoir = RTXDI_LoadGIReservoir(restir.reservoirBufferParams,
        pixel, restir.bufferIndices.secondarySurfaceReSTIRDIOutputBufferIndex);
    if (RAB_IsSurfaceValid(surface))
    {
        RTXDI_GITemporalResamplingParameters parameters = restir.temporalResamplingParams;
        // Staggered ages keep whole regions from expiring on the same frame.
        parameters.maxReservoirAge = uint(float(parameters.maxReservoirAge)
            * (0.5 + RTXDI_GetNextRandom(rng) * 0.5));
        reservoir = RTXDI_GITemporalResampling(pixel, surface, rtxdiMotion(frame, surface),
            restir.bufferIndices.temporalResamplingInputBufferIndex, reservoir, rng,
            lighting.runtimeParams, restir.reservoirBufferParams, parameters);
    }
    RTXDI_StoreGIReservoir(reservoir, restir.reservoirBufferParams, pixel,
        restir.bufferIndices.temporalResamplingOutputBufferIndex);
}

#elif RTXDI_PASS_GI_SPATIAL

[shader("raygeneration")]
void main()
{
    Frame* frame = &realtimeArgs()->frame;
    RealtimeLighting* lighting = rtxdiLighting();
    const uint2 pixel = DispatchRaysIndex().xy;
    if (pixel.x >= frame.width || pixel.y >= frame.height)
        return;
    RTXDI_RandomSamplerState rng = RTXDI_InitRandomSampler(pixel,
        lighting.runtimeParams.frameIndex, RTXDI_GI_SPATIAL_RESAMPLING_RANDOM_SEED);
    const RAB_Surface surface = RAB_GetGBufferSurface(int2(pixel), false);
    const RTXDI_GIParameters restir = lighting.restirGI;
    RTXDI_GIReservoir reservoir = RTXDI_LoadGIReservoir(restir.reservoirBufferParams,
        pixel, restir.bufferIndices.spatialResamplingInputBufferIndex);
    if (RAB_IsSurfaceValid(surface))
        reservoir = RTXDI_GISpatialResampling(pixel, surface,
            restir.bufferIndices.spatialResamplingInputBufferIndex, reservoir, rng,
            lighting.runtimeParams, restir.reservoirBufferParams,
            restir.spatialResamplingParams);
    RTXDI_StoreGIReservoir(reservoir, restir.reservoirBufferParams, pixel,
        restir.bufferIndices.spatialResamplingOutputBufferIndex);
}

#elif RTXDI_PASS_DI_SHADE || RTXDI_PASS_GI_SHADE

// The primary pass leaves each pixel's unpacked radiance in NRD's diffuse and
// specular inputs (rgb: radiance already weighted by the camera, a: hit
// distance). The shading passes add their resampled light to it; the last one
// demodulates by the material factors and packs REBLUR's inputs.
struct RealtimeLobes
{
    float4 diffuse;
    float4 specular;
};

// Adds `radiance` arriving over `distance` to a lobe. The hit distance NRD
// sees is the brightest contribution's, as RTXDI's sample does it.
void addLobeRadiance(inout float4 lobe, float3 radiance, float distance)
{
    if (!(realtimeMaxComponent(radiance) > 0.0) || !realtimeIsFinite(radiance))
        return;
    if (lobe.a <= 0.0 || realtimeLuminance(radiance) > realtimeLuminance(lobe.rgb))
        lobe.a = distance;
    lobe.rgb += radiance;
}

void storeLobes(Frame* frame, uint2 pixel, RAB_Surface surface, RealtimeLobes lobes,
    bool pack)
{
    RWTexture2D<float4> nrdDiffuse = ResourceDescriptorHeap[realtimeArgs()->targets.diffuse];
    RWTexture2D<float4> nrdSpecular = ResourceDescriptorHeap[realtimeArgs()->targets.specular];
    if (!pack)
    {
        nrdDiffuse[pixel] = lobes.diffuse;
        nrdSpecular[pixel] = lobes.specular;
        return;
    }
    if (!RAB_IsSurfaceValid(surface))
    {
        nrdDiffuse[pixel] = float4(0.0);
        nrdSpecular[pixel] = float4(0.0);
        return;
    }
    RWTexture2D<float4> diffuseFactor = ResourceDescriptorHeap[realtimeArgs()->targets.diffuseFactor];
    RWTexture2D<float4> specularFactor = ResourceDescriptorHeap[realtimeArgs()->targets.specularFactor];
    const float viewZ = surface.viewDepth;
    nrdDiffuse[pixel] = packNrdSignal(lobes.diffuse.rgb / diffuseFactor[pixel].rgb,
        lobes.diffuse.a, viewZ, 1.0);
    nrdSpecular[pixel] = packNrdSignal(lobes.specular.rgb / specularFactor[pixel].rgb, lobes.specular.a, viewZ,
        surface.material.roughness);
}

RealtimeLobes loadLobes(Frame* frame, uint2 pixel)
{
    RWTexture2D<float4> nrdDiffuse = ResourceDescriptorHeap[realtimeArgs()->targets.diffuse];
    RWTexture2D<float4> nrdSpecular = ResourceDescriptorHeap[realtimeArgs()->targets.specular];
    RealtimeLobes lobes;
    lobes.diffuse = nrdDiffuse[pixel];
    lobes.specular = nrdSpecular[pixel];
    return lobes;
}

float realtimeCameraWeight(uint2 pixel)
{
    return realtimeArgs()->lighting.surfaces[realtimeSurfaceIndex(pixel)].cameraWeight;
}

[shader("raygeneration")]
void main()
{
    Frame* frame = &realtimeArgs()->frame;
    RealtimeLighting* lighting = rtxdiLighting();
    const uint2 pixel = DispatchRaysIndex().xy;
    if (pixel.x >= frame.width || pixel.y >= frame.height)
        return;
    const RAB_Surface surface = RAB_GetGBufferSurface(int2(pixel), false);
    RealtimeLobes lobes = loadLobes(frame, pixel);
    const float cameraWeight = realtimeCameraWeight(pixel);
    const RealtimeBsdf bsdf = realtimeSurfaceBsdf(surface);

#if RTXDI_PASS_DI_SHADE
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
        const bool visible = lightSample.solidAnglePdf > 0.0
            && realtimeVisible(surface.worldPos, surface.geoNormal, direction, distance);
        // Remembered so temporal reuse can skip re-testing a light it saw.
        RTXDI_StoreVisibilityInDIReservoir(reservoir, visible ? float3(1.0) : float3(0.0),
            restir.temporalResamplingParams.enableVisibilityShortcut != 0u);
        RTXDI_StoreDIReservoir(reservoir, restir.reservoirBufferParams, pixel,
            restir.bufferIndices.shadingInputBufferIndex);
        if (visible)
        {
            const float3 incoming = lightSample.radiance
                * (RTXDI_GetDIReservoirInvPdf(reservoir) / lightSample.solidAnglePdf)
                * cameraWeight;
            const float hitDistance = min(distance, 1.0e7);
            addLobeRadiance(lobes.diffuse, bsdf.evaluateDiffuse(direction) * incoming,
                hitDistance);
            addLobeRadiance(lobes.specular, bsdf.evaluateSpecular(direction) * incoming,
                hitDistance);
        }
    }
    storeLobes(frame, pixel, surface, lobes, lighting.mode == RealtimeLightingReSTIRDI);
#else
    const RTXDI_GIParameters restir = lighting.restirGI;
    const RTXDI_GIReservoir reservoir = RTXDI_LoadGIReservoir(restir.reservoirBufferParams,
        pixel, restir.bufferIndices.finalShadingInputBufferIndex);
    if (RAB_IsSurfaceValid(surface) && RTXDI_IsValidGIReservoir(reservoir))
    {
        const float3 toSample = reservoir.position - surface.worldPos;
        const float distance = length(toSample);
        const float3 direction = distance > 0.0 ? toSample / distance : surface.normal;
        const bool visible = distance > 0.0
            && (restir.finalShadingParams.enableFinalVisibility == 0u
                || RAB_GetConservativeVisibility(surface, reservoir.position));
        if (visible)
            addLobeRadiance(lobes.diffuse, bsdf.evaluateDiffuse(direction)
                * reservoir.radiance * (reservoir.weightSum * cameraWeight), distance);
    }
    storeLobes(frame, pixel, surface, lobes, true);
#endif
}

#endif
