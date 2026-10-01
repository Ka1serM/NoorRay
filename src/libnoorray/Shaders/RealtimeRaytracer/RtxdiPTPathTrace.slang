#pragma once

#include "RtxdiPTBridge.slang"
#include "RealtimeTransparency.slang"
#include "Rtxdi/DI/InitialSampling.hlsli"
#include "Rtxdi/PT/InitialSamplingPathTracerContext.hlsli"
#include "Rtxdi/PT/HybridShiftPathTracerContext.hlsli"
#define RTXDI_PT_CONTEXT_NAME RTXDI_PathTracerContextInitial
#define RTXDI_PT_CONTEXT_TYPE RTXDI_InitialSamplingPathTracerContext
#define RTXDI_PT_CONTEXT_INIT RTXDI_InitializePathTracerContextInitial
#include "Rtxdi/PT/PathTracerContext.hlsli"
#undef RTXDI_PT_CONTEXT_NAME
#undef RTXDI_PT_CONTEXT_TYPE
#undef RTXDI_PT_CONTEXT_INIT
#define RTXDI_PT_CONTEXT_NAME RTXDI_PathTracerContextHybrid
#define RTXDI_PT_CONTEXT_TYPE RTXDI_HybridShiftPathTracerContext
#define RTXDI_PT_CONTEXT_INIT RTXDI_InitializePathTracerContextHybrid
#include "Rtxdi/PT/PathTracerContext.hlsli"
#undef RTXDI_PT_CONTEXT_NAME
#undef RTXDI_PT_CONTEXT_TYPE
#undef RTXDI_PT_CONTEXT_INIT
#include "Rtxdi/PT/PathTracerRandomContext.hlsli"
#include "Rtxdi/Utils/SampledLightData.hlsli"
#include "Rtxdi/Utils/Checkerboard.hlsli"

// Path termination of RTXDI's sample at its Medium quality preset.
static const float RealtimePTMinimumThroughput = 0.05;
static const float RealtimePTRussianRouletteContinuation = 0.8;

// Both context instantiations replay the same renderer path. The generated
// SDK headers only replace HLSL templates with their two concrete types.
RTXDI_BrdfRaySample realtimePTSampleBsdf(RAB_Surface surface,
    inout RTXDI_RandomSamplerState rng)
{
    RTXDI_BrdfRaySample ray = RTXDI_EmptyBrdfRaySample();
    const float lobe = RTXDI_GetNextRandom(rng);
    const float2 uv = float2(RTXDI_GetNextRandom(rng), RTXDI_GetNextRandom(rng));
    const RealtimeBsdfSample sample = realtimeSurfaceBsdf(surface).sample(lobe, uv);
    if (sample.valid)
    {
        ray.outDirection = sample.direction;
        ray.outPdf = sample.pdf;
        ray.brdfTimesNoL = realtimeSurfaceReflectance(surface, sample.direction);
        if (sample.specular)
            ray.properties.SetSpecular();
    }
    return ray;
}

// The surface a path found, which it reached with `pathRoughness`.
RAB_Surface realtimePTHitSurface(RealtimeHitPayload payload, RayDesc ray, float pathRoughness)
{
    const RealtimeHit hit = unpackRealtimeHit(payload);
    const float3 view = -ray.Direction;
    const float3 geoNormal = dot(hit.geometricNormal, view) >= 0.0
        ? hit.geometricNormal : -hit.geometricNormal;
    const RealtimeBsdf bsdf = makeRealtimeBsdf(hit.normal, geoNormal, view, hit.closure);
    const float depth = max(dot(hit.position - realtimeCameraPosition(false),
        realtimeArgs()->view.cameraForward), realtimeArgs()->view.nearPlane);
    RAB_Surface surface = makeRealtimeSurface(payload, bsdf, depth);
    surface.pathRoughness = pathRoughness;
    return surface;
}

void realtimePTSampleNEE(RAB_Surface surface,
    inout RTXDI_PathTracerRandomContext random,
    out RTXDI_SampledLightData sampledLight,
    out float3 radiance, out float neePdf, out float scatterPdf,
    out RAB_LightSample lightSample)
{
    sampledLight = RTXDI_SampledLightData_CreateInvalidData();
    radiance = float3(0.0);
    neePdf = 0.0;
    scatterPdf = 0.0;
    lightSample = RAB_EmptyLightSample();
    RealtimeLighting* lighting = &realtimeArgs()->lighting;
    const RTXDI_DIReservoir reservoir = RTXDI_SampleLightsForSurface(
        random.initialRandomSamplerState,
        random.initialCoherentRandomSamplerState, surface,
        lighting.secondaryInitialSamplingParams, lighting.lightBufferParams,
        lighting.localLightsRISBufferSegmentParams,
        lighting.environmentLightRISBufferSegmentParams, lighting.regir, lightSample);
    if (!RTXDI_IsValidDIReservoir(reservoir) || lightSample.solidAnglePdf <= 0.0)
        return;
    float3 direction;
    float distance;
    RAB_GetLightDirDistance(surface, lightSample, direction, distance);
    scatterPdf = realtimeSurfaceBsdf(surface).pdf(direction);
    radiance = realtimeLightSampleReflectance(surface, direction, lightSample) * lightSample.radiance
        * (RTXDI_GetDIReservoirInvPdf(reservoir) / lightSample.solidAnglePdf)
        * realtimeLightTransmittance(surface, lightSample)
        * RAB_GetMISWeightForNEE(RTXDI_GetDIReservoirLightIndex(reservoir), lightSample,
            direction, lightSample.solidAnglePdf, scatterPdf);
    sampledLight.lightData = reservoir.lightData;
    sampledLight.uvData = reservoir.uvData;
    neePdf = 1.0 / reservoir.weightSum;
}

#define PT_CONTEXT RTXDI_PathTracerContextInitial
#include "RtxdiPTPathTraceBody.inc"
#undef PT_CONTEXT
#define PT_CONTEXT RTXDI_PathTracerContextHybrid
#include "RtxdiPTPathTraceBody.inc"
#undef PT_CONTEXT
