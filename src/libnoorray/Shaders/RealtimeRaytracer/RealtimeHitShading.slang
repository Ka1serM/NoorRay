#pragma once

#include "RtxdiBridge.slang"
#include "Rtxdi/DI/InitialSampling.hlsli"

import MaterialInterface;

// The light a surface outside the G-buffer reflects toward a ray: the
// vertices of the radiance cache's paths. Every radiometric value is linear
// Rec.709 RGB.
struct RealtimeVertex
{
    uint seed;
    float3 position;
    float3 geometricNormal;
    float3 emission;
    // One light sample over every light, the environment included, which the
    // caller shadow-tests.
    RealtimeLightSample light;
};

// World-space light sampling outside the G-buffer: RIS over the ReGIR cell
// the vertex falls in and the environment, so each vertex weighs several
// lights by what they actually deliver there instead of drawing one by power.
RealtimeLightSample sampleRtxdiLight(RAB_Surface surface, inout Rng rng)
{
    RealtimeLightSample result;
    result.valid = false;
    result.environment = false;
    RealtimeLighting* lighting = &realtimeArgs()->lighting;
    RTXDI_RandomSamplerState rtxdiRng;
    rtxdiRng.seed = pcgHash(rng.state ^ 0x52545844u);
    rtxdiRng.index = 1u;
    RTXDI_RandomSamplerState coherentRng = rtxdiRng;
    coherentRng.seed = pcgHash(rtxdiRng.seed);
    RAB_LightSample lightSample;
    const RTXDI_DIReservoir reservoir = RTXDI_SampleLightsForSurface(rtxdiRng,
        coherentRng, surface, lighting.secondaryInitialSamplingParams,
        lighting.lightBufferParams, lighting.localLightsRISBufferSegmentParams,
        lighting.environmentLightRISBufferSegmentParams, lighting.regir, lightSample);
    if (!RTXDI_IsValidDIReservoir(reservoir) || !(lightSample.solidAnglePdf > 0.0))
        return result;
    RAB_GetLightDirDistance(surface, lightSample, result.direction, result.distance);
    result.environment = lightSample.kind == RealtimeLightEnvironment;
    result.diffuseScale = lightSample.diffuseScale;
    result.specularScale = lightSample.specularScale;
    result.castShadows = lightSample.castShadows != 0u;
    result.radiance = lightSample.radiance
        * (RTXDI_GetDIReservoirInvPdf(reservoir) / lightSample.solidAnglePdf);
    result.valid = realtimeIsFinite(result.radiance) && realtimeMaxComponent(result.radiance) > 0.0;
    return result;
}

float3 shadowedLight(RealtimeVertex vertex, RealtimeBsdf bsdf)
{
    if (!vertex.light.valid)
        return float3(0.0);
    const float visibility = vertex.light.castShadows ? realtimeTransmittance(vertex.position,
        vertex.geometricNormal, vertex.light.direction, vertex.light.distance) : 1.0;
    return (bsdf.evaluateDiffuse(vertex.light.direction) * vertex.light.diffuseScale
        + bsdf.evaluateSpecular(vertex.light.direction) * vertex.light.specularScale)
        * vertex.light.radiance * visibility;
}

// The BSDF at a surface hit. Surfaces are two-sided: it shades the side the
// ray arrived from.
RealtimeBsdf realtimeHitBsdf(RealtimeHit hit, float3 rayDirection)
{
    const float3 view = -normalize(rayDirection);
    const float3 geometricNormal = dot(view, hit.geometricNormal) < 0.0
        ? -hit.geometricNormal : hit.geometricNormal;
    return makeRealtimeBsdf(hit.normal, geometricNormal, view, hit.closure);
}

// Shades what a material's closest hit reported (MaterialHit.slang): its
// emission and one light sample. `pathRoughness` is the path's roughness on
// arrival (RAB_Surface::pathRoughness).
RealtimeVertex shadeRealtimeHit(RealtimeHitPayload payload, float3 rayDirection, uint seed,
    float pathRoughness)
{
    const RealtimeHit hit = unpackRealtimeHit(payload);
    const RealtimeBsdf bsdf = realtimeHitBsdf(hit, rayDirection);
    RealtimeVertex vertex;
    vertex.position = hit.position;
    vertex.geometricNormal = bsdf.geometricNormal;
    vertex.emission = max(hit.emission, 0.0);
    Rng rng;
    rng.state = seed ^ 0x85EBCA6Bu;
    // Any positive depth marks the surface valid.
    RAB_Surface surface = makeRealtimeSurface(payload, bsdf, 1.0);
    surface.pathRoughness = pathRoughness;
    vertex.light = sampleRtxdiLight(surface, rng);
    vertex.seed = rng.state;
    return vertex;
}
