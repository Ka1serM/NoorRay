#pragma once

// ReSTIR PT application callbacks. RTXDI owns path sampling, hybrid shift,
// temporal and spatial resampling; these callbacks expose NoorRay's BSDF,
// ray traversal, visibility and G-buffer to the SDK.

#include "RtxdiBridge.slang"

#define RTXDI_PT_RESERVOIR_BUFFER (realtimeArgs()->lighting.ptReservoirs)
#include "Rtxdi/PT/Reservoir.hlsli"

float3 RAB_GetReflectedBsdfRadianceForSurface(float3 samplePosition,
    float3 radiance, RAB_Surface surface)
{
    const float3 delta = samplePosition - surface.worldPos;
    const float distance = length(delta);
    return distance > 1.0e-5 ? realtimeSurfaceReflectance(surface, delta / distance)
        * radiance : float3(0.0);
}

float3 RAB_GetPTSampleTargetPdfForSurface(float3 samplePosition,
    float3 radiance, RAB_Surface surface)
{
    return RAB_GetReflectedBsdfRadianceForSurface(samplePosition, radiance, surface);
}

// RTXDI keeps each path segment's trace result for the rest of the path but
// reads only its hit distance; the path tracer uses the hit itself right away.
struct RAB_RayPayload
{
    float committedRayT;
};

float RAB_RayPayloadGetCommittedHitT(RAB_RayPayload payload)
{
    return payload.committedRayT;
}

struct RAB_PathTracerUserData
{
    uint pathType;
};

RAB_PathTracerUserData RAB_EmptyPathTracerUserData()
{
    RAB_PathTracerUserData data = {};
    return data;
}

void RAB_PathTracerUserDataSetPathType(inout RAB_PathTracerUserData data, uint type)
{
    data.pathType = type;
}

uint RAB_GetDuplicationMapCount(int2 pixel) { return 0u; }

// Analytic light geometry is absent from the mesh ray traversal, so only
// light samples find it; the environment is also found by BSDF rays.
float RAB_GetMISWeightForNEE(uint lightIndex, RAB_LightSample sample,
    float3 direction, float lightPdf, float scatterPdf)
{
    return sample.kind == RealtimeLightEnvironment
        ? realtimeEnvironmentLightWeight(direction) : 1.0;
}

void RAB_LastBounceDenoiserCallback(float3 lightPosition, RAB_Surface surface,
    inout RAB_PathTracerUserData data) {}

void RAB_ReconnectionDenoiserCallback(RTXDI_PTReservoir reservoir,
    RAB_Surface surface, inout RAB_PathTracerUserData data) {}
