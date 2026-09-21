#pragma once

#include "RealtimeArgs.slang"
#include "NRD.hlsli"

// NRD's diffuse and specular signals in the encoding of the active denoiser:
// REBLUR normalizes the hit distance, RELAX takes it in world units. With the
// denoiser off the composite reads the inputs directly, in REBLUR's encoding.
float4 packNrdSignal(float3 radiance, float hitDistance, float viewZ,
    float roughness)
{
    if (realtimeArgs()->denoiser.relax != 0u)
        return RELAX_FrontEnd_PackRadianceAndHitDist(radiance, hitDistance, true);
    return REBLUR_FrontEnd_PackRadianceAndNormHitDist(radiance,
        REBLUR_FrontEnd_GetNormHitDist(hitDistance, viewZ, realtimeArgs()->denoiser.hitDistanceParameters,
            roughness), true);
}

float3 unpackNrdRadiance(float4 signal)
{
    if (realtimeArgs()->denoiser.relax != 0u)
        return RELAX_BackEnd_UnpackRadiance(signal).rgb;
    return REBLUR_BackEnd_UnpackRadianceAndNormHitDist(signal).rgb;
}
