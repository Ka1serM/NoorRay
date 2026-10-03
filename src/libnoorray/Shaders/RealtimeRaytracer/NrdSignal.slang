#pragma once

#include "RealtimeArgs.slang"
#include "NRD.hlsli"

// RELAX diffuse/specular signals use hit distances in world units. When off,
// the composite reads the same packed inputs directly.
//
// The SH denoisers also take each lobe's lighting direction. A lobe sums
// several contributions (indirect bounce, resampled lights), so the caller
// accumulates their directions weighted by luminance; NRD's first SH band of
// the sum is exactly that accumulation, normalized here by the total.

float nrdLuminance(float3 radiance)
{
    return dot(radiance, float3(0.2126, 0.7152, 0.0722));
}

// The luminance-weighted direction of `radiance` arriving from `direction`,
// to be summed over a lobe's contributions.
float3 nrdDirectionSum(float3 radiance, float3 direction)
{
    return direction * nrdLuminance(radiance);
}

struct RealtimeLobe
{
    float3 radiance;
    float3 direction;
    float directLuminance;
    float directHitDistance;
    float indirectLuminance;
    float indirectHitDistance;

    [mutating]
    bool add(float3 contribution, float3 lightDirection)
    {
        if (!(max(contribution.x, max(contribution.y, contribution.z)) > 0.0)
            || any(isnan(contribution)) || any(isinf(contribution)))
            return false;
        radiance += contribution;
        direction += nrdDirectionSum(contribution, lightDirection);
        return true;
    }

    [mutating]
    void addDirect(float3 contribution, float3 lightDirection, float distance)
    {
        if (!add(contribution, lightDirection))
            return;
        directLuminance = nrdLuminance(contribution);
        directHitDistance = distance;
    }

    [mutating]
    void addIndirect(float3 contribution, float3 lightDirection, float distance, bool inLobe)
    {
        if (!add(contribution, lightDirection))
            return;
        indirectLuminance = nrdLuminance(contribution);
        if (inLobe)
            indirectHitDistance = distance;
    }

    float hitDistance()
    {
        if (directHitDistance <= 0.0)
            return indirectHitDistance;
        if (indirectHitDistance <= 0.0)
            return directHitDistance;
        const float directShare = min(directLuminance
            / max(directLuminance + indirectLuminance, 1.0e-6), 0.5);
        return lerp(indirectHitDistance, directHitDistance, directShare);
    }
};

// Packs one lobe. `radiance` is demodulated; `directionSum` is the sum of
// nrdDirectionSum() over the same contributions before demodulation, and
// `totalRadiance` their undemodulated sum. `sh1` is only written in SH mode.
float4 packNrdSignal(float3 radiance, float hitDistance, float viewZ, float roughness,
    float3 directionSum, float3 totalRadiance, out float4 sh1)
{
    sh1 = float4(0.0);
    const DenoiserArgs denoiser = realtimeArgs()->denoiser;
    const float3 exposed = radiance * realtimeExposureScale();
    if (denoiser.sphericalHarmonics == 0u)
        return RELAX_FrontEnd_PackRadianceAndHitDist(exposed, hitDistance, true);
    const float3 direction = directionSum / max(nrdLuminance(totalRadiance), 1.0e-6);
    return RELAX_FrontEnd_PackSh(exposed, hitDistance, direction, sh1, true);
}

float4 packNrdPlainSignal(float3 radiance, float hitDistance, float viewZ, float roughness)
{
    return RELAX_FrontEnd_PackRadianceAndHitDist(radiance * realtimeExposureScale(),
        hitDistance, true);
}

// A lobe's radiance as the non-SH denoisers pack and output it.
float3 unpackNrdRadiance(float4 signal)
{
    return RELAX_BackEnd_UnpackRadiance(signal).rgb;
}

NRD_SG unpackNrdSh(float4 sh0, float4 sh1)
{
    return RELAX_BackEnd_UnpackSh(sh0, sh1.xyz);
}
