#pragma once

import MaterialInterface;

#include "../Raytracer/ByteBuffer.slang"
#include "HitPayload.slang"
#include "../Raytracer/Ggx.slang"
#include "../Raytracer/AnalyticLights.slang"
#include "../Raytracer/Environment.slang"

// Shading for the RGB preview renderer.
//
// Surfaces are shaded from the closures their material returns
// (MaterialInterface.slang): a diffuse and a GGX reflection lobe. Gaussians
// are not part of the realtime renderer; opacity is applied during traversal
// by the materials' any-hit stages (MaterialHit.slang).

static const float RealtimePi = 3.14159265358979323846;
static const float RealtimeInvPi = 0.31830988618379067154;

// GGX lobes are never sharper than this, so every scattering event is
// covered by light sampling.
static const float RealtimeMinimumRoughness = 0.05;

float realtimeLuminance(float3 value)
{
    return dot(value, float3(0.2126, 0.7152, 0.0722));
}

float realtimeMaxComponent(float3 value)
{
    return max(value.x, max(value.y, value.z));
}

bool realtimeIsFinite(float3 value)
{
    return !any(isnan(value)) && !any(isinf(value));
}

// RGB counterpart of clampIndirectContribution(): the camera-visible
// contribution is never clamped.
float3 clampIndirectRgb(float3 contribution, float clampValue, uint depth)
{
    if (depth == 0u || !(clampValue > 0.0))
        return contribution;
    const float limit = clampValue * 3.0;
    const float sum = dot(abs(contribution), float3(1.0));
    return sum > limit ? contribution * (limit / sum) : contribution;
}

// Column-major matrix times a point, for Frame's clip matrices.
float4 transformToClip(float matrix[16], float3 position)
{
    const float4 p = float4(position, 1.0);
    return float4(
        dot(p, float4(matrix[0], matrix[4], matrix[8], matrix[12])),
        dot(p, float4(matrix[1], matrix[5], matrix[9], matrix[13])),
        dot(p, float4(matrix[2], matrix[6], matrix[10], matrix[14])),
        dot(p, float4(matrix[3], matrix[7], matrix[11], matrix[15])));
}

float realtimeSchlickWeight(float cosine)
{
    const float m = saturate(1.0 - cosine);
    const float m2 = m * m;
    return m2 * m2 * m;
}

struct RealtimeBsdfSample
{
    float3 direction;
    float3 weight;
    // Mixture pdf, for MIS against light sampling.
    float pdf;
    float roughness;
    bool specular;
    bool valid;
};

// A BSDF closure's Lambertian diffuse and GGX reflection lobes at one
// shading point.
struct RealtimeBsdf
{
    float3 normal;
    float3 geometricNormal;
    float3 view;
    float normalView;
    float3 diffuse;
    float3 specularF0;
    float3 specularF90;
    float roughness;
    float diffuseProbability;

    float3 fresnel(float cosine)
    {
        return specularF0 + (specularF90 - specularF0) * realtimeSchlickWeight(cosine);
    }

    bool above(float3 direction)
    {
        return dot(normal, direction) > 0.0 && normalView > 0.0
            && dot(geometricNormal, direction) > 0.0;
    }

    // Diffuse BSDF times the cosine term.
    float3 evaluateDiffuse(float3 direction)
    {
        return above(direction)
            ? diffuse * (dot(normal, direction) * RealtimeInvPi) : float3(0.0);
    }

    // Specular BSDF times the cosine term.
    float3 evaluateSpecular(float3 direction)
    {
        if (!above(direction))
            return float3(0.0);
        const float3 halfVector = normalize(view + direction);
        return fresnel(saturate(dot(view, halfVector)))
            * (ggxDistribution(normal, halfVector, roughness)
                * smithG2(normalView, dot(normal, direction), roughness)
                / (4.0 * normalView));
    }

    float diffusePdf(float3 direction)
    {
        return max(dot(normal, direction), 0.0) * RealtimeInvPi;
    }

    float specularPdf(float3 direction)
    {
        return dot(normal, direction) > 0.0 ? ggxReflectionPdf(normal, view,
            normalize(view + direction), roughness) : 0.0;
    }

    // Solid-angle density with which sample() produces `direction`.
    float pdf(float3 direction)
    {
        if (normalView <= 0.0)
            return 0.0;
        return diffuseProbability * diffusePdf(direction)
            + (1.0 - diffuseProbability) * specularPdf(direction);
    }

    // One lobe per sample, weighted by that lobe alone, so the continuation
    // is attributable to the diffuse or the specular signal. Lobe selection
    // keeps the estimator unbiased; MIS still uses the mixture pdf.
    RealtimeBsdfSample sample(float lobeSample, float2 directionSample)
    {
        RealtimeBsdfSample result;
        result.direction = float3(0.0);
        result.weight = float3(0.0);
        result.pdf = 0.0;
        result.roughness = 1.0;
        result.specular = false;
        result.valid = false;
        if (normalView <= 0.0)
            return result;

        float3 value;
        float lobePdf;
        if (lobeSample < diffuseProbability)
        {
            result.direction = cosineHemisphere(normal, directionSample);
            value = evaluateDiffuse(result.direction);
            lobePdf = diffuseProbability * diffusePdf(result.direction);
        }
        else
        {
            result.direction = reflect(-view, sampleVisibleNormal(normal, view,
                roughness, directionSample));
            result.roughness = roughness;
            result.specular = true;
            value = evaluateSpecular(result.direction);
            lobePdf = (1.0 - diffuseProbability) * specularPdf(result.direction);
        }
        result.pdf = pdf(result.direction);
        if (!(lobePdf > 0.0) || !(result.pdf > 0.0))
            return result;
        result.weight = value / lobePdf;
        result.valid = realtimeIsFinite(result.weight)
            && realtimeMaxComponent(result.weight) > 0.0;
        return result;
    }
};

// Lobe-selection probability of the sampler: the lobes' albedo estimates.
float realtimeDiffuseProbability(float3 diffuse, float3 specularF0, float3 specularF90,
    float normalView)
{
    const float diffuseWeight = realtimeLuminance(diffuse);
    const float specularWeight = realtimeLuminance(specularF0
        + (specularF90 - specularF0) * realtimeSchlickWeight(saturate(normalView)));
    const float totalWeight = diffuseWeight + specularWeight;
    return totalWeight > 0.0 ? diffuseWeight / totalWeight : 0.0;
}

// NRD reconstructs the hit distance of the lobe a primary path did not take
// from its 3x3 neighbours, which needs both lobes chosen often.
float realtimePrimaryDiffuseProbability(float diffuseProbability)
{
    return clamp(diffuseProbability, 0.25, 0.75);
}

float realtimeBsdfRoughness(float roughness)
{
    return clamp(roughness, RealtimeMinimumRoughness, 1.0);
}

// `geometricNormal` must face the viewer.
RealtimeBsdf makeRealtimeBsdf(float3 shadingNormal, float3 geometricNormal, float3 view,
    RealtimeClosure closure)
{
    RealtimeBsdf bsdf;
    // Keep the shading normal in the viewer's hemisphere.
    float3 normal = shadingNormal;
    if (dot(normal, geometricNormal) < 0.0)
        normal = -normal;
    normal = clampShadingNormal(geometricNormal, normal, view);
    bsdf.normal = normal;
    bsdf.geometricNormal = geometricNormal;
    bsdf.view = view;
    bsdf.normalView = dot(normal, view);
    bsdf.roughness = realtimeBsdfRoughness(closure.roughness);
    bsdf.diffuse = closure.diffuse;
    bsdf.specularF0 = closure.specularF0;
    bsdf.specularF90 = closure.specularF90;
    bsdf.diffuseProbability = realtimeDiffuseProbability(bsdf.diffuse,
        bsdf.specularF0, bsdf.specularF90, bsdf.normalView);
    return bsdf;
}

struct RealtimeLightSample
{
    // Incoming radiance, unoccluded, over the sample's pdf.
    float3 radiance;
    float3 direction;
    float distance;
    bool environment;
    bool valid;
    // The light's scale of the diffuse and specular lobes.
    float diffuseScale;
    float specularScale;
    bool castShadows;
};
