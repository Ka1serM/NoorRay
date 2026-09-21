#pragma once

import MaterialInterface;

#include "../Raytracer/ByteBuffer.slang"
#include "../Raytracer/RaytracingAbi.slang"
#include "../Raytracer/Ggx.slang"
#include "../Raytracer/AnalyticLights.slang"
#include "../Raytracer/Environment.slang"

// Shading for the RGB preview renderer.
//
// Surfaces are shaded from the closures their material returns
// (MaterialInterface.slang): a diffuse and a GGX reflection lobe. SVM
// programs are not evaluated. Gaussians
// are not part of the realtime renderer, and all geometry is traced as opaque.

static const float RealtimePi = 3.14159265358979323846;
static const float RealtimeInvPi = 0.31830988618379067154;

static const uint RealtimeMiss = 0u;
static const uint RealtimeSurfaceHit = 1u;

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

float realtimeFiniteLightWeight(Frame frame)
{
    return hasAnalyticLights(frame) ? max(frame.lightFiniteWeight, 0.0) : 0.0;
}

// Probability that light sampling picks the environment rather than one of
// the finite lights. Hit shaders and the miss path must agree on it. When
// RTXDI samples the analytic lights separately (`analyticLightsSeparate`),
// the environment gets a light sample of its own at every vertex.
float realtimeEnvironmentSelection(Frame frame, bool analyticLightsSeparate)
{
    if (frame.environment == nullptr)
        return 0.0;
    if (analyticLightsSeparate)
        return 1.0;
    Environment* environments = frame.environment;
    const float environmentWeight = max(environments[0].importanceWeight, 0.0);
    const float totalWeight = realtimeFiniteLightWeight(frame) + environmentWeight;
    return totalWeight > 0.0 ? environmentWeight / totalWeight : 0.0;
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

// `geometricNormal` must face the viewer. `shadingNormal` stands in for the
// closure's normal when its lobes carry none.
RealtimeBsdf makeRealtimeBsdf(BSDF closure, float3 shadingNormal, float3 geometricNormal,
    float3 view, float roughnessFloor)
{
    RealtimeBsdf bsdf;
    // Keep the shading normal in the viewer's hemisphere.
    float3 normal = closure.normal(shadingNormal);
    if (dot(normal, geometricNormal) < 0.0)
        normal = -normal;
    if (dot(normal, view) <= 1.0e-4)
        normal = geometricNormal;
    bsdf.normal = normal;
    bsdf.geometricNormal = geometricNormal;
    bsdf.view = view;
    bsdf.normalView = dot(normal, view);
    bsdf.roughness = clamp(max(closure.roughness(), roughnessFloor),
        RealtimeMinimumRoughness, 1.0);
    bsdf.diffuse = closure.diffuse;
    bsdf.specularF0 = closure.specularF0;
    bsdf.specularF90 = closure.specularF90;
    bsdf.diffuseProbability = realtimeDiffuseProbability(bsdf.diffuse,
        bsdf.specularF0, bsdf.specularF90, bsdf.normalView);
    return bsdf;
}

struct RealtimeLightSample
{
    float3 direction;
    float distance;
    // Unoccluded contribution per lobe: BSDF, cosine, radiance, pdf and MIS
    // weight.
    float3 diffuse;
    float3 specular;
    bool valid;
};

// One light sample, chosen between finite lights and the environment with
// the same weights the reference renderer uses, or of the environment alone
// when RTXDI samples the analytic lights (`analyticLightsSeparate`). Mesh
// lights are never uploaded for the preview, so emissive geometry is found by
// BSDF sampling.
RealtimeLightSample sampleRealtimeLight(Frame frame, RealtimeBsdf bsdf,
    float3 position, bool analyticLightsSeparate, inout Rng rng)
{
    RealtimeLightSample result;
    result.direction = float3(0.0);
    result.distance = 0.0;
    result.diffuse = float3(0.0);
    result.specular = float3(0.0);
    result.valid = false;

    const float environmentProbability = realtimeEnvironmentSelection(frame,
        analyticLightsSeparate);
    const float finiteWeight = analyticLightsSeparate ? 0.0
        : realtimeFiniteLightWeight(frame);
    const bool chooseFinite = finiteWeight > 0.0
        && (environmentProbability <= 0.0 || rng.nextFloat() >= environmentProbability);
    float3 radianceScale;
    if (chooseFinite)
    {
        const AnalyticLightSampleRgb light = sampleAnalyticLightRgb(
            makeLightTable(frame), position, rng);
        if (!light.valid || light.mesh)
            return result;
        result.direction = light.direction;
        result.distance = light.distance;
        radianceScale = light.radianceOverSelectionPdf
            / max(1.0 - environmentProbability, 1.0e-20);
    }
    else if (environmentProbability > 0.0)
    {
        Environment* environments = frame.environment;
        const Environment environment = environments[0];
        const EnvironmentSample sample = sampleEnvironmentDirection(environment, rng);
        const float lightPdf = environmentProbability * sample.pdf;
        if (!(lightPdf > 0.0))
            return result;
        result.direction = sample.direction;
        result.distance = 1.0e16;
        radianceScale = environmentRadianceRgb(environment, sample.direction, false)
            / lightPdf * powerHeuristic(lightPdf, bsdf.pdf(sample.direction));
    }
    else
        return result;

    result.diffuse = bsdf.evaluateDiffuse(result.direction) * radianceScale;
    result.specular = bsdf.evaluateSpecular(result.direction) * radianceScale;
    const float3 total = result.diffuse + result.specular;
    result.valid = realtimeIsFinite(total) && realtimeMaxComponent(total) > 0.0;
    return result;
}
