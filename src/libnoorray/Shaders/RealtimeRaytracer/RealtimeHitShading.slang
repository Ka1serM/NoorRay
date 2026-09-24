#include "RtxdiBridge.slang"
#include "Rtxdi/DI/InitialSampling.hlsli"

import MaterialInterface;

// One resolved scattering event of a path: where a realtime ray landed, the
// light sampled there and the continuation. Every radiometric value is linear
// Rec.709 RGB.
struct RealtimeVertex
{
    uint hit;
    uint instance;
    float t;
    uint seed;
    uint depth;
    // Regularizes paths after glossy scattering.
    float roughnessFloor;
    // Set by the caller when ReSTIR DI lights this hit's analytic lights in
    // screen space, so shading samples only the environment.
    uint screenSpaceLights;
    float3 position;
    float3 normal;
    float3 geometricNormal;
    float3 albedo;
    float3 emission;
    // Material terms the denoiser demodulates by.
    float3 diffuseAlbedo;
    float3 specularF0;
    float3 specularF90;
    float roughness;
    // Unoccluded light-sample contribution, split by lobe.
    float3 directDiffuse;
    float3 directSpecular;
    float3 shadowDirection;
    float shadowDistance;
    uint shadowValid;
    // A second, independently shadowed light sample: the analytic lights
    // resampled through RTXDI's world-space RIS when RTXDI owns them (the
    // sample above is then the environment's alone). Primary hits leave it
    // empty; ReSTIR DI lights them in screen space instead.
    float3 lightDiffuse;
    float3 lightSpecular;
    float3 lightDirection;
    float lightDistance;
    uint lightValid;
    float3 bsdfWeight;
    float3 nextDirection;
    uint sampleValid;
    // Set when the continuation came from the specular lobe.
    uint sampledSpecular;
    // Solid-angle pdf of nextDirection for multiple importance sampling
    // against light sampling.
    float bsdfPdf;
    float sampledRoughness;
};

// World-space light sampling at a path vertex beyond the G-buffer: RIS over
// the ReGIR cell the vertex falls in, so each vertex weighs several analytic
// lights by what they actually deliver there instead of drawing one by power.
// This is the direct light the SHaRC update paths feed the cache.
void sampleRtxdiLight(inout RealtimeVertex vertex, RealtimeBsdf bsdf, inout Rng rng)
{
    RealtimeLighting* lighting = &realtimeArgs()->lighting;
    RTXDI_RandomSamplerState rtxdiRng;
    rtxdiRng.seed = pcgHash(rng.state ^ 0x52545844u);
    rtxdiRng.index = 1u;
    RTXDI_RandomSamplerState coherentRng = rtxdiRng;
    coherentRng.seed = pcgHash(rtxdiRng.seed);
    // Any positive depth marks the surface valid.
    const RAB_Surface surface = makeRealtimeSurface(bsdf, vertex.position, 1.0);
    RAB_LightSample lightSample;
    const RTXDI_DIReservoir reservoir = RTXDI_SampleLightsForSurface(rtxdiRng,
        coherentRng, surface, lighting.secondaryInitialSamplingParams,
        lighting.lightBufferParams, lighting.localLightsRISBufferSegmentParams,
        lighting.environmentLightRISBufferSegmentParams, lighting.regir, lightSample);
    if (!RTXDI_IsValidDIReservoir(reservoir) || !(lightSample.solidAnglePdf > 0.0))
        return;
    float3 direction;
    float distance;
    RAB_GetLightDirDistance(surface, lightSample, direction, distance);
    const float3 incoming = lightSample.radiance
        * (RTXDI_GetDIReservoirInvPdf(reservoir) / lightSample.solidAnglePdf);
    vertex.lightDiffuse = bsdf.evaluateDiffuse(direction) * incoming;
    vertex.lightSpecular = bsdf.evaluateSpecular(direction) * incoming;
    const float3 total = vertex.lightDiffuse + vertex.lightSpecular;
    if (!realtimeIsFinite(total) || !(realtimeMaxComponent(total) > 0.0))
        return;
    vertex.lightDirection = direction;
    vertex.lightDistance = distance;
    vertex.lightValid = 1u;
}

// Shades what a material's closest hit reported (MaterialHit.slang): samples
// the lights and one continuation direction at the hit. A miss comes back
// with only its hit, t and the caller's state set.
RealtimeVertex shadeRealtimeHit(RealtimeHitPayload hit, float3 rayDirection, uint depth,
    uint seed, float roughnessFloor, bool screenSpaceLights)
{
    const Frame frame = realtimeArgs()->frame;
    RealtimeVertex vertex = {};
    vertex.hit = hit.hit;
    vertex.instance = hit.instance;
    vertex.t = hit.t;
    vertex.seed = seed;
    vertex.depth = depth;
    vertex.roughnessFloor = roughnessFloor;
    vertex.screenSpaceLights = screenSpaceLights ? 1u : 0u;
    if (hit.hit == RealtimeMiss)
        return vertex;

    const float3 view = -normalize(rayDirection);
    // Surfaces are two-sided: shade the side the ray arrived from.
    const float3 geometricNormal = dot(view, hit.geometricNormal) < 0.0
        ? -hit.geometricNormal : hit.geometricNormal;
    BSDF closure = BSDF();
    closure.diffuse = hit.diffuse;
    closure.specularF0 = hit.specularF0;
    closure.specularF90 = hit.specularF90;
    closure.roughnessSum = hit.roughnessSum;
    closure.roughnessWeight = hit.roughnessWeight;
    closure.normalSum = hit.normalSum;
    RealtimeBsdf bsdf = makeRealtimeBsdf(closure, hit.shadingNormal,
        geometricNormal, view, vertex.roughnessFloor);
    if (vertex.depth == 0u)
        bsdf.diffuseProbability = realtimePrimaryDiffuseProbability(bsdf.diffuseProbability);

    vertex.emission = max(hit.emission, 0.0);
    vertex.position = hit.position;
    vertex.geometricNormal = geometricNormal;
    vertex.normal = bsdf.normal;
    vertex.albedo = bsdf.diffuse + bsdf.specularF0;
    vertex.diffuseAlbedo = bsdf.diffuse;
    vertex.specularF0 = bsdf.specularF0;
    vertex.specularF90 = bsdf.specularF90;
    vertex.roughness = bsdf.roughness;

    Rng rng;
    rng.state = vertex.seed ^ (0x85EBCA6Bu * (vertex.depth + 1u));
    // With RTXDI the first sample is the environment's alone and the analytic
    // lights get their own; the image pass's primary hits leave the analytic
    // lights to ReSTIR DI, which resamples them in screen space.
    const bool rtxdi = realtimeRtxdiEnabled();
    const RealtimeLightSample light = sampleRealtimeLight(frame, bsdf,
        vertex.position, rtxdi, rng);
    if (light.valid)
    {
        vertex.directDiffuse = light.diffuse;
        vertex.directSpecular = light.specular;
        vertex.shadowDirection = light.direction;
        vertex.shadowDistance = light.distance;
        vertex.shadowValid = 1u;
    }
    if (rtxdi && vertex.screenSpaceLights == 0u && hasAnalyticLights(frame))
        sampleRtxdiLight(vertex, bsdf, rng);

    const float lobeSample = rng.nextFloat();
    const RealtimeBsdfSample sample = bsdf.sample(lobeSample, rng.nextFloat2());
    vertex.seed = rng.state;
    if (!sample.valid)
        return vertex;
    vertex.bsdfWeight = sample.weight;
    vertex.nextDirection = sample.direction;
    vertex.bsdfPdf = sample.pdf;
    vertex.sampledRoughness = sample.roughness;
    vertex.sampledSpecular = sample.specular ? 1u : 0u;
    vertex.sampleValid = 1u;
    return vertex;
}
