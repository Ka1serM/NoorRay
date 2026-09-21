#include "RtxdiBridge.slang"
#include "../Raytracer/SceneSurface.slang"
#include "Rtxdi/DI/InitialSampling.hlsli"

import MaterialInterface;

// Shading of surfaces whose material has no compiled shader: a neutral grey
// dielectric.
surfaceshader realtimeDefaultSurface(float3 normal)
{
    return mx_surface(mx_layer_bsdf(
            mx_dielectric_bsdf(1.0, float3(1.0), 1.5, float2(0.25), normal, 0),
            mx_oren_nayar_diffuse_bsdf(1.0, float3(0.8), 0.0, normal, false)),
        float3(0.0), 1.0);
}

// Evaluates the surface's material: its callable shader in the shader
// binding table, compiled from the material's MaterialX document.
surfaceshader evaluateMaterial(Material* record, RegularSurface surface)
{
    if (record == nullptr || record->shader == ~0u)
        return realtimeDefaultSurface(surface.shadingNormal);
    MaterialCall call;
    call.geometry.position = surface.position;
    call.geometry.normal = surface.shadingNormal;
    call.geometry.tangent = surface.tangent;
    call.geometry.bitangent = normalize(cross(surface.shadingNormal, surface.tangent))
        * surface.tangentSign;
    call.geometry.uv = surface.uv;
    call.geometry.color = surface.vertexColor;
    call.parameters = (uint64_t)record->shaderParameters;
    call.surface = surfaceshader();
    CallShader(record->shader, call);
    return call.surface;
}

// World-space light sampling at a path vertex beyond the G-buffer: RIS over
// the ReGIR cell the vertex falls in, so each vertex weighs several analytic
// lights by what they actually deliver there instead of drawing one by power.
// This is the direct light the SHaRC update paths feed the cache.
void sampleRtxdiLight(inout RealtimeRayPayload payload, RealtimeBsdf bsdf, inout Rng rng)
{
    RealtimeLighting* lighting = &realtimeArgs()->lighting;
    RTXDI_RandomSamplerState rtxdiRng;
    rtxdiRng.seed = pcgHash(rng.state ^ 0x52545844u);
    rtxdiRng.index = 1u;
    RTXDI_RandomSamplerState coherentRng = rtxdiRng;
    coherentRng.seed = pcgHash(rtxdiRng.seed);
    // Any positive depth marks the surface valid.
    const RAB_Surface surface = makeRealtimeSurface(bsdf, payload.position, 1.0);
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
    payload.lightDiffuse = bsdf.evaluateDiffuse(direction) * incoming;
    payload.lightSpecular = bsdf.evaluateSpecular(direction) * incoming;
    const float3 total = payload.lightDiffuse + payload.lightSpecular;
    if (!realtimeIsFinite(total) || !(realtimeMaxComponent(total) > 0.0))
        return;
    payload.lightDirection = direction;
    payload.lightDistance = distance;
    payload.lightValid = 1u;
}

// Triangle closest hit: decode the surface, call its material for the
// closures to shade, and return its light samples plus one continuation direction.
[shader("closesthit")]
void main(inout RealtimeRayPayload payload,
    in BuiltInTriangleIntersectionAttributes attributes)
{
    const Frame frame = realtimeArgs()->frame;
    payload.hit = RealtimeSurfaceHit;
    payload.instance = InstanceID();
    payload.t = RayTCurrent();
    payload.directDiffuse = float3(0.0);
    payload.directSpecular = float3(0.0);
    payload.shadowValid = 0u;
    payload.lightDiffuse = float3(0.0);
    payload.lightSpecular = float3(0.0);
    payload.lightValid = 0u;
    payload.sampleValid = 0u;
    payload.bsdfPdf = 0.0;

    const float3 view = -normalize(WorldRayDirection());
    const RegularSurface surface = loadRegularSurface(frame.scene,
        payload.instance, PrimitiveIndex(), attributes.barycentrics);
    // Surfaces are two-sided: shade the side the ray arrived from.
    const float3 geometricNormal = dot(view, surface.geometricNormal) < 0.0
        ? -surface.geometricNormal : surface.geometricNormal;
    const surfaceshader shaded = evaluateMaterial(frame.materials[surface.materialIndex],
        surface);
    RealtimeBsdf bsdf = makeRealtimeBsdf(shaded.bsdf, surface.shadingNormal,
        geometricNormal, view, payload.roughnessFloor);
    if (payload.depth == 0u)
        bsdf.diffuseProbability = realtimePrimaryDiffuseProbability(bsdf.diffuseProbability);

    payload.emission = max(shaded.emission, 0.0);
    payload.position = surface.position;
    payload.geometricNormal = geometricNormal;
    payload.normal = bsdf.normal;
    payload.albedo = bsdf.diffuse + bsdf.specularF0;
    payload.diffuseAlbedo = bsdf.diffuse;
    payload.specularF0 = bsdf.specularF0;
    payload.specularF90 = bsdf.specularF90;
    payload.roughness = bsdf.roughness;

    Rng rng;
    rng.state = payload.seed ^ (0x85EBCA6Bu * (payload.depth + 1u));
    // With RTXDI the first sample is the environment's alone and the analytic
    // lights get their own; the image pass's primary hits leave the analytic
    // lights to ReSTIR DI, which resamples them in screen space.
    const bool rtxdi = realtimeRtxdiEnabled();
    const RealtimeLightSample light = sampleRealtimeLight(frame, bsdf,
        payload.position, rtxdi, rng);
    if (light.valid)
    {
        payload.directDiffuse = light.diffuse;
        payload.directSpecular = light.specular;
        payload.shadowDirection = light.direction;
        payload.shadowDistance = light.distance;
        payload.shadowValid = 1u;
    }
    if (rtxdi && payload.screenSpaceLights == 0u && hasAnalyticLights(frame))
        sampleRtxdiLight(payload, bsdf, rng);

    const float lobeSample = rng.nextFloat();
    const RealtimeBsdfSample sample = bsdf.sample(lobeSample, rng.nextFloat2());
    payload.seed = rng.state;
    if (!sample.valid)
        return;
    payload.bsdfWeight = sample.weight;
    payload.nextDirection = sample.direction;
    payload.bsdfPdf = sample.pdf;
    payload.sampledRoughness = sample.roughness;
    payload.sampledSpecular = sample.specular ? 1u : 0u;
    payload.sampleValid = 1u;
}
