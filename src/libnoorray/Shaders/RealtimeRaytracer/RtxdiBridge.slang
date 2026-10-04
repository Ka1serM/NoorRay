#pragma once

// RTXDI application bridge (RAB_*) for the realtime renderer.
//
// RTXDI (external/RTXDI-Library) supplies ReSTIR's resampling math as HLSL
// headers and calls back into the application for everything that depends on
// the renderer: surfaces, materials, lights and visibility. This file is that
// application side. It must be included before any Rtxdi/*.hlsli header.
//
// - Surfaces come from the realtime G-buffer (RealtimeSurface records the
//   primary pass writes, ping-ponged for the previous frame) and are shaded
//   with the same diffuse + GGX model RealtimeShading.slang uses.
// - Lights are the scene's analytic lights, addressed in place: local lights
//   are the point, spot and rect records in that order, infinite lights the
//   directional ones. The environment follows them as RTXDI's environment
//   light, so ReSTIR DI owns all direct light at primary surfaces and
//   balances the environment against its BRDF samples itself. Its light
//   sample parameters are environmentUv() coordinates.
// - Buffers are device pointers in RealtimeLighting, so nothing is bound.

#ifndef RTXDI_REGIR_MODE
#define RTXDI_REGIR_MODE RTXDI_REGIR_GRID
#endif
// The spatial headers default this, but DI/TemporalResampling.hlsli reads it
// without a default, which compiles temporal bias correction out entirely.
// Restir.cpp selects ray-traced bias correction for every pass.
#define RTXDI_ALLOWED_BIAS_CORRECTION RTXDI_BIAS_CORRECTION_RAY_TRACED

#include "RealtimeShading.slang"
#include "../Raytracer/RayOrigin.slang"
#include "RealtimeArgs.slang"

#define RTXDI_RIS_BUFFER (realtimeArgs()->lighting.risBuffer)
#define RTXDI_LIGHT_RESERVOIR_BUFFER (realtimeArgs()->lighting.diReservoirs)
#define RTXDI_NEIGHBOR_OFFSETS_BUFFER (realtimeArgs()->lighting.neighborOffsets)

#include "Rtxdi/Utils/Color.hlsli"
#include "Rtxdi/Utils/Math.hlsli"
#include "Rtxdi/Utils/RandomSamplerState.hlsli"
#include "Rtxdi/Utils/TypePacking.hlsli"
#include "Rtxdi/Utils/ReservoirAddressing.hlsli"

static const uint RealtimeLightPoint = 0u;
static const uint RealtimeLightSpot = 1u;
static const uint RealtimeLightRect = 2u;
static const uint RealtimeLightDirectional = 3u;
static const uint RealtimeLightEnvironment = 4u;
static const uint RealtimeLightNone = 0xFFFFFFFFu;

// Infinite light samples sit this far along their direction; shadow rays to
// them are unbounded.
static const float RealtimeDistantLightDistance = 1.0e16;
#define RAB_DISTANT_LIGHT_DISTANCE RealtimeDistantLightDistance

// ---------------------------------------------------------------------------
// Materials and surfaces
// ---------------------------------------------------------------------------

typealias RAB_Material = RealtimeClosure;

RAB_Material RAB_EmptyMaterial()
{
    RAB_Material material;
    material.diffuse = float3(0.0);
    material.roughness = 1.0;
    material.specularF0 = float3(0.0);
    material.specularF90 = float3(0.0);
    return material;
}

bool RAB_AreMaterialsSimilar(RAB_Material a, RAB_Material b)
{
    const float roughnessThreshold = 0.5;
    const float reflectivityThreshold = 0.25;
    const float albedoThreshold = 0.25;
    if (!RTXDI_CompareRelativeDifference(a.roughness, b.roughness, roughnessThreshold))
        return false;
    if (abs(RTXDI_Luminance(a.specularF0) - RTXDI_Luminance(b.specularF0)) > reflectivityThreshold)
        return false;
    if (abs(RTXDI_Luminance(a.diffuse) - RTXDI_Luminance(b.diffuse)) > albedoThreshold)
        return false;
    return true;
}

// RTXDI's path tracer holds several surfaces across its traces, so the
// material stays packed until a BSDF needs it.
struct RAB_Surface
{
    float3 worldPos;
    // 0 marks an invalid (background or out-of-view) surface.
    float viewDepth;
    float3 normal;
    float3 geoNormal;
    float3 viewDir;
    float diffuseProbability;
    // A RealtimeClosure, packed; its roughness is clamped when unpacked.
    uint closure[5];
    // The VertexDifferential of the hit, packed; rays leaving the surface
    // continue it. Zero where the surface was not found by a traced ray.
    uint differential[9];
    // The roughest lobe the path scattered from before reaching the surface,
    // diffuse counting as 1: Unreal's path roughness, which scales lights by
    // their indirect lighting intensity. Primary surfaces have 0.
    float pathRoughness;
    // The Unreal lighting channels of the surface's instance.
    uint lightingChannels;
};

RAB_Surface RAB_EmptySurface()
{
    RAB_Surface surface;
    surface.worldPos = float3(0.0);
    surface.viewDepth = 0.0;
    surface.normal = float3(0.0, 0.0, 1.0);
    surface.geoNormal = float3(0.0, 0.0, 1.0);
    surface.viewDir = float3(0.0, 0.0, 1.0);
    surface.diffuseProbability = 1.0;
    packRealtimeClosure(surface.closure, RAB_EmptyMaterial());
    packVertexDifferential(surface.differential, zeroVertexDifferential());
    surface.pathRoughness = 0.0;
    surface.lightingChannels = 0u;
    return surface;
}

// The differential of the ray that leaves `surface` along `direction`.
VertexDifferential leaveSurface(RAB_Surface surface, float3 direction, bool specular)
{
    return leave(unpackVertexDifferential(surface.differential), -surface.viewDir,
        surface.normal, direction, specular);
}

// The differential of the ray that leaves a traced hit along `direction`,
// sampled from `bsdf`.
VertexDifferential leaveHit(RealtimeHitPayload hit, RealtimeBsdf bsdf, float3 direction,
    bool specular)
{
    return leave(unpackVertexDifferential(hit.differential), -bsdf.view, bsdf.normal, direction,
        specular);
}

bool RAB_IsSurfaceValid(RAB_Surface surface) { return surface.viewDepth > 0.0; }
float3 RAB_GetSurfaceWorldPos(RAB_Surface surface) { return surface.worldPos; }
void RAB_SetSurfaceWorldPos(inout RAB_Surface surface, float3 worldPos) { surface.worldPos = worldPos; }
float3 RAB_GetSurfaceNormal(RAB_Surface surface) { return surface.normal; }
void RAB_SetSurfaceNormal(inout RAB_Surface surface, float3 normal) { surface.normal = normal; }
float3 RAB_GetSurfaceViewDir(RAB_Surface surface) { return surface.viewDir; }
float RAB_GetSurfaceLinearDepth(RAB_Surface surface) { return surface.viewDepth; }

RAB_Material RAB_GetMaterial(RAB_Surface surface)
{
    RAB_Material material = unpackRealtimeClosure(surface.closure);
    material.roughness = realtimeBsdfRoughness(material.roughness);
    return material;
}

float RAB_GetSurfaceRoughness(RAB_Surface surface) { return RAB_GetMaterial(surface).roughness; }

// The realtime BSDF of a surface, for evaluation and sampling.
RealtimeBsdf realtimeSurfaceBsdf(RAB_Surface surface)
{
    const RAB_Material material = RAB_GetMaterial(surface);
    RealtimeBsdf bsdf;
    bsdf.normal = surface.normal;
    bsdf.geometricNormal = surface.geoNormal;
    bsdf.view = surface.viewDir;
    bsdf.normalView = dot(surface.normal, surface.viewDir);
    bsdf.diffuse = material.diffuse;
    bsdf.specularF0 = material.specularF0;
    bsdf.specularF90 = material.specularF90;
    bsdf.roughness = material.roughness;
    bsdf.diffuseProbability = surface.diffuseProbability;
    return bsdf;
}

// The surface a ray found, shaded with `bsdf`.
RAB_Surface makeRealtimeSurface(RealtimeHitPayload hit, RealtimeBsdf bsdf, float viewDepth)
{
    RAB_Surface surface;
    surface.worldPos = hit.position;
    surface.viewDepth = viewDepth;
    surface.normal = bsdf.normal;
    surface.geoNormal = bsdf.geometricNormal;
    surface.viewDir = bsdf.view;
    surface.diffuseProbability = bsdf.diffuseProbability;
    surface.closure = hit.closure;
    surface.differential = hit.differential;
    surface.pathRoughness = 0.0;
    Scene* scene = &realtimeArgs()->frame.scene;
    surface.lightingChannels = scene.instances[min(hit.instance, scene.instanceCount - 1u)].lightingChannels;
    return surface;
}

RealtimeSurface packRealtimeSurface(RAB_Surface surface)
{
    RealtimeSurface record;
    record.position = surface.worldPos;
    record.viewDepth = surface.viewDepth;
    record.normal = RTXDI_EncodeNormalizedVectorToSnorm2x16(surface.normal);
    record.geometricNormal = RTXDI_EncodeNormalizedVectorToSnorm2x16(surface.geoNormal);
    record.view = RTXDI_EncodeNormalizedVectorToSnorm2x16(surface.viewDir);
    record.closure = surface.closure;
    record.lightingChannels = surface.lightingChannels;
    return record;
}

// Records `surface` and, while paths leave surfaces, its differential.
void storeRealtimeSurface(RealtimeSurface* surfaces, RealtimeSurfaceDifferential* differentials,
    uint index, RAB_Surface surface)
{
    surfaces[index] = packRealtimeSurface(surface);
    if (realtimeArgs()->frame.maxBounces > 0u)
        differentials[index].words = surface.differential;
}

// G-buffer surfaces are primary surfaces, whose sampler picks lobes with
// realtimePrimaryDiffuseProbability(). Their differential is the caller's.
RAB_Surface unpackRealtimeSurface(RealtimeSurface record)
{
    RAB_Surface surface;
    surface.worldPos = record.position;
    surface.viewDepth = record.viewDepth;
    surface.normal = RTXDI_DecodeNormalizedVectorFromSnorm2x16(record.normal);
    surface.geoNormal = RTXDI_DecodeNormalizedVectorFromSnorm2x16(record.geometricNormal);
    surface.viewDir = RTXDI_DecodeNormalizedVectorFromSnorm2x16(record.view);
    surface.closure = record.closure;
    [unroll]
    for (uint i = 0u; i < 9u; ++i)
        surface.differential[i] = 0u;
    surface.pathRoughness = 0.0;
    surface.lightingChannels = record.lightingChannels;
    const RAB_Material material = RAB_GetMaterial(surface);
    surface.diffuseProbability = realtimePrimaryDiffuseProbability(realtimeDiffuseProbability(
        material.diffuse, material.specularF0, material.specularF90,
        dot(surface.normal, surface.viewDir)));
    return surface;
}

// Both the regular realtime tracer and RTXDI PT's path replay traverse the
// same material hit groups and any-hit opacity shaders. The hit's materials
// see `differential`, the ray's, and return the hit's in its place. `mask`
// picks the instances the kind of ray finds (RaytracingMask*).
RealtimeHitPayload traceRealtimeHitPayload(RayDesc ray, VertexDifferential differential,
    uint mask, uint rayFlags)
{
    RealtimeHitPayload hit = {};
    packVertexDifferential(hit.differential, differential);
    TraceRay(RaytracingAccelerationStructure(realtimeArgs()->frame.topLevelAS),
        rayFlags, mask, RealtimeRayTypeSurface,
        RaytracingRayTypeCount, RealtimeMissSurface, ray, hit);
    return hit;
}

// ReSTIR resamples the lighting pass's surfaces, one per lighting pixel.
// The lighting rectangle changes with the viewport while the buffers keep
// their allocation, so the previous frame's surfaces may cover a different
// rectangle than this frame's.
uint2 realtimeLightingSize()
{
    return uint2(realtimeArgs()->view.lightingWidth, realtimeArgs()->view.lightingHeight);
}

uint2 realtimeLightingSize(bool prevFrame)
{
    RealtimeView* view = &realtimeArgs()->view;
    return prevFrame ? uint2(view->previousLightingWidth, view->previousLightingHeight)
        : uint2(view->lightingWidth, view->lightingHeight);
}

bool realtimePixelInView(int2 pixel)
{
    return all(pixel >= 0) && all(pixel < int2(realtimeLightingSize()));
}

uint realtimeSurfaceIndex(uint2 pixel)
{
    return pixel.y * realtimeArgs()->view.surfaceStride + pixel.x;
}

RAB_Surface RAB_GetGBufferSurface(int2 pixelPosition, bool prevFrame)
{
    if (any(pixelPosition < 0) || any(pixelPosition >= int2(realtimeLightingSize(prevFrame))))
        return RAB_EmptySurface();
    RealtimeLighting* lighting = &realtimeArgs()->lighting;
    if (prevFrame && lighting.previousSurfacesValid == 0u)
        return RAB_EmptySurface();
    RealtimeSurface* surfaces = prevFrame ? lighting.previousSurfaces : lighting.surfaces;
    const uint index = realtimeSurfaceIndex(uint2(pixelPosition));
    RAB_Surface surface = unpackRealtimeSurface(surfaces[index]);
    if (realtimeArgs()->frame.maxBounces > 0u)
    {
        RealtimeSurfaceDifferential* differentials = prevFrame
            ? lighting.previousDifferentials : lighting.differentials;
        surface.differential = differentials[index].words;
    }
    return surface;
}

RAB_Material RAB_GetGBufferMaterial(int2 pixelPosition, bool prevFrame)
{
    return RAB_GetMaterial(RAB_GetGBufferSurface(pixelPosition, prevFrame));
}

// Reflects out-of-view positions back into the view.
int2 RAB_ClampSamplePositionIntoView(int2 pixelPosition, bool prevFrame)
{
    const int width = int(realtimeLightingSize(prevFrame).x);
    const int height = int(realtimeLightingSize(prevFrame).y);
    if (pixelPosition.x < 0) pixelPosition.x = -pixelPosition.x;
    if (pixelPosition.y < 0) pixelPosition.y = -pixelPosition.y;
    if (pixelPosition.x >= width) pixelPosition.x = 2 * width - pixelPosition.x - 1;
    if (pixelPosition.y >= height) pixelPosition.y = 2 * height - pixelPosition.y - 1;
    return pixelPosition;
}

// BSDF times cosine toward `direction`, both lobes.
float3 realtimeSurfaceReflectance(RAB_Surface surface, float3 direction)
{
    const RealtimeBsdf bsdf = realtimeSurfaceBsdf(surface);
    return bsdf.evaluateDiffuse(direction) + bsdf.evaluateSpecular(direction);
}

// BSDF times cosine toward `direction`, each lobe scaled as a light scales it.
float3 realtimeScaledReflectance(RAB_Surface surface, float3 direction, float diffuseScale,
    float specularScale)
{
    const RealtimeBsdf bsdf = realtimeSurfaceBsdf(surface);
    return bsdf.evaluateDiffuse(direction) * diffuseScale
        + bsdf.evaluateSpecular(direction) * specularScale;
}

bool RAB_SurfaceImportanceSampleBrdf(RAB_Surface surface,
    inout RTXDI_RandomSamplerState rng, out float3 dir)
{
    const float lobe = RTXDI_GetNextRandom(rng);
    float2 u;
    u.x = RTXDI_GetNextRandom(rng);
    u.y = RTXDI_GetNextRandom(rng);
    const RealtimeBsdfSample sample = realtimeSurfaceBsdf(surface).sample(lobe, u);
    dir = sample.direction;
    return sample.valid;
}

float RAB_SurfaceEvaluateBrdfPdf(RAB_Surface surface, float3 dir)
{
    return realtimeSurfaceBsdf(surface).pdf(dir);
}

// ---------------------------------------------------------------------------
// Visibility
// ---------------------------------------------------------------------------

// A ray from a surface toward `direction`, stopping short of `distance`.
RayDesc realtimeVisibilityRay(float3 position, float3 geometricNormal, float3 direction,
    float distance)
{
    RayDesc ray;
    ray.Origin = offsetRayOrigin(position, geometricNormal, direction);
    ray.Direction = direction;
    ray.TMin = 0.1;
    ray.TMax = distance < 1.0e17 ? max(distance - 2.0e-2, ray.TMin) : 1.0e30;
    return ray;
}

float realtimeRayTransmittance(RayDesc ray)
{
    // Opaque geometry is accepted as the first hit and ends the search; the
    // materials' shadow any-hit stages multiply in the fractional layers.
    RealtimeShadowPayload payload;
    payload.occluded = 1u;
    payload.transmittance = 1.0;
    TraceRay(RaytracingAccelerationStructure(realtimeArgs()->frame.topLevelAS),
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
        RaytracingMaskShadow, RealtimeRayTypeShadow, RaytracingRayTypeCount,
        RealtimeMissShadow, ray, payload);
    return payload.occluded != 0u ? 0.0 : payload.transmittance;
}

// Final visibility: every fractional and alpha-tested layer is evaluated.
// Only light that is actually shaded pays for it.
float realtimeTransmittance(float3 position, float3 geometricNormal, float3 direction,
    float distance)
{
    return realtimeRayTransmittance(realtimeVisibilityRay(position, geometricNormal,
        direction, distance));
}

// Conservative visibility, as RTXDI's sample defines it: an inline query
// against opaque geometry only, ended by the first hit. Fractional and
// alpha-tested surfaces count as clear, so no material runs; resampling and
// path sampling use it, and final shading corrects it with
// realtimeTransmittance. `mask` picks the instances that block the ray.
bool realtimeConservativelyVisible(RayDesc ray, uint mask)
{
    RayQuery<RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
    query.TraceRayInline(RaytracingAccelerationStructure(realtimeArgs()->frame.topLevelAS),
        RAY_FLAG_NONE, mask, ray);
    query.Proceed();
    return query.CommittedStatus() == COMMITTED_NOTHING;
}

bool realtimeVisible(float3 position, float3 geometricNormal, float3 direction,
    float distance)
{
    return realtimeConservativelyVisible(realtimeVisibilityRay(position, geometricNormal,
        direction, distance), RaytracingMaskShadow);
}

// Sample positions on scene geometry (ReSTIR PT's reconnection vertices)
// stop just short of their own surface.
bool RAB_GetConservativeVisibility(RAB_Surface surface, float3 samplePosition)
{
    const float3 toSample = samplePosition - surface.worldPos;
    const float distance = length(toSample);
    if (!(distance > 0.0))
        return false;
    return realtimeVisible(surface.worldPos, surface.geoNormal, toSample / distance,
        distance * 0.999);
}

// The previous frame's acceleration structure is not kept, so temporal
// visibility is tested against the current scene from the current surface.
bool RAB_GetTemporalConservativeVisibility(RAB_Surface currentSurface,
    RAB_Surface previousSurface, float3 samplePosition)
{
    return RAB_GetConservativeVisibility(currentSurface, samplePosition);
}

// ---------------------------------------------------------------------------
// Lights
// ---------------------------------------------------------------------------

struct RAB_LightInfo
{
    uint kind;
    uint index;
};

RAB_LightInfo RAB_EmptyLightInfo()
{
    RAB_LightInfo light;
    light.kind = RealtimeLightNone;
    light.index = 0u;
    return light;
}

// Light indices: [0, points) points, then spots, then rects (the local
// region), then directionals (the infinite region), then the environment.
RAB_LightInfo RAB_LoadLightInfo(uint index, bool prevFrame)
{
    Frame* frame = &realtimeArgs()->frame;
    RAB_LightInfo light;
    light.index = index;
    if (light.index < frame.pointLightCount)
    {
        light.kind = RealtimeLightPoint;
        return light;
    }
    light.index -= frame.pointLightCount;
    if (light.index < frame.spotLightCount)
    {
        light.kind = RealtimeLightSpot;
        return light;
    }
    light.index -= frame.spotLightCount;
    if (light.index < frame.rectLightCount)
    {
        light.kind = RealtimeLightRect;
        return light;
    }
    light.index -= frame.rectLightCount;
    if (light.index < frame.directionalLightCount)
    {
        light.kind = RealtimeLightDirectional;
        return light;
    }
    if (light.index == frame.directionalLightCount && frame.environment != nullptr)
    {
        light.kind = RealtimeLightEnvironment;
        return light;
    }
    return RAB_EmptyLightInfo();
}

// Lights are addressed in place and never compacted into the RIS buffer.
RAB_LightInfo RAB_LoadCompactLightInfo(uint linearIndex)
{
    return RAB_EmptyLightInfo();
}

bool RAB_StoreCompactLightInfo(uint linearIndex, RAB_LightInfo lightInfo)
{
    return false;
}

// Light records are rewritten in place, so an index names the same light in
// both frames while the counts hold; the temporal passes are reset when they
// do not.
int RAB_TranslateLightIndex(uint lightIndex, bool currentToPrevious)
{
    Frame* frame = &realtimeArgs()->frame;
    const uint count = frame.pointLightCount + frame.spotLightCount
        + frame.rectLightCount + frame.directionalLightCount
        + (frame.environment != nullptr ? 1u : 0u);
    return lightIndex < count ? int(lightIndex) : -1;
}

struct RAB_LightSample
{
    // Infinite lights sit RealtimeDistantLightDistance along their direction
    // from the surface the sample was drawn for, far enough that the
    // direction from any other surface is the same.
    float3 position;
    // Already scaled by the light's indirect lighting intensity for the
    // surface's path.
    float3 radiance;
    // Converts the unscaled BSDF weighting in RTXDI's stock hybrid-shift
    // callback to this light's diffuse/specular weighting for this surface.
    float3 hybridShiftReflectanceRatio;
    // Solid-angle density of the point given the light, or 1 for a delta
    // light whose radiance then already carries the inverse square falloff.
    float solidAnglePdf;
    uint kind;
    // The light's scale of the surface's diffuse and specular lobes.
    float diffuseScale;
    float specularScale;
    uint castShadows;
};

RAB_LightSample RAB_EmptyLightSample()
{
    RAB_LightSample sample;
    sample.position = float3(0.0);
    sample.radiance = float3(0.0);
    sample.hybridShiftReflectanceRatio = float3(0.0);
    sample.solidAnglePdf = 0.0;
    sample.kind = RealtimeLightNone;
    sample.diffuseScale = 1.0;
    sample.specularScale = 1.0;
    sample.castShadows = 1u;
    return sample;
}

// BRDF rays can only find the environment; analytic light shapes are not in
// the acceleration structure.
bool RAB_IsAnalyticLightSample(RAB_LightSample lightSample)
{
    return lightSample.kind != RealtimeLightEnvironment;
}
float RAB_LightSampleSolidAnglePdf(RAB_LightSample lightSample) { return lightSample.solidAnglePdf; }
float3 RAB_LightSamplePosition(RAB_LightSample lightSample) { return lightSample.position; }
float3 RAB_LightSampleRadiance(RAB_LightSample lightSample)
{
    return lightSample.radiance * lightSample.hybridShiftReflectanceRatio;
}

void RAB_GetLightDirDistance(RAB_Surface surface, RAB_LightSample lightSample,
    out float3 o_lightDir, out float o_lightDistance)
{
    const float3 toLight = lightSample.position - surface.worldPos;
    const float distance = length(toLight);
    o_lightDir = distance > 0.0 ? toLight / distance : float3(0.0, 0.0, 1.0);
    o_lightDistance = lightSample.kind == RealtimeLightDirectional
        || lightSample.kind == RealtimeLightEnvironment ? RealtimeDistantLightDistance : distance;
}

// Unreal's local light attenuation (the path tracer's
// ComputeAttenuationFalloff), from the distance to the light's centre. The
// exponent falloff cancels the inverse square law the light's radiance
// carries, as Unreal's does.
float localLightFalloff(LightFalloff falloff, float distanceSquared)
{
    const float normalized = distanceSquared * falloff.invRadius * falloff.invRadius;
    if (falloff.inverseSquared != 0u)
    {
        const float window = saturate(1.0 - normalized * normalized);
        return window * window;
    }
    return distanceSquared * pow(1.0 - saturate(normalized), falloff.exponent);
}

// Unreal's spot cone (FSpotLightSceneProxy's clamped angles and the path
// tracer's SpotAttenuation).
float spotAttenuation(SpotLight light, float3 fromLight)
{
    const float degrees = AnalyticLightPi / 180.0;
    const float inner = clamp(light.innerConeAngle, 0.0, 89.0) * degrees;
    const float outer = clamp(light.outerConeAngle * degrees, inner + 0.001, 89.0 * degrees + 0.001);
    const float cosOuter = cos(outer);
    const float cone = saturate((dot(normalize(light.direction), fromLight) - cosOuter)
        / (cos(inner) - cosOuter));
    return cone * cone;
}

// Sphere-shaped point and spot lights, sampled exactly as the analytic light
// sampler in AnalyticLights.slang does, but from given random numbers so a
// reservoir can replay the sample for another surface. `power` is the
// intensity; the sphere's radiance is power / (pi r^2), as in Unreal.
RAB_LightSample sampleSphereShapedLight(float3 center, float radius, float3 power,
    float3 origin, float2 uv)
{
    RAB_LightSample sample = RAB_EmptyLightSample();
    if (radius <= 0.0)
    {
        const float3 toLight = center - origin;
        const float distanceSquared = dot(toLight, toLight);
        if (distanceSquared <= 1.0e-12)
            return sample;
        sample.position = center;
        sample.radiance = power / distanceSquared;
        sample.solidAnglePdf = 1.0;
        return sample;
    }
    float3 direction;
    float distance;
    float pdf;
    if (!sampleSphereLight(origin, center, radius, uv, direction, distance, pdf)
        || !(pdf > 0.0))
        return sample;
    sample.position = origin + direction * distance;
    sample.radiance = power / max(AnalyticLightPi * radius * radius, 1.0e-20);
    sample.solidAnglePdf = pdf;
    return sample;
}

// A capsule light (a point or spot light with a source length) by area, as
// Unreal's path tracer samples one without its solid-angle bounds: the caps
// by the sphere they form, the body by the cylinder, each by its area.
RAB_LightSample sampleCapsuleLight(float3 center, float radius, float3 axis, float length,
    float3 power, float3 origin, float2 uv)
{
    RAB_LightSample sample = RAB_EmptyLightSample();
    // Unreal keeps a capsule's area above zero, as radiance is power over it.
    radius = max(radius, 1.0e-2);
    const float capsArea = radius * radius;
    const float bodyArea = 0.5 * radius * length;
    const float capsProbability = capsArea / (capsArea + bodyArea);
    float3 local;
    float3 localNormal;
    if (uv.x < capsProbability)
    {
        const float z = 1.0 - 2.0 * (uv.x / capsProbability);
        const float radial = sqrt(max(1.0 - z * z, 0.0));
        const float phi = 2.0 * AnalyticLightPi * uv.y;
        localNormal = float3(radial * cos(phi), radial * sin(phi), z);
        local = localNormal * radius;
        local.z += 0.5 * length * (z >= 0.0 ? 1.0 : -1.0);
    }
    else
    {
        const float phi = 2.0 * AnalyticLightPi * (uv.x - capsProbability) / (1.0 - capsProbability);
        localNormal = float3(cos(phi), sin(phi), 0.0);
        local = float3(radius * localNormal.xy, (uv.y - 0.5) * length);
    }
    const float3 n = normalize(axis);
    const float3 helper = abs(n.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(0.0, 1.0, 0.0);
    const float3 tangent = normalize(cross(helper, n));
    const float3 bitangent = cross(n, tangent);
    const float3 position = center + tangent * local.x + bitangent * local.y + n * local.z;
    const float3 normal = tangent * localNormal.x + bitangent * localNormal.y + n * localNormal.z;
    const float3 toPoint = position - origin;
    const float distanceSquared = dot(toPoint, toPoint);
    if (distanceSquared <= 1.0e-12)
        return sample;
    const float cosine = dot(normal, -toPoint * rsqrt(distanceSquared));
    if (cosine <= 0.0)
        return sample;
    sample.position = position;
    sample.radiance = power / (AnalyticLightPi * (capsArea + bodyArea));
    sample.solidAnglePdf = distanceSquared / (4.0 * AnalyticLightPi * cosine * (capsArea + bodyArea));
    return sample;
}

RAB_LightSample sampleLocalLight(float3 center, float radius, float3 axis, float length,
    float3 power, float3 origin, float2 uv)
{
    return length > 0.0 ? sampleCapsuleLight(center, radius, axis, length, power, origin, uv)
        : sampleSphereShapedLight(center, radius, power, origin, uv);
}

// The part of a rect light its barn doors leave visible from `origin`, as
// Unreal's GetRect (RectLight.ush) clips it: the centre's offset and the half
// extents along the light's width and height axes. False when the doors hide
// all of it.
bool visibleRect(RectLight light, float3 widthAxis, float3 heightAxis, float3 normal,
    float3 origin, out float2 center, out float2 extent)
{
    center = float2(0.0);
    extent = 0.5 * float2(max(light.width, 0.0), max(light.height, 0.0));
    const float barnCosine = cos(light.barnDoorAngle * AnalyticLightPi / 180.0);
    const float barnLength = max(light.barnDoorLength, 0.0);
    // Unreal leaves doors opened past 88 degrees out.
    if (barnCosine <= 0.035)
        return extent.x > 0.0 && extent.y > 0.0;
    const float3 toLight = light.position - origin;
    float3 shading = float3(dot(widthAxis, toLight), dot(heightAxis, toLight), dot(-normal, toLight));
    const float barnSine = sqrt(1.0 - barnCosine * barnCosine);
    const float barnDepth = min(shading.z, barnCosine * barnLength);
    const float depthRatio = barnDepth / max(1.0e-6, barnCosine * barnLength);
    const float doorProjection = barnSine * barnLength * depthRatio;
    const float2 side = sign(shading.xy);
    shading.xy = side * max(abs(shading.xy), extent + doorProjection);
    const float3 corner = float3(side * (extent + doorProjection), barnDepth);
    const float3 fromCorner = shading - corner;
    const float2 shadow = barnDepth * abs(fromCorner.xy) / max(fromCorner.z, 0.1);
    const float2 minimum = clamp(-extent + (shadow - doorProjection) * max(0.0, -side), -extent, extent);
    const float2 maximum = clamp(extent - (shadow - doorProjection) * max(0.0, side), -extent, extent);
    center = -0.5 * (minimum + maximum);
    extent = 0.5 * (maximum - minimum);
    return extent.x != 0.0 && extent.y != 0.0;
}

// Unreal's rect light: radiance over the part its barn doors leave visible,
// which is sampled by area.
RAB_LightSample sampleRectLight(RectLight light, float3 origin, float2 uv)
{
    RAB_LightSample sample = RAB_EmptyLightSample();
    const float3 normal = normalize(light.direction);
    const float3 widthAxis = normalize(light.tangent);
    const float3 heightAxis = normalize(cross(normal, widthAxis));
    const float3 toLight = light.position - origin;
    if (light.twoSided == 0u && dot(normal, toLight) > 0.0)
        return sample;
    float2 center;
    float2 extent;
    if (!visibleRect(light, widthAxis, heightAxis, normal, origin, center, extent))
        return sample;
    const float area = 4.0 * extent.x * extent.y;
    const float2 square = center + (2.0 * uv - 1.0) * extent;
    const float3 position = light.position + widthAxis * square.x + heightAxis * square.y;
    const float3 delta = position - origin;
    const float distanceSquared = dot(delta, delta);
    if (distanceSquared <= 1.0e-12)
        return sample;
    const float3 direction = delta * rsqrt(distanceSquared);
    float emitterCosine = dot(normal, -direction);
    emitterCosine = light.twoSided != 0u ? abs(emitterCosine) : max(emitterCosine, 0.0);
    if (emitterCosine <= 0.0)
        return sample;
    sample.position = position;
    sample.radiance = max(light.color, 0.0) * max(light.intensity, 0.0)
        * localLightFalloff(light.falloff, dot(toLight, toLight));
    sample.solidAnglePdf = distanceSquared / (area * emitterCosine);
    return sample;
}

// Unreal's directional light: a disk of the given angle whose radiance is
// its illuminance over the disk's solid angle.
RAB_LightSample sampleDirectionalLight(DirectionalLight light, float3 origin, float2 uv)
{
    RAB_LightSample sample = RAB_EmptyLightSample();
    const float3 centerDirection = -normalize(light.direction);
    const float3 illuminance = max(light.color, 0.0) * max(light.intensity, 0.0);
    const float sine = sin(0.5 * clamp(light.softAngle, 0.0, 180.0) * AnalyticLightPi / 180.0);
    float3 direction = centerDirection;
    sample.radiance = illuminance;
    sample.solidAnglePdf = 1.0;
    if (sine > 0.0)
    {
        const float oneMinusCosine = oneMinusCosineFromSineSquared(sine * sine);
        const float solidAngle = 2.0 * AnalyticLightPi * oneMinusCosine;
        direction = sampleConeDirection(centerDirection, oneMinusCosine, uv);
        sample.solidAnglePdf = 1.0 / solidAngle;
        sample.radiance = illuminance / solidAngle;
    }
    sample.position = origin + direction * RealtimeDistantLightDistance;
    return sample;
}

// A light's controls, applied to a sample for `surface`: Unreal's lighting
// channels, its diffuse and specular scales, whether it casts shadows, and
// its indirect lighting intensity, which grows with the roughness the path
// already went through (the path tracer's GetLightIndirectScale).
RAB_LightSample applyLightControls(RAB_LightSample sample, LightControls controls,
    RAB_Surface surface)
{
    if (controls.visible == 0u || (controls.lightingChannels & surface.lightingChannels) == 0u)
        return RAB_EmptyLightSample();
    sample.radiance *= lerp(1.0, controls.indirectLightingIntensity,
        saturate(4.0 * surface.pathRoughness));
    sample.diffuseScale = controls.diffuseScale;
    sample.specularScale = controls.specularScale;
    sample.castShadows = controls.castShadows;
    return sample;
}

RAB_LightSample RAB_SamplePolymorphicLight(RAB_LightInfo lightInfo, RAB_Surface surface,
    float2 uv)
{
    Frame* frame = &realtimeArgs()->frame;
    const float3 origin = surface.worldPos;
    RAB_LightSample sample = RAB_EmptyLightSample();
    LightControls controls;
    if (lightInfo.kind == RealtimeLightPoint)
    {
        const PointLight light = frame.pointLights[lightInfo.index];
        sample = sampleLocalLight(light.position, max(light.softRadius, 0.0), light.axis,
            max(light.sourceLength, 0.0), max(light.color, 0.0) * max(light.intensity, 0.0),
            origin, uv);
        const float3 toLight = light.position - origin;
        sample.radiance *= localLightFalloff(light.falloff, dot(toLight, toLight));
        controls = light.controls;
    }
    else if (lightInfo.kind == RealtimeLightSpot)
    {
        const SpotLight light = frame.spotLights[lightInfo.index];
        sample = sampleLocalLight(light.position, max(light.softRadius, 0.0), light.axis,
            max(light.sourceLength, 0.0), max(light.color, 0.0) * max(light.intensity, 0.0),
            origin, uv);
        // Unreal attenuates by the direction to the centre, for every sample.
        const float3 toLight = light.position - origin;
        const float distanceSquared = dot(toLight, toLight);
        sample.radiance *= localLightFalloff(light.falloff, distanceSquared)
            * (distanceSquared > 0.0 ? spotAttenuation(light, -toLight * rsqrt(distanceSquared)) : 0.0);
        controls = light.controls;
    }
    else if (lightInfo.kind == RealtimeLightRect)
    {
        const RectLight light = frame.rectLights[lightInfo.index];
        sample = sampleRectLight(light, origin, uv);
        controls = light.controls;
    }
    else if (lightInfo.kind == RealtimeLightDirectional)
    {
        const DirectionalLight light = frame.directionalLights[lightInfo.index];
        sample = sampleDirectionalLight(light, origin, uv);
        controls = light.controls;
    }
    else if (lightInfo.kind == RealtimeLightEnvironment)
    {
        const Environment environment = frame.environment[0];
        const float3 direction = environmentDirection(environment, uv);
        sample.position = origin + direction * RealtimeDistantLightDistance;
        sample.radiance = environmentRadianceRgb(environment, direction, false);
        sample.solidAnglePdf = environmentUvPerSolidAngle(environment, uv);
        controls.castShadows = environment.castShadows;
        controls.lightingChannels = 0xFFFFFFFFu;
        controls.diffuseScale = 1.0;
        controls.specularScale = 1.0;
        controls.indirectLightingIntensity = environment.indirectLightingIntensity;
        controls.visible = 1u;
    }
    else
        return sample;
    if (!realtimeIsFinite(sample.radiance) || !(sample.solidAnglePdf > 0.0))
        return RAB_EmptyLightSample();
    sample.kind = lightInfo.kind;
    sample = applyLightControls(sample, controls, surface);

    // RTXDI's stock hybrid-shift callback evaluates the unscaled BSDF and
    // multiplies it by RAB_LightSampleRadiance(). Fold NoorRay's per-light
    // lobe controls into that radiance so the callback matches the scaled
    // reflectance used by direct lighting, without changing RTXDI itself.
    float3 direction;
    float distance;
    RAB_GetLightDirDistance(surface, sample, direction, distance);
    const float3 unscaledReflectance = realtimeSurfaceReflectance(surface, direction);
    const float3 scaledReflectance = realtimeScaledReflectance(surface, direction,
        sample.diffuseScale, sample.specularScale);
    sample.hybridShiftReflectanceRatio = select(unscaledReflectance != 0.0,
        scaledReflectance / max(abs(unscaledReflectance), 1.0e-20), float3(0.0));
    return sample;
}

// Final visibility of a light sample: every fractional layer is evaluated,
// and lights that cast no shadow are never blocked.
float realtimeLightTransmittance(RAB_Surface surface, RAB_LightSample lightSample)
{
    if (!(lightSample.solidAnglePdf > 0.0))
        return 0.0;
    if (lightSample.castShadows == 0u)
        return 1.0;
    float3 direction;
    float distance;
    RAB_GetLightDirDistance(surface, lightSample, direction, distance);
    return realtimeTransmittance(surface.worldPos, surface.geoNormal, direction, distance);
}

bool RAB_GetConservativeVisibility(RAB_Surface surface, RAB_LightSample lightSample)
{
    if (lightSample.castShadows == 0u)
        return true;
    float3 direction;
    float distance;
    RAB_GetLightDirDistance(surface, lightSample, direction, distance);
    return realtimeVisible(surface.worldPos, surface.geoNormal, direction, distance);
}

bool RAB_GetTemporalConservativeVisibility(RAB_Surface currentSurface,
    RAB_Surface previousSurface, RAB_LightSample lightSample)
{
    return RAB_GetConservativeVisibility(currentSurface, lightSample);
}

float3 realtimeLightSampleReflectance(RAB_Surface surface, float3 direction,
    RAB_LightSample lightSample)
{
    return realtimeScaledReflectance(surface, direction, lightSample.diffuseScale,
        lightSample.specularScale);
}

// Radiance reflected toward the viewer per unit of the sample's solid-angle
// density; RTXDI resamples its luminance.
float3 realtimeLightSampleReflected(RAB_LightSample lightSample, RAB_Surface surface)
{
    if (!(lightSample.solidAnglePdf > 0.0))
        return float3(0.0);
    float3 direction;
    float distance;
    RAB_GetLightDirDistance(surface, lightSample, direction, distance);
    return realtimeLightSampleReflectance(surface, direction, lightSample) * lightSample.radiance
        / lightSample.solidAnglePdf;
}

float RAB_GetLightSampleTargetPdfForSurface(RAB_LightSample lightSample, RAB_Surface surface)
{
    return RTXDI_Luminance(realtimeLightSampleReflected(lightSample, surface));
}

// Power over squared distance for any receiver inside the ReGIR cell. It only
// shapes the proposal, so it merely has to be positive wherever the light can
// contribute: nowhere beyond its attenuation radius.
float RAB_GetLightTargetPdfForVolume(RAB_LightInfo light, float3 volumeCenter,
    float volumeRadius)
{
    Frame* frame = &realtimeArgs()->frame;
    float3 position;
    float power;
    float lightRadius = 0.0;
    LightFalloff falloff;
    if (light.kind == RealtimeLightPoint)
    {
        const PointLight record = frame.pointLights[light.index];
        position = record.position;
        power = record.selectionWeight;
        lightRadius = max(record.softRadius, 0.0) + 0.5 * max(record.sourceLength, 0.0);
        falloff = record.falloff;
    }
    else if (light.kind == RealtimeLightSpot)
    {
        const SpotLight record = frame.spotLights[light.index];
        position = record.position;
        power = record.selectionWeight;
        lightRadius = max(record.softRadius, 0.0) + 0.5 * max(record.sourceLength, 0.0);
        falloff = record.falloff;
    }
    else if (light.kind == RealtimeLightRect)
    {
        const RectLight record = frame.rectLights[light.index];
        position = record.position;
        power = record.selectionWeight;
        lightRadius = 0.5 * length(float2(record.width, record.height));
        falloff = record.falloff;
        // A one-sided rect cannot light a cell entirely behind it.
        if (record.twoSided == 0u && dot(normalize(record.direction),
                volumeCenter - position) < -(volumeRadius + lightRadius))
            return 0.0;
    }
    else
        return 0.0;
    // The falloff is measured from the light's centre.
    const float centerDistance = length(volumeCenter - position);
    if (falloff.invRadius > 0.0 && (centerDistance - volumeRadius) * falloff.invRadius >= 1.0)
        return 0.0;
    const float distance = max(centerDistance - volumeRadius - lightRadius, 0.0);
    return max(power, 0.0) / max(distance * distance, 0.25 * volumeRadius * volumeRadius);
}

float RAB_EvaluateLocalLightSourcePdf(uint lightIndex)
{
    RealtimeLighting* lighting = &realtimeArgs()->lighting;
    const RTXDI_LightBufferRegion region = lighting.lightBufferParams.localLightBufferRegion;
    if (lightIndex < region.firstLightIndex
        || lightIndex >= region.firstLightIndex + region.numLights)
        return 0.0;
    return lighting.localLightAlias[lightIndex - region.firstLightIndex].pdf;
}

// The environment's light reaching a path along a BSDF ray from a surface it
// reached with `pathRoughness`: scaled by the sky light's indirect lighting
// intensity, as a light sample there would be.
float3 environmentEscapedRadiance(Environment environment, float3 direction, float pathRoughness)
{
    return environmentRadianceRgb(environment, direction, false)
        * lerp(1.0, environment.indirectLightingIntensity, saturate(4.0 * pathRoughness));
}

float2 RAB_GetEnvironmentMapRandXYFromDir(float3 worldDir)
{
    return environmentUv(realtimeArgs()->frame.environment[0], worldDir);
}

// Density of the presampled environment light over its uv parameters, the
// measure RTXDI's environment light samples are drawn in.
float RAB_EvaluateEnvironmentMapSamplingPdf(float3 L)
{
    const Environment environment = realtimeArgs()->frame.environment[0];
    return environmentPdf(environment, L)
        / environmentUvPerSolidAngle(environment, environmentUv(environment, L));
}

// Share of the environment's light at `direction` that ReSTIR PT takes from
// light samples; BSDF rays that escape take the rest. The weights sum to one
// for every direction and depend on nothing a path shift changes (a
// surface's BSDF does, once a shift reconnects to it), so reused paths keep
// valid weights. BSDF densities are compared as if Lambertian.
float realtimeEnvironmentLightWeight(float3 direction)
{
    return powerHeuristic(environmentPdf(realtimeArgs()->frame.environment[0], direction),
        RealtimeInvPi);
}

// Analytic light shapes are not in the acceleration structure, so a BRDF ray
// finds a light only when nothing blocks it and RTXDI takes the environment.
bool RAB_TraceRayForLocalLight(float3 origin, float3 direction, float tMin, float tMax,
    out uint o_lightIndex, out float2 o_randXY)
{
    o_lightIndex = RTXDI_InvalidLightIndex;
    o_randXY = float2(0.0);
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = tMin;
    ray.TMax = tMax;
    return !realtimeConservativelyVisible(ray, RaytracingMaskIndirect);
}

// ---------------------------------------------------------------------------
// Frame-level helpers for the realtime passes
// ---------------------------------------------------------------------------

float3 realtimeCameraPosition(bool previous)
{
    return previous ? realtimeArgs()->view.previousCameraPosition
        : realtimeArgs()->view.cameraPosition;
}
