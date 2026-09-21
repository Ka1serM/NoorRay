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
//   directional ones. The environment is not an RTXDI light; it keeps its own
//   importance-sampled estimator with MIS against BSDF sampling.
// - Buffers are device pointers in RealtimeLighting, so nothing is bound.

#ifndef RTXDI_REGIR_MODE
#define RTXDI_REGIR_MODE RTXDI_REGIR_GRID
#endif

#include "RealtimeShading.slang"
#include "RealtimeArgs.slang"

#define RTXDI_RIS_BUFFER (realtimeArgs()->lighting.risBuffer)
#define RTXDI_LIGHT_RESERVOIR_BUFFER (realtimeArgs()->lighting.diReservoirs)
#define RTXDI_GI_RESERVOIR_BUFFER (realtimeArgs()->lighting.giReservoirs)
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
static const uint RealtimeLightNone = 0xFFFFFFFFu;

// Infinite light samples sit this far along their direction; shadow rays to
// them are unbounded.
static const float RealtimeDistantLightDistance = 1.0e16;
#define RAB_DISTANT_LIGHT_DISTANCE RealtimeDistantLightDistance

// ---------------------------------------------------------------------------
// Materials and surfaces
// ---------------------------------------------------------------------------

struct RAB_Material
{
    float3 diffuseAlbedo;
    float3 specularF0;
    float3 specularF90;
    float roughness;
};

RAB_Material RAB_EmptyMaterial()
{
    RAB_Material material;
    material.diffuseAlbedo = float3(0.0);
    material.specularF0 = float3(0.0);
    material.specularF90 = float3(0.0);
    material.roughness = 1.0;
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
    if (abs(RTXDI_Luminance(a.diffuseAlbedo) - RTXDI_Luminance(b.diffuseAlbedo)) > albedoThreshold)
        return false;
    return true;
}

struct RAB_Surface
{
    float3 worldPos;
    // 0 marks an invalid (background or out-of-view) surface.
    float viewDepth;
    float3 normal;
    float3 geoNormal;
    float3 viewDir;
    float diffuseProbability;
    RAB_Material material;
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
    surface.material = RAB_EmptyMaterial();
    return surface;
}

bool RAB_IsSurfaceValid(RAB_Surface surface) { return surface.viewDepth > 0.0; }
float3 RAB_GetSurfaceWorldPos(RAB_Surface surface) { return surface.worldPos; }
void RAB_SetSurfaceWorldPos(inout RAB_Surface surface, float3 worldPos) { surface.worldPos = worldPos; }
float3 RAB_GetSurfaceNormal(RAB_Surface surface) { return surface.normal; }
void RAB_SetSurfaceNormal(inout RAB_Surface surface, float3 normal) { surface.normal = normal; }
float3 RAB_GetSurfaceViewDir(RAB_Surface surface) { return surface.viewDir; }
float RAB_GetSurfaceLinearDepth(RAB_Surface surface) { return surface.viewDepth; }
float RAB_GetSurfaceRoughness(RAB_Surface surface) { return surface.material.roughness; }
RAB_Material RAB_GetMaterial(RAB_Surface surface) { return surface.material; }

// The realtime BSDF of a surface, for evaluation and sampling.
RealtimeBsdf realtimeSurfaceBsdf(RAB_Surface surface)
{
    RealtimeBsdf bsdf;
    bsdf.normal = surface.normal;
    bsdf.geometricNormal = surface.geoNormal;
    bsdf.view = surface.viewDir;
    bsdf.normalView = dot(surface.normal, surface.viewDir);
    bsdf.diffuse = surface.material.diffuseAlbedo;
    bsdf.specularF0 = surface.material.specularF0;
    bsdf.specularF90 = surface.material.specularF90;
    bsdf.roughness = surface.material.roughness;
    bsdf.diffuseProbability = surface.diffuseProbability;
    return bsdf;
}

RAB_Surface makeRealtimeSurface(RealtimeBsdf bsdf, float3 position, float viewDepth)
{
    RAB_Surface surface;
    surface.worldPos = position;
    surface.viewDepth = viewDepth;
    surface.normal = bsdf.normal;
    surface.geoNormal = bsdf.geometricNormal;
    surface.viewDir = bsdf.view;
    surface.diffuseProbability = bsdf.diffuseProbability;
    surface.material.diffuseAlbedo = bsdf.diffuse;
    surface.material.specularF0 = bsdf.specularF0;
    surface.material.specularF90 = bsdf.specularF90;
    surface.material.roughness = bsdf.roughness;
    return surface;
}

RealtimeSurface packRealtimeSurface(RAB_Surface surface, float cameraWeight)
{
    RealtimeSurface record;
    record.position = surface.worldPos;
    record.viewDepth = surface.viewDepth;
    record.normal = RTXDI_EncodeNormalizedVectorToSnorm2x16(surface.normal);
    record.geometricNormal = RTXDI_EncodeNormalizedVectorToSnorm2x16(surface.geoNormal);
    record.view = RTXDI_EncodeNormalizedVectorToSnorm2x16(surface.viewDir);
    record.roughness = surface.material.roughness;
    record.diffuse = surface.material.diffuseAlbedo;
    record.diffuseProbability = surface.diffuseProbability;
    record.specularF0 = surface.material.specularF0;
    record.specularF90 = surface.material.specularF90;
    record.cameraWeight = cameraWeight;
    return record;
}

RAB_Surface unpackRealtimeSurface(RealtimeSurface record)
{
    RAB_Surface surface;
    surface.worldPos = record.position;
    surface.viewDepth = record.viewDepth;
    surface.normal = RTXDI_DecodeNormalizedVectorFromSnorm2x16(record.normal);
    surface.geoNormal = RTXDI_DecodeNormalizedVectorFromSnorm2x16(record.geometricNormal);
    surface.viewDir = RTXDI_DecodeNormalizedVectorFromSnorm2x16(record.view);
    surface.diffuseProbability = record.diffuseProbability;
    surface.material.diffuseAlbedo = record.diffuse;
    surface.material.specularF0 = record.specularF0;
    surface.material.specularF90 = record.specularF90;
    surface.material.roughness = record.roughness;
    return surface;
}

bool realtimePixelInView(int2 pixel)
{
    Frame* frame = &realtimeArgs()->frame;
    return all(pixel >= 0) && pixel.x < int(frame.width) && pixel.y < int(frame.height);
}

uint realtimeSurfaceIndex(uint2 pixel)
{
    return pixel.y * realtimeArgs()->frame.width + pixel.x;
}

RAB_Surface RAB_GetGBufferSurface(int2 pixelPosition, bool prevFrame)
{
    if (!realtimePixelInView(pixelPosition))
        return RAB_EmptySurface();
    RealtimeLighting* lighting = &realtimeArgs()->lighting;
    if (prevFrame && lighting.previousSurfacesValid == 0u)
        return RAB_EmptySurface();
    RealtimeSurface* surfaces = prevFrame ? lighting.previousSurfaces : lighting.surfaces;
    return unpackRealtimeSurface(surfaces[realtimeSurfaceIndex(uint2(pixelPosition))]);
}

RAB_Material RAB_GetGBufferMaterial(int2 pixelPosition, bool prevFrame)
{
    return RAB_GetMaterial(RAB_GetGBufferSurface(pixelPosition, prevFrame));
}

// Reflects out-of-view positions back into the view.
int2 RAB_ClampSamplePositionIntoView(int2 pixelPosition, bool prevFrame)
{
    const int width = int(realtimeArgs()->frame.width);
    const int height = int(realtimeArgs()->frame.height);
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

bool realtimeVisible(float3 position, float3 geometricNormal, float3 direction,
    float distance)
{
    Frame* frame = &realtimeArgs()->frame;
    RayDesc ray;
    ray.Origin = position + geometricNormal
        * (dot(geometricNormal, direction) >= 0.0 ? 1.0e-4 : -1.0e-4);
    ray.Direction = direction;
    ray.TMin = 0.001;
    ray.TMax = distance < 1.0e15 ? max(distance - 2.0e-4, ray.TMin) : 1.0e30;
    RealtimeRayPayload payload = {};
    // Stays set unless the miss stage runs.
    payload.hit = RealtimeSurfaceHit;
    TraceRay(RaytracingAccelerationStructure(frame.topLevelAS),
        RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH
            | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
        RaytracingMaskMesh, 0, 2, 0, ray, payload);
    return payload.hit == RealtimeMiss;
}

// Sample positions on scene geometry (ReSTIR GI's) stop just short of their
// own surface.
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
// region), then directionals (the infinite region).
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
        + frame.rectLightCount + frame.directionalLightCount;
    return lightIndex < count ? int(lightIndex) : -1;
}

struct RAB_LightSample
{
    float3 position;
    // Unit direction toward the light; only meaningful for infinite lights,
    // whose position is not.
    float3 direction;
    float3 radiance;
    // Solid-angle density of the point given the light, or 1 for a delta
    // light whose radiance then already carries the inverse square falloff.
    float solidAnglePdf;
    uint kind;
};

RAB_LightSample RAB_EmptyLightSample()
{
    RAB_LightSample sample;
    sample.position = float3(0.0);
    sample.direction = float3(0.0, 0.0, 1.0);
    sample.radiance = float3(0.0);
    sample.solidAnglePdf = 0.0;
    sample.kind = RealtimeLightNone;
    return sample;
}

bool RAB_IsAnalyticLightSample(RAB_LightSample lightSample) { return true; }
float RAB_LightSampleSolidAnglePdf(RAB_LightSample lightSample) { return lightSample.solidAnglePdf; }
float3 RAB_LightSamplePosition(RAB_LightSample lightSample) { return lightSample.position; }
float3 RAB_LightSampleRadiance(RAB_LightSample lightSample) { return lightSample.radiance; }

void RAB_GetLightDirDistance(RAB_Surface surface, RAB_LightSample lightSample,
    out float3 o_lightDir, out float o_lightDistance)
{
    if (lightSample.kind == RealtimeLightDirectional)
    {
        o_lightDir = lightSample.direction;
        o_lightDistance = RealtimeDistantLightDistance;
        return;
    }
    const float3 toLight = lightSample.position - surface.worldPos;
    o_lightDistance = length(toLight);
    o_lightDir = o_lightDistance > 0.0 ? toLight / o_lightDistance : float3(0.0, 0.0, 1.0);
}

float spotAttenuation(SpotLight light, float3 fromLight)
{
    const float inner = light.innerConeAngle * AnalyticLightPi / 180.0;
    const float outer = light.outerConeAngle * AnalyticLightPi / 180.0;
    const float cosine = dot(normalize(light.direction), fromLight);
    const float cone = saturate((cosine - cos(outer)) / max(cos(inner) - cos(outer), 1.0e-6));
    return cone * cone * (3.0 - 2.0 * cone);
}

// Sphere-shaped point and spot lights, sampled exactly as the analytic light
// sampler in AnalyticLights.slang does, but from given random numbers so a
// reservoir can replay the sample for another surface.
RAB_LightSample sampleSphereShapedLight(float3 center, float radius, float3 radiance,
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
        sample.direction = toLight * rsqrt(distanceSquared);
        sample.radiance = radiance / distanceSquared;
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
    sample.direction = direction;
    sample.radiance = radiance / max(AnalyticLightPi * radius * radius, 1.0e-20);
    sample.solidAnglePdf = pdf;
    return sample;
}

RAB_LightSample RAB_SamplePolymorphicLight(RAB_LightInfo lightInfo, RAB_Surface surface,
    float2 uv)
{
    Frame* frame = &realtimeArgs()->frame;
    const float3 origin = surface.worldPos;
    RAB_LightSample sample = RAB_EmptyLightSample();
    if (lightInfo.kind == RealtimeLightPoint)
    {
        const PointLight light = frame.pointLights[lightInfo.index];
        sample = sampleSphereShapedLight(light.position, max(light.softRadius, 0.0),
            max(light.color, 0.0) * max(light.intensity, 0.0), origin, uv);
    }
    else if (lightInfo.kind == RealtimeLightSpot)
    {
        const SpotLight light = frame.spotLights[lightInfo.index];
        sample = sampleSphereShapedLight(light.position, max(light.softRadius, 0.0),
            max(light.color, 0.0) * max(light.intensity, 0.0), origin, uv);
        sample.radiance *= spotAttenuation(light, -sample.direction);
    }
    else if (lightInfo.kind == RealtimeLightRect)
    {
        const RectLight light = frame.rectLights[lightInfo.index];
        const float width = max(light.width, 0.0);
        const float height = max(light.height, 0.0);
        const float area = width * height;
        if (area <= 0.0)
            return sample;
        const float3 normal = normalize(light.direction);
        const float3 tangent = normalize(light.tangent);
        const float3 bitangent = normalize(cross(normal, tangent));
        const float2 square = uv - 0.5;
        const float3 position = light.position + tangent * (square.x * width)
            + bitangent * (square.y * height);
        const float3 delta = position - origin;
        const float distanceSquared = dot(delta, delta);
        if (distanceSquared <= 1.0e-12)
            return sample;
        const float distance = sqrt(distanceSquared);
        const float3 direction = delta / distance;
        float emitterCosine = dot(normal, -direction);
        emitterCosine = light.twoSided != 0u ? abs(emitterCosine) : max(emitterCosine, 0.0);
        if (emitterCosine <= 0.0)
            return sample;
        float barnDoorMask = 1.0;
        const float barnDoorLength = max(light.barnDoorLength, 0.0);
        if (barnDoorLength > 0.0 && light.barnDoorAngle < 89.9)
        {
            const float3 outgoing = -direction;
            const float forward = max(abs(dot(normal, outgoing)), 1.0e-5);
            const float3 local = position - light.position;
            const float expansion = barnDoorLength * tan(max(light.barnDoorAngle, 0.0)
                * AnalyticLightPi / 180.0);
            const float projectedU = dot(local, tangent) + barnDoorLength
                * dot(outgoing, tangent) / forward;
            const float projectedV = dot(local, bitangent) + barnDoorLength
                * dot(outgoing, bitangent) / forward;
            if (abs(projectedU) > width * 0.5 + expansion
                || abs(projectedV) > height * 0.5 + expansion)
                barnDoorMask = 0.0;
        }
        sample.position = position;
        sample.direction = direction;
        sample.radiance = max(light.color, 0.0) * (max(light.intensity, 0.0) * barnDoorMask);
        sample.solidAnglePdf = distanceSquared / (area * emitterCosine);
    }
    else if (lightInfo.kind == RealtimeLightDirectional)
    {
        const DirectionalLight light = frame.directionalLights[lightInfo.index];
        const float3 centerDirection = -normalize(light.direction);
        const float diameter = clamp(light.softAngle, 0.0, 180.0);
        float3 radiance = max(light.color, 0.0) * max(light.intensity, 0.0);
        sample.direction = centerDirection;
        sample.solidAnglePdf = 1.0;
        if (diameter > 0.0)
        {
            const float halfAngle = 0.5 * diameter * AnalyticLightPi / 180.0;
            const float oneMinusCosine = 1.0 - cos(halfAngle);
            sample.direction = sampleConeDirection(centerDirection, oneMinusCosine, uv);
            sample.solidAnglePdf = 1.0 / max(2.0 * AnalyticLightPi * oneMinusCosine, 1.0e-20);
            radiance /= max(AnalyticLightPi * sin(halfAngle) * sin(halfAngle), 1.0e-20);
        }
        sample.position = origin + sample.direction * RealtimeDistantLightDistance;
        sample.radiance = radiance;
    }
    else
        return sample;
    sample.kind = lightInfo.kind;
    if (!realtimeIsFinite(sample.radiance) || !(sample.solidAnglePdf > 0.0))
        return RAB_EmptyLightSample();
    return sample;
}

bool RAB_GetConservativeVisibility(RAB_Surface surface, RAB_LightSample lightSample)
{
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

// Radiance reflected toward the viewer per unit of the sample's solid-angle
// density; RTXDI resamples its luminance.
float3 realtimeLightSampleReflected(RAB_LightSample lightSample, RAB_Surface surface)
{
    if (!(lightSample.solidAnglePdf > 0.0))
        return float3(0.0);
    float3 direction;
    float distance;
    RAB_GetLightDirDistance(surface, lightSample, direction, distance);
    return realtimeSurfaceReflectance(surface, direction) * lightSample.radiance
        / lightSample.solidAnglePdf;
}

float RAB_GetLightSampleTargetPdfForSurface(RAB_LightSample lightSample, RAB_Surface surface)
{
    return RTXDI_Luminance(realtimeLightSampleReflected(lightSample, surface));
}

// Power over squared distance for any receiver inside the ReGIR cell. It only
// shapes the proposal, so it merely has to be positive wherever the light can
// contribute.
float RAB_GetLightTargetPdfForVolume(RAB_LightInfo light, float3 volumeCenter,
    float volumeRadius)
{
    Frame* frame = &realtimeArgs()->frame;
    float3 position;
    float power;
    float lightRadius = 0.0;
    if (light.kind == RealtimeLightPoint)
    {
        const PointLight record = frame.pointLights[light.index];
        position = record.position;
        power = record.selectionWeight;
        lightRadius = max(record.softRadius, 0.0);
    }
    else if (light.kind == RealtimeLightSpot)
    {
        const SpotLight record = frame.spotLights[light.index];
        position = record.position;
        power = record.selectionWeight;
        lightRadius = max(record.softRadius, 0.0);
    }
    else if (light.kind == RealtimeLightRect)
    {
        const RectLight record = frame.rectLights[light.index];
        position = record.position;
        power = record.selectionWeight;
        lightRadius = 0.5 * length(float2(record.width, record.height));
        // A one-sided rect cannot light a cell entirely behind it.
        if (record.twoSided == 0u && dot(normalize(record.direction),
                volumeCenter - position) < -(volumeRadius + lightRadius))
            return 0.0;
    }
    else
        return 0.0;
    const float distance = max(length(volumeCenter - position) - volumeRadius - lightRadius, 0.0);
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

// The environment is not an RTXDI light, so BRDF rays never find one and
// never need its density.
float2 RAB_GetEnvironmentMapRandXYFromDir(float3 worldDir) { return float2(0.0); }
float RAB_EvaluateEnvironmentMapSamplingPdf(float3 L) { return 0.0; }

// Analytic light shapes are not in the acceleration structure; BRDF samples
// cannot find them. Reporting a hit keeps RTXDI from trying the environment.
bool RAB_TraceRayForLocalLight(float3 origin, float3 direction, float tMin, float tMax,
    out uint o_lightIndex, out float2 o_randXY)
{
    o_lightIndex = RTXDI_InvalidLightIndex;
    o_randXY = float2(0.0);
    return true;
}

// ---------------------------------------------------------------------------
// ReSTIR GI
// ---------------------------------------------------------------------------

// ReSTIR GI resamples the diffuse lobe's indirect light only; the specular
// lobe keeps its path-traced estimate, which glossy reflections need.
float RAB_GetGISampleTargetPdfForSurface(float3 samplePosition, float3 sampleRadiance,
    RAB_Surface surface)
{
    const float3 direction = normalize(samplePosition - surface.worldPos);
    return RTXDI_Luminance(realtimeSurfaceBsdf(surface).evaluateDiffuse(direction)
        * sampleRadiance);
}

bool RAB_ValidateGISampleWithJacobian(inout float jacobian)
{
    // Sold-angle to area jacobians this far from one mean the sample moved
    // too much between the surfaces to be trusted.
    if (jacobian > 10.0 || jacobian < 1.0 / 10.0)
        return false;
    jacobian = clamp(jacobian, 1.0 / 3.0, 3.0);
    return true;
}

// ---------------------------------------------------------------------------
// Frame-level helpers for the realtime passes
// ---------------------------------------------------------------------------

bool realtimeRtxdiEnabled()
{
    return realtimeArgs()->lighting.enabled != 0u;
}

float3 realtimeCameraPosition(bool previous)
{
    return previous ? realtimeArgs()->view.previousCameraPosition
        : realtimeArgs()->view.cameraPosition;
}
