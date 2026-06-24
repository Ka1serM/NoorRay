#ifndef RAYTRACING_COMMON_SLANG
#define RAYTRACING_COMMON_SLANG

#include "Shared.h"
#include "Bindings.slang"
#include "Core/Core.slang"
#include "Core/RenderPolicy.slang"
#include "Core/Accumulation.slang"
#include "Core/Payload.slang"
#include "Geometry/Intersection.slang"
#include "Geometry/SurfaceData.slang"
#include "Bxdfs/Brdf.slang"
#include "Bxdfs/PbrBxdf.slang"
#include "Lights/DirectionalLight.slang"
#include "Lights/SceneLight.slang"
#include "Cameras/PerspectiveCamera.slang"
#include "Cameras/ThinLensCamera.slang"
#include "Cameras/OrthographicCamera.slang"
#include "Cameras/FisheyeCamera.slang"
#include "Cameras/RealisticCamera.slang"
#include "Samplers/R2Sampler.slang"
#include "Materials/PbrMaterial.slang"
#include "Lights/EnvironmentLight.slang"

[[vk::binding(0, 1)]] RWStructuredBuffer<int> queueCounters;

#ifdef USE_RAY_QUERY
[[vk::binding(0, 2)]] RaytracingAccelerationStructure topLevelAS;
#endif

struct PathState
{
    float3 Throughput; uint RngState;
    float3 Radiance;   uint Depth;
    float3 RayOrigin;  uint Flags;
    float3 RayDirection; uint PrimaryObjectIndex;
    float3 PrimaryAlbedo; uint PackedCounters;
    float3 PrimaryNormal; float LastBsdfPdf;
    float3 PrimaryPosition; uint Pad0;
};

struct PathRayWorkItem
{
    float3 Origin;
    uint   SampleIndex;
    float3 Direction;
    uint   Pad0;
};

struct HitWorkItem
{
    float3 RayDirection;
    uint   SampleIndex;
    float2 Barycentrics;
    uint   InstanceIndex;
    uint   PrimitiveIndex;
    uint   Pad0;
};

// Contribution is pre-multiplied by path throughput so ConnectCompute needs
// no read of pathStates — one shadow ray per path eliminates the write race.
struct ShadowWorkItem
{
    float3 Origin;
    float  TMin;
    float3 Direction;
    float  TMax;
    float3 Contribution; // = throughput * f * Li * cosTheta * MIS / (selPdf * lightPdf)
    uint   SampleIndex;
};

[[vk::binding(1, 1)]] RWStructuredBuffer<PathState, Std430DataLayout>         pathStates;
[[vk::binding(2, 1)]] RWStructuredBuffer<PathRayWorkItem, Std430DataLayout>   inputRayQueue;
[[vk::binding(3, 1)]] RWStructuredBuffer<PathRayWorkItem, Std430DataLayout>   outputRayQueue;
[[vk::binding(4, 1)]] RWStructuredBuffer<HitWorkItem, Std430DataLayout>       hitQueue;
[[vk::binding(5, 1)]] RWStructuredBuffer<ShadowWorkItem, Std430DataLayout>    shadowQueue;

static const uint QUEUE_COUNTER_ACTIVE = 0u;
static const uint QUEUE_COUNTER_NEXT   = 1u;
static const uint QUEUE_COUNTER_HIT    = 2u;
static const uint QUEUE_COUNTER_SHADOW = 3u;
static const uint QUEUE_GROUP_SIZE     = 128u;

static const uint PATH_COUNTER_DIFFUSE_SHIFT      = 0u;
static const uint PATH_COUNTER_SPECULAR_SHIFT     = 8u;
static const uint PATH_COUNTER_TRANSMISSION_SHIFT = 16u;
static const uint PATH_COUNTER_HIT_SHIFT          = 24u;
static const uint PATH_COUNTER_MASK               = 0xFFu;

struct TraceHit
{
    uint   InstanceIndex;
    uint   PrimitiveIndex;
    float3 Barycentrics;
    float  T;
};

uint appendNextRay(PathRayWorkItem item)
{
    int index;
    InterlockedAdd(queueCounters[QUEUE_COUNTER_NEXT], 1, index);
    outputRayQueue[uint(index)] = item;
    return uint(index);
}

uint appendHit(HitWorkItem item)
{
    int index;
    InterlockedAdd(queueCounters[QUEUE_COUNTER_HIT], 1, index);
    hitQueue[uint(index)] = item;
    return uint(index);
}

uint appendShadow(ShadowWorkItem item)
{
    int index;
    InterlockedAdd(queueCounters[QUEUE_COUNTER_SHADOW], 1, index);
    shadowQueue[uint(index)] = item;
    return uint(index);
}

bool popQueueIndex(uint counterIndex, out uint workIndex)
{
    int previousCount;
    InterlockedAdd(queueCounters[counterIndex], -1, previousCount);
    if (previousCount <= 0) {
        queueCounters[counterIndex] = 0;
        workIndex = 0u;
        return false;
    }
    workIndex = uint(previousCount - 1);
    return true;
}

int2 computeShadingScreenSize(int2 screenSize) { return effectiveShadingScreenSize(screenSize); }

uint computeShadingPixelCount(int2 screenSize)
{
    int2 s = computeShadingScreenSize(screenSize);
    return uint(max(s.x, 1) * max(s.y, 1));
}

uint computePrimarySampleCount()
{
    return uint(useInteractivePathTracingPreview() ? 1 : effectivePrimarySampleCount());
}

uint computeSampleSlot(int2 shadingPixel, int2 shadingScreenSize, uint sampleIndex, uint sampleCount)
{
    return (uint(shadingPixel.y * shadingScreenSize.x + shadingPixel.x) * sampleCount) + sampleIndex;
}

uint packPathCounters(uint diffuse, uint specular, uint transmission, bool hitAnything)
{
    return ((diffuse      & PATH_COUNTER_MASK) << PATH_COUNTER_DIFFUSE_SHIFT)      |
           ((specular     & PATH_COUNTER_MASK) << PATH_COUNTER_SPECULAR_SHIFT)     |
           ((transmission & PATH_COUNTER_MASK) << PATH_COUNTER_TRANSMISSION_SHIFT) |
           ((hitAnything ? 1u : 0u)            << PATH_COUNTER_HIT_SHIFT);
}

uint getDiffuseCount(PathState state)      { return (state.PackedCounters >> PATH_COUNTER_DIFFUSE_SHIFT)      & PATH_COUNTER_MASK; }
uint getSpecularCount(PathState state)     { return (state.PackedCounters >> PATH_COUNTER_SPECULAR_SHIFT)     & PATH_COUNTER_MASK; }
uint getTransmissionCount(PathState state) { return (state.PackedCounters >> PATH_COUNTER_TRANSMISSION_SHIFT) & PATH_COUNTER_MASK; }
bool getHitAnything(PathState state)       { return ((state.PackedCounters >> PATH_COUNTER_HIT_SHIFT) & 1u) != 0u; }

void setHitAnything(inout PathState state, bool v)
{
    state.PackedCounters = (state.PackedCounters & ~(1u << PATH_COUNTER_HIT_SHIFT)) | ((v ? 1u : 0u) << PATH_COUNTER_HIT_SHIFT);
}

void incrementDiffuseCount(inout PathState state)      { state.PackedCounters += (1u << PATH_COUNTER_DIFFUSE_SHIFT); }
void incrementSpecularCount(inout PathState state)     { state.PackedCounters += (1u << PATH_COUNTER_SPECULAR_SHIFT); }
void incrementTransmissionCount(inout PathState state) { state.PackedCounters += (1u << PATH_COUNTER_TRANSMISSION_SHIFT); }

void initializePathState(uint sampleIndex, float3 origin, float3 direction, uint rngState)
{
    PathState state;
    state.Throughput        = float3(1.0);
    state.RngState          = rngState == 0u ? 1u : rngState;
    state.Radiance          = float3(0.0);
    state.Depth             = 0u;
    state.RayOrigin         = origin;
    state.Flags             = 0u;
    state.RayDirection      = direction;
    state.PrimaryObjectIndex = InvalidInstance;
    state.PrimaryAlbedo     = float3(0.0);
    state.PackedCounters    = 0u;
    state.PrimaryNormal     = float3(0.0);
    state.LastBsdfPdf       = 0.0;
    state.PrimaryPosition   = float3(0.0);
    state.Pad0              = 0u;
    pathStates[sampleIndex] = state;
}

bool traceSceneRay(float3 rayOrigin, float3 rayDirection, float tMin, float tMax, out TraceHit traceHit)
{
#ifdef USE_RAY_QUERY
    RayDesc ray;
    ray.Origin    = rayOrigin;
    ray.Direction = rayDirection;
    ray.TMin      = tMin;
    ray.TMax      = tMax;

    RayQuery<RAY_FLAG_FORCE_OPAQUE> rayQuery;
    rayQuery.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xFF, ray);
    while (rayQuery.Proceed()) { }

    if (rayQuery.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
        traceHit.InstanceIndex  = InvalidInstance;
        traceHit.PrimitiveIndex = InvalidInstance;
        traceHit.Barycentrics   = float3(0.0);
        traceHit.T              = tMax;
        return false;
    }

    float2 bary = rayQuery.CommittedTriangleBarycentrics();
    traceHit.InstanceIndex  = rayQuery.CommittedInstanceID();
    traceHit.PrimitiveIndex = rayQuery.CommittedPrimitiveIndex();
    traceHit.Barycentrics   = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
    traceHit.T              = rayQuery.CommittedRayT();
    return true;
#else
    HitInfo hit = traceScene(rayOrigin, rayDirection, tMin, tMax);
    traceHit.InstanceIndex  = hit.instanceIndex;
    traceHit.PrimitiveIndex = hit.primitiveIndex;
    traceHit.Barycentrics   = hit.barycentrics;
    traceHit.T              = hit.t;
    return hit.instanceIndex != InvalidInstance;
#endif
}

bool traceShadowPath(float3 rayOrigin, float3 rayDirection, float tMin, float tMax)
{
#ifdef USE_RAY_QUERY
    RayDesc ray;
    ray.Origin    = rayOrigin;
    ray.Direction = rayDirection;
    ray.TMin      = tMin;
    ray.TMax      = tMax;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE> rayQuery;
    rayQuery.TraceRayInline(topLevelAS, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE, 0xFF, ray);
    while (rayQuery.Proceed()) { }
    return rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
#else
    return traceShadowRay(rayOrigin, rayDirection, tMin, tMax);
#endif
}

// ── NEE light selection ───────────────────────────────────────────────────────

// Power-proportional weights for each light category at a given surface point.
struct NEEWeights {
    float dir;    // directional sun
    float env;    // environment map
    float scene;  // aggregate of all active scene lights
    float total;
};

NEEWeights computeNEEWeights(float3 worldPos, float3 shadingNormal)
{
    NEEWeights w;

    w.dir = 0.0;
    if (hasDirectionalLight(sceneSettings.environment)) {
        float3 lightDir = getDirectionalLightDirection(sceneSettings.environment);
        float  NdotL    = max(dot(shadingNormal, lightDir), 0.0);
        w.dir = sceneSettings.environment.directionalIntensity * NdotL;
    }

    w.env = 0.0;
    if (sceneSettings.environment.textureIndex >= 0 && sceneSettings.environment.cdfTextureIndex >= 0)
        w.env = sceneSettings.environment.lightingExposureScale;

    w.scene = 0.0;
    uint lightCount, lightStride;
    sceneLightsBuffer.GetDimensions(lightCount, lightStride);
    for (uint i = 0; i < lightCount; i++) {
        if (!isSceneLightActive(int(i))) continue;
        LightGpu light = sceneLightsBuffer[i];
        float3 d = light.position - worldPos;
        float dist2 = max(dot(d, d), 0.01);
        w.scene += luminance(light.color) * light.intensity / dist2;
    }

    w.total = w.dir + w.env + w.scene;
    return w;
}

// Approximate env selection probability for use in the miss shader where
// no surface position is available — scene point lights are negligible from
// directions that escape to infinity, so they are excluded.
float approximateEnvSelectionProb()
{
    float wDir = hasDirectionalLight(sceneSettings.environment)
        ? sceneSettings.environment.directionalIntensity : 0.0;
    float wEnv = (sceneSettings.environment.textureIndex >= 0 && sceneSettings.environment.cdfTextureIndex >= 0)
        ? sceneSettings.environment.lightingExposureScale : 0.0;
    float wTotal = wDir + wEnv;
    if (wTotal <= EPSILON || wEnv <= EPSILON) return 0.0;
    return wEnv / wTotal;
}

// ── Single NEE shadow ray per bounce ─────────────────────────────────────────
//
// One light is selected stochastically (power-weighted) and ONE shadow work
// item is queued with the contribution PRE-MULTIPLIED by the path throughput.
// Because each path vertex emits at most one shadow ray, ConnectCompute's
// write to pathStates[SampleIndex].Radiance is never concurrent — the race
// condition that existed when multiple rays were queued is eliminated by
// design.

void queueNEEShadowRay(
    uint   sampleIndex,
    float3 throughput,
    float3 worldPos,
    float3 worldShadowPos,
    float3 shadingNormal,
    float3 geometricNormal,
    float3 viewDir,
    float3 albedo,
    float  specular,
    float  metallic,
    float  roughness,
    float  swSpecular,
    inout uint rngState)
{
    NEEWeights weights = computeNEEWeights(worldPos, shadingNormal);
    if (weights.total <= EPSILON) return;

    float u = rand(rngState) * weights.total;

    // ── Directional (delta) light ──────────────────────────────────────────
    if (u < weights.dir) {
        float selProb = weights.dir / weights.total;
        float3 sunDir = sampleDirectionalLightDirection(sceneSettings.environment, float2(rand(rngState), rand(rngState)));
        float NdotLGeom = max(dot(geometricNormal, sunDir), 0.0);
        float NdotL = max(dot(shadingNormal, sunDir), 0.0);
        if (NdotLGeom <= 0.0) return;
        if (NdotL <= 0.0) return;

        PbrBxdf bxdf;
        BxdfEval eval = bxdf.evalOpaque(viewDir, shadingNormal, geometricNormal,
            sunDir, albedo, specular, metallic, roughness, swSpecular);

        // Delta light: no BSDF-side MIS (BSDF cannot sample a delta direction).
        float3 Li = float3(sceneSettings.environment.directionalIntensity);
        float3 contribution = throughput * eval.f * Li * NdotL / max(selProb, EPSILON);

        float tMin;
        ShadowWorkItem item;
        item.Origin       = offsetShadowRayOrigin(worldPos, worldShadowPos, geometricNormal, sunDir, NdotLGeom, tMin);
        item.TMin         = tMin;
        item.Direction    = sunDir;
        item.TMax         = 1000.0;
        item.Contribution = contribution;
        item.SampleIndex  = sampleIndex;
        appendShadow(item);
        return;
    }
    u -= weights.dir;

    // ── Environment map (area light) ──────────────────────────────────────
    if (u < weights.env) {
        float selProb = weights.env / weights.total;
        float lightPdf;
        float3 lightDir = sampleEnvironmentDirection(sceneSettings.environment, rngState, lightPdf);
        if (lightPdf <= 0.0) return;
        float NdotLGeom = max(dot(geometricNormal, lightDir), 0.0);
        float NdotL = max(dot(shadingNormal, lightDir), 0.0);
        if (NdotLGeom <= 0.0) return;
        if (NdotL <= 0.0) return;

        float3 Li = sampleEnvironmentMapColor(lightDir, sceneSettings.environment, sceneSettings.environment.lightingExposureScale);
        PbrBxdf bxdf;
        BxdfEval eval = bxdf.evalOpaque(viewDir, shadingNormal, geometricNormal,
            lightDir, albedo, specular, metallic, roughness, swSpecular);
        if (eval.pdf <= 0.0) return;

        // Area light: balance heuristic MIS between NEE and BSDF paths.
        float jointPdf = selProb * lightPdf;
        float wNEE     = powerHeuristic(jointPdf, eval.pdf);
        float3 contribution = throughput * eval.f * Li * NdotL * wNEE / max(jointPdf, EPSILON);

        float tMin;
        ShadowWorkItem item;
        item.Origin       = offsetShadowRayOrigin(worldPos, worldShadowPos, geometricNormal, lightDir, NdotLGeom, tMin);
        item.TMin         = tMin;
        item.Direction    = lightDir;
        item.TMax         = 1000.0;
        item.Contribution = contribution;
        item.SampleIndex  = sampleIndex;
        appendShadow(item);
        return;
    }
    u -= weights.env;

    // ── Scene lights (delta point / spot / rect) ───────────────────────────
    if (weights.scene <= EPSILON) return;
    float groupSelProb = weights.scene / weights.total;

    // Pick one scene light proportional to estimated irradiance.
    uint lightCount, lightStride;
    sceneLightsBuffer.GetDimensions(lightCount, lightStride);

    float uLight   = rand(rngState) * weights.scene;
    float accum    = 0.0;
    int   chosen   = -1;
    float chosenW  = 0.0;
    for (uint i = 0; i < lightCount; i++) {
        if (!isSceneLightActive(int(i))) continue;
        LightGpu light = sceneLightsBuffer[i];
        float3 d = light.position - worldPos;
        float  w = luminance(light.color) * light.intensity / max(dot(d, d), 0.01);
        accum += w;
        if (uLight <= accum && chosen < 0) {
            chosen  = int(i);
            chosenW = w;
        }
    }
    if (chosen < 0) return;

    float lightSelProb = groupSelProb * (chosenW / max(weights.scene, EPSILON));

    LightLiSample ls = sampleSceneLight(chosen, worldPos, float2(rand(rngState), rand(rngState)));
    if (ls.pdf <= EPSILON || dot(ls.Li, ls.Li) <= EPSILON) return;
    float NdotLGeom = max(dot(geometricNormal, ls.wi), 0.0);
    float NdotL = max(dot(shadingNormal, ls.wi), 0.0);
    if (NdotLGeom <= 0.0) return;
    if (NdotL <= 0.0) return;

    PbrBxdf bxdf;
    BxdfEval eval = bxdf.evalOpaque(viewDir, shadingNormal, geometricNormal,
        ls.wi, albedo, specular, metallic, roughness, swSpecular);
    if (eval.pdf <= 0.0) return;

    // Scene lights have pdf=1 (delta), so no MIS needed — divide by joint sel prob only.
    float3 contribution = throughput * eval.f * ls.Li * NdotL / max(lightSelProb * ls.pdf, EPSILON);

    float tMin;
    LightGpu chosenLight = sceneLightsBuffer[chosen];
    ShadowWorkItem item;
    item.Origin       = offsetShadowRayOrigin(worldPos, worldShadowPos, geometricNormal, ls.wi, NdotLGeom, tMin);
    item.TMin         = tMin;
    item.Direction    = ls.wi;
    item.TMax         = max(chosenLight.range, 1000.0);
    item.Contribution = contribution;
    item.SampleIndex  = sampleIndex;
    appendShadow(item);
}

// ── Environment miss radiance with corrected MIS ──────────────────────────────

float3 evaluateMissRadiance(float3 worldRayDirection, uint depth, uint previousFlags, float lastBsdfPdf, out uint missFlags)
{
    missFlags = uint(BounceType.RAY_TERMINATED);

    bool hiddenCameraEnvironment =
        sceneSettings.environment.visible == 0 ||
        sceneSettings.renderSettings.transparentBackground != 0;
    bool cameraVisibilityRay =
        depth == 0u ||
        (previousFlags & (uint(BounceType.RAY_TRANSPARENT) | uint(BounceType.BOUNCE_TRANSMIT))) != 0u;

    float3 envColor = depth == 0u
        ? evaluateEnvironmentVisibleRadiance(worldRayDirection, sceneSettings.environment)
        : evaluateEnvironmentRadiance(worldRayDirection, sceneSettings.environment);

    if (hiddenCameraEnvironment && cameraVisibilityRay) {
        envColor = float3(0.0);
        missFlags |= uint(BounceType.ENV_TRANSPARENT);
    }

    if (depth > 0u && lastBsdfPdf > 0.0) {
        // MIS: weight the BSDF-sampled env contribution against the NEE joint PDF.
        // The joint PDF accounts for the probability of having selected the env map
        // AND the probability of sampling this particular direction from it.
        float envSelProb  = approximateEnvSelectionProb();
        float envDirPdf   = evaluateEnvironmentPdfForDirection(worldRayDirection, sceneSettings.environment);
        float neeJointPdf = envSelProb * envDirPdf;
        float wBsdf       = powerHeuristic(lastBsdfPdf, neeJointPdf);
        envColor *= wBsdf;
    }

    return envColor;
}

// ── BSDF sampling helpers ─────────────────────────────────────────────────────

void handleDielectricBSDF(float3 viewDir, float3 shadingNormal, float roughness, float ior, float3 transmissionColor, inout Payload payload)
{
    float3 shadingFacingNormal = dot(shadingNormal, viewDir) < 0.0 ? -shadingNormal : shadingNormal;
    float3 halfVector = sampleVisibleHalfVectorGGX(viewDir, shadingFacingNormal, roughness, payload.RngState);
    float  VdotH      = max(dot(viewDir, halfVector), 0.0);
    float3 incident   = fastNormalize(-viewDir);
    float  etaI = 1.0, etaT = ior;

    if (dot(incident, shadingNormal) > 0.0) { etaI = ior; etaT = 1.0; }

    // Exact dielectric Fresnel for reflection/refraction split.
    float  reflectProb  = max(fresnelDielectric(VdotH, etaI, etaT), EPSILON);
    float3 refractedDir = refract(incident, halfVector, etaI / etaT);
    bool   totalInternalReflection = dot(refractedDir, refractedDir) < EPSILON;

    if (totalInternalReflection || rand(payload.RngState) < reflectProb) {
        float3 reflectedDir = reflect(-viewDir, halfVector);
        float3 brdf = evaluateSpecularBRDF(shadingFacingNormal, viewDir, reflectedDir, float3(reflectProb), roughness, halfVector);
        float  pdf  = max(pdfSpecular(viewDir, shadingFacingNormal, halfVector, roughness), EPSILON);
        payload.Attenuation  = (brdf * max(dot(shadingFacingNormal, reflectedDir), 0.0)) / max(pdf * reflectProb, EPSILON);
        payload.NextDirection = reflectedDir;
        payload.LastBsdfPdf  = pdf;
        payload.Flags       |= uint(BounceType.BOUNCE_SPECULAR);
    } else {
        payload.Attenuation  = transmissionColor / max(1.0 - reflectProb, EPSILON);
        payload.NextDirection = refractedDir;
        payload.LastBsdfPdf  = 0.0;
        payload.Flags       |= uint(BounceType.BOUNCE_TRANSMIT);
    }
}

void handleOpaqueBSDF(
    float3 viewDir,
    float3 shadingNormal,
    float3 geometricNormal,
    float3 albedo,
    float  metallic,
    float  specular,
    float  roughness,
    inout Payload payload)
{
    float  dielectricF0 = dielectricF0FromSpecular(specular);
    float3 F0           = lerp(float3(dielectricF0), albedo, metallic);
    float  swSpecular   = computeSpecularSamplingWeight(viewDir, shadingNormal, albedo, F0, metallic, roughness, specular);

    float3 halfVector;
    bool   sampledSpecular = rand(payload.RngState) < swSpecular;
    if (sampledSpecular) {
        halfVector = sampleVisibleHalfVectorGGX(viewDir, shadingNormal, roughness, payload.RngState);
        payload.NextDirection = reflect(-viewDir, halfVector);
        for (int i = 0; i < 2 && dot(geometricNormal, payload.NextDirection) <= 0.0; ++i) {
            halfVector = sampleVisibleHalfVectorGGX(viewDir, shadingNormal, roughness, payload.RngState);
            payload.NextDirection = reflect(-viewDir, halfVector);
        }
    } else {
        payload.NextDirection = sampleDiffuse(shadingNormal, payload.RngState);
        halfVector = fastNormalize(viewDir + payload.NextDirection);
    }

    if (dot(geometricNormal, payload.NextDirection) <= 0.0) {
        payload.NextDirection = sampleDiffuse(geometricNormal, payload.RngState);
        halfVector = fastNormalize(viewDir + payload.NextDirection);
        sampledSpecular = false;
    }

    float  VdotH = max(dot(viewDir, halfVector), 0.0);
    float  NdotL = max(dot(shadingNormal, payload.NextDirection), 0.0);
    float  NdotV = max(dot(shadingNormal, viewDir), 0.0);
    float  VdotL = dot(viewDir, payload.NextDirection);
    float3 F     = fresnelMaterial(F0, metallic, VdotH, specular);

    float3 diffuseBRDF  = (1.0 - F) * evaluateDiffuseBRDF(albedo, metallic, roughness, NdotL, NdotV, VdotL);
    float3 specularBRDF = evaluateSpecularBRDF(shadingNormal, viewDir, payload.NextDirection, F, roughness, halfVector);
    float3 combinedBRDF = diffuseBRDF + specularBRDF;

    float specPdf    = sampledSpecular ? pdfSpecular(viewDir, shadingNormal, halfVector, roughness) : 0.0;
    float diffPdf    = pdfDiffuse(shadingNormal, payload.NextDirection);
    float combinedPdf = max(swSpecular * specPdf + (1.0 - swSpecular) * diffPdf, EPSILON);

    payload.Attenuation  = combinedBRDF * NdotL / combinedPdf;
    payload.LastBsdfPdf  = combinedPdf;
    payload.Flags       |= sampledSpecular ? uint(BounceType.BOUNCE_SPECULAR) : uint(BounceType.BOUNCE_DIFFUSE);
}

void enqueueContinuationRay(uint sampleIndex, PathState state)
{
    PathRayWorkItem nextRay;
    nextRay.Origin      = state.RayOrigin;
    nextRay.SampleIndex = sampleIndex;
    nextRay.Direction   = state.RayDirection;
    nextRay.Pad0        = 0u;
    appendNextRay(nextRay);
    pathStates[sampleIndex] = state;
}

#endif
