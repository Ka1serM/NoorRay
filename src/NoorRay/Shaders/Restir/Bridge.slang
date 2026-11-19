import NoorRHI;
import Wavefront.Records;
import Wavefront.Direct;
import Wavefront.Extend;
import Wavefront.SurfaceShading;
import Common.Math;
import Common.LinkConstants;
import Common.Random;
import Common.RayCone;
import Common.Sampling;
import Lights.Lights;
import Lights.Environment;
import Bsdf.Principled;
import Materials.Material;
import Scene.Camera;
import Scene.SceneSurface;
import Spectral.Wavelengths;

#include "NoorRay/Shaders/Restir/RestirRecords.h"

static const float restirDistantLightDistance = 1.0e6;
static const uint restirEnvironmentLight = environmentLight & 0x7fffffffu;

static RestirFrame restirFrame;
static SceneRecord restirScene;
static CameraRecord restirCamera;
static Wavelengths restirWavelengths;
static float restirWavelengthSample;
static uint restirPixelSeed;

void useRestirWavelengths(float u) {
    restirWavelengthSample = u;
    restirWavelengths = sampleWavelengths(restirScene.spectral, u, false);
}

#define RTXDI_SLANG
#define RTXDI_PT_RESERVOIR_BUFFER restirFrame.reservoirs
#define RTXDI_NEIGHBOR_OFFSETS_BUFFER restirFrame.neighborOffsets
#define RAB_DISTANT_LIGHT_DISTANCE restirDistantLightDistance
#define RTXDI_PT_BEGIN_SHIFT(reservoir) useRestirWavelengths(reservoir.wavelengthSample)
#define RTXDI_PT_FINISH_INITIAL_SAMPLE(reservoir) reservoir.wavelengthSample = restirWavelengthSample

struct RAB_Surface {
    float3 position;
    float depth;
    float3 normal;
    uint material;
    float3 view;
    uint cone;
    uint instance;
    uint primitive;
    float2 barycentrics;
};

struct RAB_Material {
    uint index;
};

struct RAB_LightInfo {
    uint light;
};

struct RAB_LightSample {
    float3 position;
    float solidAnglePdf;
    float4 radiance;
    bool delta;
};

struct RAB_RayPayload {
    float committedRayT;
};

struct RAB_PathTracerUserData {
    uint pathType;
};

#include "Rtxdi/PT/InitialSamplingPathTracerContext.hlsli"
#include "Rtxdi/PT/HybridShiftPathTracerContext.hlsli"
#define RTXDI_PT_CONTEXT_NAME RTXDI_PathTracerContextInitial
#define RTXDI_PT_CONTEXT_TYPE RTXDI_InitialSamplingPathTracerContext
#define RTXDI_PT_CONTEXT_INIT RTXDI_InitializePathTracerContextInitial
#include "Rtxdi/PT/PathTracerContext.hlsli"
#undef RTXDI_PT_CONTEXT_NAME
#undef RTXDI_PT_CONTEXT_TYPE
#undef RTXDI_PT_CONTEXT_INIT
#define RTXDI_PT_CONTEXT_NAME RTXDI_PathTracerContextHybrid
#define RTXDI_PT_CONTEXT_TYPE RTXDI_HybridShiftPathTracerContext
#define RTXDI_PT_CONTEXT_INIT RTXDI_InitializePathTracerContextHybrid
#include "Rtxdi/PT/PathTracerContext.hlsli"
#undef RTXDI_PT_CONTEXT_NAME
#undef RTXDI_PT_CONTEXT_TYPE
#undef RTXDI_PT_CONTEXT_INIT
#include "Rtxdi/PT/PathTracerRandomContext.hlsli"
#include "Rtxdi/PT/InitialSampling.hlsli"
#include "Rtxdi/PT/TemporalResampling.hlsli"
#include "Rtxdi/PT/SpatialResampling.hlsli"

void beginRestir(RestirFrame frame, uint2 pixel) {
    restirFrame = frame;
    const WavefrontFrame wavefront = frame.wavefront[0];
    restirScene = wavefront.scene[0];
    restirCamera = wavefront.camera[0];
    restirPixelSeed = hashCombine(hashCombine(pixel.x, pixel.y), frame.seed);
}

uint restirPixelIndex(uint2 pixel) { return pixel.y * restirFrame.width + pixel.x; }

float3 previousCameraPosition(CameraRecord camera) {
    return float3(camera.previousCameraToWorld[0].w, camera.previousCameraToWorld[1].w, camera.previousCameraToWorld[2].w);
}

struct RestirFilter : IHitFilter {
    float3 direction;
    uint seed;
    HitVerdict accepted;

    [mutating]
    HitVerdict filter(RayCandidate candidate) {
        if (!hasAlphaTest)
            return accepted;
        const uint material = triangleMaterial(restirScene, candidate.instanceId, candidate.primitiveIndex);
        if ((restirScene.materials[material].flags & materialFlagAlphaTest) == 0)
            return accepted;
        const Surface surface =
            loadSurface(restirScene, candidate.instanceId, candidate.primitiveIndex, candidate.barycentrics, direction, candidate.t, makeRayCone(0.0, 0.0));
        Random random = makeRandom(hashCombine(hashCombine(seed, candidate.instanceId), candidate.primitiveIndex));
        return random.next() < evaluateOpacity(restirScene, surface.context, surface.material) ? accepted : HitVerdict.Ignore;
    }
};

RestirFilter makeRestirFilter(float3 direction, uint salt, HitVerdict accepted) {
    RestirFilter filter;
    filter.direction = direction;
    filter.seed = hashCombine(restirPixelSeed, salt);
    filter.accepted = accepted;
    return filter;
}

bool traceRestirSurface(RayDesc ray, uint mask, uint salt, out RayHit hit) {
    RestirFilter filter = makeRestirFilter(ray.Direction, salt, HitVerdict.Accept);
    hit = (RayHit)0;
    return traceRayInline(restirScene.tlas, RAY_FLAG_NONE, mask, ray, filter, hit);
}

bool isRestirVisible(RAB_Surface surface, float3 target) {
    const float3 offset = target - surface.position;
    const float distance = length(offset);
    if (distance <= 0.0)
        return false;
    RayDesc ray;
    ray.Direction = offset / distance;
    ray.Origin = offsetAlong(surface.position, surface.normal, ray.Direction);
    ray.TMin = 0.0;
    ray.TMax = distance * 0.999;
    RestirFilter filter = makeRestirFilter(ray.Direction, 1u, HitVerdict.AcceptAndEnd);
    RayHit hit;
    return !traceRayInline(restirScene.tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, instanceMaskShadow, ray, filter, hit);
}

RAB_Surface makeRestirSurface(Surface surface, float2 barycentrics, float distance, RayCone cone) {
    RAB_Surface result;
    result.position = surface.context.position.xyz;
    result.depth = distance;
    result.normal = surface.context.geometricNormal.xyz;
    result.material = surface.material;
    result.view = surface.context.incoming.xyz;
    result.cone = packRayCone(cone);
    result.instance = surface.instance;
    result.primitive = surface.primitive;
    result.barycentrics = barycentrics;
    return result;
}

bool RAB_IsSurfaceValid(RAB_Surface surface) { return surface.depth > 0.0; }

RestirSurfaceRecord packRestirSurface(RAB_Surface surface) {
    RestirSurfaceRecord record;
    record.position = surface.position;
    record.depth = surface.depth;
    record.normal = RAB_IsSurfaceValid(surface) ? packOctahedral(surface.normal) : 0u;
    record.view = packOctahedral(surface.view);
    record.material = surface.material;
    record.cone = surface.cone;
    record.instance = surface.instance;
    record.primitive = surface.primitive;
    record.barycentrics = surface.barycentrics;
    return record;
}

RAB_Surface unpackRestirSurface(RestirSurfaceRecord record) {
    RAB_Surface surface;
    surface.position = record.position;
    surface.depth = record.depth;
    surface.normal = unpackOctahedral(record.normal);
    surface.material = record.material;
    surface.view = unpackOctahedral(record.view);
    surface.cone = record.cone;
    surface.instance = record.instance;
    surface.primitive = record.primitive;
    surface.barycentrics = record.barycentrics;
    return surface;
}

Surface loadRestirSurface(RAB_Surface surface) {
    return loadSurface(restirScene, surface.instance, surface.primitive, surface.barycentrics, -surface.view, surface.depth, unpackRayCone(surface.cone));
}

BsdfInputs restirMaterial(Surface surface) { return evaluateMaterial(restirScene, surface.context, surface.material); }

PrincipledBsdf makeRestirBsdf(Surface surface, BsdfInputs inputs) {
    const ShadingContext context = surface.context;
    return makePrincipled(inputs, context.incoming.xyz, context.geometricNormal.xyz, isBackfacing(context), restirWavelengths, sceneEnergyLut(restirScene),
                          false);
}

PrincipledBsdf restirBsdf(RAB_Surface surface) {
    const Surface loaded = loadRestirSurface(surface);
    return makeRestirBsdf(loaded, restirMaterial(loaded));
}

BsdfValue evaluateRestirBsdf(RAB_Surface surface, float3 direction) { return restirBsdf(surface).evaluate(direction); }

RAB_Surface RAB_EmptySurface() { return (RAB_Surface)0; }

float3 RAB_GetSurfaceWorldPos(RAB_Surface surface) { return surface.position; }

void RAB_SetSurfaceWorldPos(inout RAB_Surface surface, float3 position) { surface.position = position; }

float3 RAB_GetSurfaceNormal(RAB_Surface surface) { return surface.normal; }

void RAB_SetSurfaceNormal(inout RAB_Surface surface, float3 normal) { surface.normal = normal; }

float3 RAB_GetSurfaceViewDir(RAB_Surface surface) { return surface.view; }

float RAB_GetSurfaceLinearDepth(RAB_Surface surface) { return surface.depth; }

float RAB_GetSurfaceRoughness(RAB_Surface surface) { return restirScene.materials[surface.material].metallicRoughnessIorSpecular.y; }

RAB_Material RAB_GetMaterial(RAB_Surface surface) {
    RAB_Material material;
    material.index = surface.material;
    return material;
}

bool RAB_AreMaterialsSimilar(RAB_Material a, RAB_Material b) { return a.index == b.index; }

RAB_Surface RAB_GetGBufferSurface(int2 pixel, bool previous) {
    if (any(pixel < 0) || pixel.x >= int(restirFrame.width) || pixel.y >= int(restirFrame.height))
        return RAB_EmptySurface();
    const Ptr<RestirSurfaceRecord> surfaces = previous ? restirFrame.previousSurfaces : restirFrame.surfaces;
    if (isNull(surfaces))
        return RAB_EmptySurface();
    return unpackRestirSurface(surfaces[restirPixelIndex(uint2(pixel))]);
}

int2 RAB_ClampSamplePositionIntoView(int2 pixel, bool previous) { return clamp(pixel, int2(0, 0), int2(restirFrame.width, restirFrame.height) - 1); }

float RAB_SurfaceEvaluateBrdfPdf(RAB_Surface surface, float3 direction) { return evaluateRestirBsdf(surface, direction).pdf; }

float4 RAB_GetReflectedBsdfRadianceForSurface(float3 samplePosition, float4 radiance, RAB_Surface surface) {
    const float3 offset = samplePosition - surface.position;
    const float distance = length(offset);
    if (distance <= 0.0)
        return float4(0.0, 0.0, 0.0, 0.0);
    return evaluateRestirBsdf(surface, offset / distance).value * radiance;
}

float4 RAB_GetPTSampleTargetPdfForSurface(float3 samplePosition, float4 radiance, RAB_Surface surface) {
    return RAB_GetReflectedBsdfRadianceForSurface(samplePosition, radiance, surface);
}

bool RAB_GetConservativeVisibility(RAB_Surface surface, float3 samplePosition) { return isRestirVisible(surface, samplePosition); }

uint restirLight(uint index) { return index == restirEnvironmentLight ? environmentLight : index; }

uint rtxdiLight(uint light) { return light & 0x7fffffffu; }

RAB_LightInfo RAB_LoadLightInfo(uint index, bool previous) {
    RAB_LightInfo info;
    info.light = restirLight(index);
    return info;
}

int RAB_TranslateLightIndex(uint index, bool currentToPrevious) { return int(index); }

RAB_LightSample RAB_EmptyLightSample() { return (RAB_LightSample)0; }

float3 RAB_LightSamplePosition(RAB_LightSample sample) { return sample.position; }

float4 RAB_LightSampleRadiance(RAB_LightSample sample) { return sample.radiance; }

float RAB_LightSampleSolidAnglePdf(RAB_LightSample sample) { return sample.solidAnglePdf; }

RAB_LightSample RAB_SamplePolymorphicLight(RAB_LightInfo info, RAB_Surface surface, float2 uv) {
    LightSample sample = emptyLightSample();
    if (info.light == environmentLight) {
        sample.direction = environmentDirection(restirScene, uv);
        sample.distance = floatMax;
        sample.radiance = environmentRadiance(restirScene, sample.direction);
        sample.pdf = environmentPdf(restirScene, sample.direction);
    } else
        sample = sampleLightRecord(restirScene.lights[info.light], surface.position, uv);
    RAB_LightSample result = RAB_EmptyLightSample();
    if (sample.pdf <= 0.0)
        return result;
    result.position = surface.position + sample.direction * min(sample.distance, restirDistantLightDistance);
    result.solidAnglePdf = sample.pdf;
    result.radiance = restirWavelengths.upsample(sample.radiance);
    result.delta = sample.delta;
    return result;
}

bool RAB_GetConservativeVisibility(RAB_Surface surface, RAB_LightSample sample) { return isRestirVisible(surface, sample.position); }

bool isBsdfReachable(uint light) { return light == environmentLight || restirScene.lights[light].meta.x == lightTypeTriangle; }

float RAB_GetMISWeightForNEE(uint index, RAB_LightSample sample, float3 direction, float lightPdf, float scatterPdf) {
    const uint light = restirLight(index);
    if (sample.delta || !isBsdfReachable(light))
        return 1.0;
    return powerHeuristic(lightSelection(restirScene, light) * lightPdf, scatterPdf);
}

float RAB_RayPayloadGetCommittedHitT(RAB_RayPayload payload) { return payload.committedRayT; }

RAB_PathTracerUserData RAB_EmptyPathTracerUserData() { return (RAB_PathTracerUserData)0; }

void RAB_PathTracerUserDataSetPathType(inout RAB_PathTracerUserData data, uint type) { data.pathType = type; }

uint RAB_GetDuplicationMapCount(int2 pixel) { return 0u; }

void RAB_LastBounceDenoiserCallback(float3 lightPosition, RAB_Surface surface, inout RAB_PathTracerUserData data) {}

void RAB_ReconnectionDenoiserCallback(RTXDI_PTReservoir reservoir, RAB_Surface surface, inout RAB_PathTracerUserData data) {}

struct RestirLightSample {
    RTXDI_SampledLightData light;
    RAB_LightSample sample;
    float4 radiance;
    float selection;
    float scatterPdf;
};

bool sampleRestirLight(RAB_Surface surface, PrincipledBsdf bsdf, inout RTXDI_RandomSamplerState random, out RestirLightSample result) {
    result = (RestirLightSample)0;
    const float4 u = float4(RTXDI_GetNextRandom(random), RTXDI_GetNextRandom(random), RTXDI_GetNextRandom(random), RTXDI_GetNextRandom(random));
    const uint light = selectLight(restirScene, u.w, result.selection);
    if (light == invalidIndex)
        return false;
    float environmentDensity;
    const float2 uv = light == environmentLight ? environmentUv(restirScene, sampleEnvironment(restirScene, u.xyz, environmentDensity)) : u.xy;
    RTXDI_SampledLightData_SetLightData(result.light, rtxdiLight(light));
    RTXDI_SampledLightData_SetUVData(result.light, uv);
    result.sample = RAB_SamplePolymorphicLight(RAB_LoadLightInfo(rtxdiLight(light), false), surface, RTXDI_SampledLightData_GetUVDataFloat2(result.light));
    if (result.sample.solidAnglePdf <= 0.0)
        return false;
    const float3 direction = normalize(result.sample.position - surface.position);
    const BsdfValue value = bsdf.evaluate(direction);
    if (all(value.value <= 0.0) || !RAB_GetConservativeVisibility(surface, result.sample))
        return false;
    result.scatterPdf = value.pdf;
    const float weight = RAB_GetMISWeightForNEE(rtxdiLight(light), result.sample, direction, result.sample.solidAnglePdf, value.pdf);
    result.radiance = value.value * result.sample.radiance * (weight / (result.selection * result.sample.solidAnglePdf));
    return true;
}

RTXDI_BrdfRaySample makeBrdfRaySample(BsdfSample sample) {
    RTXDI_BrdfRaySample ray = RTXDI_EmptyBrdfRaySample();
    ray.outDirection = sample.direction;
    ray.outPdf = sample.pdf;
    ray.brdfTimesNoL = sample.weight * sample.pdf;
    if ((sample.flags & bsdfFlagDelta) != 0)
        ray.properties.SetDelta();
    if ((sample.flags & bsdfFlagDiffuse) == 0)
        ray.properties.SetSpecular();
    if ((sample.flags & bsdfFlagTransmission) != 0)
        ray.properties.SetTransmission();
    return ray;
}

float emissionWeight(BsdfSample sample, Surface hit, float3 origin, float distance) {
    if ((sample.flags & bsdfFlagDelta) != 0 || hit.light == invalidIndex)
        return 1.0;
    return powerHeuristic(sample.pdf, lightPdf(restirScene, hit.light, origin, sample.direction, distance, dot(hit.worldNormal, sample.direction)));
}

float environmentWeight(BsdfSample sample, float3 origin) {
    if ((sample.flags & bsdfFlagDelta) != 0)
        return 1.0;
    return powerHeuristic(sample.pdf, lightPdf(restirScene, environmentLight, origin, sample.direction, floatMax, 1.0));
}

interface IRestirPathContext {
    [mutating] void BeginPathState();
    RAB_Surface GetIntersectionSurface();
    [mutating] void SetBrdfRaySample(RTXDI_BrdfRaySample sample);
    bool ValidContinuationRayBrdfOverPdf();
    float4 GetContinuationRayBrdfOverPdf();
    [mutating] void MultiplyPathThroughput(float4 factor);
    float4 GetPathThroughput();
    bool ShouldRunRussianRoulette();
    [mutating] void RecordRussianRouletteProbability(float probability);
    [mutating] void SetContinuationRay(RayDesc ray);
    [mutating] bool AnalyzePathReconnectibilityBeforeTrace();
    [mutating] void SetTraceResult(RAB_RayPayload payload);
    [mutating] void RecordPathRadianceMiss(inout RTXDI_RandomSamplerState random);
    [mutating] bool RecordEnvironmentMapLightSample(float4 radiance, RAB_Surface previous, inout RTXDI_RandomSamplerState random);
    [mutating] void RecordPathIntersection(RAB_Surface surface);
    bool IsPathTerminated();
    bool ShouldSampleEmissiveSurfaces();
    [mutating] bool RecordEmissiveLightSample(float4 radiance, RAB_Surface previous, inout RTXDI_RandomSamplerState random);
    bool ShouldSampleNee();
    [mutating] bool RecordNeeLightSample(RTXDI_SampledLightData light, float4 radiance, float neePdf, float scatterPdf, RAB_LightSample sample,
                                         inout RTXDI_RandomSamplerState random);
    uint GetBounceDepth();
    uint GetMaxPathBounce();
    [mutating] void IncreaseBounceDepth();
}

extension RTXDI_PathTracerContextInitial : IRestirPathContext {}
extension RTXDI_PathTracerContextHybrid : IRestirPathContext {}

float rouletteSurvival(float4 throughput) { return clamp(max(max(throughput.x, throughput.y), max(throughput.z, throughput.w)), 0.05, 0.95); }

void RAB_PathTrace<C : IRestirPathContext>(inout C context, inout RTXDI_PathTracerRandomContext random, inout RAB_PathTracerUserData data) {
    const uint rouletteStart = restirFrame.wavefront[0].settings.y + 1;
    RAB_Surface surface = context.GetIntersectionSurface();
    RayHit hit = (RayHit)0;
    hit.instanceId = surface.instance;
    hit.primitiveIndex = surface.primitive;
    hit.barycentrics = surface.barycentrics;
    hit.t = surface.depth;
    float3 direction = -surface.view;
    RayCone cone = unpackRayCone(surface.cone);
    BsdfSample sample = (BsdfSample)0;
    bool bounced = false;
    while (true) {
        const Surface loaded = loadSurface(restirScene, hit.instanceId, hit.primitiveIndex, hit.barycentrics, direction, hit.t, cone);
        const RAB_Surface previous = surface;
        if (bounced) {
            surface = makeRestirSurface(loaded, hit.barycentrics, hit.t, cone);
            context.RecordPathIntersection(surface);
            if (context.IsPathTerminated())
                break;
        }
        const BsdfInputs inputs = restirMaterial(loaded);
        const PrincipledBsdf bsdf = makeRestirBsdf(loaded, inputs);
        if (bounced) {
            if (context.ShouldSampleEmissiveSurfaces()) {
                const float4 emission = emittedRadiance(inputs, restirWavelengths);
                if (any(emission > 0.0))
                    context.RecordEmissiveLightSample(emission * emissionWeight(sample, loaded, previous.position, hit.t), previous,
                                                      random.initialRandomSamplerState);
            }
            if (context.ShouldSampleNee()) {
                RestirLightSample light;
                if (sampleRestirLight(surface, bsdf, random.initialRandomSamplerState, light))
                    context.RecordNeeLightSample(light.light, light.radiance, light.selection, light.scatterPdf, light.sample,
                                                 random.initialRandomSamplerState);
            }
            context.IncreaseBounceDepth();
        }
        if (context.GetBounceDepth() > context.GetMaxPathBounce())
            break;
        context.BeginPathState();
        const float3 u = float3(RTXDI_GetNextRandom(random.replayRandomSamplerState), RTXDI_GetNextRandom(random.replayRandomSamplerState),
                                RTXDI_GetNextRandom(random.replayRandomSamplerState));
        sample = bsdf.sample(u);
        context.SetBrdfRaySample(makeBrdfRaySample(sample));
        if (!context.ValidContinuationRayBrdfOverPdf())
            break;
        context.MultiplyPathThroughput(context.GetContinuationRayBrdfOverPdf());
        if (context.ShouldRunRussianRoulette() && context.GetBounceDepth() >= rouletteStart) {
            const float survival = rouletteSurvival(context.GetPathThroughput());
            if (RTXDI_GetNextRandom(random.initialRandomSamplerState) >= survival)
                break;
            context.MultiplyPathThroughput(float4(1.0 / survival));
            context.RecordRussianRouletteProbability(survival);
        }
        RayDesc ray;
        ray.Origin = offsetAlong(surface.position, surface.normal, sample.direction);
        ray.Direction = sample.direction;
        ray.TMin = 0.0;
        ray.TMax = floatMax;
        context.SetContinuationRay(ray);
        if (!context.AnalyzePathReconnectibilityBeforeTrace())
            break;
        const bool found = traceRestirSurface(ray, instanceMaskIndirect, 2u + context.GetBounceDepth(), hit);
        RAB_RayPayload payload;
        payload.committedRayT = found ? hit.t : restirDistantLightDistance;
        context.SetTraceResult(payload);
        if (!found) {
            context.RecordPathRadianceMiss(random.initialRandomSamplerState);
            const float4 environment = restirWavelengths.upsample(environmentRadiance(restirScene, ray.Direction));
            if (any(environment > 0.0))
                context.RecordEnvironmentMapLightSample(environment * environmentWeight(sample, surface.position), surface, random.initialRandomSamplerState);
            break;
        }
        const bool delta = (sample.flags & bsdfFlagDelta) != 0;
        const float roughness = bsdf.roughness();
        cone = propagate(delta ? cone : widenRayCone(cone, roughness * roughness), hit.t);
        direction = ray.Direction;
        bounced = true;
    }
}
