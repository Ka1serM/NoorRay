import NoorRHI;
import Lights.SgLighting;
import Wavefront.Records;
import Wavefront.PathState;
import Wavefront.Direct;
import Wavefront.SurfaceShading;
import Wavefront.Media;
import Common.Math;
import Common.LinkConstants;
import Common.Random;
import Common.RayCone;
import Common.Sampling;
import Lights.Lights;
import Lights.Environment;
import Bsdf.Principled;
import Materials.Closure;
import Materials.Material;
import Materials.Graph;
import Scene.Camera;
import Scene.SceneSurface;
import Spectral.Wavelengths;

#include "NoorRay/Shaders/Restir/RestirRecords.h"

static const float restirDistantLightDistance = 1.0e6;
static const uint restirEnvironmentLight = environmentLight & 0x7fffffffu;
//! The context Init draws two Gaussian footprint thresholds from the replay sampler before the first BSDF sample, as RTXDI_GetRngForShading expects.
static const uint restirReplayInitDraws = 4u;
static const uint restirSamplesPerBounce = 3u;
static const uint restirInitialPass = 2u;

static const uint restirSurfaceBackfacing = 1u;
//! The surface of a replay result carries its BSDF value toward the target of the shift instead of a closure.
static const uint restirSurfaceCached = 2u;
//! The instance of a pixel whose camera ray met a holdout.
static const uint restirInstanceHoldout = 0xfffffffeu;
//! Bits 8 to 13 of the surface flags hold the light set of the surface, whose lights it samples.
static const uint restirSurfaceLightSetShift = 8u;
//! The material field of a replay record keeps the light set from bit 24 and this bit for a surface above the reconnection roughness threshold.
static const uint restirReplayRough = 1u << 31;

//! The fields of a replay job in the path flags, above the wavefront flags: the last bounce to trace, the reconnection and path lengths, and two flags.
static const uint replayMaxBounceShift = 16u;
static const uint replayRcLengthShift = 20u;
static const uint replayPathLengthShift = 24u;
static const uint replayFlagRough = 1u << 28;
static const uint replayFlagNee = 1u << 29;

static RestirFrame restirFrame;
static SceneRecord restirScene;
static CameraRecord restirCamera;
static Wavelengths restirWavelengths;
static float restirWavelengthSample;
static uint restirPixelSeed;
//! The stats of the visibility cache, for the receivers of the volume forks that Shade gave a cell.
static Ptr<uint> restirVisibility;

void useRestirWavelengths(float u) {
    restirWavelengthSample = u;
    restirWavelengths = sampleWavelengths(restirScene.spectral, u, false);
}

#define RTXDI_SLANG
#define RTXDI_PT_APPLICATION_PATH_TRACER
#define RTXDI_PT_RESERVOIR_BUFFER restirFrame.reservoirs
#define RTXDI_NEIGHBOR_OFFSETS_BUFFER restirFrame.neighborOffsets
#define RAB_DISTANT_LIGHT_DISTANCE restirDistantLightDistance
#define RTXDI_PT_BEGIN_SHIFT(reservoir) useRestirWavelengths(reservoir.wavelengthSample)
#define RTXDI_PT_FINISH_INITIAL_SAMPLE(reservoir) reservoir.wavelengthSample = restirWavelengthSample
#define RTXDI_LIGHT_RESERVOIR_BUFFER restirFrame.lightReservoirs
#define RTXDI_ALLOWED_BIAS_CORRECTION RTXDI_BIAS_CORRECTION_RAY_TRACED

//! normal is the geometric normal on the ray side; closure indexes restirFrame.closures; bsdf and bsdfPdf cache the BSDF of a replay.
struct RAB_Surface {
    float3 position;
    float depth;
    float3 normal;
    uint material;
    float3 view;
    uint closure;
    uint flags;
    float4 bsdf;
    float bsdfPdf;
    //! The roughness of the closure, which picks the reconnection vertices; a replay result knows only whether it is above the threshold.
    float roughness;
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
    //! A sample of a light without Cast Shadow, which nothing occludes.
    bool unshadowed;
    //! The solid angle per uv area of the light at the sample, which turns a ReSTIR DI target into a density over the uv square.
    float jacobian;
    uint light;
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
#include "Rtxdi/DI/ReservoirStorage.hlsli"
#include "Rtxdi/DI/TemporalResampling.hlsli"
#include "Rtxdi/DI/SpatialResampling.hlsli"

void beginRestir(RestirFrame frame) {
    restirFrame = frame;
    const WavefrontFrame wavefront = frame.wavefront[0];
    restirScene = wavefront.scene[0];
    restirCamera = wavefront.camera[0];
    restirPixelSeed = wavefront.settings.z;
    restirVisibility = wavefront.visibility.cells;
}

uint restirPixelCount() { return restirFrame.width * restirFrame.height; }

uint restirPixelIndex(uint2 pixel) { return pixel.y * restirFrame.width + pixel.x; }

uint2 restirPixelCoordinate(uint index) { return uint2(index % restirFrame.width, index / restirFrame.width); }

static const uint restirTile = 8u;

//! The index-th pixel in 8x8 tiles, so that the threads of a group resample neighbouring pixels.
uint2 restirPixel(RestirFrame frame, uint index) {
    const uint bandPixels = frame.width * restirTile;
    const uint band = index / bandPixels;
    const uint rows = min(restirTile, frame.height - band * restirTile);
    const uint inBand = index - band * bandPixels;
    const uint column = inBand / (restirTile * rows);
    const uint columns = min(restirTile, frame.width - column * restirTile);
    const uint inTile = inBand - column * restirTile * rows;
    return uint2(column * restirTile + inTile % columns, band * restirTile + inTile / columns);
}

//! The index of the surface and closure of a pixel in the half of this frame or of the previous frame.
uint restirSurfaceIndex(uint pixel, bool previous) { return (restirFrame.current ^ (previous ? 1u : 0u)) * restirPixelCount() + pixel; }

float3 previousCameraPosition(CameraRecord camera) {
    return float3(camera.previousCameraToWorld[0].w, camera.previousCameraToWorld[1].w, camera.previousCameraToWorld[2].w);
}

//! The opacity of an alpha-tested surface for a shadow ray; a graph material runs its graph through the alpha graphs that the pipeline cache links in.
float restirOpacity(MaterialRecord material, Surface surface, RayCandidate candidate, uint seed) {
    if ((material.flags & materialFlagGraph) == 0)
        return fixedOpacity(material, surface.context);
    MaterialGraphCall call;
    call.context = surface.context;
    setCamera(call.context, restirCamera);
    call.context.rayType = float4(0.0, 1.0, 0.0, 0.0);
    call.material = graphMaterial(restirScene.graphParameters + material.graph.y, material.textures);
    call.scene = restirFrame.wavefront[0].scene;
    call.instance = candidate.instanceId;
    call.primitive = candidate.primitiveIndex;
    call.seed = seed;
    call.output = neutralGraphOutput(call.context);
    return saturate(1.0 - runAlphaGraph(material.graph.z, call).output.transparency);
}

//! Alpha-tested surfaces are tested stochastically.
struct RestirFilter : IHitFilter {
    float3 direction;
    uint seed;

    [mutating]
    HitVerdict filter(RayCandidate candidate) {
        if (!hasAlphaTest)
            return HitVerdict.AcceptAndEnd;
        const MaterialRecord material = restirScene.materials[triangleMaterial(restirScene, candidate.instanceId, candidate.primitiveIndex)];
        if ((material.flags & materialFlagAlphaTest) == 0)
            return HitVerdict.AcceptAndEnd;
        const Surface surface =
            loadSurface(restirScene, candidate.instanceId, candidate.primitiveIndex, candidate.barycentrics, direction, candidate.t, makeRayCone(0.0, 0.0));
        const uint hitSeed = hashCombine(hashCombine(seed, candidate.instanceId), candidate.primitiveIndex);
        Random random = makeRandom(hitSeed);
        return random.next() < restirOpacity(material, surface, candidate, hitSeed) ? HitVerdict.AcceptAndEnd : HitVerdict.Ignore;
    }
};

bool RAB_IsSurfaceValid(RAB_Surface surface) { return surface.depth > 0.0; }

bool isRestirRayVisible(float3 origin, float3 direction, float tMax) {
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0;
    ray.TMax = tMax;
    RestirFilter filter;
    filter.direction = direction;
    filter.seed = hashCombine(restirPixelSeed, asuint(origin.x) ^ asuint(origin.y));
    RayHit hit;
    return !traceRayInline(restirScene.tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, instanceMaskShadow, ray, filter, hit);
}

bool isRestirVisible(RAB_Surface surface, float3 target) {
    if (!RAB_IsSurfaceValid(surface))
        return false;
    const float3 offset = target - surface.position;
    const float distance = length(offset);
    if (distance <= 0.0)
        return false;
    const float3 direction = offset / distance;
    return isRestirRayVisible(offsetAlong(surface.position, surface.normal, direction), direction, distance * 0.999);
}

//! The fog along a segment of a ReSTIR path. Grid volumes take the seed from the ray, so that the same segment gives the same estimate on every evaluation.
float4 restirTransmittance(float3 origin, float3 direction, float distance) {
    if (!hasVolumes || restirScene.counts.w == 0)
        return float4(1.0, 1.0, 1.0, 1.0);
    const uint seed = hashCombine(hashCombine(asuint(origin.x) ^ asuint(direction.z), asuint(origin.y) ^ asuint(direction.x)), asuint(origin.z) ^ asuint(direction.y));
    return mediumTransmittance(restirScene, origin, direction, distance, restirWavelengths, seed);
}

RAB_Surface RAB_EmptySurface() { return (RAB_Surface)0; }

//! A path vertex; its closure is in the path pool, so the surface keeps only its roughness.
RAB_Surface makeRestirSurface(Surface surface, float depth, PackedClosure closure) {
    RAB_Surface result = RAB_EmptySurface();
    result.position = surface.context.position.xyz;
    result.depth = depth;
    result.normal = surface.context.geometricNormal.xyz;
    result.material = surface.material;
    result.view = surface.context.incoming.xyz;
    result.closure = invalidIndex;
    result.roughness = closureRoughness(closure);
    result.flags = (isBackfacing(surface.context) ? restirSurfaceBackfacing : 0u) | (surface.instanceFlags >> lightSetShift) << restirSurfaceLightSetShift;
    return result;
}

RestirSurfaceRecord packRestirSurface(RAB_Surface surface, uint instance) {
    RestirSurfaceRecord record;
    record.position = surface.position;
    record.depth = surface.depth;
    record.normal = RAB_IsSurfaceValid(surface) ? packOctahedral(surface.normal) : 0u;
    record.view = packOctahedral(surface.view);
    record.material = surface.material | ((surface.flags & restirSurfaceBackfacing) != 0 ? 0x80000000u : 0u);
    record.instance = instance;
    return record;
}

RAB_Surface unpackRestirSurface(RestirSurfaceRecord record, uint index) {
    RAB_Surface surface = RAB_EmptySurface();
    surface.position = record.position;
    surface.depth = record.depth;
    surface.normal = unpackOctahedral(record.normal);
    surface.material = record.material & 0x7fffffffu;
    surface.view = unpackOctahedral(record.view);
    surface.closure = index;
    surface.roughness = closureRoughness(restirFrame.closures[index]);
    surface.flags = (record.material & 0x80000000u) != 0 ? restirSurfaceBackfacing : 0u;
    if (record.instance < restirScene.counts.x)
        surface.flags |= (restirScene.instances[record.instance].meta.y >> lightSetShift) << restirSurfaceLightSetShift;
    return surface;
}

//! The fog between the camera and the primary vertex of a pixel, or behind a pixel without surface.
float4 restirCameraTransmittance(RAB_Surface surface) {
    if (!RAB_IsSurfaceValid(surface))
        return restirTransmittance(cameraPosition(restirCamera), -surface.view, floatMax);
    return restirTransmittance(surface.position, surface.view, surface.depth);
}

float3 restirMotion(RestirFrame frame, RAB_Surface surface, uint2 pixel) {
    float2 film;
    if (!projectToFilm(restirCamera, restirCamera.previousWorldToCamera, surface.position, film))
        return float3(0.0, 0.0, 0.0);
    const float2 previousPixel = film * float2(frame.width, frame.height) - 0.5;
    return float3(previousPixel - float2(pixel), length(surface.position - previousCameraPosition(restirCamera)) - surface.depth);
}

PrincipledBsdf makeRestirBsdf(BsdfInputs inputs, RAB_Surface surface) {
    return makePrincipled(inputs, surface.view, surface.normal, (surface.flags & restirSurfaceBackfacing) != 0, restirWavelengths, sceneEnergyLut(restirScene),
                          false);
}

PrincipledBsdf restirBsdf(RAB_Surface surface) { return makeRestirBsdf(unpackClosure(restirFrame.closures[surface.closure]), surface); }

//! A replay result knows only its BSDF value toward the target of its shift, which is the only direction that the shift evaluates there.
BsdfValue evaluateRestirBsdf(RAB_Surface surface, float3 direction) {
    if ((surface.flags & restirSurfaceCached) == 0)
        return restirBsdf(surface).evaluate(direction);
    BsdfValue value;
    value.value = surface.bsdf;
    value.pdf = surface.bsdfPdf;
    return value;
}

float3 RAB_GetSurfaceWorldPos(RAB_Surface surface) { return surface.position; }

void RAB_SetSurfaceWorldPos(inout RAB_Surface surface, float3 position) { surface.position = position; }

float3 RAB_GetSurfaceNormal(RAB_Surface surface) { return surface.normal; }

void RAB_SetSurfaceNormal(inout RAB_Surface surface, float3 normal) { surface.normal = normal; }

float3 RAB_GetSurfaceViewDir(RAB_Surface surface) { return surface.view; }

float RAB_GetSurfaceLinearDepth(RAB_Surface surface) { return surface.depth; }

float RAB_GetSurfaceRoughness(RAB_Surface surface) { return surface.roughness; }

RAB_Material RAB_GetMaterial(RAB_Surface surface) {
    RAB_Material material;
    material.index = surface.material;
    return material;
}

bool RAB_AreMaterialsSimilar(RAB_Material a, RAB_Material b) { return a.index == b.index; }

RAB_Surface RAB_GetGBufferSurface(int2 pixel, bool previous) {
    if (any(pixel < 0) || pixel.x >= int(restirFrame.width) || pixel.y >= int(restirFrame.height) || (previous && restirFrame.history == 0))
        return RAB_EmptySurface();
    const uint index = restirSurfaceIndex(restirPixelIndex(uint2(pixel)), previous);
    return unpackRestirSurface(restirFrame.surfaces[index], index);
}

int2 RAB_ClampSamplePositionIntoView(int2 pixel, bool previous) { return clamp(pixel, int2(0, 0), int2(restirFrame.width, restirFrame.height) - 1); }

float RAB_SurfaceEvaluateBrdfPdf(RAB_Surface surface, float3 direction) { return evaluateRestirBsdf(surface, direction).pdf; }

float4 RAB_GetReflectedBsdfRadianceForSurface(float3 samplePosition, float4 radiance, RAB_Surface surface) {
    if (!RAB_IsSurfaceValid(surface))
        return float4(0.0, 0.0, 0.0, 0.0);
    const float3 offset = samplePosition - surface.position;
    const float distance = length(offset);
    if (distance <= 0.0)
        return float4(0.0, 0.0, 0.0, 0.0);
    const float3 direction = offset / distance;
    return evaluateRestirBsdf(surface, direction).value * radiance * restirTransmittance(surface.position, direction, distance);
}

float4 RAB_GetPTSampleTargetPdfForSurface(float3 samplePosition, float4 radiance, RAB_Surface surface) {
    return RAB_GetReflectedBsdfRadianceForSurface(samplePosition, radiance, surface);
}

bool RAB_GetConservativeVisibility(RAB_Surface surface, float3 samplePosition) { return isRestirVisible(surface, samplePosition); }

//! The RTXDI light index of a light sample that the BSDF found, which takes its own MIS weight; the environment, whose index has the bit too, never is one.
static const uint restirBsdfLightFlag = 0x40000000u;

uint restirLight(uint index) { return index == restirEnvironmentLight ? environmentLight : index & ~restirBsdfLightFlag; }

bool isRestirBsdfLight(uint index) { return index != restirEnvironmentLight && (index & restirBsdfLightFlag) != 0; }

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
    } else if (info.light < restirScene.counts.y)
        sample = sampleLightRecord(restirScene.lights[info.light], surface.position, uv);
    RAB_LightSample result = RAB_EmptyLightSample();
    if (sample.pdf <= 0.0)
        return result;
    result.position = surface.position + sample.direction * min(sample.distance, restirDistantLightDistance);
    result.solidAnglePdf = sample.pdf;
    result.radiance = restirWavelengths.upsample(sample.radiance);
    result.delta = sample.delta;
    result.unshadowed = !castsShadow(restirScene, info.light);
    result.light = info.light;
    //! The equirect map spans 2 pi^2 sin(theta) steradians per uv area; the other lights map uv uniformly to their sample density.
    result.jacobian = info.light == environmentLight ? 2.0 * pi * pi * sin(pi * uv.y) : sample.delta ? 1.0 : 1.0 / sample.pdf;
    return result;
}

RAB_LightInfo RAB_EmptyLightInfo() { return (RAB_LightInfo)0; }

//! The density of the environment sampling of the light tree over the uv square of the equirect map.
float environmentUvPdf(float2 uv) {
    if (!hasEnvironmentTable(restirScene))
        return 0.5 * pi * sin(pi * uv.y);
    const uint width = restirScene.environment.z;
    const uint height = restirScene.environment.w;
    const uint x = min(uint(uv.x * float(width)), width - 1);
    const uint y = min(uint(uv.y * float(height)), height - 1);
    return restirScene.environmentAlias[y * width + x].pdf * float(width * height);
}

float restirLightShare(uint light, RAB_Surface surface, float3 direction, float diffuseFraction) {
    return visibleShare(hiddenRays(restirScene, light), diffuseFraction, dot(surface.normal, direction) <= 0.0);
}

//! The ReSTIR DI target over the uv square of the light: the reflected light without visibility, at the fixed wavelengths of the frame.
//! The target of a light sample over the uv square of its light, with the BSDF of the surface built once by the caller.
float lightSampleTarget(RAB_LightSample sample, RAB_Surface surface, PrincipledBsdf bsdf) {
    if (!RAB_IsSurfaceValid(surface) || sample.solidAnglePdf <= 0.0 || !lightReachesBounce(restirScene, sample.light, 0u)
        || !reachesSet(restirScene, sample.light, restirLightSet(surface)))
        return 0.0;
    const float3 direction = normalize(sample.position - surface.position);
    float diffuseFraction;
    const float4 value = bsdf.evaluate(direction, diffuseFraction).value;
    const float4 reflected = value * sample.radiance * restirLightShare(sample.light, surface, direction, diffuseFraction);
    return max(dot(reflected, float4(0.25, 0.25, 0.25, 0.25)), 0.0) * sample.jacobian;
}

float RAB_GetLightSampleTargetPdfForSurface(RAB_LightSample sample, RAB_Surface surface) {
    if (!RAB_IsSurfaceValid(surface) || sample.solidAnglePdf <= 0.0)
        return 0.0;
    return lightSampleTarget(sample, surface, restirBsdf(surface));
}

bool RAB_GetConservativeVisibility(RAB_Surface surface, RAB_LightSample sample) {
    return RAB_IsSurfaceValid(surface) && (sample.unshadowed || isRestirVisible(surface, sample.position));
}

bool RAB_GetTemporalConservativeVisibility(RAB_Surface surface, RAB_Surface temporalSurface, RAB_LightSample sample) {
    return RAB_GetConservativeVisibility(temporalSurface, sample);
}

uint restirLightSet(RAB_Surface surface) { return (surface.flags >> restirSurfaceLightSetShift) & (lightSetCount - 1u); }

//! Paths reach the environment and triangle lights; sampleRestirBsdfLight reaches the analytic area lights.
bool isBsdfReachable(uint light) {
    if (light == environmentLight)
        return true;
    if (light >= restirScene.counts.y)
        return false;
    const LightRecord record = restirScene.lights[light];
    const uint type = record.meta.x;
    const bool area = type == lightTypeRect || type == lightTypeDisk || type == lightTypeSphere || (type == lightTypeSpot && record.position.w > 0.0);
    return type == lightTypeTriangle || area;
}

//! ReSTIR surfaces keep no lobe weights, so the light tree weighs lights for a diffuse receiver; quantized as the path stores it, so NEE and MIS agree.
LightReceiver restirReceiver(RAB_Surface surface) { return unpackReceiver(packReceiver(diffuseReceiver(surface.normal, surface.view)), restirVisibility); }

float restirNeeWeight(RAB_Surface surface, uint set, uint light, RAB_LightSample sample, float lightPdf, float scatterPdf) {
    if (sample.delta || !isBsdfReachable(light))
        return 1.0;
    return powerHeuristic(lightSelection(restirScene, set, light, surface.position, restirReceiver(surface)) * lightPdf, scatterPdf);
}

float RAB_GetMISWeightForNEE(RAB_Surface surface, uint index, RAB_LightSample sample, float3 direction, float lightPdf, float scatterPdf) {
    const uint set = restirLightSet(surface);
    const uint light = restirLight(index);
    if (isRestirBsdfLight(index))
        return sample.delta ? 0.0 : powerHeuristic(scatterPdf, lightSelection(restirScene, set, light, surface.position, restirReceiver(surface)) * lightPdf);
    return restirNeeWeight(surface, set, light, sample, lightPdf, scatterPdf);
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
    //! The pdf that RTXDI takes out of the target: the light selection, or for a light that the BSDF found its pdf over the light pdf.
    float neePdf;
    float scatterPdf;
    //! The diffuse share of the scattered light, for the lighting split of the denoiser.
    float diffuseFraction;
};

//! bounce is the Cycles bounce of the vertex: 0 at the primary surface, where a light whose Max Bounces is lower gives nothing.
bool sampleRestirLight(RAB_Surface surface, PrincipledBsdf bsdf, uint bounce, inout RTXDI_RandomSamplerState random, out RestirLightSample result) {
    result = (RestirLightSample)0;
    const float4 u = float4(RTXDI_GetNextRandom(random), RTXDI_GetNextRandom(random), RTXDI_GetNextRandom(random), RTXDI_GetNextRandom(random));
    float selection;
    uint choice;
    const uint light = selectLight(restirScene, restirLightSet(surface), surface.position, restirReceiver(surface), u.w, selection, choice);
    if (light == invalidIndex || !lightReachesBounce(restirScene, light, bounce))
        return false;
    float environmentDensity;
    const float2 uv = light == environmentLight ? environmentUv(restirScene, sampleEnvironment(restirScene, u.xyz, environmentDensity)) : u.xy;
    RTXDI_SampledLightData_SetLightData(result.light, rtxdiLight(light));
    RTXDI_SampledLightData_SetUVData(result.light, uv);
    result.sample = RAB_SamplePolymorphicLight(RAB_LoadLightInfo(rtxdiLight(light), false), surface, RTXDI_SampledLightData_GetUVDataFloat2(result.light));
    if (result.sample.solidAnglePdf <= 0.0)
        return false;
    const float3 direction = normalize(result.sample.position - surface.position);
    const BsdfValue value = bsdf.evaluate(direction, result.diffuseFraction);
    if (all(value.value <= 0.0) || !RAB_GetConservativeVisibility(surface, result.sample))
        return false;
    result.scatterPdf = value.pdf;
    const float weight = restirNeeWeight(surface, restirLightSet(surface), light, result.sample, result.sample.solidAnglePdf, value.pdf);
    const float4 transmittance = restirTransmittance(surface.position, direction, length(result.sample.position - surface.position));
    result.radiance = value.value * result.sample.radiance * transmittance
                      * (weight * restirLightShare(light, surface, direction, result.diffuseFraction) / (selection * result.sample.solidAnglePdf));
    result.neePdf = selection;
    return true;
}

//! The second strategy for analytic area lights, which paths never hit: a BSDF direction that meets one becomes a light sample with its uv.
bool sampleRestirBsdfLight(RAB_Surface surface, PrincipledBsdf bsdf, uint bounce, inout RTXDI_RandomSamplerState random, out RestirLightSample result) {
    result = (RestirLightSample)0;
    const BsdfSample scattered = bsdf.sample(sampleRestirRandom(random));
    if ((scattered.flags & bsdfFlagDelta) != 0 || all(scattered.direction == 0.0))
        return false;
    const uint set = restirLightSet(surface);
    const LightHit hit = intersectLights(restirScene, set, surface.position, scattered.direction, floatMax);
    float2 uv;
    if (hit.light == invalidIndex || !lightReachesBounce(restirScene, hit.light, bounce)
        || isHiddenFrom(hiddenRays(restirScene, hit.light), bsdfLightPathEvent(scattered.flags))
        || !lightSampleUv(restirScene.lights[hit.light], surface.position, surface.position + scattered.direction * hit.distance, uv))
        return false;
    RTXDI_SampledLightData_SetLightData(result.light, rtxdiLight(hit.light) | restirBsdfLightFlag);
    RTXDI_SampledLightData_SetUVData(result.light, uv);
    result.sample = RAB_SamplePolymorphicLight(RAB_LoadLightInfo(hit.light, false), surface, RTXDI_SampledLightData_GetUVDataFloat2(result.light));
    if (result.sample.solidAnglePdf <= 0.0 || result.sample.delta)
        return false;
    const float3 direction = normalize(result.sample.position - surface.position);
    const BsdfValue value = bsdf.evaluate(direction, result.diffuseFraction);
    if (value.pdf <= 0.0 || all(value.value <= 0.0) || !RAB_GetConservativeVisibility(surface, result.sample))
        return false;
    result.scatterPdf = value.pdf;
    const float weight =
        powerHeuristic(value.pdf, lightSelection(restirScene, set, hit.light, surface.position, restirReceiver(surface)) * result.sample.solidAnglePdf);
    const float4 transmittance = restirTransmittance(surface.position, direction, length(result.sample.position - surface.position));
    result.radiance = value.value * result.sample.radiance * transmittance * (weight / value.pdf);
    result.neePdf = value.pdf / result.sample.solidAnglePdf;
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

bool isValidThroughput(float4 throughput) {
    const float luminance = RTXDI_Luminance(throughput);
    return luminance > 0.0 && !isinf(luminance) && !isnan(luminance);
}

//! The MIS weight of emission that the BSDF sample of a path hit, against light sampling; previousPdf 0 marks a delta sample.
float restirEmissionWeight(PathState path, Surface hit, float distance) {
    if (path.previousPdf <= 0.0 || hit.light == invalidIndex)
        return 1.0;
    return powerHeuristic(path.previousPdf, lightPdf(restirScene, 0u, hit.light, path.origin, path.lightReceiver(restirVisibility), path.direction, distance,
                                                     dot(hit.worldNormal, path.direction)));
}

float restirEnvironmentWeight(PathState path) {
    if (path.previousPdf <= 0.0)
        return 1.0;
    return powerHeuristic(path.previousPdf,
                          lightPdf(restirScene, 0u, environmentLight, path.origin, path.lightReceiver(restirVisibility), path.direction, floatMax, 1.0));
}

float4 restirEnvironment(float3 direction, uint lightPath) {
    if (isHiddenFrom(hiddenRays(restirScene, environmentLight), lightPath))
        return float4(0.0, 0.0, 0.0, 0.0);
    return restirWavelengths.upsample(environmentRadiance(restirScene, direction));
}

//! The replay sampler before the BSDF sample at bounceDepth: the initial path and every replay of it draw the same numbers there.
RTXDI_RandomSamplerState restirBsdfRandom(uint seed, uint randomIndex, uint bounceDepth) {
    return RTXDI_CreateRandomSamplerFromDirectSeed(seed, randomIndex + restirReplayInitDraws + restirSamplesPerBounce * (bounceDepth - 2));
}

float3 sampleRestirRandom(inout RTXDI_RandomSamplerState random) {
    return float3(RTXDI_GetNextRandom(random), RTXDI_GetNextRandom(random), RTXDI_GetNextRandom(random));
}

//! The ray cone of a ray that leaves a surface: the cone that reached it, widened by the roughness unless the sample is delta.
RayCone restirContinuationCone(RayCone cone, bool delta, float roughness) { return delta ? cone : widenRayCone(cone, roughness * roughness); }

RTXDI_PTInitialSamplingRuntimeParameters initialRuntime() {
    RTXDI_PTInitialSamplingRuntimeParameters runtime = RTXDI_EmptyPTInitialSamplingRuntimeParameters();
    runtime.cameraPos = cameraPosition(restirCamera);
    runtime.prevCameraPos = previousCameraPosition(restirCamera);
    runtime.prevPrevCameraPos = runtime.prevCameraPos;
    return runtime;
}

//! A vertex of the initial context in the path record: position and octahedral normal.
float4 packRestirVertex(RAB_Surface surface) { return float4(surface.position, asfloat(packOctahedral(surface.normal))); }

RAB_Surface unpackRestirVertex(float4 packed) {
    RAB_Surface surface = RAB_EmptySurface();
    surface.position = packed.xyz;
    surface.normal = unpackOctahedral(asuint(packed.w));
    return surface;
}

//! The record keeps the reconnection lengths and path length in 4 bits each, the state flags and the found bit, then the index of the initial sampler.
RestirPathRecord packInitialContext(RTXDI_PathTracerContextInitial context, uint randomIndex) {
    const RTXDI_InitialSamplingPathTracerContext inner = context.ctx;
    const uint packed = (inner.ptState.rcVertexLength & 15u) | ((inner.selectedPathLength & 15u) << 4) | ((inner.selectedRcVertexLength & 15u) << 8) |
                        ((inner.ptState.ptStateFlags & 0x1fu) << 12) | (inner.foundRcVertex ? 1u << 17 : 0u) | (randomIndex << 18);
    RestirPathRecord record;
    record.sums = float4(inner.runningWeightSum, inner.brdfPdfBeforeRcVertex, inner.russianRoulettePdf, asfloat(packed));
    record.mainVertex = packRestirVertex(inner.mainPathRcVertexSurface);
    record.selectedVertex = packRestirVertex(inner.selectedRcSurface);
    record.pdfs = float4(inner.mainPathRcWiPdf, inner.mainPathPartialJacobian, inner.selectedRcWiPdf, inner.selectedPartialJacobian);
    record.radiance = inner.radiance;
    record.targetFunction = inner.selectedTargetFunction;
    return record;
}

//! The context of a path at the hit or miss of its ray: the record, the throughputs in the pool, and the ray of the path as the BSDF sample and trace result.
RTXDI_PathTracerContextInitial unpackInitialContext(RestirPathRecord record, PathState path, float distance, out uint randomIndex) {
    const RTXDI_PTParameters pt = restirFrame.pt;
    RTXDI_PathTracerContextInitial context = (RTXDI_PathTracerContextInitial)0;
    const uint packed = asuint(record.sums.w);
    context.ctx.ptState.initialBounceDepth = 2;
    context.ctx.ptState.bounceDepth = path.depth() + 1;
    context.ctx.ptState.maxPathBounce = pt.initialSampling.maxBounceDepth;
    context.ctx.ptState.pathThroughput = path.throughput;
    context.ctx.ptState.rcPathThroughput = float4(path.radiance, path.diffuseFraction);
    context.ctx.ptState.rcVertexLength = packed & 15u;
    context.ctx.ptState.ptStateFlags = (packed >> 12) & 0x1fu;
    RTXDI_BrdfRaySample ray = RTXDI_EmptyBrdfRaySample();
    ray.outDirection = path.direction;
    ray.outPdf = path.previousPdf;
    if (path.hasFlag(pathFlagDelta))
        ray.properties.SetDelta();
    context.ctx.ptState.brdfRaySample = ray;
    context.ctx.ptState.continuationRay.Origin = path.origin;
    context.ctx.ptState.continuationRay.Direction = path.direction;
    context.ctx.ptState.traceResult.committedRayT = distance;
    context.ctx.ptState.intersectionSurface.position = path.origin;
    context.ctx.selectedPathLength = (packed >> 4) & 15u;
    context.ctx.selectedRcVertexLength = (packed >> 8) & 15u;
    context.ctx.foundRcVertex = ((packed >> 17) & 1u) != 0;
    context.ctx.runningWeightSum = record.sums.x;
    context.ctx.brdfPdfBeforeRcVertex = record.sums.y;
    context.ctx.russianRoulettePdf = record.sums.z;
    context.ctx.mainPathRcVertexSurface = unpackRestirVertex(record.mainVertex);
    context.ctx.selectedRcSurface = unpackRestirVertex(record.selectedVertex);
    context.ctx.mainPathRcWiPdf = record.pdfs.x;
    context.ctx.mainPathPartialJacobian = record.pdfs.y;
    context.ctx.selectedRcWiPdf = record.pdfs.z;
    context.ctx.selectedPartialJacobian = record.pdfs.w;
    context.ctx.radiance = record.radiance;
    context.ctx.selectedTargetFunction = record.targetFunction;
    context.ctx.reconnectionMode = pt.reconnection.reconnectionMode;
    context.ctx.roughnessThreshold = pt.reconnection.roughnessThreshold;
    context.ctx.distanceThreshold = pt.reconnection.distanceThreshold;
    randomIndex = packed >> 18;
    return context;
}

//! The replay hook of the hybrid shift: call k of a pass reads the result of replay k, or starts it; an inverse shift starts after its forward replays.
static uint replayBase;
static uint replaySlotBase;
static uint replayForward;
static uint replayForwardCalls;
static uint replayInverseCalls;
static uint replayStage;
static uint replayParity;
static uint replayEmitted;
static bool replayForwardPending;

void beginReplays(uint local, uint forward, uint stage, uint parity) {
    replayBase = local * forward * 2;
    replaySlotBase = local * forward;
    replayForward = forward;
    replayForwardCalls = 0;
    replayInverseCalls = 0;
    replayStage = stage;
    replayParity = parity;
    replayEmitted = 0;
    replayForwardPending = false;
}

RAB_Surface unpackReplaySurface(RestirReplayRecord record) {
    RAB_Surface surface = RAB_EmptySurface();
    surface.position = record.position;
    surface.depth = 1.0;
    surface.normal = unpackOctahedral(record.normal);
    surface.material = record.material & 0xffffffu;
    surface.closure = invalidIndex;
    surface.roughness = (record.material & restirReplayRough) != 0 ? 1.0 : 0.0;
    surface.flags = restirSurfaceCached | ((record.material >> 24u) & (lightSetCount - 1u)) << restirSurfaceLightSetShift;
    surface.bsdf = float4(unpackHalf2(record.bsdf.x), unpackHalf2(record.bsdf.y));
    surface.bsdfPdf = record.bsdfPdf;
    return surface;
}

//! Samples the BSDF at the base surface of the shift with the replay sampler of the sample and queues the ray as a replay job in a pool slot.
bool startReplay(RTXDI_PTReservoir sample, RAB_Surface base, uint index, uint slot) {
    if (!RAB_IsSurfaceValid(base))
        return false;
    const PrincipledBsdf bsdf = restirBsdf(base);
    RTXDI_RandomSamplerState random = restirBsdfRandom(sample.randomSeed, sample.randomIndex, 2);
    const BsdfSample scattered = bsdf.sample(sampleRestirRandom(random));
    if (scattered.pdf <= 0.0 || !isValidThroughput(scattered.weight))
        return false;
    const bool delta = (scattered.flags & bsdfFlagDelta) != 0;
    const bool nee = RTXDI_ConnectsToNeeLight(sample);
    const uint maxBounce = min(sample.rcVertexLength - 1, sample.pathLength);
    const bool rough = RAB_GetSurfaceRoughness(base) > restirFrame.pt.reconnection.roughnessThreshold;
    const WavefrontFrame wavefront = restirFrame.wavefront[0];
    const PathPoolRecord pool = wavefront.pool;
    const RayCone cone = propagate(cameraRayCone(restirCamera), base.depth);
    const uint packedCone = packRayCone(restirContinuationCone(cone, delta, bsdf.roughness()));
    pool.origin[slot] = float4(offsetAlong(base.position, base.normal, scattered.direction), asfloat(packedCone));
    pool.direction[slot] = float4(scattered.direction, delta ? 0.0 : scattered.pdf);
    pool.receiver[slot] = packReceiver(restirReceiver(base));
    pool.throughput[slot] = scattered.weight;
    const float3 target = nee ? float3(sample.radiance.y, sample.radiance.z, 0.0) : sample.translatedWorldPosition;
    pool.radiance[slot] = float4(target, asfloat(sample.randomIndex));
    const uint flags = 1u | (delta ? pathFlagDelta : 0u) | (maxBounce << replayMaxBounceShift) | (min(sample.rcVertexLength, 15u) << replayRcLengthShift) |
                       (sample.pathLength << replayPathLengthShift) | (rough ? replayFlagRough : 0u) | (nee ? replayFlagNee : 0u);
    pool.state[slot] = uint4(index, sample.randomSeed, flags, asuint(sample.wavelengthSample));
    pool.lightPath[slot] = uint2(nextLightPath(0u, bsdfLightPathEvent(scattered.flags)), 0u);
    pool.minRayPdf[slot] = delta ? floatMax : scattered.pdf;
    pushActive(wavefront, replayParity, slot);
    return true;
}

void RAB_RandomReplay(RTXDI_PTReservoir sample, inout RAB_Surface surface, inout float4 targetFunction, inout RAB_PathTracerUserData data) {
    const bool inverse = data.pathType == RTXDI_PTPathTraceInvocationType_TemporalInverse || data.pathType == RTXDI_PTPathTraceInvocationType_SpatialInverse;
    const uint order = inverse ? replayInverseCalls++ : replayForwardCalls++;
    const RAB_Surface base = surface;
    surface = RAB_EmptySurface();
    targetFunction = float4(0.0, 0.0, 0.0, 0.0);
    if (order >= replayForward)
        return;
    const uint call = inverse ? replayForward + order : order;
    const uint index = replayBase + call;
    if (replayStage != 0) {
        const uint state = restirFrame.replayStates[index];
        if ((state & 3u) == restirReplayDone) {
            const RestirReplayRecord record = restirFrame.replays[index];
            if ((state & restirReplayHasSurface) != 0)
                surface = unpackReplaySurface(record);
            targetFunction = record.radiance;
            return;
        }
        if ((state & 3u) == restirReplayInFlight)
            return;
    }
    if (replayStage + 1 == restirStages || (inverse && replayForwardPending))
        return;
    if (!startReplay(sample, base, index, replaySlotBase + order))
        return;
    replayEmitted |= 1u << call;
    replayForwardPending = replayForwardPending || !inverse;
}

//! After a pass that started replays: the started ones are in flight, and in the first stage every other replay of the pixel is still to start.
bool endReplays() {
    if (replayEmitted == 0)
        return false;
    for (uint call = 0; call < replayForward * 2; ++call) {
        const bool emitted = (replayEmitted & (1u << call)) != 0;
        if (emitted || replayStage == 0)
            restirFrame.replayStates[replayBase + call] = emitted ? restirReplayInFlight : restirReplayEmpty;
    }
    return true;
}
