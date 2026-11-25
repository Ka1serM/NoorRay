import NoorRHI;
import Lights.SgLighting;
import Wavefront.Records;
import Common.Math;
import Common.Random;
import Common.RayCone;
import Scene.Camera;
import Spectral.Wavelengths;

//! Sobol patterns: 0 film and lens, 1 wavelength, then patternsPerBounce per depth: light, BSDF and roulette, medium point, medium light.
static const uint patternCamera = 0u;
static const uint patternWavelength = 1u;
static const uint patternsPerBounce = 4u;
static const uint patternFirstBounce = 2u;
//! The subsurface probe of each depth has a pattern outside the bounce patterns, because the medium patterns of a depth precede its surface hit.
static const uint patternSubsurface = 0x53535300u;

WavefrontFrame loadFrame(WavefrontArguments arguments) { return arguments.frame[0]; }

Ptr<uint> activeQueue(QueueRecord queues, uint parity) { return parity == 0 ? queues.active0 : queues.active1; }
Ptr<uint> shadowQueue(QueueRecord queues, uint parity) { return parity == 0 ? queues.shadow0 : queues.shadow1; }

struct PathState {
    float3 origin;
    RayCone cone;
    float3 direction;
    float previousPdf;
    float4 throughput;
    float3 radiance;
    //! The diffuse share of the light that reaches the primary vertex along this path, for the lighting split of the denoiser.
    float diffuseFraction;
    //! The direction in which the path left its primary vertex, or zero when it is not known, for the RELAX spherical harmonics.
    float3 primaryDirection;
    //! The packed light receiver of the vertex that the ray left, a medium one at the camera: NEE and MIS weigh the light tree for it.
    uint4 receiver;
    uint pixel;
    uint sampleIndex;
    uint flags;
    uint hero;

    uint depth() { return flags & pathDepthMask; }
    LightReceiver lightReceiver(Ptr<uint> visibility) { return unpackReceiver(receiver, visibility); }
    [mutating]
    void setReceiver(LightReceiver value) { receiver = packReceiver(value); }
    bool hasFlag(uint flag) { return (flags & flag) != 0; }
    [mutating]
    void setFlag(uint flag, bool value) { flags = value ? (flags | flag) : (flags & ~flag); }
    uint lightSet() { return (flags & pathLightSetMask) >> pathLightSetShift; }
    [mutating]
    void setLightSet(uint set) { flags = (flags & ~pathLightSetMask) | (min(set, lightSetCount - 1) << pathLightSetShift); }
    [mutating]
    void advanceDepth() { flags = (flags & ~pathDepthMask) | min(depth() + 1, pathDepthMask); }
};

//! A camera sample without a ray: it already wrote its black sample and only waits for its slot to be freed.
bool isDeadPath(PathState path) { return path.depth() == pathDepthMask; }

PathState loadPath(PathPoolRecord pool, uint slot) {
    const float4 origin = pool.origin[slot];
    const float4 direction = pool.direction[slot];
    const uint4 state = pool.state[slot];
    PathState path;
    path.origin = origin.xyz;
    path.cone = unpackRayCone(asuint(origin.w));
    path.direction = direction.xyz;
    path.previousPdf = direction.w;
    path.throughput = pool.throughput[slot];
    const float4 radiance = pool.radiance[slot];
    path.radiance = radiance.xyz;
    path.diffuseFraction = radiance.w;
    const uint primaryDirection = pool.primaryDirection[slot];
    path.primaryDirection = primaryDirection == noPrimaryDirection ? float3(0.0, 0.0, 0.0) : unpackOctahedral(primaryDirection);
    path.receiver = pool.receiver[slot];
    path.pixel = state.x;
    path.sampleIndex = state.y;
    path.flags = state.z;
    path.hero = state.w;
    return path;
}

void storeRay(PathPoolRecord pool, uint slot, PathState path) {
    pool.origin[slot] = float4(path.origin, asfloat(packRayCone(path.cone)));
    pool.direction[slot] = float4(path.direction, path.previousPdf);
    pool.receiver[slot] = path.receiver;
}

void storePath(PathPoolRecord pool, uint slot, PathState path) {
    storeRay(pool, slot, path);
    pool.throughput[slot] = path.throughput;
    pool.radiance[slot] = float4(path.radiance, path.diffuseFraction);
    pool.primaryDirection[slot] = all(path.primaryDirection == 0.0) ? noPrimaryDirection : packOctahedral(path.primaryDirection);
    pool.state[slot] = uint4(path.pixel, path.sampleIndex, path.flags, path.hero);
}

//! The bits of a light path: the last event in bits 0 to 4, then the diffuse, glossy, transmission and transparent bounce counts of Cycles in 6 bits each.
static const uint lightPathDiffuse = 1u;
static const uint lightPathGlossy = 2u;
static const uint lightPathSingular = 4u;
static const uint lightPathTransmission = 8u;
static const uint lightPathVolumeScatter = 16u;
static const uint lightPathEvents = 31u;
static const uint lightPathDiffuseShift = 8u;
static const uint lightPathGlossyShift = 14u;
static const uint lightPathTransmissionShift = 20u;
static const uint lightPathTransparentShift = 26u;
static const uint lightPathCountMask = 63u;

uint lightPathCount(uint lightPath, uint shift) { return (lightPath >> shift) & lightPathCountMask; }

uint addLightPathCount(uint lightPath, uint shift) { return lightPathCount(lightPath, shift) < lightPathCountMask ? lightPath + (1u << shift) : lightPath; }

//! As Cycles counts: a transmission counts as transmission, a diffuse or glossy reflection as diffuse or glossy, a volume scatter as neither.
uint nextLightPath(uint previous, uint event) {
    uint counts = previous & ~lightPathEvents;
    if ((event & lightPathTransmission) != 0)
        counts = addLightPathCount(counts, lightPathTransmissionShift);
    else if ((event & lightPathDiffuse) != 0)
        counts = addLightPathCount(counts, lightPathDiffuseShift);
    else if ((event & lightPathGlossy) != 0)
        counts = addLightPathCount(counts, lightPathGlossyShift);
    return counts | event;
}

//! The instances that the ray in flight sees, as Cycles' Ray Visibility.
uint rayInstanceMask(uint lightPath, bool camera) {
    if (camera)
        return instanceMaskCamera;
    if ((lightPath & lightPathVolumeScatter) != 0)
        return instanceMaskVolumeScatter;
    if ((lightPath & lightPathTransmission) != 0)
        return instanceMaskTransmission;
    return (lightPath & lightPathDiffuse) != 0 ? instanceMaskDiffuse : instanceMaskGlossy;
}

//! Whether a light with these hidden rays is dark for the ray in flight.
bool isHiddenFrom(uint hidden, uint lightPath) {
    if ((lightPath & lightPathVolumeScatter) != 0)
        return (hidden & lightHiddenVolumeScatter) != 0;
    if ((lightPath & lightPathDiffuse) != 0)
        return (hidden & lightHiddenDiffuse) != 0;
    return (hidden & ((lightPath & lightPathTransmission) != 0 ? lightHiddenTransmission : lightHiddenGlossy)) != 0;
}

//! x is the light path of the ray in flight, y the light path at the vertex that sent the last shadow ray.
uint loadLightPath(PathPoolRecord pool, uint slot) { return pool.lightPath[slot].x; }

void recordLightPath(PathPoolRecord pool, uint slot, uint event) {
    const uint previous = pool.lightPath[slot].x;
    pool.lightPath[slot] = uint2(nextLightPath(previous, event), previous);
}

//! Whether no scatter ray follows the last vertex: then its light sample carries the full weight, as no ray can find the light.
bool isLastVertex(WavefrontFrame frame, PathState path) { return path.depth() + 1 >= (frame.settings.x & 0xffu) && frame.bounces.w == 0; }

//! Whether the diffuse, glossy or transmission count of the ray in flight passed its limit: Cycles then ends the path at the vertex after its emission.
bool reachedKindBounces(WavefrontFrame frame, uint lightPath) {
    return lightPathCount(lightPath, lightPathDiffuseShift) > frame.bounces.x || lightPathCount(lightPath, lightPathGlossyShift) > frame.bounces.y
           || lightPathCount(lightPath, lightPathTransmissionShift) > frame.bounces.z;
}

//! Keeps the smallest pdf of the non-delta scatters of the path for Filter Glossy.
void recordScatterPdf(PathPoolRecord pool, uint slot, float pdf) {
    if (pdf > 0.0)
        pool.minRayPdf[slot] = min(pool.minRayPdf[slot], pdf);
}

//! The salt of the opacity decision at a surface: each transparent layer of a ray decides on its own.
uint opacitySalt(PathPoolRecord pool, uint slot) { return 2u + (lightPathCount(pool.lightPath[slot].x, lightPathTransparentShift) << 16); }

//! A surface that the stochastic opacity lets through counts as a transparent bounce.
void recordTransparentBounce(PathPoolRecord pool, uint slot) {
    const uint2 lightPath = pool.lightPath[slot];
    pool.lightPath[slot] = uint2(addLightPathCount(lightPath.x, lightPathTransparentShift), lightPath.y);
}

Wavelengths pathWavelengths(SceneRecord scene, PathState path) {
    if (path.hasFlag(pathFlagRgb))
        return rgbWavelengths();
    return sampleWavelengths(scene.spectral, unpackHero(path.hero), path.hasFlag(pathFlagSecondaryTerminated));
}

SobolSampler pathSampler(WavefrontFrame frame, PathState path, uint pattern) {
    SobolSampler sampler;
    sampler.index = path.sampleIndex;
    sampler.seed = hashCombine(path.pixel, frame.settings.w);
    sampler.dimension = pattern;
    return sampler;
}

float4 bouncePattern(WavefrontFrame frame, PathState path, uint offset) {
    SobolSampler sampler = pathSampler(frame, path, patternFirstBounce + path.depth() * patternsPerBounce + offset);
    return sampler.next4();
}

float4 subsurfacePattern(WavefrontFrame frame, PathState path) {
    SobolSampler sampler = pathSampler(frame, path, patternSubsurface + path.depth());
    return sampler.next4();
}

//! A per-ray seed for hashed decisions that must not depend on traversal order (Gaussian acceptance).
uint raySeed(WavefrontFrame frame, PathState path, uint salt) {
    return hashCombine(hashCombine(hashCombine(path.pixel, path.sampleIndex), frame.settings.z), path.depth() * 16 + salt);
}

//! The RGB film value of a spectral contribution, scaled down to the indirect clamp of limits.x or the direct clamp of filter.y when set.
float3 filmValue(WavefrontFrame frame, Wavelengths wavelengths, float4 contribution, bool indirect) {
    const float4 value = sanitize(contribution);
    const float3 rgb = max(sanitize(wavelengths.toFilm(value)), 0.0);
    const float limit = indirect ? frame.limits.x : frame.filter.y;
    //! Cycles bounds the mean of the RGB components; the mean over the wavelengths has that value for white light without the colour noise of the samples.
    const float mean = max(dot(value, wavelengths.weights) * 0.25, 0.0);
    if (limit <= 0.0 || mean <= limit)
        return rgb;
    return rgb * (limit / mean);
}

//! Emission that a path hits: as in Cycles, the light that the primary vertex scatters toward the camera is direct, so it is indirect from depth 2.
void addRadiance(WavefrontFrame frame, inout PathState path, Wavelengths wavelengths, float4 contribution) {
    path.radiance += filmValue(frame, wavelengths, contribution, path.depth() > 1);
}

void addFilm(AtomicPtr<float> target, float3 value) {
    if (all(value == 0.0))
        return;
    atomicAdd(target, value.x);
    atomicAdd(target + 1, value.y);
    atomicAdd(target + 2, value.z);
}

//! Adds light to a pixel without counting a sample; the shadow rays of ended paths still arrive this way.
void splatRadiance(FilmRecord film, uint pixel, float3 value) { addFilm(film.frame + pixel * 4, value); }

//! Adds to one bucket of the lighting split when the frame denoises; the diffuse light is the rest of the pixel sum.
void splitLighting(FilmRecord film, uint pixel, uint bucket, float3 value) {
    if (film.splitsLighting != 0)
        addFilm(film.lighting + pixel * lightingStride + bucket, value);
}

//! Adds the primary hit distance of a path to the hit distance sums, weighted by the diffuse and specular shares.
void splitHitDistance(FilmRecord film, uint pixel, float distance, float diffuseFraction) {
    const AtomicPtr<float> target = film.lighting + pixel * lightingStride;
    const float diffuse = saturate(diffuseFraction);
    const float hit = min(distance, 65000.0);
    if (diffuse > 0.0) {
        atomicAdd(target + lightingDiffuseHit, hit * diffuse);
        atomicAdd(target + lightingDiffuseHit + 1, diffuse);
    }
    if (diffuse < 1.0) {
        atomicAdd(target + lightingSpecularHit, hit * (1.0 - diffuse));
        atomicAdd(target + lightingSpecularHit + 1, 1.0 - diffuse);
    }
}

void splitHitDistance(FilmRecord film, PathState path, float distance) {
    if (film.splitsLighting != 0 && !path.hasFlag(pathFlagFork))
        splitHitDistance(film, path.pixel, distance, path.diffuseFraction);
}

//! Light that a path gathered before its primary scattering vertex: emission seen by the camera and the background.
void splitPrimaryLight(FilmRecord film, uint pixel, float3 value) { splitLighting(film, pixel, lightingEmission, value); }

//! Light that a path gathered after its primary vertex, split by the diffuse share of the direction it left that vertex in; a zero direction adds no direction.
void splitPathLight(FilmRecord film, uint pixel, float3 value, float diffuseFraction, float3 direction) {
    const float diffuse = saturate(diffuseFraction);
    splitLighting(film, pixel, lightingSpecular, value * (1.0 - diffuse));
    if (all(direction == 0.0))
        return;
    const float3 moment = direction * luminance(value);
    splitLighting(film, pixel, lightingDiffuseSh, moment * diffuse);
    splitLighting(film, pixel, lightingSpecularSh, moment * (1.0 - diffuse));
}

//! Fog light in front of the primary vertex, which the denoiser filters without demodulation.
void splitFogLight(FilmRecord film, uint pixel, float3 value) { splitLighting(film, pixel, lightingFog, value); }

void splitPathLight(FilmRecord film, PathState path, float3 value, float diffuseFraction, float3 direction) {
    if (path.hasFlag(pathFlagCameraFork))
        splitFogLight(film, path.pixel, value);
    else
        splitPathLight(film, path.pixel, value, diffuseFraction, direction);
}

//! Counts a camera sample that sees no surface and no background, for the alpha of the film.
void countTransparentSample(FilmRecord film, uint pixel) { atomicAdd(film.frame + film.width * film.height * 4 + pixel, 1.0); }

//! Writes the finished path into the frame film with atomics: several paths can share a pixel.
void finishPath(FilmRecord film, PathState path) {
    const AtomicPtr<float> pixel = film.frame + path.pixel * 4;
    const float3 value = sanitize(path.radiance);
    atomicAdd(pixel, value.x);
    atomicAdd(pixel + 1, value.y);
    atomicAdd(pixel + 2, value.z);
    if (!path.hasFlag(pathFlagFork))
        atomicAdd(pixel + 3, 1.0);
    if (path.depth() == 0 && !path.hasFlag(pathFlagCameraFork))
        splitPrimaryLight(film, path.pixel, value);
    else
        splitPathLight(film, path, value, path.diffuseFraction, path.primaryDirection);
}

//! At the primary vertex of a denoised frame, moves the light so far into the pixel, so the path carries only light the vertex scatters.
void flushPrimaryLight(FilmRecord film, inout PathState path) {
    if (film.splitsLighting == 0 || path.depth() != 0)
        return;
    const float3 value = sanitize(path.radiance);
    splatRadiance(film, path.pixel, value);
    splitPrimaryLight(film, path.pixel, value);
    path.radiance = float3(0.0, 0.0, 0.0);
}

//! Writes the primary AOVs of the first sample of a pixel at its first scattering vertex.
void writePrimaryAovs(FilmRecord film, PathState path, float3 albedo, float3 normal, float depth, uint objectId) {
    if (path.depth() != 0 || !path.hasFlag(pathFlagPrimarySample) || path.hasFlag(pathFlagVolumeGuide))
        return;
    film.albedo[path.pixel] = float4(albedo, 1.0);
    film.normalDepth[path.pixel] = float4(normal, depth);
    film.objectIds[path.pixel] = objectId;
}

//! The camera distance of the point at t on a camera ray, which after transparent hits starts behind the last of them.
float cameraRayDistance(WavefrontFrame frame, PathState path, uint slot, float t) {
    if (path.depth() != 0 || lightPathCount(loadLightPath(frame.pool, slot), lightPathTransparentShift) == 0)
        return t;
    return t + max(dot(path.origin - cameraPosition(frame.camera[0]), path.direction), 0.0);
}

//! The depth and object of the first surface of a primary sample that scatters in a thin medium before it; normal and albedo keep those of an earlier frame.
void writePrimarySurfaceDepth(FilmRecord film, PathState path, float depth, uint objectId) {
    if (path.depth() != 0 || !path.hasFlag(pathFlagPrimarySample) || path.hasFlag(pathFlagVolumeGuide))
        return;
    film.normalDepth[path.pixel].w = depth;
    film.objectIds[path.pixel] = objectId;
}

//! The demodulation AOVs of a denoised frame at the primary vertex of the first sample of a pixel.
void writeDemodulation(FilmRecord film, PathState path, float3 diffuseAlbedo, float3 specularAlbedo, float roughness) {
    if (film.splitsLighting == 0 || path.depth() != 0 || !path.hasFlag(pathFlagPrimarySample) || path.hasFlag(pathFlagVolumeGuide))
        return;
    film.demodulation[path.pixel * 2] = float4(diffuseAlbedo, roughness);
    film.demodulation[path.pixel * 2 + 1] = float4(specularAlbedo, 0.0);
}

void pushActive(WavefrontFrame frame, uint parity, uint slot) {
    const uint index = atomicAdd(frame.queues.counters + counterNextActive, 1u);
    activeQueue(frame.queues, parity ^ 1)[index] = slot;
}

void pushShadow(WavefrontFrame frame, uint parity, uint slot) {
    const uint index = atomicAdd(frame.queues.counters + counterShadow, 1u);
    shadowQueue(frame.queues, parity)[index] = slot;
}

//! The hit record that Extend writes for Shade.
struct PathHit {
    float t;
    float2 barycentrics;
    uint primitive;
    uint kind;
    uint instance;
};

PathHit loadHit(PathPoolRecord pool, uint slot) {
    const uint4 packed = pool.hit[slot];
    const uint instance = pool.hitInstance[slot];
    PathHit hit;
    hit.t = asfloat(packed.x);
    hit.barycentrics = float2(asfloat(packed.y), asfloat(packed.z));
    hit.primitive = packed.w;
    hit.kind = instance >> 30;
    hit.instance = instance & 0x3fffffffu;
    return hit;
}

void storeHit(PathPoolRecord pool, uint slot, PathHit hit) {
    pool.hit[slot] = uint4(asuint(hit.t), asuint(hit.barycentrics.x), asuint(hit.barycentrics.y), hit.primitive);
    pool.hitInstance[slot] = (hit.kind << 30) | (hit.instance & 0x3fffffffu);
}

//! Russian roulette from rouletteStart on, by the throughput maximum; returns false when the path ends.
bool continuePath(WavefrontFrame frame, inout PathState path, float u) {
    if (path.depth() + 1 >= (frame.settings.x & 0xffu)) {
        if (frame.bounces.w == 0)
            return false;
        path.setFlag(pathFlagFinalRay, true);
        return true;
    }
    if (path.depth() < frame.settings.y)
        return true;
    const float survival = clamp(max(max(path.throughput.x, path.throughput.y), max(path.throughput.z, path.throughput.w)), 0.05, 0.95);
    if (u >= survival)
        return false;
    path.throughput /= survival;
    return true;
}
