import NoorRHI;
import Wavefront.Records;
import Common.Math;
import Common.Random;
import Common.RayCone;
import Spectral.Wavelengths;

//! Sobol patterns per path: 0 film and lens, 1 wavelength and spare, then patternsPerBounce per depth (light, BSDF and roulette, medium and spare).
static const uint patternCamera = 0u;
static const uint patternWavelength = 1u;
static const uint patternsPerBounce = 3u;
static const uint patternFirstBounce = 2u;

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
    uint pixel;
    uint sampleIndex;
    uint flags;
    uint hero;

    uint depth() { return flags & pathDepthMask; }
    bool hasFlag(uint flag) { return (flags & flag) != 0; }
    [mutating]
    void setFlag(uint flag, bool value) { flags = value ? (flags | flag) : (flags & ~flag); }
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
    path.radiance = pool.radiance[slot].xyz;
    path.pixel = state.x;
    path.sampleIndex = state.y;
    path.flags = state.z;
    path.hero = state.w;
    return path;
}

void storeRay(PathPoolRecord pool, uint slot, PathState path) {
    pool.origin[slot] = float4(path.origin, asfloat(packRayCone(path.cone)));
    pool.direction[slot] = float4(path.direction, path.previousPdf);
}

void storePath(PathPoolRecord pool, uint slot, PathState path) {
    storeRay(pool, slot, path);
    pool.throughput[slot] = path.throughput;
    pool.radiance[slot] = float4(path.radiance, 0.0);
    pool.state[slot] = uint4(path.pixel, path.sampleIndex, path.flags, path.hero);
}

Wavelengths pathWavelengths(SceneRecord scene, PathState path) {
    if (path.hasFlag(pathFlagRgb))
        return rgbWavelengths();
    return sampleWavelengths(scene.spectral, unpackHero(path.hero), path.hasFlag(pathFlagSecondaryTerminated));
}

SobolSampler pathSampler(WavefrontFrame frame, PathState path, uint pattern) {
    SobolSampler sampler;
    sampler.index = path.sampleIndex;
    sampler.seed = hashCombine(path.pixel, frame.settings.z);
    sampler.dimension = pattern;
    return sampler;
}

float4 bouncePattern(WavefrontFrame frame, PathState path, uint offset) {
    SobolSampler sampler = pathSampler(frame, path, patternFirstBounce + path.depth() * patternsPerBounce + offset);
    return sampler.next4();
}

//! A per-ray seed for hashed decisions that must not depend on traversal order (Gaussian acceptance).
uint raySeed(WavefrontFrame frame, PathState path, uint salt) { return hashCombine(hashCombine(hashCombine(path.pixel, path.sampleIndex), frame.settings.z), path.depth() * 4 + salt); }

//! The RGB film value of a spectral contribution; indirect contributions are clamped to frame.limits.x when it is set.
float3 filmValue(WavefrontFrame frame, Wavelengths wavelengths, float4 contribution, bool indirect) {
    float3 rgb = max(sanitize(wavelengths.toFilm(contribution)), 0.0);
    const float clampValue = frame.limits.x;
    if (clampValue > 0.0 && indirect) {
        const float peak = maxComponent(rgb);
        if (peak > clampValue)
            rgb *= clampValue / peak;
    }
    return rgb;
}

void addRadiance(WavefrontFrame frame, inout PathState path, Wavelengths wavelengths, float4 contribution) {
    path.radiance += filmValue(frame, wavelengths, contribution, path.depth() > 0);
}

//! Adds light to a pixel without counting a sample; the shadow rays of ended paths still arrive this way.
void splatRadiance(FilmRecord film, uint pixel, float3 value) {
    const AtomicPtr<float> target = film.frame + pixel * 4;
    atomicAdd(target, value.x);
    atomicAdd(target + 1, value.y);
    atomicAdd(target + 2, value.z);
}

//! Writes the finished path into the frame film with atomics: several paths can share a pixel.
void finishPath(FilmRecord film, PathState path) {
    const AtomicPtr<float> pixel = film.frame + path.pixel * 4;
    const float3 value = sanitize(path.radiance);
    atomicAdd(pixel, value.x);
    atomicAdd(pixel + 1, value.y);
    atomicAdd(pixel + 2, value.z);
    atomicAdd(pixel + 3, 1.0);
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
    if (path.depth() + 1 >= frame.settings.x)
        return false;
    if (path.depth() < frame.settings.y)
        return true;
    const float survival = clamp(max(max(path.throughput.x, path.throughput.y), max(path.throughput.z, path.throughput.w)), 0.05, 0.95);
    if (u >= survival)
        return false;
    path.throughput /= survival;
    return true;
}
