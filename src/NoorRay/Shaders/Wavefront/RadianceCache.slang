import NoorRHI;
import Wavefront.Records;
import Spectral.Wavelengths;
import Common.Random;

//! SHaRC atomics need real buffer pointers, which the WebGPU heap does not have, so there the cache is always off.
#if !defined(NOORRHI_WEBGPU)
#define SHARC_UPDATE 1
#define SHARC_PROPAGATION_DEPTH 2
#define SHARC_SEPARATE_EMISSIVE 1
#define SHARC_ENABLE_RESPONSIVE_LIGHTING 0
#define SHARC_ENABLE_GLSL 0
#define RW_STRUCTURED_BUFFER(name, type) type* name
#define BUFFER_AT_OFFSET(name, offset) name[offset]
#include "SharcCommon.h"

T* radianceCachePointer<T, U>(Ptr<U> pointer) { return (T*)(uint64_t(pointer.low) | uint64_t(pointer.high) << 32); }

SharcParameters radianceCacheParameters(RadianceCacheRecord cache) {
    SharcParameters parameters;
    parameters.hashGridParameters.cameraPosition = cache.camera;
    parameters.hashGridParameters.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
    parameters.hashGridParameters.sceneScale = cache.sceneScale;
    parameters.hashGridParameters.levelBias = SHARC_GRID_LEVEL_BIAS;
    parameters.hashGridData.capacity = cache.capacity;
    parameters.hashGridData.hashEntriesBuffer = radianceCachePointer<uint64_t>(cache.hashes);
    parameters.radianceScale = cache.radianceScale;
    parameters.accumulationBuffer = radianceCachePointer<SharcAccumulationData>(cache.accumulation);
    parameters.resolvedBuffer = radianceCachePointer<SharcPackedData>(cache.resolved);
    return parameters;
}

SharcHitData radianceCacheHit(float3 position, float3 normal, float3 emission) {
    SharcHitData hit;
    hit.positionWorld = position;
    hit.normalWorld = normal;
    hit.emissive = emission;
    return hit;
}

SharcState loadRadianceCacheState(RadianceCacheState stored) {
    SharcState state;
    state.cacheIndices[0] = stored.indices.x;
    state.cacheIndices[1] = stored.indices.y;
    state.sampleWeights[0] = stored.weight0;
    state.sampleWeights[1] = stored.weight1;
    state.pathLength = stored.length;
    return state;
}

//! The SHaRC fields of the state; pending and direct stay as they were.
RadianceCacheState storeRadianceCacheState(SharcState state, RadianceCacheState stored) {
    stored.indices = uint2(state.cacheIndices[0], state.cacheIndices[1]);
    stored.weight0 = state.sampleWeights[0];
    stored.weight1 = state.sampleWeights[1];
    stored.length = state.pathLength;
    return stored;
}
#endif

static const uint radianceCacheStopped = 0xffffffffu;

bool isRadianceCacheOn(RadianceCacheRecord cache) {
#if defined(NOORRHI_WEBGPU)
    return false;
#else
    return !isNull(cache.hashes);
#endif
}

//! A rule of the path alone, so replays end as their initial paths did; after a delta bounce the cache would show sharply, so the path goes on.
bool mayEndInRadianceCache(RadianceCacheRecord cache, uint bounceDepth, bool deltaBounce) {
    return isRadianceCacheOn(cache) && bounceDepth >= cache.depth && !deltaBounce;
}

//! The cached outgoing radiance at a vertex without its emission, upsampled to the wavelengths of the path. False when the cache has no entry there.
bool queryRadianceCache(RadianceCacheRecord cache, Wavelengths wavelengths, float3 position, float3 normal, out float4 radiance) {
    radiance = float4(0.0, 0.0, 0.0, 0.0);
#if defined(NOORRHI_WEBGPU)
    return false;
#else
    float3 rgb;
    if (!SharcGetCachedRadiance(radianceCacheParameters(cache), radianceCacheHit(position, normal, float3(0.0, 0.0, 0.0)), rgb, false))
        return false;
    radiance = wavelengths.upsample(rgb);
    return true;
#endif
}

//! The pixels whose initial paths update the cache this frame: one per block, at an offset that changes each frame.
bool isRadianceCacheUpdatePixel(RadianceCacheRecord cache, uint2 pixel, uint frameIndex) {
    if (!isRadianceCacheOn(cache))
        return false;
    const uint stride = cache.updateStride;
    const uint2 block = pixel / stride;
    const uint offset = hashCombine(hashCombine(block.x, block.y), frameIndex) % (stride * stride);
    return all(pixel == block * stride + uint2(offset % stride, offset / stride));
}

//! The RGB weight of a spectral throughput: its film value over the film value of a flat spectrum with the same wavelengths.
float3 radianceCacheWeight(Wavelengths wavelengths, float4 throughput) {
    return max(wavelengths.toFilm(throughput), 0.0) / max(wavelengths.toFilm(float4(1.0, 1.0, 1.0, 1.0)), 1e-4);
}

//! Starts the update of a path at its primary vertex.
void beginRadianceCacheUpdate(RadianceCacheRecord cache, uint slot) {
#if !defined(NOORRHI_WEBGPU)
    SharcState state;
    SharcInit(state);
    RadianceCacheState stored;
    stored.pending = float3(1.0, 1.0, 1.0);
    stored.direct = float3(0.0, 0.0, 0.0);
    stored.unused = 0;
    cache.states[slot] = storeRadianceCacheState(state, stored);
#endif
}

//! Adds the light of a vertex to the cache: direct is its NEE and emission the MIS-weighted emission that the previous vertex sees, both local and spectral.
void updateRadianceCacheHit(RadianceCacheRecord cache, Wavelengths wavelengths, uint slot, float3 position, float3 normal, float4 direct, float4 emission,
                            float random) {
#if !defined(NOORRHI_WEBGPU)
    RadianceCacheState stored = cache.states[slot];
    if (stored.length == radianceCacheStopped)
        return;
    SharcState state = loadRadianceCacheState(stored);
    const SharcHitData hit = radianceCacheHit(position, normal, max(wavelengths.toFilm(emission), 0.0));
    const bool continues = SharcUpdateHit(radianceCacheParameters(cache), state, hit, max(wavelengths.toFilm(direct), 0.0), random);
    stored = storeRadianceCacheState(state, stored);
    if (!continues)
        stored.length = radianceCacheStopped;
    cache.states[slot] = stored;
#endif
}

//! Gives the vertices of the update the light that arrives along the last segment: a miss, or the vertex where the path ends in the cache.
void updateRadianceCacheEnd(RadianceCacheRecord cache, Wavelengths wavelengths, uint slot, float4 radiance) {
#if !defined(NOORRHI_WEBGPU)
    RadianceCacheState stored = cache.states[slot];
    if (stored.length == radianceCacheStopped)
        return;
    SharcUpdateMiss(radianceCacheParameters(cache), loadRadianceCacheState(stored), max(wavelengths.toFilm(radiance), 0.0));
    stored.length = radianceCacheStopped;
    cache.states[slot] = stored;
#endif
}

//! The throughput of the segment that the update path takes next.
void updateRadianceCacheThroughput(RadianceCacheRecord cache, Wavelengths wavelengths, uint slot, float4 throughput) {
#if !defined(NOORRHI_WEBGPU)
    RadianceCacheState stored = cache.states[slot];
    if (stored.length == radianceCacheStopped)
        return;
    SharcState state = loadRadianceCacheState(stored);
    SharcSetThroughput(state, radianceCacheWeight(wavelengths, throughput));
    cache.states[slot] = storeRadianceCacheState(state, stored);
#endif
}

//! The wavefront path tracer defers the weight of a scatter, because Shadow adds the NEE of the vertex later with the weights of the vertex.
void deferRadianceCacheThroughput(RadianceCacheRecord cache, Wavelengths wavelengths, uint slot, float4 throughput) {
#if !defined(NOORRHI_WEBGPU)
    RadianceCacheState stored = cache.states[slot];
    if (stored.length == radianceCacheStopped)
        return;
    stored.pending *= radianceCacheWeight(wavelengths, throughput);
    cache.states[slot] = stored;
#endif
}

//! Applies the deferred scatter weight at the next vertex of the update, before the update adds light there.
void applyRadianceCachePending(RadianceCacheRecord cache, uint slot) {
#if !defined(NOORRHI_WEBGPU)
    RadianceCacheState stored = cache.states[slot];
    if (stored.length == radianceCacheStopped)
        return;
    SharcState state = loadRadianceCacheState(stored);
    SharcSetThroughput(state, stored.pending);
    stored = storeRadianceCacheState(state, stored);
    stored.pending = float3(1.0, 1.0, 1.0);
    stored.direct = float3(0.0, 0.0, 0.0);
    cache.states[slot] = stored;
#endif
}

//! Keeps the NEE of the vertex, local and spectral, for Shadow.
void setRadianceCacheDirect(RadianceCacheRecord cache, Wavelengths wavelengths, uint slot, float4 direct) {
#if !defined(NOORRHI_WEBGPU)
    RadianceCacheState stored = cache.states[slot];
    if (stored.length == radianceCacheStopped)
        return;
    stored.direct = max(wavelengths.toFilm(direct), 0.0);
    cache.states[slot] = stored;
#endif
}

//! Adds the NEE that Shadow found unoccluded to the entry of its vertex and to the entry before, as SharcUpdateHit adds direct light; the vertex counted its sample already.
void addRadianceCacheDirect(RadianceCacheRecord cache, uint slot) {
#if !defined(NOORRHI_WEBGPU)
    const RadianceCacheState stored = cache.states[slot];
    if (stored.length == radianceCacheStopped || stored.length == 0 || all(stored.direct <= 0.0))
        return;
    const SharcParameters parameters = radianceCacheParameters(cache);
    const float3 axis = float3(0.0, 0.0, 1.0);
    if (stored.indices.x != HASH_GRID_INVALID_CACHE_INDEX)
        SharcAddVoxelData(parameters, stored.indices.x, stored.direct, float3(1.0, 1.0, 1.0), axis, 0.0, 0);
    if (stored.length > 1 && stored.indices.y != HASH_GRID_INVALID_CACHE_INDEX)
        SharcAddVoxelData(parameters, stored.indices.y, stored.direct, stored.weight1, axis, 0.0, 0);
#endif
}

//! Ends the update without more light, as at a vertex that it does not model.
void stopRadianceCacheUpdate(RadianceCacheRecord cache, uint slot) {
#if !defined(NOORRHI_WEBGPU)
    cache.states[slot].length = radianceCacheStopped;
#endif
}

//! Resolves or clears one cache entry.
void resolveRadianceCache(RadianceCacheRecord cache, uint mode, uint frameIndex, uint entry) {
#if !defined(NOORRHI_WEBGPU)
    if (mode == radianceCacheModeClear) {
        cache.hashes[entry] = uint2(0, 0);
        cache.accumulation[entry] = uint4(0, 0, 0, 0);
        cache.resolved[entry] = uint4(0, 0, 0, 0);
        return;
    }
    SharcResolveParameters resolve;
    resolve.cameraPositionPrev = cache.previousCamera;
    resolve.accumulationFrameNum = cache.accumulationFrames;
    resolve.responsiveFrameNum = cache.accumulationFrames;
    resolve.staleFrameNumMax = 64;
    resolve.frameIndex = frameIndex;
    SharcResolveEntry(entry, radianceCacheParameters(cache), resolve);
#endif
}
