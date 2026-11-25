import NoorRHI;
import Lights.SgLighting;
import Wavefront.Records;
import Wavefront.PathState;
import Wavefront.Queues;
import Wavefront.RadianceCache;
import Common.Math;
import Common.Random;
import Common.RayCone;
import Scene.Camera;
import Spectral.Wavelengths;

uint2 pixelCoordinate(FilmRecord film, uint pixel) { return uint2(pixel % film.width, pixel / film.width); }

//! A 16x16 ordered dither in [0, 1) that spreads the first sample of a budget below one path per pixel over the frames.
float tileDither(uint2 pixel) {
    const uint x = pixel.x & 15u;
    const uint y = pixel.y & 15u;
    const uint v = x ^ y;
    const uint bits = ((v & 1u) << 7) | ((x & 1u) << 6) | ((v & 2u) << 4) | ((x & 2u) << 3) | ((v & 4u) << 1) | (x & 4u) | ((v & 8u) >> 2) | ((x & 8u) >> 3);
    return (float(bits) + 0.5) / 256.0;
}

float blackmanHarrisIntegral(float t) {
    return 0.35875 * t + (-0.48829 * sin(twoPi * t) + 0.07064 * sin(2.0 * twoPi * t) - 0.01168 / 3.0 * sin(3.0 * twoPi * t)) / twoPi;
}

//! Inverts the integral by Newton steps kept inside the bracket, which bisect where the window is flat.
float sampleBlackmanHarris(float u) {
    const float target = u * 0.35875;
    float low = 0.0;
    float high = 1.0;
    float t = u;
    for (uint i = 0; i < 8; ++i) {
        const float error = blackmanHarrisIntegral(t) - target;
        if (error > 0.0)
            high = t;
        else
            low = t;
        const float next = t - error / max(blackmanHarris(t), 1e-6);
        t = next > low && next < high ? next : 0.5 * (low + high);
    }
    return t;
}

//! Cycles' pixel filter by importance sampling each axis, so every camera sample has weight one and the mean of the samples is the filtered pixel.
float2 samplePixelFilter(float2 u, float width) {
    if (width <= 0.0)
        return u;
    return 0.5 + (float2(sampleBlackmanHarris(u.x), sampleBlackmanHarris(u.y)) - 0.5) * width;
}

//! Per pixel: spreads the path budget evenly, turns its share into whole samples with a carried credit, and appends them to the sample list.
void buildSampleList(WavefrontFrame frame, uint pixel) {
    const FilmRecord film = frame.film;
    if (pixel >= film.width * film.height)
        return;
    uint4 state = film.pixels[pixel];
    const float rate = frame.limits.y / float(film.width * film.height);
    float credit = state.z == 0 && asfloat(state.y) == 0.0 ? tileDither(pixelCoordinate(film, pixel)) : asfloat(state.y);
    credit += rate;
    const uint count = min(uint(floor(credit)), 16u);
    credit -= float(count);
    state.y = asuint(min(credit, 1.0));
    if (count != 0) {
        const uint first = atomicAdd(frame.queues.counters + counterSamples, count);
        for (uint j = 0; j < count; ++j)
            if (first + j < frame.queues.sampleCapacity)
                frame.queues.samples[first + j] = uint2(pixel | (j == 0 ? 0x80000000u : 0u), state.z + j);
        state.z += count;
    }
    film.pixels[pixel] = state;
}

//! Starts a camera path in a pool slot. Returns false when the camera produced no ray; the sample then counts as black.
bool spawnPath(WavefrontFrame frame, CameraRecord camera, uint slot, uint2 entry) {
    const FilmRecord film = frame.film;
    const SceneRecord scene = frame.scene[0];
    PathState path;
    path.pixel = entry.x & 0x7fffffffu;
    path.sampleIndex = entry.y;
    const bool primarySample = (entry.x & 0x80000000u) != 0;
    const bool rasterized = primarySample && frame.raster.enabled != 0;
    path.flags = pathFlagDelta | (primarySample ? pathFlagPrimarySample : 0u) | (rasterized ? pathFlagRasterPrimary : 0u) |
                 (frame.raster.rgb != 0 ? pathFlagRgb : 0u);
    path.radiance = float3(0.0, 0.0, 0.0);
    path.diffuseFraction = 1.0;
    path.primaryDirection = float3(0.0, 0.0, 0.0);
    path.setReceiver(mediumReceiver());
    SobolSampler cameraSampler = pathSampler(frame, path, patternCamera);
    const float4 u = cameraSampler.next4();
    SobolSampler wavelengthSampler = pathSampler(frame, path, patternWavelength);
    const float hero = wavelengthSampler.next();
    path.hero = packHero(hero);
    const Wavelengths wavelengths = pathWavelengths(scene, path);
    const uint2 coordinate = pixelCoordinate(film, path.pixel);
    const float2 jitter = rasterized ? frame.raster.jitter : (frame.limits.z >= 0.0 ? frame.limits.zw : samplePixelFilter(u.xy, frame.filter.x));
    const float2 filmPosition = (float2(coordinate) + jitter) / float2(film.width, film.height);
    const CameraRay ray = generateCameraRay(camera, filmPosition, u.zw, wavelengths.lambda.x);
    path.origin = ray.origin;
    path.direction = ray.direction;
    path.cone = cameraRayCone(camera);
    path.previousPdf = 0.0;
    path.throughput = float4(ray.weight, ray.weight, ray.weight, ray.weight);
    if (ray.dispersive)
        path.setFlag(pathFlagSecondaryTerminated, true);
    if (!ray.valid || ray.weight <= 0.0) {
        if (frame.options.y == 0)
            finishPath(film, path);
        path.flags |= pathDepthMask;
        storePath(frame.pool, slot, path);
        return false;
    }
    //! ReSTIR PT updates the cache in its own path kernels.
    if (primarySample && frame.options.y == 0 && isRadianceCacheUpdatePixel(frame.cache, coordinate, frame.cache.frameIndex)) {
        path.setFlag(pathFlagCacheUpdate, true);
        beginRadianceCacheUpdate(frame.cache, slot);
    }
    storePath(frame.pool, slot, path);
    frame.pool.lightPath[slot] = uint2(0u, 0u);
    frame.pool.minRayPdf[slot] = floatMax;
    return true;
}


//! Fills the first slots of the pool; a camera sample without a ray stays in the queue as a dead path that Extend frees.
void generatePath(WavefrontFrame frame, uint index) {
    spawnPath(frame, frame.camera[0], index, frame.queues.samples[index]);
    frame.queues.active0[index] = index;
}

//! Refills ended slots with the next samples of the list, into the queue of the next iteration.
void regeneratePath(WavefrontFrame frame, uint parity, uint index) {
    const uint slot = frame.queues.free[index];
    const uint available = min(atomicLoad(frame.queues.counters + counterSamples), frame.queues.sampleCapacity);
    const CameraRecord camera = frame.camera[0];
    for (uint attempt = 0; attempt < 8; ++attempt) {
        const uint sample = atomicAdd(frame.queues.counters + counterSampleCursor, 1u);
        if (sample >= available)
            return;
        if (spawnPath(frame, camera, slot, frame.queues.samples[sample])) {
            pushActive(frame, parity, slot);
            return;
        }
    }
}

//! At the end of the frame: paths still in flight write what they have gathered.
void finishInFlightPath(WavefrontFrame frame, uint parity, uint index) {
    const uint slot = activeQueue(frame.queues, parity)[index];
    const PathState path = loadPath(frame.pool, slot);
    if (!isDeadPath(path))
        finishPath(frame.film, path);
}
