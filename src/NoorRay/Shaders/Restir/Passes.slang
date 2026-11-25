import Lights.Lights;
import NoorRHI;
import Lights.SgLighting;
import Restir.Bridge;
import Wavefront.RadianceCache;
import Wavefront.Records;
import Wavefront.PathState;
import Wavefront.Queues;
import Wavefront.Extend;
import Wavefront.Media;
import Wavefront.Direct;
import Wavefront.SurfaceShading;
import Common.LinkConstants;
import Common.Math;
import Common.Random;
import Common.RayCone;
import Lights.Environment;
import Bsdf.Principled;
import Materials.Closure;
import Scene.Camera;
import Scene.SceneSurface;
import Spectral.Wavelengths;

static const uint restirTemporalPass = 4u;
static const uint restirSpatialPass = 5u;
static const uint restirShadePass = 6u;
static const uint restirReplaySeedPass = 3u;
//! Light that scatters more than once in a volume changes slowly over the image, so a quarter of the forks trace, at four times the weight.
static const float restirForkProbability = 0.25;

Ptr<uint> pendingList(RestirFrame frame, uint list) { return frame.pending + list * frame.pendingCapacity; }

//! The number of pixels that a resampling stage runs: the sweep in the first stage, then the pixels that the stage before left waiting.
uint resampleCount(RestirFrame frame, RestirArguments arguments) {
    return arguments.stage == 0 ? arguments.count : atomicLoad(frame.counters + (arguments.stage + 2) % restirStages);
}

//! Traces the surfaces of a path or replay ray; a miss goes to the first bin, so that the next kernel gets every path in one range.
void restirExtendPath(RestirFrame frame, uint parity, uint mode, uint index) {
    const WavefrontFrame wavefront = frame.wavefront[0];
    const PathPoolRecord pool = wavefront.pool;
    const uint slot = activeQueue(wavefront.queues, parity)[index];
    uint4 state = pool.state[slot];
    PathHit hit = (PathHit)0;
    hit.kind = hitKindMiss;
    hit.t = floatMax;
    if ((state.z & pathDepthMask) == pathDepthMask) {
        storeHit(pool, slot, hit);
        pushBin(wavefront, slot, mediumBin);
        return;
    }
    const SceneRecord scene = wavefront.scene[0];
    RayDesc ray;
    ray.Origin = pool.origin[slot].xyz;
    ray.Direction = pool.direction[slot].xyz;
    ray.TMin = 0.0;
    const bool primary = mode == restirPathInitial && (state.z & pathDepthMask) == 0;
    ray.TMax = primary ? cameraFarDistance(wavefront.camera[0], ray.Origin, ray.Direction) : floatMax;
    SurfaceFilter filter;
    filter.scene = scene;
    filter.alphaT = floatMax;
    filter.alphaInstance = invalidIndex;
    filter.alphaPrimitive = 0;
    filter.alphaBarycentrics = float2(0.0, 0.0);
    RayHit meshHit = (RayHit)0;
    const uint mask = rayInstanceMask(loadLightPath(pool, slot), primary);
    if (traceRayInline(scene.tlas, RAY_FLAG_NONE, mask, ray, filter, meshHit)) {
        hit.t = meshHit.t;
        hit.kind = hitKindMesh;
        hit.instance = meshHit.instanceId;
        hit.primitive = meshHit.primitiveIndex;
        hit.barycentrics = meshHit.barycentrics;
    }
    const bool alpha = hasAlphaTest && filter.alphaT < hit.t;
    if (alpha) {
        hit.t = filter.alphaT;
        hit.kind = hitKindMesh;
        hit.instance = filter.alphaInstance;
        hit.primitive = filter.alphaPrimitive;
        hit.barycentrics = filter.alphaBarycentrics;
    }
    if (alpha != ((state.z & pathFlagAlphaCandidate) != 0)) {
        state.z ^= pathFlagAlphaCandidate;
        pool.state[slot] = state;
    }
    storeHit(pool, slot, hit);
    pushBin(wavefront, slot, hit.kind == hitKindMesh ? surfaceBin(scene, hit.instance, hit.primitive) : mediumBin);
}

//! A hit that the stochastic opacity of its alpha-tested material lets through: the ray goes on from behind it with the same depth.
bool skipTransparentHit(WavefrontFrame frame, uint parity, uint slot, inout PathState path, Surface surface, RayCone cone, PackedClosure closure) {
    if (!hasAlphaTest || !path.hasFlag(pathFlagAlphaCandidate))
        return false;
    path.setFlag(pathFlagAlphaCandidate, false);
    if (passesOpacity(frame, path, closureOpacity(closure), opacitySalt(frame.pool, slot)))
        return false;
    recordTransparentBounce(frame.pool, slot);
    path.origin = offsetAlong(surface.context.position.xyz, surface.worldNormal, path.direction);
    path.cone = cone;
    storePath(frame.pool, slot, path);
    pushActive(frame, parity, slot);
    return true;
}

//! ReSTIR PT resamples surface paths only: each fog segment of an initial path lights its points, which add to the film unresampled.
void lightRestirSegment(WavefrontFrame frame, PathState path, MediumRange range) {
    const float4 lightU = bouncePattern(frame, path, 3);
    const VolumeDirect plan = planVolumeDirect(restirScene, range, path.origin, path.direction, bouncePattern(frame, path, 2).xy, lightU, true);
    const uint count = mediumPointCount(path);
    float4 value = float4(0.0, 0.0, 0.0, 0.0);
    for (uint point = 0; point < count; ++point) {
        const MediumLight light = sampleMediumLight(restirScene, path, restirWavelengths, range, stratifyMediumPoint(plan, point, count),
                                                    rotateMediumLightU(lightU, point), hashCombine(raySeed(frame, path, 7), point),
                                                    (frame.settings.x >> 8) == 0);
        value += light.emission / float(count);
        if (any(light.contribution > 0.0) && isRestirRayVisible(light.position, light.direction, light.distance))
            value += light.contribution / float(count) * restirTransmittance(light.position, light.direction, light.distance);
    }
    const float3 film = filmValue(frame, restirWavelengths, value, path.depth() > 0);
    splatRadiance(frame.film, path.pixel, film);
    if (path.depth() == 0)
        splitFogLight(frame.film, path.pixel, film);
    else
        splitPathLight(frame.film, path.pixel, film, 1.0, float3(0.0, 0.0, 0.0));
}

//! Delta tracking of the segment as the wavefront path tracer does it; each scatter is a candidate fork of the pixel, and the pixel keeps one at random.
void offerRestirFork(RestirFrame frame, uint slot, PathState path, MediumRange range, float tMax) {
    const WavefrontFrame wavefront = frame.wavefront[0];
    if ((wavefront.settings.x >> 8) == 0)
        return;
    float4 absorbed;
    const MediumEvent event =
        trackMedium(restirScene, range, path.origin, path.direction, tMax, restirWavelengths, raySeed(wavefront, path, 1), true, path.throughput, absorbed);
    if (event.kind != mediumEventScatter)
        return;
    RestirForkRecord fork = frame.forks[path.pixel];
    const uint count = fork.frame == frame.runtime.frameIndex ? fork.count + 1 : 1u;
    Random random = makeRandom(raySeed(wavefront, path, 10));
    if (random.next() * float(count) < 1.0) {
        fork.origin = float4(path.origin, asfloat(packRayCone(path.cone)));
        fork.direction = float4(path.direction, path.previousPdf);
        fork.throughput = path.throughput;
        const uint flags = (path.flags & ~(pathFlagPrimarySample | pathFlagAlphaCandidate | pathVolumeMask)) | pathFlagFork |
                           (path.depth() == 0 ? pathFlagCameraFork : 0u);
        fork.state = uint4(path.pixel, path.sampleIndex, flags, path.hero);
        fork.t = event.t;
        fork.volume = event.volume;
        fork.lightPath = loadLightPath(wavefront.pool, slot);
    }
    fork.count = count;
    fork.frame = frame.runtime.frameIndex;
    frame.forks[path.pixel] = fork;
}

//! The fog of a path ray segment: it lights the film, offers a fork, and attenuates the prefix and the suffix past the reconnection vertex.
void crossRestirFog(RestirFrame frame, uint slot, inout PathState path, float tMax) {
    if (!hasVolumes || restirScene.counts.w == 0)
        return;
    const MediumRange range = mediumRange(restirScene, path.origin, path.direction, tMax);
    if (!hasMedium(range))
        return;
    //! The pool throughput leaves out the fog of the camera segment, which the shade pass applies to the reservoir.
    PathState lit = path;
    if (path.depth() > 0) {
        const uint index = restirSurfaceIndex(path.pixel, false);
        lit.throughput *= restirCameraTransmittance(unpackRestirSurface(frame.surfaces[index], index));
    }
    lightRestirSegment(frame.wavefront[0], lit, range);
    offerRestirFork(frame, slot, lit, range, tMax);
    if (path.depth() == 0)
        return;
    const float4 transmittance = restirTransmittance(path.origin, path.direction, tMax);
    path.throughput *= transmittance;
    if (path.depth() + 1 <= (asuint(frame.paths[slot].sums.w) & 15u))
        return;
    path.radiance *= transmittance.xyz;
    path.diffuseFraction *= transmittance.w;
}

//! Moves the fork of a pixel to a pool slot as a path at its volume scatter in the medium bin; the segment point lit the scatter already.
void startRestirFork(RestirFrame frame, uint parity, uint pixel, uint slot) {
    const RestirForkRecord fork = frame.forks[pixel];
    if (fork.frame != frame.runtime.frameIndex || fork.count == 0)
        return;
    Random random = makeRandom(hashCombine(hashCombine(pixel, frame.runtime.frameIndex), restirReplaySeedPass));
    if (random.next() >= restirForkProbability)
        return;
    const WavefrontFrame wavefront = frame.wavefront[0];
    const PathPoolRecord pool = wavefront.pool;
    pool.origin[slot] = fork.origin;
    pool.direction[slot] = fork.direction;
    pool.throughput[slot] = fork.throughput * (float(fork.count) / restirForkProbability);
    pool.radiance[slot] = float4(0.0, 0.0, 0.0, 1.0);
    pool.primaryDirection[slot] = noPrimaryDirection;
    pool.receiver[slot] = packReceiver(mediumReceiver());
    pool.state[slot] = fork.state;
    pool.lightPath[slot] = uint2(fork.lightPath, fork.lightPath);
    pool.minRayPdf[slot] = floatMax;
    PathHit hit = (PathHit)0;
    hit.kind = hitKindVolume;
    hit.t = fork.t;
    hit.instance = fork.volume;
    storeHit(pool, slot, hit);
    pushBin(wavefront, slot, mediumBin);
    activeQueue(wavefront.queues, parity)[atomicAdd(wavefront.queues.counters + counterActive, 1u)] = slot;
}

RTXDI_RandomSamplerState restirReplaySampler(uint2 pixel) { return RTXDI_InitRandomSampler(pixel, restirFrame.runtime.frameIndex, restirReplaySeedPass); }

RTXDI_RandomSamplerState restirInitialSampler(uint2 pixel, uint index) {
    RTXDI_RandomSamplerState random = RTXDI_InitRandomSampler(pixel, restirFrame.runtime.frameIndex, restirInitialPass);
    random.index = index;
    return random;
}

//! The random number of the SHaRC update at a vertex; it is not drawn from the initial sampler, so the update pixels sample their paths as the others do.
float radianceCacheRandom(uint2 pixel, uint bounceDepth) {
    Random random = makeRandom(hashCombine(hashCombine(hashCombine(pixel.x, pixel.y), restirFrame.runtime.frameIndex), bounceDepth));
    return random.next();
}

//! The end of an initial path: its context becomes the initial reservoir of the pixel.
void finishInitialPath(RTXDI_PathTracerContextInitial context, uint2 pixel, inout RTXDI_RandomSamplerState random) {
    const RTXDI_PTParameters pt = restirFrame.pt;
    const RTXDI_PTReservoir candidate = InitializePTReservoirFromPTContext(context, restirReplaySampler(pixel));
    RTXDI_PTReservoir reservoir = RTXDI_EmptyPTReservoir();
    RTXDI_CombinePTReservoirs(reservoir, candidate, RTXDI_GetNextRandom(random), context.GetSelectedTargetFunction());
    FinalizeResampling(reservoir, context.GetSelectedTargetFunction());
    RTXDI_StorePTReservoir(reservoir, pt.reservoirBuffer, pixel, pt.bufferIndices.initialPathTracerOutputBufferIndex);
}

//! Samples the BSDF of the vertex with the replay sampler and queues the ray, as the loop of RTXDI_PathTrace does. Returns false when the path ends.
bool continueInitialPath(WavefrontFrame frame, uint parity, uint slot, inout PathState path, inout RTXDI_PathTracerContextInitial context, RAB_Surface surface,
                         PrincipledBsdf bsdf, RayCone cone, uint2 pixel, inout RTXDI_RandomSamplerState random) {
    context.BeginPathState();
    const RTXDI_RandomSamplerState replay = restirReplaySampler(pixel);
    RTXDI_RandomSamplerState sampler = restirBsdfRandom(replay.seed, replay.index, context.GetBounceDepth());
    const BsdfSample sample = bsdf.sample(sampleRestirRandom(sampler));
    context.SetBrdfRaySample(makeBrdfRaySample(sample));
    if (!context.ValidContinuationRayBrdfOverPdf())
        return false;
    float4 segment = context.GetContinuationRayBrdfOverPdf();
    context.MultiplyPathThroughput(segment);
    if (context.ShouldRunRussianRoulette() && context.GetBounceDepth() >= frame.settings.y + 1) {
        const float4 throughput = context.GetPathThroughput();
        const float survival = clamp(max(max(throughput.x, throughput.y), max(throughput.z, throughput.w)), 0.05, 0.95);
        if (RTXDI_GetNextRandom(random) >= survival)
            return false;
        context.MultiplyPathThroughput(float4(1.0 / survival));
        context.RecordRussianRouletteProbability(survival);
        segment /= survival;
    }
    if (isRadianceCacheUpdatePixel(restirFrame.cache, pixel, restirFrame.runtime.frameIndex))
        updateRadianceCacheThroughput(restirFrame.cache, restirWavelengths, slot, segment);
    RayDesc ray;
    ray.Origin = offsetAlong(surface.position, surface.normal, sample.direction);
    ray.Direction = sample.direction;
    ray.TMin = 0.0;
    ray.TMax = floatMax;
    context.SetContinuationRay(ray);
    if (!context.AnalyzePathReconnectibilityBeforeTrace())
        return false;
    const bool delta = (sample.flags & bsdfFlagDelta) != 0;
    path.origin = ray.Origin;
    path.setReceiver(restirReceiver(surface));
    path.direction = ray.Direction;
    path.previousPdf = delta ? 0.0 : sample.pdf;
    path.setFlag(pathFlagDelta, delta);
    path.cone = restirContinuationCone(cone, delta, bsdf.roughness());
    recordLightPath(frame.pool, slot, bsdfLightPathEvent(sample.flags));
    path.advanceDepth();
    path.throughput = context.ctx.ptState.pathThroughput;
    const float4 suffix = context.ctx.ptState.rcPathThroughput;
    path.radiance = suffix.xyz;
    path.diffuseFraction = suffix.w;
    storePath(frame.pool, slot, path);
    restirFrame.paths[slot] = packInitialContext(context, random.index);
    pushActive(frame, parity, slot);
    return true;
}

//! The primary vertex of a pixel: it writes the G-buffer surface and closure and starts the initial context there.
void beginInitialPath(WavefrontFrame frame, uint parity, uint slot, inout PathState path, Surface loaded, PackedClosure closure, RayCone cone, uint2 pixel,
                      uint instance) {
    RAB_Surface surface = makeRestirSurface(loaded, length(loaded.context.position.xyz - cameraPosition(restirCamera)), closure);
    const uint index = restirSurfaceIndex(path.pixel, false);
    restirFrame.surfaces[index] = packRestirSurface(surface, instance);
    restirFrame.closures[index] = closure;
    surface.closure = index;
    const RTXDI_PTParameters pt = restirFrame.pt;
    RTXDI_PathTracerRandomContext randomContext =
        RTXDI_InitializePathTracerRandomContext(pixel, restirFrame.runtime.frameIndex, restirInitialPass, restirReplaySeedPass);
    RTXDI_PathTracerContextInitial context =
        RTXDI_InitializePathTracerContextInitial(GetPTContextParams(pt.initialSampling, initialRuntime(), pt.reconnection), surface, randomContext);
    RTXDI_RandomSamplerState random = randomContext.initialRandomSamplerState;
    const PrincipledBsdf bsdf = makeRestirBsdf(unpackClosure(closure), surface);
    RestirLightSample light;
    float4 direct = float4(0.0, 0.0, 0.0, 0.0);
    if (sampleRestirLight(surface, bsdf, 0u, random, light)) {
        context.RecordPrimaryNeeLightSample(light.light, light.radiance, light.neePdf, light.sample, random);
        direct += light.radiance;
    }
    if (sampleRestirBsdfLight(surface, bsdf, 0u, random, light)) {
        context.RecordPrimaryNeeLightSample(light.light, light.radiance, light.neePdf, light.sample, random);
        direct += light.radiance;
    }
    if (isRadianceCacheUpdatePixel(restirFrame.cache, pixel, restirFrame.runtime.frameIndex)) {
        beginRadianceCacheUpdate(restirFrame.cache, slot);
        updateRadianceCacheHit(restirFrame.cache, restirWavelengths, slot, loaded.context.position.xyz, loaded.context.geometricNormal.xyz, direct,
                               float4(0.0, 0.0, 0.0, 0.0), radianceCacheRandom(pixel, 1));
    }
    if (continueInitialPath(frame, parity, slot, path, context, surface, bsdf, cone, pixel, random))
        return;
    finishInitialPath(context, pixel, random);
    pushFree(frame, slot);
}

//! A pixel whose camera ray found no surface has no G-buffer surface and an empty reservoir.
void clearRestirPixel(PathState path, uint2 pixel, uint instance) {
    RAB_Surface surface = RAB_EmptySurface();
    surface.view = -path.direction;
    restirFrame.surfaces[restirSurfaceIndex(path.pixel, false)] = packRestirSurface(surface, instance);
    const RTXDI_PTParameters pt = restirFrame.pt;
    RTXDI_StorePTReservoir(RTXDI_EmptyPTReservoir(), pt.reservoirBuffer, pixel, pt.bufferIndices.initialPathTracerOutputBufferIndex);
}

//! The primary hit distance of an initial path; the pool holds the reconnection throughput, so the diffuse share comes from the G-buffer BSDF.
void splitRestirHitDistance(RestirFrame frame, PathState path, float distance) {
    const FilmRecord film = frame.wavefront[0].film;
    if (film.splitsLighting == 0 || path.depth() != 1)
        return;
    const uint index = restirSurfaceIndex(path.pixel, false);
    float diffuseFraction;
    restirBsdf(unpackRestirSurface(frame.surfaces[index], index)).evaluate(path.direction, diffuseFraction);
    splitHitDistance(film, path.pixel, distance, diffuseFraction);
}

//! One wavefront step of an initial path at its hit or miss: the vertex of the loop in RTXDI_PathTrace, with the context kept in the path record.
void restirBouncePath(RestirFrame frame, uint parity, uint slot) {
    const WavefrontFrame wavefront = frame.wavefront[0];
    const PathPoolRecord pool = wavefront.pool;
    PathState path = loadPath(pool, slot);
    const uint2 pixel = restirPixelCoordinate(path.pixel);
    const PathHit hit = loadHit(pool, slot);
    useRestirWavelengths(unpackHero(path.hero));
    const bool miss = hit.kind == hitKindMiss;
    if (!isDeadPath(path))
        crossRestirFog(frame, slot, path, miss ? floatMax : hit.t);
    if (miss && (isDeadPath(path) || path.depth() == 0)) {
        clearRestirPixel(path, pixel, invalidIndex);
        pushFree(wavefront, slot);
        return;
    }
    if (!miss && path.depth() == 0) {
        const RayCone cone = propagate(path.cone, hit.t);
        const Surface loaded = loadSurface(restirScene, hit.instance, hit.primitive, hit.barycentrics, path.direction, hit.t, cone);
        const PackedClosure closure = pool.closures[slot];
        if (skipTransparentHit(wavefront, parity, slot, path, loaded, cone, closure))
            return;
        if (isHoldout(closure, loaded.instanceFlags)) {
            clearRestirPixel(path, pixel, restirInstanceHoldout);
            pushFree(wavefront, slot);
            return;
        }
        beginInitialPath(wavefront, parity, slot, path, loaded, closure, cone, pixel, hit.instance);
        return;
    }
    uint randomIndex;
    RTXDI_PathTracerContextInitial context = unpackInitialContext(frame.paths[slot], path, miss ? restirDistantLightDistance : hit.t, randomIndex);
    RTXDI_RandomSamplerState random = restirInitialSampler(pixel, randomIndex);
    const bool updates = isRadianceCacheUpdatePixel(frame.cache, pixel, frame.runtime.frameIndex);
    if (miss) {
        splitRestirHitDistance(frame, path, floatMax);
        context.RecordPathRadianceMiss(random);
        const float4 environment = restirEnvironment(path.direction, loadLightPath(pool, slot)) * restirEnvironmentWeight(path);
        if (any(environment > 0.0))
            context.RecordEnvironmentMapLightSample(environment, RAB_EmptySurface(), random);
        if (updates)
            updateRadianceCacheEnd(frame.cache, restirWavelengths, slot, environment);
        finishInitialPath(context, pixel, random);
        pushFree(wavefront, slot);
        return;
    }
    const RayCone cone = propagate(path.cone, hit.t);
    const Surface loaded = loadSurface(restirScene, hit.instance, hit.primitive, hit.barycentrics, path.direction, hit.t, cone);
    const PackedClosure closure = pool.closures[slot];
    if (skipTransparentHit(wavefront, parity, slot, path, loaded, cone, closure))
        return;
    splitRestirHitDistance(frame, path, hit.t);
    const RAB_Surface surface = makeRestirSurface(loaded, cameraRayDistance(wavefront, path, slot, hit.t), closure);
    context.RecordPathIntersection(surface);
    bool continued = false;
    if (!context.IsPathTerminated()) {
        const BsdfInputs inputs = unpackClosure(closure);
        const PrincipledBsdf bsdf = makeRestirBsdf(inputs, surface);
        float4 emission = float4(0.0, 0.0, 0.0, 0.0);
        if (context.ShouldSampleEmissiveSurfaces())
            emission = emittedRadiance(inputs, restirWavelengths) * restirEmissionWeight(path, loaded, hit.t);
        const float3 cachePosition = loaded.context.position.xyz;
        const float3 cacheNormal = loaded.context.geometricNormal.xyz;
        float4 cached;
        if (mayEndInRadianceCache(frame.cache, context.GetBounceDepth(), path.hasFlag(pathFlagDelta))
            && queryRadianceCache(frame.cache, restirWavelengths, cachePosition, cacheNormal, cached)) {
            const float4 ending = emission + cached;
            if (any(ending > 0.0))
                context.RecordEmissiveLightSample(ending, RAB_EmptySurface(), random);
            if (updates)
                updateRadianceCacheEnd(frame.cache, restirWavelengths, slot, ending);
            finishInitialPath(context, pixel, random);
            pushFree(wavefront, slot);
            return;
        }
        if (any(emission > 0.0))
            context.RecordEmissiveLightSample(emission, RAB_EmptySurface(), random);
        float4 direct = float4(0.0, 0.0, 0.0, 0.0);
        if (context.ShouldSampleNee()) {
            //! RTXDI counts the primary surface as depth 1, and Cycles counts it as bounce 0.
            const uint bounce = context.GetBounceDepth() - 1u;
            RestirLightSample light;
            if (sampleRestirLight(surface, bsdf, bounce, random, light)) {
                context.RecordNeeLightSample(light.light, light.radiance, light.neePdf, light.scatterPdf, light.sample, random);
                direct += light.radiance;
            }
            if (sampleRestirBsdfLight(surface, bsdf, bounce, random, light)) {
                context.RecordNeeLightSample(light.light, light.radiance, light.neePdf, light.scatterPdf, light.sample, random);
                direct += light.radiance;
            }
        }
        if (updates)
            updateRadianceCacheHit(frame.cache, restirWavelengths, slot, cachePosition, cacheNormal, direct, emission,
                                   radianceCacheRandom(pixel, context.GetBounceDepth()));
        context.IncreaseBounceDepth();
        if (context.GetBounceDepth() <= context.GetMaxPathBounce())
            continued = continueInitialPath(wavefront, parity, slot, path, context, surface, bsdf, cone, pixel, random);
    }
    if (continued)
        return;
    finishInitialPath(context, pixel, random);
    pushFree(wavefront, slot);
}

//! A path that is still in flight after the last bounce ends with the samples it has; a camera ray that only passed transparent hits found no surface.
void finishInFlightInitialPath(RestirFrame frame, uint slot) {
    const PathState path = loadPath(frame.wavefront[0].pool, slot);
    const uint2 pixel = restirPixelCoordinate(path.pixel);
    if (path.depth() == 0) {
        clearRestirPixel(path, pixel, invalidIndex);
        return;
    }
    useRestirWavelengths(unpackHero(path.hero));
    uint randomIndex;
    const RTXDI_PathTracerContextInitial context = unpackInitialContext(frame.paths[slot], path, restirDistantLightDistance, randomIndex);
    RTXDI_RandomSamplerState random = restirInitialSampler(pixel, randomIndex);
    finishInitialPath(context, pixel, random);
}

void writeReplay(RestirFrame frame, uint index, RAB_Surface surface, float4 radiance, bool hasSurface) {
    if (!hasSurface && all(radiance == 0.0))
        return;
    RestirReplayRecord record;
    record.position = surface.position;
    record.normal = packOctahedral(surface.normal);
    record.radiance = radiance;
    const float4 bsdf = min(surface.bsdf, 65504.0);
    record.bsdf = uint2(packHalf2(bsdf.xy), packHalf2(bsdf.zw));
    const bool rough = surface.roughness > frame.pt.reconnection.roughnessThreshold;
    record.material = surface.material | restirLightSet(surface) << 24u | (rough ? restirReplayRough : 0u);
    record.bsdfPdf = surface.bsdfPdf;
    frame.replays[index] = record;
    frame.replayStates[index] = restirReplayDone | (hasSurface ? restirReplayHasSurface : 0u);
}

//! The BSDF value and pdf at the vertex before the reconnection vertex toward the target of the shift: the light sample or the reconnection vertex.
BsdfValue replayTargetBsdf(RAB_Surface surface, PrincipledBsdf bsdf, bool nee, float3 target) {
    float3 position = target;
    if (nee) {
        RTXDI_SampledLightData light;
        light.lightData = asuint(target.x);
        light.uvData = asuint(target.y);
        const RAB_LightInfo info = RAB_LoadLightInfo(RTXDI_SampledLightData_GetLightIndex(light), false);
        const RAB_LightSample sample = RAB_SamplePolymorphicLight(info, surface, RTXDI_SampledLightData_GetUVDataFloat2(light));
        if (sample.solidAnglePdf <= 0.0)
            return (BsdfValue)0;
        position = sample.position;
    }
    const float3 offset = position - surface.position;
    const float distance = length(offset);
    if (distance <= 0.0)
        return (BsdfValue)0;
    return bsdf.evaluate(offset / distance);
}

//! One wavefront step of a replay at its hit or miss, as RTXDI_PathTrace replays a hybrid shift; a failed shift writes nothing, read as zero.
void restirReplayPath(RestirFrame frame, uint parity, uint slot) {
    const WavefrontFrame wavefront = frame.wavefront[0];
    const PathPoolRecord pool = wavefront.pool;
    PathState path = loadPath(pool, slot);
    const PathHit hit = loadHit(pool, slot);
    useRestirWavelengths(asfloat(path.hero));
    const uint index = path.pixel;
    const uint bounceDepth = path.depth() + 1;
    const uint maxBounce = (path.flags >> replayMaxBounceShift) & 15u;
    const uint rcLength = (path.flags >> replayRcLengthShift) & 15u;
    const uint pathLength = (path.flags >> replayPathLengthShift) & 15u;
    const bool rough = path.hasFlag(replayFlagRough);
    const RTXDI_PTReconnectionParameters reconnection = frame.pt.reconnection;
    path.throughput *= restirTransmittance(path.origin, path.direction, hit.kind == hitKindMiss ? floatMax : hit.t);
    if (hit.kind == hitKindMiss) {
        if (bounceDepth == pathLength && !rough && bounceDepth != 2)
            writeReplay(frame, index, RAB_EmptySurface(), restirEnvironment(path.direction, loadLightPath(pool, slot)) * restirEnvironmentWeight(path) * path.throughput, false);
        return;
    }
    const RayCone cone = propagate(path.cone, hit.t);
    const Surface loaded = loadSurface(restirScene, hit.instance, hit.primitive, hit.barycentrics, path.direction, hit.t, cone);
    const PackedClosure closure = pool.closures[slot];
    if (skipTransparentHit(wavefront, parity, slot, path, loaded, cone, closure))
        return;
    RAB_Surface surface = makeRestirSurface(loaded, hit.t, closure);
    const BsdfInputs inputs = unpackClosure(closure);
    const PrincipledBsdf bsdf = makeRestirBsdf(inputs, surface);
    float4 cached = float4(0.0, 0.0, 0.0, 0.0);
    const bool cacheEnd = mayEndInRadianceCache(frame.cache, bounceDepth, path.hasFlag(pathFlagDelta))
                          && queryRadianceCache(frame.cache, restirWavelengths, loaded.context.position.xyz, loaded.context.geometricNormal.xyz, cached);
    //! The initial path went on from this vertex, so a shifted path that ends here in the cache is not its image.
    if (cacheEnd && bounceDepth < pathLength)
        return;
    const bool far = hit.t > reconnection.distanceThreshold && !path.hasFlag(pathFlagDelta);
    const bool roughHit = RAB_GetSurfaceRoughness(surface) > reconnection.roughnessThreshold;
    float4 radiance = float4(0.0, 0.0, 0.0, 0.0);
    if (bounceDepth == maxBounce) {
        if ((rough || rcLength > pathLength) && far && (rcLength > pathLength || (rough && roughHit)))
            return;
        if (rcLength <= pathLength) {
            const BsdfValue target = replayTargetBsdf(surface, bsdf, path.hasFlag(replayFlagNee), path.radiance);
            surface.bsdf = target.value;
            surface.bsdfPdf = target.pdf;
            writeReplay(frame, index, surface, path.throughput, true);
            return;
        }
        if (bounceDepth == pathLength) {
            const float4 emission = emittedRadiance(inputs, restirWavelengths);
            radiance = (emission * restirEmissionWeight(path, loaded, hit.t) + cached) * path.throughput;
        }
    }
    if (bounceDepth + 1 > maxBounce) {
        writeReplay(frame, index, RAB_EmptySurface(), radiance, false);
        return;
    }
    RTXDI_RandomSamplerState random = restirBsdfRandom(path.sampleIndex, asuint(path.diffuseFraction), bounceDepth + 1);
    const BsdfSample sample = bsdf.sample(sampleRestirRandom(random));
    if (sample.pdf <= 0.0 || !isValidThroughput(sample.weight))
        return;
    if (rough && far && roughHit)
        return;
    const bool delta = (sample.flags & bsdfFlagDelta) != 0;
    path.throughput *= sample.weight;
    path.origin = offsetAlong(surface.position, surface.normal, sample.direction);
    path.setReceiver(restirReceiver(surface));
    path.direction = sample.direction;
    path.previousPdf = delta ? 0.0 : sample.pdf;
    path.setFlag(pathFlagDelta, delta);
    path.setFlag(replayFlagRough, roughHit);
    path.cone = restirContinuationCone(cone, delta, bsdf.roughness());
    recordLightPath(pool, slot, bsdfLightPathEvent(sample.flags));
    path.advanceDepth();
    storeRay(pool, slot, path);
    pool.throughput[slot] = path.throughput;
    pool.state[slot] = uint4(path.pixel, path.sampleIndex, path.flags, path.hero);
    pushActive(wavefront, parity, slot);
}

RTXDI_PTReservoir resampleTemporal(RestirFrame frame, uint2 pixel) {
    const RTXDI_PTParameters pt = frame.pt;
    RTXDI_PTTemporalResamplingRuntimeParameters runtime = RTXDI_EmptyPTTemporalResamplingRuntimeParameters();
    runtime.pixelPosition = pixel;
    runtime.reservoirPosition = pixel;
    runtime.motionVector = restirMotion(frame, RAB_GetGBufferSurface(int2(pixel), false), pixel);
    runtime.cameraPos = cameraPosition(restirCamera);
    runtime.prevCameraPos = previousCameraPosition(restirCamera);
    runtime.prevPrevCameraPos = runtime.prevCameraPos;
    bool selectedPrevious = false;
    RAB_PathTracerUserData data = RAB_EmptyPathTracerUserData();
    return RTXDI_PTTemporalResampling(pt.temporalResampling, runtime, pt.hybridShift, pt.reconnection, frame.runtime, pt.reservoirBuffer,
                                      RTXDI_InitRandomSampler(pixel, frame.runtime.frameIndex, restirTemporalPass), pt.bufferIndices, selectedPrevious, data);
}

RTXDI_PTReservoir resampleSpatial(RestirFrame frame, uint2 pixel) {
    const RTXDI_PTParameters pt = frame.pt;
    RTXDI_PTSpatialResamplingRuntimeParameters runtime = RTXDI_EmptyPTSpatialResamplingRuntimeParameters();
    runtime.pixelPosition = pixel;
    runtime.reservoirPosition = pixel;
    runtime.cameraPos = cameraPosition(restirCamera);
    runtime.prevCameraPos = previousCameraPosition(restirCamera);
    runtime.prevPrevCameraPos = runtime.prevCameraPos;
    runtime.viewportSize = uint2(frame.width, frame.height);
    bool resampled = false;
    RAB_PathTracerUserData data = RAB_EmptyPathTracerUserData();
    return RTXDI_PTSpatialResampling(runtime, pt.spatialResampling, pt.hybridShift, pt.reconnection, pt.reservoirBuffer, pt.bufferIndices, frame.runtime,
                                     RTXDI_InitRandomSampler(pixel, frame.runtime.frameIndex, restirSpatialPass), resampled, data);
}

//! One stage of the resampling of a pixel: it stores the reservoir, or it waits in the next pending list for the replays that it started.
void resamplePixel(RestirFrame frame, RestirArguments arguments, uint index) {
    const uint local = arguments.stage == 0 ? index : pendingList(frame, (arguments.stage + 2) % restirStages)[index];
    const uint2 pixel = restirPixel(frame, arguments.start + local);
    beginReplays(local, arguments.forward, arguments.stage, arguments.parity);
    const bool temporal = arguments.mode == restirModeTemporal;
    const RTXDI_PTReservoir reservoir = temporal ? resampleTemporal(frame, pixel) : resampleSpatial(frame, pixel);
    if (endReplays()) {
        const uint list = arguments.stage % restirStages;
        pendingList(frame, list)[atomicAdd(frame.counters + list, 1u)] = local;
        return;
    }
    const RTXDI_PTParameters pt = frame.pt;
    RTXDI_StorePTReservoir(reservoir, pt.reservoirBuffer, pixel,
                           temporal ? pt.bufferIndices.temporalResamplingOutputBufferIndex : pt.bufferIndices.spatialResamplingOutputBufferIndex);
}

//! The guides of a pixel without a surface: in fog, the guides of the middle of the fog, so the denoiser filters the fog light of the pixel.
void writeRestirFogGuides(FilmRecord film, uint pixelIndex, RAB_Surface surface) {
    film.objectIds[pixelIndex] = invalidIndex;
    const MediumRange range = hasVolumes && restirScene.counts.w != 0 ? mediumRange(restirScene, cameraPosition(restirCamera), -surface.view, floatMax)
                                                                       : (MediumRange)0;
    if (!hasMedium(range)) {
        film.albedo[pixelIndex] = float4(0.0, 0.0, 0.0, 1.0);
        film.normalDepth[pixelIndex] = float4(surface.view, 0.0);
        return;
    }
    film.albedo[pixelIndex] = float4(1.0, 1.0, 1.0, 1.0);
    film.normalDepth[pixelIndex] = float4(surface.view, 0.5 * (range.tStart + range.tEnd));
    if (film.splitsLighting == 0)
        return;
    film.demodulation[pixelIndex * 2] = float4(1.0, 1.0, 1.0, 1.0);
    film.demodulation[pixelIndex * 2 + 1] = float4(0.0, 0.0, 0.0, 0.0);
}

//! The nearest light that the camera sees before tMax.
float4 restirCameraLight(float3 direction, float tMax) {
    if ((sceneFlags(restirScene) & sceneFlagCameraLights) == 0)
        return float4(0.0, 0.0, 0.0, 0.0);
    const LightHit hit = intersectLights(restirScene, 0u, cameraPosition(restirCamera), direction, tMax);
    if (hit.light == invalidIndex || (restirScene.lights[hit.light].linking.z & lightFlagCamera) == 0)
        return float4(0.0, 0.0, 0.0, 0.0);
    return restirWavelengths.upsample(hit.radiance);
}

//! Returns the light that the camera sees at the primary vertex, and writes the primary AOVs.
float3 shadeRestirSurface(FilmRecord film, uint pixelIndex, RestirSurfaceRecord record, uint index, out float diffuseShare) {
    const RAB_Surface surface = unpackRestirSurface(record, index);
    diffuseShare = 1.0;
    const float4 transmittance = restirCameraTransmittance(surface);
    if (!RAB_IsSurfaceValid(surface)) {
        writeRestirFogGuides(film, pixelIndex, surface);
        if (record.instance == restirInstanceHoldout)
            return float3(0.0, 0.0, 0.0);
        return restirWavelengths.toFilm(restirWavelengths.upsample(backgroundRadiance(restirScene, -surface.view)) * transmittance + restirCameraLight(-surface.view, floatMax));
    }
    const BsdfInputs inputs = unpackClosure(restirFrame.closures[index]);
    const PrincipledBsdf bsdf = makeRestirBsdf(inputs, surface);
    const float3 albedo = saturate(restirWavelengths.toFilm(bsdf.albedo()));
    const float3 diffuseAlbedo = saturate(restirWavelengths.toFilm(bsdf.diffuseAlbedo()));
    const float total = albedo.x + albedo.y + albedo.z;
    diffuseShare = total > 0.0 ? saturate((diffuseAlbedo.x + diffuseAlbedo.y + diffuseAlbedo.z) / total) : 1.0;
    film.albedo[pixelIndex] = float4(albedo, 1.0);
    film.normalDepth[pixelIndex] = float4(safeNormalize(inputs.normalRotation.xyz), surface.depth);
    film.objectIds[pixelIndex] = record.instance;
    if (film.splitsLighting != 0) {
        const DemodulationAlbedo demodulation =
            demodulationAlbedo(inputs, surface.view, surface.normal, (surface.flags & restirSurfaceBackfacing) != 0, sceneEnergyLut(restirScene));
        film.demodulation[pixelIndex * 2] = float4(demodulation.diffuse, bsdf.roughness());
        film.demodulation[pixelIndex * 2 + 1] = float4(demodulation.specularGuide(film.approximatesSpecular != 0), 0.0);
    }
    return restirWavelengths.toFilm(emittedRadiance(inputs, restirWavelengths) * transmittance + restirCameraLight(-surface.view, surface.depth));
}

//! The reservoir knows the direction that its path left the primary vertex in only when it reconnects at the next vertex; else the albedo ratio, no direction.
float reconnectionDiffuseShare(RAB_Surface surface, RTXDI_PTReservoir reservoir, float albedoShare, out float3 direction) {
    direction = float3(0.0, 0.0, 0.0);
    if (reservoir.rcVertexLength != 2u)
        return albedoShare;
    float3 position = reservoir.translatedWorldPosition;
    if (RTXDI_ConnectsToNeeLight(reservoir)) {
        const RTXDI_SampledLightData light = RTXDI_GetSampledLightData(reservoir);
        const RAB_LightInfo info = RAB_LoadLightInfo(RTXDI_SampledLightData_GetLightIndex(light), false);
        const RAB_LightSample sample = RAB_SamplePolymorphicLight(info, surface, RTXDI_SampledLightData_GetUVDataFloat2(light));
        if (sample.solidAnglePdf <= 0.0)
            return albedoShare;
        position = sample.position;
    }
    direction = safeNormalize(position - surface.position);
    float diffuseFraction;
    restirBsdf(surface).evaluate(direction, diffuseFraction);
    return diffuseFraction;
}

void shadeRestir(RestirFrame frame, uint index) {
    const uint2 pixel = restirPixel(frame, index);
    const RTXDI_PTParameters pt = frame.pt;
    RTXDI_RandomSamplerState random = RTXDI_InitRandomSampler(pixel, frame.runtime.frameIndex, restirShadePass);
    useRestirWavelengths(RTXDI_GetNextRandom(random));
    const uint pixelIndex = restirPixelIndex(pixel);
    const uint surfaceIndex = restirSurfaceIndex(pixelIndex, false);
    const FilmRecord film = frame.wavefront[0].film;
    float diffuseShare;
    const float3 primary = max(sanitize(shadeRestirSurface(film, pixelIndex, frame.surfaces[surfaceIndex], surfaceIndex, diffuseShare)), 0.0);
    float3 scattered = float3(0.0, 0.0, 0.0);
    float3 direction = float3(0.0, 0.0, 0.0);
    const RTXDI_PTReservoir reservoir = RTXDI_LoadPTReservoir(pt.reservoirBuffer, pixel, pt.bufferIndices.finalShadingInputBufferIndex);
    if (RTXDI_IsValidPTReservoir(reservoir)) {
        useRestirWavelengths(reservoir.wavelengthSample);
        const RAB_Surface surface = unpackRestirSurface(frame.surfaces[surfaceIndex], surfaceIndex);
        scattered = max(sanitize(restirWavelengths.toFilm(reservoir.targetFunction * reservoir.weightSum * restirCameraTransmittance(surface))), 0.0);
        diffuseShare = reconnectionDiffuseShare(surface, reservoir, diffuseShare, direction);
    }
    splatRadiance(film, pixelIndex, primary + scattered);
    if (restirScene.sceneCenter.w == environmentTransparent || frame.surfaces[surfaceIndex].instance == restirInstanceHoldout) {
        const RAB_Surface surface = unpackRestirSurface(frame.surfaces[surfaceIndex], surfaceIndex);
        if (!RAB_IsSurfaceValid(surface) && RTXDI_GetNextRandom(random) < dot(restirCameraTransmittance(surface), float4(0.25, 0.25, 0.25, 0.25)))
            countTransparentSample(film, pixelIndex);
    }
    atomicAdd(film.frame + pixelIndex * 4 + 3, 1.0);
    splitPrimaryLight(film, pixelIndex, primary);
    splitPathLight(film, pixelIndex, scattered, diffuseShare, direction);
}
