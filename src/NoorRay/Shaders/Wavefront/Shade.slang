import NoorRHI;
import Lights.SgLighting;
import Wavefront.Records;
import Wavefront.PathState;
import Wavefront.Queues;
import Wavefront.Direct;
import Wavefront.SurfaceShading;
import Wavefront.Subsurface;
import Wavefront.VisibilityCache;
import Wavefront.RadianceCache;
import Common.Math;
import Common.LinkConstants;
import Common.Random;
import Common.RayCone;
import Common.Sampling;
import Lights.Lights;
import Bsdf.EnergyLut;
import Bsdf.Principled;
import Materials.Closure;
import Scene.Camera;
import Scene.SceneSurface;
import Spectral.Wavelengths;

//! Cycles' Filter Glossy: after scatters of small pdf, the glossy lobes take at least the alpha sqrt(1 - pdf * filter_glossy) / 2.
BsdfInputs blurGlossy(WavefrontFrame frame, BsdfInputs inputs, float minRayPdf) {
    const float blur = frame.filter.z * minRayPdf;
    if (frame.filter.z <= 0.0 || blur >= 1.0)
        return inputs;
    const float roughness = sqrt(sqrt(1.0 - blur) * 0.5);
    inputs.metallicRoughnessIorSpecular.y = max(inputs.metallicRoughnessIorSpecular.y, roughness);
    inputs.coat.y = max(inputs.coat.y, roughness);
    return inputs;
}

//! The primary vertex of a pixel for ReSTIR DI, which lights it after the paths end; a zero throughput keeps the surface for the neighbors only.
void writeDirectVertex(WavefrontFrame frame, PathState path, Surface surface, PackedClosure closure, uint instance, float4 throughput, uint flags) {
    const DirectLightingRecord direct = frame.direct;
    const uint index = direct.base + path.pixel;
    const float3 position = surface.context.position.xyz;
    RestirSurfaceRecord record;
    record.position = position;
    record.depth = length(position - cameraPosition(frame.camera[0]));
    record.normal = packOctahedral(surface.context.geometricNormal.xyz);
    record.view = packOctahedral(surface.context.incoming.xyz);
    record.material = surface.material | (isBackfacing(surface.context) ? 0x80000000u : 0u);
    record.instance = instance;
    direct.surfaces[index] = record;
    direct.closures[index] = closure;
    DirectVertex vertex;
    vertex.throughput = throughput;
    vertex.receiver = path.receiver;
    vertex.hero = path.hero;
    vertex.flags = flags;
    vertex.frame = direct.frame;
    vertex.unused = 0;
    direct.vertices[path.pixel] = vertex;
}

//! resamplesDirect hands the light sample of the primary vertex of each pixel to ReSTIR DI.
void shadeSurfacePath<let resamplesDirect : bool>(WavefrontFrame frame, uint parity, uint slot) {
    const SceneRecord scene = frame.scene[0];
    PathState path = loadPath(frame.pool, slot);
    const PathHit hit = loadHit(frame.pool, slot);
    Wavelengths wavelengths = pathWavelengths(scene, path);
    const RayCone cone = propagate(path.cone, hit.t);
    const Surface surface = loadSurface(scene, hit.instance, hit.primitive, hit.barycentrics, path.direction, hit.t, cone);
    const ShadingContext context = surface.context;
    const float3 position = context.position.xyz;
    const PackedClosure closure = frame.pool.closures[slot];
    const float3 view = -path.direction;

    if (hasAlphaTest && path.hasFlag(pathFlagAlphaCandidate)) {
        path.setFlag(pathFlagAlphaCandidate, false);
        if (!passesOpacity(frame, path, closureOpacity(closure), opacitySalt(frame.pool, slot))) {
            recordTransparentBounce(frame.pool, slot);
            path.origin = offsetAlong(position, surface.worldNormal, path.direction);
            path.cone = cone;
            storePath(frame.pool, slot, path);
            pushActive(frame, parity, slot);
            return;
        }
    }

    const bool updates = path.hasFlag(pathFlagCacheUpdate);
    if (updates)
        applyRadianceCachePending(frame.cache, slot);
    if (isHoldout(closure, surface.instanceFlags)) {
        if (path.depth() == 0 && !path.hasFlag(pathFlagFork))
            countTransparentSample(frame.film, path.pixel);
        writePrimaryAovs(frame.film, path, float3(0.0, 0.0, 0.0), view, cameraRayDistance(frame, path, slot, hit.t), hit.instance);
        if (path.hasFlag(pathFlagCacheUpdate))
            stopRadianceCacheUpdate(frame.cache, slot);
        finishPath(frame.film, path);
        storePath(frame.pool, slot, path);
        pushFree(frame, slot);
        return;
    }
    const BsdfInputs inputs = blurGlossy(frame, unpackClosure(closure), frame.pool.minRayPdf[slot]);
    const float4 emission = emittedRadiance(inputs, wavelengths);
    float4 emitted = float4(0.0, 0.0, 0.0, 0.0);
    //! Light linking: an emitter that does not reach the set of the previous vertex adds nothing there, but the camera sees it.
    if (any(emission > 0.0) && (path.depth() == 0 || reachesSet(scene, surface.light, path.lightSet()))) {
        float weight = 1.0;
        if (!path.hasFlag(pathFlagDelta) && path.previousPdf > 0.0 && surface.light != invalidIndex) {
            const float pdf = lightPdf(scene, path.lightSet(), surface.light, path.origin, path.lightReceiver(frame.visibility.cells), path.direction, hit.t,
                                       dot(surface.worldNormal, path.direction));
            weight = powerHeuristic(path.previousPdf, pdf);
        }
        emitted = emission * weight;
        addRadiance(frame, path, wavelengths, path.throughput * emitted);
    }
    flushPrimaryLight(frame.film, path);
    if (path.hasFlag(pathFlagFinalRay) || reachedKindBounces(frame, loadLightPath(frame.pool, slot))) {
        if (updates)
            updateRadianceCacheEnd(frame.cache, wavelengths, slot, emitted);
        finishPath(frame.film, path);
        storePath(frame.pool, slot, path);
        pushFree(frame, slot);
        return;
    }
    //! The cache holds the light that leaves the vertex without its emission, so the path ends there.
    float4 cached;
    if (mayEndInRadianceCache(frame.cache, path.depth() + 1, path.hasFlag(pathFlagDelta))
        && queryRadianceCache(frame.cache, wavelengths, position, context.geometricNormal.xyz, cached)) {
        addRadiance(frame, path, wavelengths, path.throughput * cached);
        if (updates)
            updateRadianceCacheEnd(frame.cache, wavelengths, slot, emitted + cached);
        finishPath(frame.film, path);
        storePath(frame.pool, slot, path);
        pushFree(frame, slot);
        return;
    }
    //! Shadow adds the NEE of the vertex when its ray arrives.
    if (updates)
        updateRadianceCacheHit(frame.cache, wavelengths, slot, position, context.geometricNormal.xyz, float4(0.0, 0.0, 0.0, 0.0), emitted, 0.5);

    const bool backfacing = isBackfacing(context);
    SurfaceScatter scatter;
    scatter.bsdf = makePrincipled(inputs, view, context.geometricNormal.xyz, backfacing, wavelengths, sceneEnergyLut(scene),
                                    hasDispersion && !path.hasFlag(pathFlagSecondaryTerminated));
    const float3 albedo = saturate(wavelengths.toFilm(scatter.bsdf.albedo()));
    writePrimaryAovs(frame.film, path, albedo, safeNormalize(inputs.normalRotation.xyz), cameraRayDistance(frame, path, slot, hit.t), hit.instance);
    if (frame.film.splitsLighting != 0 && path.depth() == 0 && path.hasFlag(pathFlagPrimarySample)) {
        const DemodulationAlbedo demodulation = demodulationAlbedo(inputs, view, context.geometricNormal.xyz, backfacing, sceneEnergyLut(scene));
        writeDemodulation(frame.film, path, demodulation.diffuse, demodulation.specularGuide(frame.film.approximatesSpecular != 0), scatter.bsdf.roughness());
    }

    path.setLightSet(surface.instanceFlags >> lightSetShift);
    const float4 lightU = bouncePattern(frame, path, 0);
    const float4 bsdfU = bouncePattern(frame, path, 1);
    const float3 shadowOrigin = spawnOrigin(surface, view);
    //! As Cycles' Christensen-Burley, subsurface scattering after a diffuse bounce stays diffuse.
    float subsurfaceProbability = 0.0;
    float4 subsurfaceWeights = float4(0.0, 0.0, 0.0, 0.0);
    float4 radius = float4(0.0, 0.0, 0.0, 0.0);
    if (hasSubsurface && inputs.subsurface.w > 0.0 && lightPathCount(loadLightPath(frame.pool, slot), lightPathDiffuseShift) == 0) {
        radius = subsurfaceRadius(inputs.subsurface.xyz, wavelengths);
        const float4 weights = select(radius > 0.0, float4(1.0, 1.0, 1.0, 1.0) * inputs.subsurface.w, float4(0.0, 0.0, 0.0, 0.0));
        subsurfaceWeights = scatter.bsdf.splitSubsurface(weights, subsurfaceProbability);
    }
    //! A slot has one shadow ray, so a subsurface path lights its entry or its exit by a coin flip; each light sample scales by the inverse of its chance.
    LightReceiver receiver = depthReceiver(frame, path, scatter.bsdf.lightReceiver(view));
    receiver.cell = visibilityCell(frame, position, receiver.normal, path.lightSet());
    path.setReceiver(receiver);
    PathState lit = path;
    lit.throughput /= 1.0 - 0.5 * subsurfaceProbability;
    //! A cache update pixel lights its primary vertex by NEE, so the cache learns that light too.
    const bool direct = resamplesDirect && path.depth() == 0 && path.hasFlag(pathFlagPrimarySample) && !updates;
    const bool dispersive = hasDispersion && !path.hasFlag(pathFlagSecondaryTerminated);
    const uint directFlags = (isLastVertex(frame, path) ? directVertexLast : 0u) | (path.hasFlag(pathFlagSecondaryTerminated) ? directVertexSecondaryTerminated : 0u)
                             | (path.hasFlag(pathFlagRgb) ? directVertexRgb : 0u) | (dispersive ? directVertexDispersion : 0u);
    if (bsdfU.z < subsurfaceProbability) {
        if (updates) {
            stopRadianceCacheUpdate(frame.cache, slot);
            path.setFlag(pathFlagCacheUpdate, false);
            lit.setFlag(pathFlagCacheUpdate, false);
        }
        if (direct)
            writeDirectVertex(frame, path, surface, closure, hit.instance, float4(0.0, 0.0, 0.0, 0.0), directFlags);
        const float4 diskU = subsurfacePattern(frame, path);
        if (diskU.w < 0.5)
            queueDirectLight(frame, parity, slot, scene, lit, wavelengths, scatter, shadowOrigin, lightU);
        const SubsurfaceExit exit =
            probeSubsurface(scene, hit.instance, position, context.geometricNormal.xyz, backfacing, radius, diskU.xyz, raySeed(frame, path, 5));
        path.throughput *= subsurfaceWeights / subsurfaceProbability * exit.weight;
        LambertScatter lambert;
        lambert.normal = exit.normal;
        lambert.geometricNormal = exit.geometricNormal;
        const BsdfSample exitSample = lambert.sampleScatter(bsdfU.xy);
        if (!exit.found || exitSample.pdf <= 0.0) {
            finishPath(frame.film, path);
            storePath(frame.pool, slot, path);
            pushFree(frame, slot);
            return;
        }
        const float3 exitOrigin = offsetRayOrigin(exit.position, exit.geometricNormal);
        if (diskU.w >= 0.5) {
            lit.throughput = path.throughput * 2.0;
            lit.setReceiver(depthReceiver(frame, path, diffuseReceiver(exit.geometricNormal, exit.geometricNormal)));
            queueDirectLight(frame, parity, slot, scene, lit, wavelengths, lambert, exitOrigin, lightU);
        }
        if (path.depth() == 0) {
            path.diffuseFraction = 1.0;
            path.primaryDirection = exitSample.direction;
        }
        path.setFlag(pathFlagDelta, false);
        path.previousPdf = exitSample.pdf;
        recordScatterPdf(frame.pool, slot, exitSample.pdf);
        path.origin = exitOrigin;
        path.setReceiver(depthReceiver(frame, path, diffuseReceiver(exit.geometricNormal, exit.geometricNormal)));
        path.direction = exitSample.direction;
        path.cone = widenRayCone(cone, 1.0);
        recordLightPath(frame.pool, slot, lightPathDiffuse);
        continueOrFinish(frame, parity, slot, path, bsdfU.w);
        return;
    }
    if (direct)
        writeDirectVertex(frame, path, surface, closure, hit.instance, lit.throughput, directFlags);
    else
        queueDirectLight(frame, parity, slot, scene, lit, wavelengths, scatter, shadowOrigin, lightU);

    float diffuseFraction;
    const BsdfSample sample = scatter.bsdf.sample(float3(bsdfU.xy, bsdfU.z - subsurfaceProbability), diffuseFraction);
    if (sample.pdf <= 0.0 || all(sample.weight <= 0.0)) {
        finishPath(frame.film, path);
        storePath(frame.pool, slot, path);
        pushFree(frame, slot);
        return;
    }
    path.throughput *= sample.weight;
    if (updates)
        deferRadianceCacheThroughput(frame.cache, wavelengths, slot, sample.weight);
    if (path.depth() == 0) {
        path.diffuseFraction = diffuseFraction;
        path.primaryDirection = sample.direction;
    }
    if ((sample.flags & bsdfFlagDispersive) != 0)
        path.setFlag(pathFlagSecondaryTerminated, true);
    const bool delta = (sample.flags & bsdfFlagDelta) != 0;
    path.setFlag(pathFlagDelta, delta);
    path.previousPdf = delta ? 0.0 : sample.pdf;
    if (!delta)
        recordScatterPdf(frame.pool, slot, sample.pdf);
    path.origin = spawnOrigin(surface, sample.direction);
    path.direction = sample.direction;
    const float roughness = scatter.bsdf.roughness();
    path.cone = delta ? cone : widenRayCone(cone, roughness * roughness);
    recordLightPath(frame.pool, slot, bsdfLightPathEvent(sample.flags));
    continueOrFinish(frame, parity, slot, path, bsdfU.w);
}

//! Resolves the alpha occluder that Shadow found: an opaque one drops the shadow ray, a clear one moves it past the hit for the next pass.
void resolveShadowAlphaPath(WavefrontFrame frame, uint parity, uint slot) {
    const SceneRecord scene = frame.scene[0];
    PathState path = loadPath(frame.pool, slot);
    uint4 state = frame.pool.state[slot];
    state.z &= ~pathFlagShadowAlpha;
    frame.pool.state[slot] = state;
    const PathHit hit = loadHit(frame.pool, slot);
    const float4 origin = frame.pool.shadowOrigin[slot];
    const float3 direction = frame.pool.shadowDirection[slot].xyz;
    if (passesOpacity(frame, path, closureOpacity(frame.pool.closures[slot]), 3 + hit.primitive))
        return;
    const Surface surface = loadSurface(scene, hit.instance, hit.primitive, hit.barycentrics, direction, hit.t, makeRayCone(0.0, 0.0));
    const float3 moved = offsetAlong(surface.context.position.xyz, surface.worldNormal, direction);
    const float remaining = origin.w >= floatMax ? floatMax : max(origin.w - length(moved - origin.xyz), 0.0);
    frame.pool.shadowOrigin[slot] = float4(moved, remaining);
    pushShadow(frame, parity, slot);
}
