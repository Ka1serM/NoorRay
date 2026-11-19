import NoorRHI;
import Wavefront.Records;
import Wavefront.PathState;
import Wavefront.Queues;
import Wavefront.Media;
import Common.Math;
import Common.LinkConstants;
import Common.Sampling;
import Lights.Lights;
import Lights.Environment;
import Scene.Gaussians;
import Scene.SceneSurface;
import Spectral.Wavelengths;
import Raster.VisibilityBuffer;

//! Accepts opaque-material triangles of non-opaque meshes and keeps the nearest alpha-tested candidate for its shade kernel.
struct SurfaceFilter : IHitFilter {
    SceneRecord scene;
    float alphaT;
    uint alphaInstance;
    uint alphaPrimitive;
    float2 alphaBarycentrics;

    [mutating]
    HitVerdict filter(RayCandidate candidate) {
        const uint material = triangleMaterial(scene, candidate.instanceId, candidate.primitiveIndex);
        if (!hasAlphaTest || (scene.materials[material].flags & materialFlagAlphaTest) == 0)
            return HitVerdict.Accept;
        if (candidate.t < alphaT) {
            alphaT = candidate.t;
            alphaInstance = candidate.instanceId;
            alphaPrimitive = candidate.primitiveIndex;
            alphaBarycentrics = candidate.barycentrics;
        }
        return HitVerdict.Ignore;
    }
};

//! Tests Gaussians stochastically and keeps the nearest accepted peak. An accepted Gaussian commits its proxy triangle, so traversal skips every proxy
//! behind it.
struct GaussianFilter : IHitFilter {
    SceneRecord scene;
    uint seed;
    float t;
    uint cloud;
    uint gaussian;

    [mutating]
    HitVerdict filter(RayCandidate candidate) {
        const GaussianCandidate tested = testGaussian(scene, candidate, 0.0, t, seed);
        if (!tested.accepted)
            return HitVerdict.Ignore;
        t = tested.t;
        cloud = candidate.instanceId;
        gaussian = tested.gaussian;
        return HitVerdict.Accept;
    }
};

bool isAlphaTested(SceneRecord scene, uint instance, uint primitive) {
    return (scene.materials[triangleMaterial(scene, instance, primitive)].flags & materialFlagAlphaTest) != 0;
}

uint surfaceMask(PathState path) { return path.depth() == 0 ? instanceMaskCamera : instanceMaskIndirect; }

float misWeight(PathState path, float lightPdf) {
    if (path.hasFlag(pathFlagDelta) || path.previousPdf <= 0.0)
        return 1.0;
    return powerHeuristic(path.previousPdf, lightPdf);
}

//! Emission of analytic area lights that the ray passes before tMax. Lights do not occlude, the camera does not see them, and volumes do not attenuate them.
float4 lightsAlongRay(SceneRecord scene, PathState path, Wavelengths wavelengths, float tMax) {
    float4 total = float4(0.0, 0.0, 0.0, 0.0);
    if (path.depth() == 0)
        return total;
    float start = 0.0;
    for (uint step = 0; step < 4; ++step) {
        const LightHit hit = intersectLights(scene, path.origin + path.direction * start, path.direction, tMax - start);
        if (hit.light == invalidIndex)
            break;
        const float distance = start + hit.distance;
        const float pdf = lightPdf(scene, hit.light, path.origin, path.direction, distance, hit.emitterCosine);
        total += wavelengths.upsample(hit.radiance) * misWeight(path, pdf);
        start = distance * (1.0 + 1e-5) + 1e-5;
    }
    return total;
}

void extendPath(WavefrontFrame frame, uint parity, uint index) {
    const uint slot = activeQueue(frame.queues, parity)[index];
    PathState path = loadPath(frame.pool, slot);
    if (isDeadPath(path)) {
        frame.pool.pathBin[slot] = invalidIndex;
        pushFree(frame, slot);
        return;
    }
    const SceneRecord scene = frame.scene[0];
    const Wavelengths wavelengths = pathWavelengths(scene, path);

    RayDesc ray;
    ray.Origin = path.origin;
    ray.Direction = path.direction;
    ray.TMin = 0.0;
    ray.TMax = floatMax;
    SurfaceFilter filter;
    filter.scene = scene;
    filter.alphaT = floatMax;
    filter.alphaInstance = invalidIndex;
    filter.alphaPrimitive = 0;
    filter.alphaBarycentrics = float2(0.0, 0.0);
    RayHit meshHit = (RayHit)0;
    RasterCoverage coverage = RasterCoverage.NeedsTrace;
    if (path.hasFlag(pathFlagRasterPrimary)) {
        path.setFlag(pathFlagRasterPrimary, false);
        const uint2 pixel = uint2(path.pixel % frame.film.width, path.pixel / frame.film.width);
        coverage = loadRasterHit(frame.raster, scene, pixel, path.origin, path.direction, meshHit);
    }
    const bool found = coverage == RasterCoverage.NeedsTrace ? traceRayInline(scene.tlas, RAY_FLAG_NONE, surfaceMask(path), ray, filter, meshHit)
                                                             : coverage == RasterCoverage.Mesh;
    //! Gaussians trace after the surfaces, bounded by them, with back faces culled so each proxy is tested once, where the ray enters it.
    GaussianFilter gaussians;
    gaussians.scene = scene;
    gaussians.seed = raySeed(frame, path, 0);
    gaussians.t = floatMax;
    gaussians.cloud = invalidIndex;
    gaussians.gaussian = 0;
    if (hasGaussians) {
        ray.TMax = min(found ? meshHit.t : floatMax, filter.alphaT);
        RayHit gaussianHit;
        traceRayInline(scene.gaussianTlas, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xffu, ray, gaussians, gaussianHit);
    }

    PathHit hit;
    hit.t = found ? meshHit.t : floatMax;
    hit.kind = found ? hitKindMesh : hitKindMiss;
    hit.instance = found ? meshHit.instanceId : 0;
    hit.primitive = found ? meshHit.primitiveIndex : 0;
    hit.barycentrics = found ? meshHit.barycentrics : float2(0.0, 0.0);
    path.setFlag(pathFlagAlphaCandidate, false);
    if (hasAlphaTest && filter.alphaT < hit.t) {
        hit.t = filter.alphaT;
        hit.kind = hitKindMesh;
        hit.instance = filter.alphaInstance;
        hit.primitive = filter.alphaPrimitive;
        hit.barycentrics = filter.alphaBarycentrics;
        path.setFlag(pathFlagAlphaCandidate, true);
    }
    if (hasAlphaTest && coverage == RasterCoverage.Mesh && isAlphaTested(scene, hit.instance, hit.primitive))
        path.setFlag(pathFlagAlphaCandidate, true);
    if (hasGaussians && gaussians.t < hit.t) {
        hit.t = gaussians.t;
        hit.kind = hitKindGaussian;
        hit.instance = gaussians.cloud;
        hit.primitive = gaussians.gaussian;
        hit.barycentrics = float2(0.0, 0.0);
        path.setFlag(pathFlagAlphaCandidate, false);
    }

    addRadiance(frame, path, wavelengths, path.throughput * lightsAlongRay(scene, path, wavelengths, hit.t));

    if (hasVolumes && scene.counts.w != 0) {
        float4 absorbed;
        const MediumEvent event = trackMedium(scene, path.origin, path.direction, hit.t, wavelengths, raySeed(frame, path, 1), path.throughput, absorbed);
        addRadiance(frame, path, wavelengths, absorbed);
        if (event.kind == mediumEventAbsorb) {
            finishPath(frame.film, path);
            storePath(frame.pool, slot, path);
            frame.pool.pathBin[slot] = invalidIndex;
            pushFree(frame, slot);
            return;
        }
        if (event.kind == mediumEventScatter) {
            hit.t = event.t;
            hit.kind = hitKindVolume;
            hit.instance = event.volume;
            hit.primitive = 0;
            path.setFlag(pathFlagAlphaCandidate, false);
        }
    }

    if (hit.kind == hitKindMiss) {
        const float environment = environmentProbability(scene);
        const float3 environmentSeen = path.depth() == 0 ? backgroundRadiance(scene, path.direction) : environmentRadiance(scene, path.direction);
        float4 background = wavelengths.upsample(environmentSeen) *
                            misWeight(path, environment * environmentPdf(scene, path.direction));
        uint directional;
        const float3 sun = directionalRadiance(scene, path.direction, directional);
        if (directional != invalidIndex && path.depth() > 0)
            background += wavelengths.upsample(sun) * misWeight(path, lightPdf(scene, directional, path.origin, path.direction, floatMax, 1.0));
        addRadiance(frame, path, wavelengths, path.throughput * background);
        finishPath(frame.film, path);
        storePath(frame.pool, slot, path);
        frame.pool.pathBin[slot] = invalidIndex;
        pushFree(frame, slot);
        return;
    }

    storeHit(frame.pool, slot, hit);
    storePath(frame.pool, slot, path);
    pushBin(frame, slot, hit.kind == hitKindMesh ? surfaceBin : mediumBin);
}
