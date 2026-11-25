import NoorRHI;
import Lights.SgLighting;
import Wavefront.Records;
import Wavefront.PathState;
import Wavefront.Queues;
import Wavefront.Media;
import Wavefront.Direct;
import Wavefront.Shadow;
import Wavefront.RadianceCache;
import Bsdf.Principled;
import Common.Math;
import Common.Random;
import Common.LinkConstants;
import Common.Sampling;
import Lights.Lights;
import Lights.Environment;
import Scene.Gaussians;
import Scene.Camera;
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

//! Tests Gaussians stochastically and keeps the nearest accepted peak. With commits, an accepted Gaussian commits its proxy entry, so traversal culls
//! every proxy that the ray enters later, though such a proxy can hold a nearer peak: that pass only bounds the peak, and a pass without commits
//! below the bound makes it exact.
struct GaussianFilter : IHitFilter {
    SceneRecord scene;
    uint seed;
    bool commits;
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
        return commits ? HitVerdict.Accept : HitVerdict.Ignore;
    }
};

bool isAlphaTested(SceneRecord scene, uint instance, uint primitive) {
    return (scene.materials[triangleMaterial(scene, instance, primitive)].flags & materialFlagAlphaTest) != 0;
}

float misWeight(PathState path, float lightPdf) {
    if (path.hasFlag(pathFlagDelta) || path.previousPdf <= 0.0)
        return 1.0;
    return powerHeuristic(path.previousPdf, lightPdf);
}

//! Emission of analytic area lights that the ray passes before tMax. Lights do not occlude, and volumes do not attenuate them.
float4 lightsAlongRay(WavefrontFrame frame, SceneRecord scene, PathState path, uint lightPath, Wavelengths wavelengths, float tMax) {
    float4 total = float4(0.0, 0.0, 0.0, 0.0);
    const bool camera = path.depth() == 0;
    if (camera && (sceneFlags(scene) & sceneFlagCameraLights) == 0)
        return total;
    float start = 0.0;
    for (uint step = 0; step < 4; ++step) {
        const LightHit hit = intersectLights(scene, path.lightSet(), path.origin + path.direction * start, path.direction, tMax - start);
        if (hit.light == invalidIndex)
            break;
        const float distance = start + hit.distance;
        start = distance * (1.0 + 1e-5) + 1e-5;
        if (camera ? (scene.lights[hit.light].linking.z & lightFlagCamera) == 0
                   : isHiddenFrom(hiddenRays(scene, hit.light), lightPath))
            continue;
        const float pdf =
            lightPdf(scene, path.lightSet(), hit.light, path.origin, path.lightReceiver(frame.visibility.cells), path.direction, distance, hit.emitterCosine);
        total += wavelengths.upsample(hit.radiance) * misWeight(path, pdf);
    }
    return total;
}

//! Cycles' volume direct light: segment points by one-sample MIS of equiangular sampling towards a light and an exponential of the majorant on the segment.
struct VolumeDirect {
    bool active;
    bool mis;
    bool equiangular;
    Equiangular sampler;
    float u;
};

//! Wavefront scenes with alpha test light the scatter of delta tracking in ShadeMedium instead, because their shadow ray has no material pass.
VolumeDirect planVolumeDirect(SceneRecord scene, MediumRange range, float3 origin, float3 direction, float2 pointU, float4 lightU, bool active) {
    VolumeDirect plan;
    plan.active = active;
    plan.mis = false;
    plan.equiangular = false;
    plan.u = pointU.x;
    if (!plan.active || range.volume >= scene.counts.w)
        return plan;
    const float3 middle = origin + direction * (0.5 * (range.tStart + range.tEnd));
    const LightSample light = sampleLights(scene, scene.volumes[range.volume].kind >> lightSetShift, middle, mediumReceiver(), lightU);
    if (light.pdf <= 0.0 || light.distance >= floatMax)
        return plan;
    plan.sampler = makeEquiangular(origin, direction, middle + light.direction * light.distance, range.tStart, range.tEnd);
    plan.mis = plan.sampler.isValid();
    plan.equiangular = plan.mis && pointU.y >= 0.5;
    return plan;
}

//! The camera segment lights several points, since its fog light is seen directly and one gobo-lit point per frame leaves most pixels dark.
static const uint mediumCameraPoints = 2;

uint mediumPointCount(PathState path) { return path.depth() == 0 ? mediumCameraPoints : 1; }

//! Point i of count: the segment random in stratum i, and the light randoms rotated by a rank-1 lattice, so each point alone is unbiased.
VolumeDirect stratifyMediumPoint(VolumeDirect plan, uint point, uint count) {
    plan.u = (float(point) + plan.u) / float(count);
    return plan;
}

float4 rotateMediumLightU(float4 u, uint point) { return frac(u + float(point) * float4(0.7548777, 0.5698403, 0.6180340, 0.4142136)); }

//! The segment point and its light toward one light sample, before the shadow ray; contribution is zero when there is nothing to trace.
struct MediumLight {
    float4 emission;
    float4 contribution;
    float3 position;
    float3 direction;
    float distance;
    float diffuseFraction;
};

//! ends is set when no phase sample of the segment can find the light, which then takes the full weight.
MediumLight sampleMediumLight(SceneRecord scene, PathState path, Wavelengths wavelengths, MediumRange range, VolumeDirect plan, float4 u, uint seed,
                              bool ends) {
    MediumLight result = (MediumLight)0;
    result.diffuseFraction = 1.0;
    const float opticalDepth = range.majorant * (range.tEnd - range.tStart);
    const float mass = opticalDepth < 1e-4 ? opticalDepth * (1.0 - 0.5 * opticalDepth) : 1.0 - exp(-opticalDepth);
    float equiangularPdf = 0.0;
    float t = plan.equiangular ? clamp(plan.sampler.sample(plan.u, equiangularPdf), range.tStart, range.tEnd)
                               : range.tStart - log(1.0 - plan.u * mass) / range.majorant;
    t = min(t, range.tEnd);
    const float distancePdf = range.majorant * exp(-range.majorant * (t - range.tStart)) / mass;
    if (plan.mis && !plan.equiangular)
        equiangularPdf = plan.sampler.pdf(t);
    //! One-sample MIS picks each strategy half the time, so the power heuristic weight over the chosen pdf is 2 pdf / (sum of squares).
    const float chosen = plan.equiangular ? equiangularPdf : distancePdf;
    const float pointWeight = plan.mis ? 2.0 * chosen / (distancePdf * distancePdf + equiangularPdf * equiangularPdf) : 1.0 / distancePdf;
    if (!(pointWeight > 0.0) || isinf(pointWeight))
        return result;
    const MediumPoint point = trackToPoint(scene, range, path.origin, path.direction, t, wavelengths, seed);
    result.emission = path.throughput * point.transmittance * point.medium.emission * pointWeight;
    if (point.medium.volume == invalidIndex)
        return result;
    const VolumeRecord volume = scene.volumes[point.medium.volume];
    result.position = path.origin + path.direction * t;
    const LightSample light = sampleLights(scene, volume.kind >> lightSetShift, result.position, mediumReceiver(), u);
    if (light.pdf <= 0.0 || all(light.radiance <= 0.0) || !lightReachesBounce(scene, light.light, path.depth())
        || (hiddenRays(scene, light.light) & lightHiddenVolumeScatter) != 0)
        return result;
    PhaseScatter phase;
    phase.direction = path.direction;
    phase.g = volume.boundsMax.w;
    phase.alpha = volume.draineAlpha;
    phase.albedo = float4(1.0, 1.0, 1.0, 1.0);
    const BsdfValue value = phase.evaluateScatter(light.direction, result.diffuseFraction);
    const float weight = pointWeight * (light.delta || ends ? 1.0 : powerHeuristic(light.pdf, value.pdf)) / light.pdf;
    const float4 contribution = path.throughput * point.transmittance * point.medium.scattering * value.value * wavelengths.upsample(light.radiance) * weight;
    if (all(contribution <= 0.0) || !isFiniteValue(contribution.xyz) || isnan(contribution.w) || isinf(contribution.w))
        return result;
    result.contribution = contribution;
    result.direction = light.direction;
    result.distance = castsShadow(scene, light.light) ? shadowDistance(light.distance) : 0.0;
    return result;
}

//! The primary AOVs of a miss: in fog, the guides of the middle of the fog, so the denoiser filters the fog light of the pixel.
void writeMissAovs(WavefrontFrame frame, PathState path, uint slot, MediumRange range) {
    const FilmRecord film = frame.film;
    if (!hasMedium(range)) {
        writePrimaryAovs(film, path, float3(0.0, 0.0, 0.0), -path.direction, 0.0, invalidIndex);
        return;
    }
    writePrimaryAovs(film, path, float3(1.0, 1.0, 1.0), -path.direction, cameraRayDistance(frame, path, slot, 0.5 * (range.tStart + range.tEnd)), invalidIndex);
    writeDemodulation(film, path, float3(1.0, 1.0, 1.0), float3(0.0, 0.0, 0.0), 1.0);
}

//! The guides of the first sample of a pixel whose camera ray passes a medium that lets less than half through before the first surface.
void writeMediumGuide(WavefrontFrame frame, SceneRecord scene, MediumRange range, inout PathState path, uint slot, float tEnd) {
    const FilmRecord film = frame.film;
    if (film.splitsLighting == 0 || path.depth() != 0 || !path.hasFlag(pathFlagPrimarySample))
        return;
    const MediumGuide guide = mediumGuide(scene, range, path.origin, path.direction, tEnd);
    if (!(guide.t > 0.0))
        return;
    writePrimaryAovs(film, path, guide.albedo, -path.direction, cameraRayDistance(frame, path, slot, guide.t), invalidIndex);
    writeDemodulation(film, path, guide.albedo, float3(0.0, 0.0, 0.0), 1.0);
    path.setFlag(pathFlagVolumeGuide, true);
}

//! Lights the segment points with their own shadow rays, as the path may still shade the surface behind; returns their mean emission.
float4 lightMediumSegment(WavefrontFrame frame, SceneRecord scene, PathState path, Wavelengths wavelengths, MediumRange range, VolumeDirect plan, float4 u,
                          bool scatters) {
    const bool ends = !scatters || isLastVertex(frame, path);
    const uint count = mediumPointCount(path);
    float4 emission = float4(0.0, 0.0, 0.0, 0.0);
    float4 contribution = float4(0.0, 0.0, 0.0, 0.0);
    for (uint point = 0; point < count; ++point) {
        const MediumLight light = sampleMediumLight(scene, path, wavelengths, range, stratifyMediumPoint(plan, point, count), rotateMediumLightU(u, point),
                                                    hashCombine(raySeed(frame, path, 7), point), ends);
        emission += light.emission / float(count);
        if (all(light.contribution <= 0.0)
            || !isUnoccluded(scene, light.position, light.direction, light.distance, hashCombine(raySeed(frame, path, 8), point)))
            continue;
        contribution += light.contribution / float(count)
                        * mediumTransmittance(scene, light.position, light.direction, light.distance, wavelengths, hashCombine(raySeed(frame, path, 9), point));
    }
    if (all(contribution <= 0.0))
        return emission;
    const float3 film = filmValue(frame, wavelengths, contribution, path.depth() > 0);
    splatRadiance(frame.film, path.pixel, film);
    if (path.depth() == 0)
        splitFogLight(frame.film, path.pixel, film);
    else
        splitPathLight(frame.film, path, film, path.diffuseFraction, path.primaryDirection);
    return emission;
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
    const uint lightPath = loadLightPath(frame.pool, slot);

    RayDesc ray;
    ray.Origin = path.origin;
    ray.Direction = path.direction;
    ray.TMin = 0.0;
    ray.TMax = path.depth() == 0 ? cameraFarDistance(frame.camera[0], path.origin, path.direction) : floatMax;
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
    const bool found = coverage == RasterCoverage.NeedsTrace ? traceRayInline(scene.tlas, RAY_FLAG_NONE, rayInstanceMask(lightPath, path.depth() == 0), ray, filter, meshHit)
                                                             : coverage == RasterCoverage.Mesh;
    //! Gaussians trace after the surfaces, bounded by them, with back faces culled so each proxy is tested once, where the ray enters it. A proxy
    //! encloses its peak, so the second pass up to the bound of the first tests every proxy with a nearer peak; the hash of the seed accepts the same
    //! Gaussians in both, so the nearest accepted peak does not depend on traversal order.
    GaussianFilter gaussians;
    gaussians.scene = scene;
    gaussians.seed = raySeed(frame, path, 0);
    gaussians.commits = true;
    gaussians.t = floatMax;
    gaussians.cloud = invalidIndex;
    gaussians.gaussian = 0;
    if (hasGaussians) {
        ray.TMax = min(found ? meshHit.t : ray.TMax, filter.alphaT);
        gaussians.t = ray.TMax;
        RayHit gaussianHit;
        if (traceRayInline(scene.gaussianTlas, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xffu, ray, gaussians, gaussianHit)) {
            gaussians.commits = false;
            ray.TMax = gaussians.t;
            traceRayInline(scene.gaussianTlas, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xffu, ray, gaussians, gaussianHit);
        }
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
    if (hasGaussians && gaussians.cloud != invalidIndex) {
        hit.t = gaussians.t;
        hit.kind = hitKindGaussian;
        hit.instance = gaussians.cloud;
        hit.primitive = gaussians.gaussian;
        hit.barycentrics = float2(0.0, 0.0);
        path.setFlag(pathFlagAlphaCandidate, false);
    }

    if (path.depth() == 1)
        splitHitDistance(frame.film, path, hit.t);
    addRadiance(frame, path, wavelengths, path.throughput * lightsAlongRay(frame, scene, path, lightPath, wavelengths, hit.t));

    const MediumRange range = hasVolumes && scene.counts.w != 0 ? mediumRange(scene, path.origin, path.direction, hit.t) : (MediumRange)0;
    if (hasVolumes && hasMedium(range)) {
        writeMediumGuide(frame, scene, range, path, slot, hit.t);
        const float4 pointU = bouncePattern(frame, path, 2);
        const float4 lightU = bouncePattern(frame, path, 3);
        const VolumeDirect plan = planVolumeDirect(scene, range, path.origin, path.direction, pointU.xy, lightU, !hasAlphaTest);
        const bool final = path.hasFlag(pathFlagFinalRay);
        const bool below = ((path.flags & pathVolumeMask) >> pathVolumeShift) < (frame.settings.x >> 8);
        const bool scatters = !final && (!plan.active || below || frame.bounces.w != 0);
        if (plan.active && !final) {
            const float4 emission = lightMediumSegment(frame, scene, path, wavelengths, range, plan, lightU, scatters);
            if (!scatters)
                addRadiance(frame, path, wavelengths, emission);
        }
        float4 absorbed;
        const MediumEvent event =
            trackMedium(scene, range, path.origin, path.direction, hit.t, wavelengths, raySeed(frame, path, 1), scatters, path.throughput, absorbed);
        addRadiance(frame, path, wavelengths, absorbed);
        if (event.kind == mediumEventAbsorb) {
            finishPath(frame.film, path);
            storePath(frame.pool, slot, path);
            frame.pool.pathBin[slot] = invalidIndex;
            pushFree(frame, slot);
            return;
        }
        if (event.kind == mediumEventScatter) {
            if (hit.kind == hitKindMesh)
                writePrimarySurfaceDepth(frame.film, path, cameraRayDistance(frame, path, slot, hit.t), hit.instance);
            else
                writeMissAovs(frame, path, slot, range);
            hit.t = event.t;
            hit.kind = hitKindVolume;
            hit.instance = event.volume;
            hit.primitive = 0;
            //! The weight of the direct light at the scatter, which ShadeMedium reads: none when the segment point took it.
            hit.barycentrics = float2(plan.active ? 0.0 : 1.0, 0.0);
            path.setFlag(pathFlagAlphaCandidate, false);
        }
    }

    if (hit.kind == hitKindMiss) {
        writeMissAovs(frame, path, slot, range);
        if (path.depth() == 0 && !path.hasFlag(pathFlagFork) && scene.sceneCenter.w == environmentTransparent)
            countTransparentSample(frame.film, path.pixel);
        float3 environmentSeen = path.depth() == 0 ? backgroundRadiance(scene, path.direction) : environmentRadiance(scene, path.direction);
        if (path.depth() > 0 && isHiddenFrom(hiddenRays(scene, environmentLight), lightPath))
            environmentSeen = float3(0.0, 0.0, 0.0);
        float4 background = wavelengths.upsample(environmentSeen)
                            * misWeight(path, lightPdf(scene, path.lightSet(), environmentLight, path.origin, path.lightReceiver(frame.visibility.cells),
                                                       path.direction, floatMax, 1.0));
        uint directional;
        const float3 sun = directionalRadiance(scene, path.lightSet(), path.direction, directional);
        if (directional != invalidIndex && path.depth() > 0 && !isHiddenFrom(hiddenRays(scene, directional), lightPath))
            background += wavelengths.upsample(sun)
                          * misWeight(path, lightPdf(scene, path.lightSet(), directional, path.origin, path.lightReceiver(frame.visibility.cells),
                                                     path.direction, floatMax, 1.0));
        addRadiance(frame, path, wavelengths, path.throughput * background);
        if (path.hasFlag(pathFlagCacheUpdate)) {
            applyRadianceCachePending(frame.cache, slot);
            updateRadianceCacheEnd(frame.cache, wavelengths, slot, background);
        }
        finishPath(frame.film, path);
        storePath(frame.pool, slot, path);
        frame.pool.pathBin[slot] = invalidIndex;
        pushFree(frame, slot);
        return;
    }

    storeHit(frame.pool, slot, hit);
    storePath(frame.pool, slot, path);
    pushBin(frame, slot, hit.kind == hitKindMesh ? surfaceBin(scene, hit.instance, hit.primitive) : mediumBin);
}
