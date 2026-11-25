import NoorRHI;
import Lights.SgLighting;
import Wavefront.Records;
import Wavefront.RadianceCache;
import Wavefront.PathState;
import Wavefront.Queues;
import Wavefront.Direct;
import Common.Math;
import Common.RayCone;
import Common.LinkConstants;
import Common.Sampling;
import Scene.Gaussians;
import Scene.Volumes;
import Spectral.Wavelengths;

void scatterInMedium(WavefrontFrame frame, uint parity, uint slot, SceneRecord scene, inout PathState path, Wavelengths wavelengths, float3 position, float g,
                     float alpha, float4 albedo, bool volume, float directWeight) {
    if (path.hasFlag(pathFlagFinalRay) || reachedKindBounces(frame, loadLightPath(frame.pool, slot))) {
        finishPath(frame.film, path);
        storePath(frame.pool, slot, path);
        pushFree(frame, slot);
        return;
    }
    PhaseScatter phase;
    phase.direction = path.direction;
    phase.g = g;
    phase.alpha = alpha;
    phase.albedo = albedo;
    path.setReceiver(depthReceiver(frame, path, mediumReceiver()));
    //! The cache keeps surface light only, so an update ends at a volume scatter.
    if (path.hasFlag(pathFlagCacheUpdate)) {
        stopRadianceCacheUpdate(frame.cache, slot);
        path.setFlag(pathFlagCacheUpdate, false);
    }
    PathState lit = path;
    lit.throughput *= directWeight;
    const uint volumeBounces = (path.flags & pathVolumeMask) >> pathVolumeShift;
    const bool limit = volume && volumeBounces >= (frame.settings.x >> 8);
    const bool ends = limit && frame.bounces.w == 0;
    if (directWeight > 0.0)
        queueDirectLight(frame, parity, slot, scene, lit, wavelengths, phase, position, bouncePattern(frame, path, 0), ends);
    if (ends) {
        finishPath(frame.film, path);
        storePath(frame.pool, slot, path);
        pushFree(frame, slot);
        return;
    }
    if (volume && !limit)
        path.flags = (path.flags & ~pathVolumeMask) | (volumeBounces + 1) << pathVolumeShift;
    //! As Cycles, the scatter at the volume bounce limit traces one more ray for the light that light sampling misses.
    if (limit)
        path.setFlag(pathFlagFinalRay, true);
    const float4 u = bouncePattern(frame, path, 1);
    float pdf;
    const float3 next = sampleHenyeyGreenstein(path.direction, g, u.xy, pdf);
    path.throughput *= albedo * (alpha != 0.0 ? draine(dot(path.direction, next), g, alpha) / max(pdf, 1e-20) : 1.0);
    path.previousPdf = pdf;
    recordScatterPdf(frame.pool, slot, pdf);
    path.setFlag(pathFlagDelta, false);
    path.origin = position;
    path.direction = next;
    path.cone = makeRayCone(path.cone.width, halfPi);
    recordLightPath(frame.pool, slot, lightPathVolumeScatter);
    continueOrFinish(frame, parity, slot, path, u.w);
}

//! The special bin: Gaussian hits (emissive colour, or isotropic scattering when litGaussians) and volume scattering events found by Extend.
void shadeMediumPath(WavefrontFrame frame, uint parity, uint bin, uint slot) {
    const SceneRecord scene = frame.scene[0];
    PathState path = loadPath(frame.pool, slot);
    const PathHit hit = loadHit(frame.pool, slot);
    const Wavelengths wavelengths = pathWavelengths(scene, path);
    path.cone = propagate(path.cone, hit.t);

    if (hasGaussians && hit.kind == hitKindGaussian) {
        const GaussianSurface gaussian = loadGaussianSurface(scene, hit.instance, hit.primitive, path.origin, path.direction, hit.t);
        writePrimaryAovs(frame.film, path, saturate(gaussian.color), gaussian.normal, cameraRayDistance(frame, path, slot, hit.t), hit.instance | objectIdGaussian);
        writeDemodulation(frame.film, path, saturate(gaussian.color), float3(0.0, 0.0, 0.0), 1.0);
        if (!litGaussians) {
            addRadiance(frame, path, wavelengths, path.throughput * wavelengths.upsample(gaussian.color));
            finishPath(frame.film, path);
            storePath(frame.pool, slot, path);
            pushFree(frame, slot);
            return;
        }
        path.setLightSet(0);
        const float4 albedo = saturate(wavelengths.upsample(saturate(gaussian.color)));
        scatterInMedium(frame, parity, slot, scene, path, wavelengths, gaussian.position, 0.0, 0.0, albedo, false, 1.0);
        return;
    }

    if (hasVolumes && hit.kind == hitKindVolume) {
        const float3 position = path.origin + path.direction * hit.t;
        const bool known = hit.instance < scene.counts.w;
        const float g = known ? scene.volumes[hit.instance].boundsMax.w : 0.0;
        const float alpha = known ? scene.volumes[hit.instance].draineAlpha : 0.0;
        path.setLightSet(known ? scene.volumes[hit.instance].kind >> lightSetShift : 0);
        scatterInMedium(frame, parity, slot, scene, path, wavelengths, position, g, alpha, float4(1.0, 1.0, 1.0, 1.0), true, hit.barycentrics.x);
        return;
    }

    finishPath(frame.film, path);
    pushFree(frame, slot);
}
