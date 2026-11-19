import NoorRHI;
import Wavefront.Records;
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
                     float4 albedo) {
    PhaseScatter phase;
    phase.direction = path.direction;
    phase.g = g;
    phase.albedo = albedo;
    queueDirectLight(frame, parity, slot, scene, path, wavelengths, phase, position, bouncePattern(frame, path, 0));
    const float4 u = bouncePattern(frame, path, 1);
    float pdf;
    const float3 next = sampleHenyeyGreenstein(path.direction, g, u.xy, pdf);
    path.throughput *= albedo;
    path.previousPdf = pdf;
    path.setFlag(pathFlagDelta, false);
    path.origin = position;
    path.direction = next;
    path.cone = makeRayCone(path.cone.width, halfPi);
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
        writePrimaryAovs(frame.film, path, saturate(gaussian.color), gaussian.normal, hit.t, hit.instance | objectIdGaussian);
        if (!litGaussians) {
            addRadiance(frame, path, wavelengths, path.throughput * wavelengths.upsample(gaussian.color));
            finishPath(frame.film, path);
            storePath(frame.pool, slot, path);
            pushFree(frame, slot);
            return;
        }
        scatterInMedium(frame, parity, slot, scene, path, wavelengths, gaussian.position, 0.0, saturate(wavelengths.upsample(saturate(gaussian.color))));
        return;
    }

    if (hasVolumes && hit.kind == hitKindVolume) {
        const float3 position = path.origin + path.direction * hit.t;
        const float g = hit.instance < scene.counts.w ? scene.volumes[hit.instance].boundsMax.w : 0.0;
        writePrimaryAovs(frame.film, path, float3(1.0, 1.0, 1.0), -path.direction, hit.t, invalidIndex);
        scatterInMedium(frame, parity, slot, scene, path, wavelengths, position, g, float4(1.0, 1.0, 1.0, 1.0));
        return;
    }

    finishPath(frame.film, path);
    pushFree(frame, slot);
}
