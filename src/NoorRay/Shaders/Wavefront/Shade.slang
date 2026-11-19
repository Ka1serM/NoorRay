import NoorRHI;
import Wavefront.Records;
import Wavefront.PathState;
import Wavefront.Queues;
import Wavefront.Direct;
import Wavefront.SurfaceShading;
import Common.Math;
import Common.LinkConstants;
import Common.Random;
import Common.RayCone;
import Common.Sampling;
import Lights.Lights;
import Bsdf.EnergyLut;
import Bsdf.Principled;
import Materials.Material;
import Scene.SceneSurface;
import Spectral.Wavelengths;

void shadeSurfacePath(WavefrontFrame frame, uint parity, uint bin, uint slot) {
    const SceneRecord scene = frame.scene[0];
    PathState path = loadPath(frame.pool, slot);
    const PathHit hit = loadHit(frame.pool, slot);
    Wavelengths wavelengths = pathWavelengths(scene, path);
    const RayCone cone = propagate(path.cone, hit.t);
    const Surface surface = loadSurface(scene, hit.instance, hit.primitive, hit.barycentrics, path.direction, hit.t, cone);
    const ShadingContext context = surface.context;
    const float3 position = context.position.xyz;

    if (hasAlphaTest && path.hasFlag(pathFlagAlphaCandidate)) {
        path.setFlag(pathFlagAlphaCandidate, false);
        if (!passesOpacity(frame, path, surface, 2)) {
            path.origin = offsetAlong(position, surface.worldNormal, path.direction);
            path.cone = cone;
            storePath(frame.pool, slot, path);
            pushActive(frame, parity, slot);
            return;
        }
    }

    const BsdfInputs inputs = evaluateMaterial(scene, context, surface.material);
    const float4 emission = emittedRadiance(inputs, wavelengths);
    if (any(emission > 0.0)) {
        float weight = 1.0;
        if (!path.hasFlag(pathFlagDelta) && path.previousPdf > 0.0 && surface.light != invalidIndex) {
            const float pdf = lightPdf(scene, surface.light, path.origin, path.direction, hit.t, dot(surface.worldNormal, path.direction));
            weight = powerHeuristic(path.previousPdf, pdf);
        }
        addRadiance(frame, path, wavelengths, path.throughput * emission * weight);
    }

    const float3 view = -path.direction;
    const bool backfacing = isBackfacing(context);
    SurfaceScatter scatter;
    scatter.bsdf = makePrincipled(inputs, view, context.geometricNormal.xyz, backfacing, wavelengths, sceneEnergyLut(scene),
                                    hasDispersion && !path.hasFlag(pathFlagSecondaryTerminated));
    writePrimaryAovs(frame.film, path, saturate(wavelengths.toFilm(scatter.bsdf.albedo())), safeNormalize(inputs.normalRotation.xyz), hit.t,
                     hit.instance);

    const float4 lightU = bouncePattern(frame, path, 0);
    const float3 shadowOrigin = spawnOrigin(surface, view);
    queueDirectLight(frame, parity, slot, scene, path, wavelengths, scatter, shadowOrigin, lightU);

    const float4 bsdfU = bouncePattern(frame, path, 1);
    const BsdfSample sample = scatter.bsdf.sample(bsdfU.xyz);
    if (sample.pdf <= 0.0 || all(sample.weight <= 0.0)) {
        finishPath(frame.film, path);
        storePath(frame.pool, slot, path);
        pushFree(frame, slot);
        return;
    }
    path.throughput *= sample.weight;
    if ((sample.flags & bsdfFlagDispersive) != 0)
        path.setFlag(pathFlagSecondaryTerminated, true);
    const bool delta = (sample.flags & bsdfFlagDelta) != 0;
    path.setFlag(pathFlagDelta, delta);
    path.previousPdf = delta ? 0.0 : sample.pdf;
    path.origin = spawnOrigin(surface, sample.direction);
    path.direction = sample.direction;
    const float roughness = scatter.bsdf.roughness();
    path.cone = delta ? cone : widenRayCone(cone, roughness * roughness);
    continueOrFinish(frame, parity, slot, path, bsdfU.w);
}

//! Resolves the alpha-tested occluder that Shadow found: an opaque decision drops the shadow ray, a transparent one moves it past the hit for the next Shadow pass.
void resolveShadowAlphaPath(WavefrontFrame frame, uint parity, uint bin, uint slot) {
    const SceneRecord scene = frame.scene[0];
    PathState path = loadPath(frame.pool, slot);
    uint4 state = frame.pool.state[slot];
    state.z &= ~pathFlagShadowAlpha;
    frame.pool.state[slot] = state;
    const PathHit hit = loadHit(frame.pool, slot);
    const float4 origin = frame.pool.shadowOrigin[slot];
    const float3 direction = frame.pool.shadowDirection[slot].xyz;
    const Surface surface = loadSurface(scene, hit.instance, hit.primitive, hit.barycentrics, direction, hit.t, makeRayCone(0.0, 0.0));
    if (passesOpacity(frame, path, surface, 3 + hit.primitive))
        return;
    const float3 moved = offsetAlong(surface.context.position.xyz, surface.worldNormal, direction);
    const float remaining = origin.w >= floatMax ? floatMax : max(origin.w - length(moved - origin.xyz), 0.0);
    frame.pool.shadowOrigin[slot] = float4(moved, remaining);
    pushShadow(frame, parity, slot);
}
