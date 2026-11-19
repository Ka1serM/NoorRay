import NoorRHI;
import Wavefront.Records;
import Wavefront.PathState;
import Wavefront.Queues;
import Common.Math;
import Common.Sampling;
import Lights.Lights;
import Bsdf.Principled;
import Scene.SceneSurface;
import Scene.Volumes;
import Spectral.Wavelengths;

interface IScatterFunction {
    BsdfValue evaluateScatter(float3 wi);
};

struct SurfaceScatter : IScatterFunction {
    PrincipledBsdf bsdf;
    BsdfValue evaluateScatter(float3 wi) { return bsdf.evaluate(wi); }
};

struct PhaseScatter : IScatterFunction {
    float3 direction;
    float g;
    float4 albedo;
    BsdfValue evaluateScatter(float3 wi) {
        const float phase = henyeyGreenstein(dot(direction, wi), g);
        BsdfValue value;
        value.value = albedo * phase;
        value.pdf = phase;
        return value;
    }
};

//! Samples one light for the vertex and writes the shadow ray of the path; position is already offset for surfaces. Returns whether a shadow ray was queued.
bool queueDirectLight<S : IScatterFunction>(WavefrontFrame frame, uint parity, uint slot, SceneRecord scene, PathState path, Wavelengths wavelengths, S scatter,
                                            float3 position, float4 u) {
    const LightSample light = sampleLights(scene, position, u);
    if (light.pdf <= 0.0 || all(light.radiance <= 0.0))
        return false;
    const BsdfValue value = scatter.evaluateScatter(light.direction);
    if (all(value.value <= 0.0))
        return false;
    const float weight = light.delta ? 1.0 : powerHeuristic(light.pdf, value.pdf);
    const float4 contribution = path.throughput * value.value * wavelengths.upsample(light.radiance) * (weight / light.pdf);
    if (all(contribution <= 0.0) || !isFiniteValue(contribution.xyz) || isnan(contribution.w) || isinf(contribution.w))
        return false;
    const float distance = light.distance >= floatMax ? floatMax : light.distance * (1.0 - 1e-4);
    frame.pool.shadowOrigin[slot] = float4(position, distance);
    frame.pool.shadowDirection[slot] = float4(light.direction, float(path.depth()));
    frame.pool.shadowRadiance[slot] = contribution;
    pushShadow(frame, parity, slot);
    return true;
}

float3 offsetAlong(float3 position, float3 normal, float3 direction) { return offsetRayOrigin(position, dot(direction, normal) >= 0.0 ? normal : -normal); }

//! The origin of a ray that leaves the surface: on the viewer side it starts from the lifted hit, so a coarse mesh shows no shadow terminator.
float3 spawnOrigin(Surface surface, float3 direction) {
    const float3 facing = surface.context.geometricNormal.xyz;
    if (dot(direction, facing) > 0.0)
        return offsetRayOrigin(surface.liftedPosition, facing);
    return offsetRayOrigin(surface.context.position.xyz, -facing);
}

//! Writes the primary AOVs of the first sample of a pixel at its first scattering vertex.
void writePrimaryAovs(FilmRecord film, PathState path, float3 albedo, float3 normal, float depth, uint objectId) {
    if (path.depth() != 0 || !path.hasFlag(pathFlagPrimarySample))
        return;
    film.albedo[path.pixel] = float4(albedo, 1.0);
    film.normalDepth[path.pixel] = float4(normal, depth);
    film.objectIds[path.pixel] = objectId;
}

//! Ends the path or queues it for the next Extend, after Russian roulette.
void continueOrFinish(WavefrontFrame frame, uint parity, uint slot, inout PathState path, float roulette) {
    if (all(path.throughput <= 0.0) || !continuePath(frame, path, roulette)) {
        finishPath(frame.film, path);
        storePath(frame.pool, slot, path);
        pushFree(frame, slot);
        return;
    }
    path.advanceDepth();
    storePath(frame.pool, slot, path);
    pushActive(frame, parity, slot);
}
