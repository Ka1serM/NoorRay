import NoorRHI;
import Lights.SgLighting;
import Wavefront.Records;
import Wavefront.PathState;
import Wavefront.Queues;
import Common.Math;
import Common.Sampling;
import Lights.Lights;
import Wavefront.VisibilityCache;
import Wavefront.RadianceCache;
import Bsdf.Principled;
import Scene.SceneSurface;
import Scene.Volumes;
import Spectral.Wavelengths;

//! diffuseFraction is the diffuse share of the value, for the lighting split of the denoiser.
interface IScatterFunction {
    BsdfValue evaluateScatter(float3 wi, out float diffuseFraction);
    //! The share of the value that a light with these hidden rays gives.
    float lightShare(uint hidden, float3 wi, float diffuseFraction);
};

struct SurfaceScatter : IScatterFunction {
    PrincipledBsdf bsdf;
    BsdfValue evaluateScatter(float3 wi, out float diffuseFraction) { return bsdf.evaluate(wi, diffuseFraction); }
    float lightShare(uint hidden, float3 wi, float diffuseFraction) {
        return visibleShare(hidden, diffuseFraction, dot(bsdf.geometricNormal, wi) <= 0.0);
    }
};

struct PhaseScatter : IScatterFunction {
    float3 direction;
    float g;
    float alpha;
    float4 albedo;
    //! The phase is Draine's, and the pdf that of the Henyey-Greenstein sampling of scatterInMedium.
    BsdfValue evaluateScatter(float3 wi, out float diffuseFraction) {
        diffuseFraction = 1.0;
        const float cosTheta = dot(direction, wi);
        BsdfValue value;
        value.value = albedo * draine(cosTheta, g, alpha);
        value.pdf = henyeyGreenstein(cosTheta, g);
        return value;
    }
    float lightShare(uint hidden, float3 wi, float diffuseFraction) { return (hidden & lightHiddenVolumeScatter) != 0 ? 0.0 : 1.0; }
};

//! Samples a light and queues the shadow ray of the vertex; position is offset already, ends marks no scatter ray after. True when queued.
bool queueDirectLight<S : IScatterFunction>(WavefrontFrame frame, uint parity, uint slot, SceneRecord scene, PathState path, Wavelengths wavelengths, S scatter,
                                            float3 position, float4 u, bool ends = false) {
    const LightReceiver receiver = path.lightReceiver(frame.visibility.cells);
    const LightSample light = sampleLights(scene, path.lightSet(), position, receiver, u);
    if (light.pdf <= 0.0 || all(light.radiance <= 0.0) || !lightReachesBounce(scene, light.light, path.depth()))
        return false;
    float diffuseFraction;
    const BsdfValue value = scatter.evaluateScatter(light.direction, diffuseFraction);
    if (all(value.value <= 0.0))
        return false;
    const bool last = ends || isLastVertex(frame, path);
    const float weight = light.delta || last ? 1.0 : powerHeuristic(light.pdf, value.pdf);
    const float4 local = value.value * wavelengths.upsample(light.radiance)
                         * (weight * scatter.lightShare(hiddenRays(scene, light.light), light.direction, diffuseFraction) / light.pdf);
    const float4 contribution = path.throughput * local;
    if (all(contribution <= 0.0) || !isFiniteValue(contribution.xyz) || isnan(contribution.w) || isinf(contribution.w))
        return false;
    if (path.hasFlag(pathFlagCacheUpdate))
        setRadianceCacheDirect(frame.cache, wavelengths, slot, local);
    const float distance = castsShadow(scene, light.light) ? shadowDistance(light.distance) : 0.0;
    frame.pool.shadowOrigin[slot] = float4(position, distance);
    //! w is the vertex depth; a primary vertex stores -1 - its diffuse share instead, because the path itself may have moved on when Shadow runs.
    frame.pool.shadowDirection[slot] = float4(light.direction, path.depth() == 0 ? -1.0 - diffuseFraction : float(path.depth()));
    frame.pool.shadowRadiance[slot] = contribution;
    frame.pool.shadowEntry[slot] = visibilityEntry(receiver.cell, light.choice);
    pushShadow(frame, parity, slot);
    return true;
}

//! Below the light tree depth a vertex picks its lights by the light tree, deeper ones by power alone; NEE and MIS read the choice from the same receiver.
LightReceiver depthReceiver(WavefrontFrame frame, PathState path, LightReceiver receiver) {
    receiver.byPower = path.depth() >= frame.lighting.x;
    return receiver;
}

float3 offsetAlong(float3 position, float3 normal, float3 direction) { return offsetRayOrigin(position, dot(direction, normal) >= 0.0 ? normal : -normal); }

//! The origin of a ray that leaves the surface: on the viewer side it starts from the lifted hit, so a coarse mesh shows no shadow terminator.
float3 spawnOrigin(Surface surface, float3 direction) {
    const float3 facing = surface.context.geometricNormal.xyz;
    if (dot(direction, facing) > 0.0)
        return offsetRayOrigin(surface.liftedPosition, facing);
    return offsetRayOrigin(surface.context.position.xyz, -facing);
}

//! The Light Path event of a BSDF sample: Cycles counts a singular event as glossy.
uint bsdfLightPathEvent(uint flags) {
    const uint side = (flags & bsdfFlagTransmission) != 0 ? lightPathTransmission : 0u;
    if ((flags & bsdfFlagDiffuse) != 0)
        return side | lightPathDiffuse;
    return side | lightPathGlossy | ((flags & bsdfFlagDelta) != 0 ? lightPathSingular : 0u);
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
