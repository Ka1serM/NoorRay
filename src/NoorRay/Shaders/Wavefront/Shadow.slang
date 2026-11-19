import NoorRHI;
import Wavefront.Records;
import Wavefront.PathState;
import Wavefront.Queues;
import Wavefront.Media;
import Common.Math;
import Common.LinkConstants;
import Scene.Gaussians;
import Scene.SceneSurface;
import Spectral.Wavelengths;

//! Opaque surfaces end the search; alpha-tested surfaces are kept as the nearest candidate for their shade kernel.
struct ShadowFilter : IHitFilter {
    SceneRecord scene;
    float alphaT;
    uint alphaInstance;
    uint alphaPrimitive;
    float2 alphaBarycentrics;

    [mutating]
    HitVerdict filter(RayCandidate candidate) {
        const uint material = triangleMaterial(scene, candidate.instanceId, candidate.primitiveIndex);
        if (!hasAlphaTest || (scene.materials[material].flags & materialFlagAlphaTest) == 0)
            return HitVerdict.AcceptAndEnd;
        if (candidate.t < alphaT) {
            alphaT = candidate.t;
            alphaInstance = candidate.instanceId;
            alphaPrimitive = candidate.primitiveIndex;
            alphaBarycentrics = candidate.barycentrics;
        }
        return HitVerdict.Ignore;
    }
};

//! An accepted Gaussian ends the search. Back faces are culled, so the Gaussian that scatters the shadow ray from inside does not shadow itself.
struct GaussianShadowFilter : IHitFilter {
    SceneRecord scene;
    uint seed;
    float tMax;

    [mutating]
    HitVerdict filter(RayCandidate candidate) { return testGaussian(scene, candidate, 0.0, tMax, seed).accepted ? HitVerdict.AcceptAndEnd : HitVerdict.Ignore; }
};

//! bin is 1 on the last alpha layer, where an unresolved alpha-tested occluder counts as opaque. The light goes straight into the frame film, because the path may have ended in Shade.
void traceShadow(WavefrontFrame frame, uint parity, uint lastLayer, uint index) {
    const uint slot = shadowQueue(frame.queues, parity)[index];
    const SceneRecord scene = frame.scene[0];
    const float4 origin = frame.pool.shadowOrigin[slot];
    const float3 direction = frame.pool.shadowDirection[slot].xyz;
    PathState path = loadPath(frame.pool, slot);

    RayDesc ray;
    ray.Origin = origin.xyz;
    ray.Direction = direction;
    ray.TMin = 0.0;
    ray.TMax = origin.w;
    ShadowFilter filter;
    filter.scene = scene;
    filter.alphaT = floatMax;
    filter.alphaInstance = invalidIndex;
    filter.alphaPrimitive = 0;
    filter.alphaBarycentrics = float2(0.0, 0.0);
    RayHit hit;
    if (traceRayInline(scene.tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, instanceMaskShadow, ray, filter, hit))
        return;
    if (litGaussians) {
        GaussianShadowFilter gaussians;
        gaussians.scene = scene;
        gaussians.seed = raySeed(frame, path, 4);
        gaussians.tMax = min(origin.w, filter.alphaT);
        ray.TMax = gaussians.tMax;
        if (traceRayInline(scene.gaussianTlas, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xffu, ray, gaussians, hit))
            return;
    }
    if (hasAlphaTest && filter.alphaInstance != invalidIndex) {
        if (lastLayer != 0)
            return;
        PathHit alpha;
        alpha.t = filter.alphaT;
        alpha.barycentrics = filter.alphaBarycentrics;
        alpha.primitive = filter.alphaPrimitive;
        alpha.kind = hitKindMesh;
        alpha.instance = filter.alphaInstance;
        storeHit(frame.pool, slot, alpha);
        uint4 state = frame.pool.state[slot];
        state.z |= pathFlagShadowAlpha;
        frame.pool.state[slot] = state;
        const uint queued = atomicAdd(frame.queues.counters + counterNextShadow, 1u);
        shadowQueue(frame.queues, parity ^ 1)[queued] = slot;
        pushBin(frame, slot, surfaceBin);
        return;
    }
    const Wavelengths wavelengths = pathWavelengths(scene, path);
    float4 contribution = frame.pool.shadowRadiance[slot];
    if (hasVolumes && scene.counts.w != 0)
        contribution *= mediumTransmittance(scene, origin.xyz, direction, origin.w, wavelengths, raySeed(frame, path, 5));
    const float vertexDepth = frame.pool.shadowDirection[slot].w;
    splatRadiance(frame.film, path.pixel, filmValue(frame, wavelengths, contribution, vertexDepth > 0.5));
}
