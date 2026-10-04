#include "GaussianSplat.slang"

// Gaussian proxy geometry is non-opaque. Any-hit normally performs stochastic
// opacity/ellipsoid acceptance and rejects candidates before closest-hit,
// avoiding a raygen-side brute-force scan over the splat array.
[shader("anyhit")]
void main(inout RayPayload payload)
{
    const Frame frame = *root.arguments;
    if (InstanceID() == payload.splatInstance)
    {
        IgnoreHit();
        return;
    }
    const GaussianSplat splat = loadGaussianSplat(frame, InstanceID());
    GaussianHashSampler sampler;
    sampler.seed = payload.seed;
    // The proxy is instanced by the splat's R*S transform. Density is defined
    // in that normalized object space; using world rays made acceptance depend
    // on the scene's distance from the world origin.
    const GaussianHit accepted = sampleGaussian(sampler, splat.records, splat.id,
        ObjectRayOrigin(), ObjectRayDirection(), RayTMin(), RayTCurrent(),
        gaussianCutoffDistanceSq(splat.records), InstanceID() + 17u);
    if (!accepted.valid)
    {
        IgnoreHit();
        return;
    }
    payload.hit = 2u;
    payload.splatInstance = InstanceID();
    payload.gaussianOpacity = gaussianFloat(splat.records, splat.id,
        GaussianOpacityOffset);
    payload.t = accepted.t;
}
