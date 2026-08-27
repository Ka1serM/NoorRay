#include "RaytracingAbi.slang"
#include "../Gaussian/Gaussian.slang"

// Gaussian proxy geometry is non-opaque. Any-hit normally performs stochastic
// opacity/ellipsoid acceptance and rejects candidates before closest-hit,
// avoiding a raygen-side brute-force scan over the splat array. The overdraw
// diagnostic instead counts every proxy candidate from its canonical primary
// ray, so it must not be affected by opacity roulette.
[shader("anyhit")]
void main(inout RayPayload payload)
{
    const FrameParams frame = *root.arguments;
    if (frame.gaussianOverdrawEnabled != 0u
        && frame.gaussianOverdrawImage != 0xFFFFFFFFu)
    {
        RWStructuredBuffer<uint> overdraw = DescriptorHandle<RWStructuredBuffer<uint>>(
            uint2(frame.gaussianOverdrawImage, 0));
        uint previous;
        const uint2 pixel = DispatchRaysIndex().xy;
        InterlockedAdd(overdraw[pixel.y * frame.width + pixel.x], 1u, previous);
        IgnoreHit();
        return;
    }
    if (frame.gaussianRecords == 0xFFFFFFFFu || frame.gaussianCount == 0u)
    {
        IgnoreHit();
        return;
    }
    ByteAddressBuffer records = DescriptorHandle<ByteAddressBuffer>(
        uint2(frame.gaussianRecords, 0));
    const uint gaussianId = InstanceIndex() - frame.gaussianInstanceBase;
    if (gaussianId == payload.gaussianId)
    {
        IgnoreHit();
        return;
    }
    if (gaussianId >= frame.gaussianCount)
    {
        IgnoreHit();
        return;
    }
    GaussianHashSampler sampler;
    sampler.seed = payload.seed;
    // The shared proxy BLAS is instanced by each Gaussian's R*S transform.
    // Density is defined in that normalized object space; using world rays
    // made acceptance depend on the scene's distance from the world origin.
    const GaussianHit accepted = sampleGaussian(sampler, records, gaussianId,
        ObjectRayOrigin(), ObjectRayDirection(), RayTMin(), RayTCurrent(),
        frame.gaussianCutoffDistanceSq, gaussianId + 17u);
    if (!accepted.valid)
    {
        IgnoreHit();
        return;
    }
    payload.hit = 2u;
    payload.gaussianId = gaussianId;
    payload.gaussianOpacity = gaussianFloat(records, gaussianId,
        GaussianOpacityOffset);
    payload.t = accepted.t;
}
