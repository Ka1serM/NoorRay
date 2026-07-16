#ifdef NR_OPTIX_PTX_BUILD

#include <optix_device.h>

#include "Raytracing/SceneData.h"
#include "Samplers/OwenSobolSampler.h"
#include "Samplers/RandomSampler.h"

extern "C"
{
extern __constant__ KernelParams params;
}

extern "C" __global__ void __anyhit__gaussian()
{
    const uint32_t instanceId = optixGetInstanceId();
    const uint32_t globalGaussianId = instanceId;

    // Shadow rays start inside the Gaussian that produced the scattering
    // event. Exclude that exact instance instead of relying on a world-space
    // epsilon, which cannot cover differently scaled Gaussians robustly.
    if (globalGaussianId == optixGetPayload_2())
    {
        optixIgnoreIntersection();
        return;
    }

    const float opacity = params.scene.gaussianOpacities[globalGaussianId];
    if (opacity <= 0.0f)
    {
        optixIgnoreIntersection();
        return;
    }

    // The instance transform carries each Gaussian's true (untruncated) R*S;
    // object-space coordinates are already Mahalanobis space. Compute the
    // closest-point-on-ray-to-origin distance.
    const float3 rayOrigin = optixGetObjectRayOrigin();
    const float3 rayDir    = optixGetObjectRayDirection();
    const float tClosest = -(rayOrigin.x * rayDir.x + rayOrigin.y * rayDir.y + rayOrigin.z * rayDir.z)
                         / fmaxf(rayDir.x * rayDir.x + rayDir.y * rayDir.y + rayDir.z * rayDir.z, 1e-8f);
    // OptiX does not normalize the object-space ray direction, so this t is
    // numerically identical to the world-space ray parameter — usable as the
    // shading hit distance without any extra transform.
    const float hitT = fmaxf(tClosest, optixGetRayTmin());
    const float maxHitT = __uint_as_float(optixGetPayload_1());
    if (hitT >= maxHitT)
    {
        optixIgnoreIntersection();
        return;
    }

    const float px = rayOrigin.x + hitT * rayDir.x;
    const float py = rayOrigin.y + hitT * rayDir.y;
    const float pz = rayOrigin.z + hitT * rayDir.z;
    const float distanceSq = px * px + py * py + pz * pz;

    // Precomputed cutoffDistanceSq (host-side, see Raytracer::renderFrame).
    if (distanceSq >= params.frame.cutoffDistanceSq)
    {
        optixIgnoreIntersection();
        return;
    }

    // Eq. 2: α_i = opacity * exp(-0.5 * distance²)
    // ── Russian roulette ────────────────────────────────────────────────
    // Accept with probability alpha so that expected value matches 3DGS
    // over operator compositing:
    //   E[C] = Σᵢ αᵢ·cᵢ·Πⱼ<ᵢ(1-αⱼ)
    // Payload layout:
    //   0: sampleIndex (set before traversal, preserved)
    //   1: world-space hit distance (float as uint) of the accepted gaussian's
    //      closest-approach point, set only on acceptance (init to tMax)
    //   2: accepted gaussianId (init to InvalidIndex)
    const uint32_t sampleIndex = optixGetPayload_0();

    const uint32_t pathSeed = hashCombine32(sampleIndex, params.depth);
    const uint32_t gaussianSeed = hashCombine32(globalGaussianId, 0u);
    const OwenSobolSampler sampler({
        params.frame.totalAccumulated, hashCombine32(pathSeed, gaussianSeed)});
    const float xi = sampler.sample1D(SampleDimension::Opacity);
    // alpha <= opacity, so low-opacity rejections avoid evaluating the
    // exponential.
    if (xi >= opacity)
    {
        optixIgnoreIntersection();
        return;
    }
    const float alpha = opacity * __expf(-0.5f * distanceSq);

    if (xi < alpha)
    {
        // Accept: record gaussian as closest hit, traversal terminates here.
        optixSetPayload_1(__float_as_uint(hitT));
        optixSetPayload_2(globalGaussianId);
    }
    else
    {
        // Reject: continue traversal to next gaussian or mesh.
        optixIgnoreIntersection();
    }
}

#endif
