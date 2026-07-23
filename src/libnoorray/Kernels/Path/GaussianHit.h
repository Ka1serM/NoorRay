#include <optix_device.h>

#include "Raytracing/Gpu/SceneData.h"
#include "Samplers/OwenSobolSampler.h"
#include "Samplers/RandomSampler.h"

extern "C"
{
extern __constant__ KernelParams params;
}

extern "C" __global__ void __anyhit__gaussian()
{
    const uint32_t localGaussianId = optixGetInstanceIndex();
    const uint32_t* const instanceOffsets = params.scene.gaussianInstanceOffsets;
    // A single Gaussian instance needs no outer-IAS lookup: its IAS index is
    // already the packed global Gaussian ID.  Large splat scenes spend most
    // of their time here, so avoid transform-list traversal and an extra
    // global load for every proxy candidate.
    const uint32_t globalGaussianId = instanceOffsets == nullptr
        ? localGaussianId
        : instanceOffsets[optixGetInstanceIdFromHandle(
            optixGetTransformListHandle(0))] + localGaussianId;

    // The Cryptomatte query keeps its sentinel in payload 0, which is never
    // changed by traversal. Surface AOV and beauty queries retain their normal
    // stochastic opacity sampling.
    const bool opaqueAov = params.frame.aovQuery != 0u
        && optixGetPayload_0() == OpaqueAovGaussianSample;
    const float opacity = params.scene.gaussianOpacities[globalGaussianId];
    if (!opaqueAov && opacity <= 0.0f)
    {
        optixIgnoreIntersection();
        return;
    }

    // If xi is above opacity it is necessarily above alpha, independent of
    // the proxy hit distance.  Evaluate that rejection before the object-space
    // closest-point calculation; dense splat scenes hit this path constantly.
    float xi = 0.0f;
    if (!opaqueAov)
    {
        const uint32_t sampleIndex = optixGetPayload_0();
        // The first Sobol dimension is a bit-reversed Van der Corput sequence,
        // retaining 1-D low discrepancy without a direction-number walk.
        const uint32_t gaussianSeed = hashCombine32(sampleIndex, globalGaussianId);
        const OwenSobolSampler sampler({
            params.frame.totalAccumulated, gaussianSeed});
        xi = sampler.sample1D(SampleDimension::PixelX);
        if (xi >= opacity)
        {
            optixIgnoreIntersection();
            return;
        }
    }

    // The instance transform carries each Gaussian's true (untruncated) R*S;
    // object-space coordinates are already Mahalanobis space. Compute the
    // closest-point-on-ray-to-origin distance.
    const float3 rayOrigin = optixGetObjectRayOrigin();
    const float3 rayDir    = optixGetObjectRayDirection();
    const float tClosest = -(rayOrigin.x * rayDir.x + rayOrigin.y * rayDir.y + rayOrigin.z * rayDir.z)
                         / (rayDir.x * rayDir.x + rayDir.y * rayDir.y + rayDir.z * rayDir.z);
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

    if (opaqueAov)
    {
        optixSetPayload_1(__float_as_uint(hitT));
        optixSetPayload_2(globalGaussianId);
        return;
    }

    // Eq. 2: α_i = opacity * exp(-0.5 * distance²)
    // ── Russian roulette ────────────────────────────────────────────────
    // Accept with probability alpha so that expected value matches 3DGS
    // over operator compositing:
    //   E[C] = Σᵢ αᵢ·cᵢ·Πⱼ<ᵢ(1-αⱼ)
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
