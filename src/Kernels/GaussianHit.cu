#ifdef NR_OPTIX_PTX_BUILD

#include <optix_device.h>

#include "Mesh/GaussianCutoff.h"
#include "Raytracing/SceneData.h"

extern "C"
{
extern __constant__ KernelParams params;
}

extern "C" __global__ void __anyhit__gaussian()
{
    const uint32_t instanceId = optixGetInstanceId();
    const uint32_t globalGaussianId = instanceId;

    if (globalGaussianId >= params.scene.gaussianCount)
        return;

    const float opacity = params.scene.gaussianOpacities[globalGaussianId];
    if (opacity <= 0.0f)
    {
        optixIgnoreIntersection();
        return;
    }

    // Mahalanobis distance in object space: the instance transform carries
    // each Gaussian's true (untruncated) R*S — see GaussianAsset.cpp — so
    // object-space coordinates here are already whitened and distanceSq is
    // directly the squared Mahalanobis distance, with no per-hit correction
    // factor. Compute closest-point-on-ray-to-origin distance.
    const float3 rayOrigin = optixGetObjectRayOrigin();
    const float3 rayDir    = optixGetObjectRayDirection();
    const float tClosest = -(rayOrigin.x * rayDir.x + rayOrigin.y * rayDir.y + rayOrigin.z * rayDir.z)
                         / fmaxf(rayDir.x * rayDir.x + rayDir.y * rayDir.y + rayDir.z * rayDir.z, 1e-8f);
    // OptiX does not normalize the object-space ray direction, so this t is
    // numerically identical to the world-space ray parameter — usable as the
    // shading hit distance without any extra transform.
    const float hitT = fmaxf(tClosest, 0.0f);
    const float px = rayOrigin.x + hitT * rayDir.x;
    const float py = rayOrigin.y + hitT * rayDir.y;
    const float pz = rayOrigin.z + hitT * rayDir.z;
    const float distanceSq = px * px + py * py + pz * pz;

    // The shared proxy geometry bounds exactly the GaussianCutoffSigma²
    // support region (see GaussianCutoff.h), so this rejects before alpha
    // even needs evaluating — and guarantees alpha has already decayed to
    // near-zero by the time the ray reaches the proxy's facets, so the
    // icosahedron's edges never show up as visible seams.
    constexpr float cutoffDistanceSq = GaussianCutoffSigma * GaussianCutoffSigma;
    if (distanceSq >= cutoffDistanceSq)
    {
        optixIgnoreIntersection();
        return;
    }

    // Eq. 2: α_i = opacity * exp(-0.5 * distance²)
    const float alpha = opacity * __expf(-0.5f * distanceSq);
    if (alpha <= 0.0f)
    {
        optixIgnoreIntersection();
        return;
    }

    // ── Russian roulette ────────────────────────────────────────────────
    // Accept with probability alpha so that expected value matches 3DGS
    // over operator compositing:
    //   E[C] = Σᵢ αᵢ·cᵢ·Πⱼ<ᵢ(1-αⱼ)
    // Fully opaque gaussians (alpha ~ 1) always accept.
    if (alpha >= 1.0f)
    {
        optixSetPayload_1(__float_as_uint(hitT));
        optixSetPayload_2(globalGaussianId);
        optixSetPayload_3(__float_as_uint(1.0f));
        return; // traversal terminates here (no optixIgnoreIntersection)
    }

    // Payload layout:
    //   0: sampleIndex (set before traversal, preserved)
    //   1: world-space hit distance (float as uint) of the accepted gaussian's
    //      closest-approach point, set only on acceptance (init to tMax)
    //   2: accepted gaussianId (init to InvalidIndex)
    //   3: accepted gaussianAlpha (float as uint, init to 0)
    const uint32_t sampleIndex = optixGetPayload_0();

    // Stateless hash-based RNG key: decorrelates across pixels, accumulated
    // frames (sub-samples), and gaussians along the same ray.
    uint32_t key = sampleIndex
        ^ (params.frame.totalAccumulated << 8)
        ^ (globalGaussianId << 16);
    key ^= key >> 16;
    key *= 0x85ebca6bu;
    key ^= key >> 16;
    const float xi = (static_cast<float>(key & 0x00ffffffu) + 0.5f) / 16777216.0f;

    if (xi < alpha)
    {
        // Accept: record gaussian as closest hit, traversal terminates here.
        optixSetPayload_1(__float_as_uint(hitT));
        optixSetPayload_2(globalGaussianId);
        optixSetPayload_3(__float_as_uint(alpha));
    }
    else
    {
        // Reject: continue traversal to next gaussian or mesh.
        optixIgnoreIntersection();
    }
}

#endif
