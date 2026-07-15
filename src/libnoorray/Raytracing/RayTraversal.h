#pragma once

#include "CUDA/Annotations.h"
#include "Raytracing/Types.h"

#include <optix.h>

#if defined(NR_GPU_CODE)
#include <optix_device.h>
#endif


struct RayHit
{
    float t = 1e30f;
    float u = 0.0f;
    float v = 0.0f;
    uint32_t instanceIndex = InvalidIndex;
    uint32_t primitiveIndex = InvalidIndex;
};

// Payload layout for optixTraverse (gaussian-enabled path):
//   0: sampleIndex (uint32, preserved through traversal)
//   1: before acceptance, maximum allowed world-space Gaussian hit distance
//      (normally the nearest mesh hit); after acceptance, accepted Gaussian t
//   2: accepted gaussianId (uint32, init to InvalidIndex, set by any-hit on accept)
//
// We trace meshes first and use that t as a hard visibility bound for Gaussian
// sampling. This prevents a stochastic Gaussian hit behind an opaque mesh from
// replacing the mesh hit and making solid objects randomly disappear.
//
// This can't be collapsed into one optixTraverse call: Gaussians are instanced
// as a shared triangulated proxy shape (for fast RT), and the any-hit samples
// opacity at the analytic closest-approach point, not at wherever the ray
// happens to cross that proxy's triangles. OptiX's own hit-narrowing tracks
// the proxy-crossing distance, so in a merged trace, accepting a Gaussian
// narrows the search using a distance nearer than its true (farther) shading
// point — silently culling any mesh triangle that lies in between, even
// though that mesh is unambiguously the closer, opaque surface. That's biased
// (it recurs at a fixed, non-vanishing rate — roughly the Gaussian's alpha —
// so it does not average out with more samples), not just noise: it was
// confirmed by deliberately reproducing it. The mesh bound has to come from
// an authoritative closest-hit query first.

static constexpr uint32_t GaussianPayloadCount = 3;
static constexpr uint32_t GaussianSbtIndex     = 1; // sbtOffset for Gaussian hitgroup

NR_GPU inline RayHit intersectRay(
    const TlasHandle accel,
    const glm::vec3 origin,
    const glm::vec3 direction,
    const float tMin,
    const float tMax,
    const uint32_t sampleIndex = 0,
    const bool gaussianEnabled = true,
    const bool meshVisibilityBoundEnabled = true,
    const uint32_t excludedGaussianId = InvalidIndex)
{
    RayHit hit{};
#if defined(NR_GPU_DEVICE_COMPILE)
    if (gaussianEnabled)
    {
        RayHit meshHit{};
        if (meshVisibilityBoundEnabled)
        {
            // No TERMINATE_ON_FIRST_HIT: this must be the true closest mesh
            // hit, since it's also used directly as the final surface hit
            // when no Gaussian wins below (see comment above).
            optixTraverse(
                accel,
                make_float3(origin.x, origin.y, origin.z),
                make_float3(direction.x, direction.y, direction.z),
                tMin,
                tMax,
                0.0f,
                MeshVisibility,
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                0,
                1,
                0);
            if (optixHitObjectIsHit())
            {
                meshHit.t = optixHitObjectGetRayTmax();
                meshHit.u = __uint_as_float(optixHitObjectGetAttribute_0());
                meshHit.v = __uint_as_float(optixHitObjectGetAttribute_1());
                meshHit.instanceIndex = optixHitObjectGetInstanceIndex();
                meshHit.primitiveIndex = optixHitObjectGetPrimitiveIndex();
            }
        }

        const float gaussianTMax = meshHit.instanceIndex != InvalidIndex ? meshHit.t : tMax;
        uint32_t payload0 = sampleIndex;
        uint32_t payload1 = __float_as_uint(gaussianTMax);
        // Before acceptance payload 2 identifies a Gaussian that must not
        // intersect this ray. Any-hit overwrites it only when another
        // Gaussian is accepted.
        uint32_t payload2 = excludedGaussianId;
        optixTraverse(
            accel,
            make_float3(origin.x, origin.y, origin.z),
            make_float3(direction.x, direction.y, direction.z),
            tMin,
            gaussianTMax,
            0.0f,
            GaussianVisibility,
            OPTIX_RAY_FLAG_NONE,
            0,
            1,
            0,
            payload0, payload1, payload2);

        const float gaussianT = __uint_as_float(payload1);
        if (gaussianT < gaussianTMax)
        {
            // Gaussian accepted via Russian roulette.
            hit.instanceIndex = payload2;
            hit.t = gaussianT;
        }
        else if (meshHit.instanceIndex != InvalidIndex)
        {
            hit = meshHit;
        }
    }
    else
    {
        // Simple mesh-only traversal — no Gaussian payload, no any-hit.
        // No TERMINATE_ON_FIRST_HIT: needs the true closest hit, not an
        // arbitrary BVH-order intersection.
        optixTraverse(
            accel,
            make_float3(origin.x, origin.y, origin.z),
            make_float3(direction.x, direction.y, direction.z),
            tMin,
            tMax,
            0.0f,
            MeshVisibility,
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            0,
            1,
            0);
        if (optixHitObjectIsHit())
        {
            hit.t = optixHitObjectGetRayTmax();
            hit.u = __uint_as_float(optixHitObjectGetAttribute_0());
            hit.v = __uint_as_float(optixHitObjectGetAttribute_1());
            hit.instanceIndex = optixHitObjectGetInstanceIndex();
            hit.primitiveIndex = optixHitObjectGetPrimitiveIndex();
        }
    }
#endif
    return hit;
}

NR_GPU inline bool testOcclusion(
    const TlasHandle accel,
    const glm::vec3 origin,
    const glm::vec3 direction,
    const float tMin,
    const float tMax)
{
#if defined(NR_GPU_DEVICE_COMPILE)
    optixTraverse(
        accel,
        make_float3(origin.x, origin.y, origin.z),
        make_float3(direction.x, direction.y, direction.z),
        tMin,
        tMax,
        0.0f,
        MeshVisibility,
        OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
        0,
        1,
        0);
    return optixHitObjectIsHit();
#else
    return false;
#endif
}
