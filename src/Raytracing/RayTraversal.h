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
    bool hit = false;
    bool isGaussianHit = false;
    float gaussianAlpha = 0.0f;
};

// Payload layout for optixTraverse (gaussian-enabled path):
//   0: sampleIndex (uint32, preserved through traversal)
//   1: world-space hit distance (float as uint) of the accepted gaussian's
//      closest-approach point; init to tMax, set by any-hit on accept
//   2: accepted gaussianId (uint32, init to InvalidIndex, set by any-hit on accept)
//   3: accepted gaussianAlpha (float as uint, init to 0)
//
// The any-hit program uses Russian roulette: accept with prob = density alpha.
// On acceptance, payload[2,3] are set and the traversal terminates at the
// gaussian proxy triangle (no optixIgnoreIntersection).  On rejection the
// traversal continues to the next gaussian or mesh.

static constexpr uint32_t GaussianPayloadCount = 4;
static constexpr uint32_t GaussianSbtIndex     = 1; // sbtOffset for Gaussian hitgroup

NR_GPU inline RayHit intersectRay(
    const TlasHandle accel,
    const glm::vec3 origin,
    const glm::vec3 direction,
    const float tMin,
    const float tMax,
    const uint32_t sampleIndex = 0,
    const bool gaussianEnabled = true)
{
    RayHit hit{};
#if defined(NR_GPU_DEVICE_COMPILE)
    if (gaussianEnabled)
    {
        uint32_t payload0 = sampleIndex;
        uint32_t payload1 = __float_as_uint(tMax); // unused
        uint32_t payload2 = InvalidIndex;
        uint32_t payload3 = 0;
        optixTraverse(
            accel,
            make_float3(origin.x, origin.y, origin.z),
            make_float3(direction.x, direction.y, direction.z),
            tMin,
            tMax,
            0.0f,
            0x03,
            OPTIX_RAY_FLAG_NONE,
            0,
            1,
            0,
            payload0, payload1, payload2, payload3);

        const uint32_t gaussianId = payload2;
        if (gaussianId != InvalidIndex)
        {
            // Gaussian accepted via Russian roulette.
            hit.hit = true;
            hit.isGaussianHit = true;
            hit.instanceIndex = gaussianId;
            hit.gaussianAlpha = __uint_as_float(payload3);
            hit.t = __uint_as_float(payload1);
        }
        else if (optixHitObjectIsHit())
        {
            // Mesh hit (no gaussian accepted along this ray).
            hit.hit = true;
            hit.t = optixHitObjectGetRayTmax();
            hit.u = __uint_as_float(optixHitObjectGetAttribute_0());
            hit.v = __uint_as_float(optixHitObjectGetAttribute_1());
            hit.instanceIndex = optixHitObjectGetInstanceIndex();
            hit.primitiveIndex = optixHitObjectGetPrimitiveIndex();
        }
    }
    else
    {
        // Simple mesh-only traversal — no Gaussian payload, no any-hit.
        optixTraverse(
            accel,
            make_float3(origin.x, origin.y, origin.z),
            make_float3(direction.x, direction.y, direction.z),
            tMin,
            tMax,
            0.0f,
            0x01,
            OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            0,
            1,
            0);
        if (optixHitObjectIsHit())
        {
            hit.hit = true;
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
        0x01,
        OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
        0,
        1,
        0);
    return optixHitObjectIsHit();
#else
    return false;
#endif
}
