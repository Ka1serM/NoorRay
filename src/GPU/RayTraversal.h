#pragma once

#include "GPU/Annotations.h"
#include "Kernels/Types.h"

#include <optix.h>

#if defined(NR_GPU_CODE)
#include <optix_device.h>
#endif

using SceneTraversable = OptixTraversableHandle;

struct RayHit
{
    float t = 1e30f;
    float u = 0.0f;
    float v = 0.0f;
    uint32_t instanceIndex = InvalidIndex;
    uint32_t primitiveIndex = InvalidIndex;
    bool hit = false;
};

NR_GPU inline RayHit intersectRay(
    const SceneTraversable accel,
    const glm::vec3 origin,
    const glm::vec3 direction,
    const float tMin,
    const float tMax)
{
    RayHit hit{};
#if defined(NR_GPU_DEVICE_COMPILE)
    optixTraverse(
        accel,
        make_float3(origin.x, origin.y, origin.z),
        make_float3(direction.x, direction.y, direction.z),
        tMin,
        tMax,
        0.0f,
        0xff,
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
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
#endif
    return hit;
}

NR_GPU inline bool testOcclusion(
    const SceneTraversable accel,
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
        0xff,
        OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
        0,
        1,
        0);
    return optixHitObjectIsHit();
#else
    return false;
#endif
}
