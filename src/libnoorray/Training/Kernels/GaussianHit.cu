#ifdef NR_OPTIX_PTX_BUILD

#include <optix_device.h>

#include "Samplers/RandomSampler.h"
#include "Training/GaussianTrainData.h"

extern "C"
{
extern __constant__ GaussianTrainingKernelParams params;
}

extern "C" __global__ void __anyhit__trainingGaussian()
{
    const uint32_t gaussianId = optixGetInstanceId();
    const float opacity = 1.0f
        / (1.0f + __expf(-params.train.opacityLogit[gaussianId]));
    if (opacity <= 0.0f)
    {
        optixIgnoreIntersection();
        return;
    }

    const float3 rayOrigin = optixGetObjectRayOrigin();
    const float3 rayDirection = optixGetObjectRayDirection();
    const float directionLengthSq = rayDirection.x * rayDirection.x
        + rayDirection.y * rayDirection.y + rayDirection.z * rayDirection.z;
    const float tClosest = -(rayOrigin.x * rayDirection.x
        + rayOrigin.y * rayDirection.y + rayOrigin.z * rayDirection.z)
        / fmaxf(directionLengthSq, 1.0e-8f);
    const float hitT = fmaxf(tClosest, optixGetRayTmin());
    if (hitT >= __uint_as_float(optixGetPayload_1()))
    {
        optixIgnoreIntersection();
        return;
    }

    const float px = rayOrigin.x + hitT * rayDirection.x;
    const float py = rayOrigin.y + hitT * rayDirection.y;
    const float pz = rayOrigin.z + hitT * rayDirection.z;
    const float distanceSq = px * px + py * py + pz * pz;
    if (distanceSq >= params.frame.cutoffDistanceSq)
    {
        optixIgnoreIntersection();
        return;
    }

    const uint32_t sampleIndex = optixGetPayload_0();
    const uint32_t sampleCounter = hashCombine32(
        params.frame.totalAccumulated, params.depth);
    const uint32_t pathKey = hashCombine32(sampleIndex, sampleCounter);
    const uint32_t bits = hashCombine32(gaussianId, pathKey);
    const float xi = (static_cast<float>(bits >> 8u) + 0.5f)
        * (1.0f / 16777216.0f);
    if (xi >= opacity)
    {
        optixIgnoreIntersection();
        return;
    }

    const float alpha = opacity * __expf(-0.5f * distanceSq);
    if (xi < alpha)
    {
        optixSetPayload_1(__float_as_uint(hitT));
        optixSetPayload_2(gaussianId);
    }
    else
    {
        optixIgnoreIntersection();
    }
}

#endif
