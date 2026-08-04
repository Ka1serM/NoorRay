#pragma once

#include <optix_device.h>

#include "Backend/OptiX/ABI/Geometry.h"
#include "Rendering/Sampling/OwenSobolSampler.h"

extern "C"
{
extern __constant__ KernelParams params;
}

NR_GPU inline uint32_t gaussianGlobalId()
{
    const uint32_t localGaussianId = optixGetInstanceIndex();
    const uint32_t* const instanceOffsets = params.scene.gaussianInstanceOffsets;
    return instanceOffsets == nullptr
        ? localGaussianId
        : instanceOffsets[optixGetInstanceIdFromHandle(
            optixGetTransformListHandle(0))] + localGaussianId;
}

NR_GPU inline float gaussianDistanceSq(
    const float3 origin, const float3 direction, float& hitT)
{
    const float directionSq = direction.x * direction.x
        + direction.y * direction.y + direction.z * direction.z;
    const float projectedT = directionSq > 0.0f
        ? -(origin.x * direction.x + origin.y * direction.y
            + origin.z * direction.z) / directionSq
        : 0.0f;
    hitT = fmaxf(projectedT, optixGetRayTmin());

    const float hitX = fmaf(direction.x, hitT, origin.x);
    const float hitY = fmaf(direction.y, hitT, origin.y);
    const float hitZ = fmaf(direction.z, hitT, origin.z);
    return fmaf(hitX, hitX, fmaf(hitY, hitY, hitZ * hitZ));
}

// Beauty rays carry a pointer payload. They only need alpha testing here;
// closest-hit derives the Gaussian ID again and performs the shading.
extern "C" __global__ void __anyhit__gaussian_path()
{
    const uint32_t globalGaussianId = gaussianGlobalId();
    PathTracePayload* const payload = getPathTracePayload<>();
    const float opacity = params.scene.gaussianOpacities[globalGaussianId];
    if (opacity <= 0.0f)
    {
        optixIgnoreIntersection();
        return;
    }

    const uint32_t scramble = hashCombine32(
        payload->gaussianSampleIndex, globalGaussianId);
    const OwenSobolSampler sampler({
        params.frame.totalAccumulated,
        params.frame.sampleSeed == 0u
            ? scramble : hashCombine32(scramble, params.frame.sampleSeed)});
    const float xi = sampler.sample1D(SampleDimension::PixelX);
    if (xi >= opacity)
    {
        optixIgnoreIntersection();
        return;
    }

    const float3 origin = optixGetObjectRayOrigin();
    const float3 direction = optixGetObjectRayDirection();
    float hitT = 0.0f;
    const float distanceSq = gaussianDistanceSq(origin, direction, hitT);
    if (hitT >= payload->gaussianTMax
        || distanceSq >= params.frame.cutoffDistanceSq
        || xi >= opacity * __expf(-0.5f * distanceSq))
    {
        optixIgnoreIntersection();
    }
}

// Traversal-only rays carry a scalar sample/tmax payload. The opaque AOV
// sentinel deliberately skips opacity and random sampling.
extern "C" __global__ void __anyhit__gaussian_query()
{
    const uint32_t globalGaussianId = gaussianGlobalId();
    const bool opaqueAov = params.frame.aovQuery != 0u
        && optixGetPayload_0() == OpaqueAovGaussianSample
        && optixGetPayload_2() == OpaqueAovQueryMarker;

    // Shadow rays carry the Gaussian that spawned them in payload 2.
    if (!opaqueAov && globalGaussianId == optixGetPayload_2())
    {
        optixIgnoreIntersection();
        return;
    }

    float opacity = 0.0f;
    float xi = 0.0f;
    if (!opaqueAov)
    {
        opacity = params.scene.gaussianOpacities[globalGaussianId];
        if (opacity <= 0.0f)
        {
            optixIgnoreIntersection();
            return;
        }

        const uint32_t scramble = hashCombine32(
            optixGetPayload_0(), globalGaussianId);
        const OwenSobolSampler sampler({
            params.frame.totalAccumulated,
            params.frame.sampleSeed == 0u
                ? scramble : hashCombine32(scramble, params.frame.sampleSeed)});
        xi = sampler.sample1D(SampleDimension::PixelX);
        if (xi >= opacity)
        {
            optixIgnoreIntersection();
            return;
        }
    }

    const float3 origin = optixGetObjectRayOrigin();
    const float3 direction = optixGetObjectRayDirection();
    float hitT = 0.0f;
    const float distanceSq = gaussianDistanceSq(origin, direction, hitT);
    if (hitT >= __uint_as_float(optixGetPayload_1())
        || distanceSq >= params.frame.cutoffDistanceSq)
    {
        optixIgnoreIntersection();
        return;
    }

    if (opaqueAov || xi < opacity * __expf(-0.5f * distanceSq))
    {
        optixSetPayload_1(__float_as_uint(hitT));
        optixSetPayload_2(globalGaussianId);
    }
    else
    {
        optixIgnoreIntersection();
    }
}
