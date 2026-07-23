#include <optix_device.h>

#include <glm/gtc/quaternion.hpp>

#include "Raytracing/Gpu/Types.h"
#include "Samplers/RandomSampler.h"
#include "Samplers/OwenSobolSampler.h"
#include "Training/GaussianTrainData.h"
#include "Training/Kernels/Backward.h"

extern "C"
{
extern __constant__ GaussianTrainingKernelParams params;
}

extern "C" __global__ void __anyhit__trainingGaussian()
{
    const uint32_t sceneInstance = optixGetInstanceIdFromHandle(
        optixGetTransformListHandle(0));
    const uint32_t gaussianId =
        params.scene.gaussianInstanceOffsets[sceneInstance] + optixGetInstanceIndex();
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
        / directionLengthSq;
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
    }
    else
    {
        optixIgnoreIntersection();
    }
}

NR_GPU inline RayHit intersectTrainingRay(
    const TlasHandle accel, const glm::vec3 origin, const glm::vec3 direction,
    const float tMin, const float tMax, const uint32_t sampleIndex,
    const bool meshVisibilityBoundEnabled)
{
    RayHit hit{};
    RayHit meshHit{};
    if (meshVisibilityBoundEnabled)
    {
        optixTraverse(accel, make_float3(origin.x, origin.y, origin.z),
            make_float3(direction.x, direction.y, direction.z), tMin, tMax, 0.0f,
            MeshVisibility, OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0);
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
    optixTraverse(accel, make_float3(origin.x, origin.y, origin.z),
        make_float3(direction.x, direction.y, direction.z), tMin, gaussianTMax, 0.0f,
        GaussianVisibility, OPTIX_RAY_FLAG_NONE, 0, 1, 0, payload0, payload1);

    const float gaussianT = __uint_as_float(payload1);
    if (gaussianT < gaussianTMax && optixHitObjectIsHit())
    {
        hit.t = gaussianT;
        const uint32_t sceneInstance = optixGetInstanceIdFromHandle(
            optixHitObjectGetTransformListHandle(0));
        hit.instanceIndex = params.scene.gaussianInstanceOffsets[sceneInstance]
            + optixHitObjectGetInstanceIndex();
    }
    else if (meshHit.instanceIndex != InvalidIndex)
    {
        hit = meshHit;
    }
    return hit;
}

NR_GPU inline bool makeTrainingRay(
    const GaussianTrainingKernelParams& params, const uint32_t pixel,
    glm::vec3& origin, glm::vec3& direction)
{
    const uint32_t x = pixel % params.frame.width;
    const uint32_t y = pixel / params.frame.width;
    const OwenSobolSampler sampler({
        params.frame.totalAccumulated, hashCombine32(x, y)});
    glm::vec2 jitter(0.5f, 0.5f);
    if (params.frame.frameIndex != 0)
        jitter = sampler.sample2D(PixelSampleDimensions);

    origin = glm::vec3(params.train.cameraToWorld[3]);
    const glm::vec3 cameraDirection(
        (static_cast<float>(x) + jitter.x - params.train.cx) / params.train.fx,
        -(static_cast<float>(y) + jitter.y - params.train.cy) / params.train.fy,
        -1.0f);
    direction = glm::normalize(glm::vec3(
        params.train.cameraToWorld * glm::vec4(cameraDirection, 0.0f)));
    return true;
}

extern "C" __global__ void __raygen__trainingPath()
{
    const uint32_t pixel = NR_GPU_OPTIX_LAUNCH_ID;
    const uint32_t pixelCount = params.frame.width * params.frame.height;
    if (pixel >= pixelCount)
        return;

    glm::vec3 origin{};
    glm::vec3 direction{};
    if (!makeTrainingRay(params, pixel, origin, direction))
        return;

    const RayHit hit = intersectTrainingRay(params.train.tlas, origin, direction,
        0.001f, 1000.0f, pixel, params.scene.meshInstanceCount > 0);
    if (hit.instanceIndex == InvalidIndex || hit.primitiveIndex != InvalidIndex)
        return;

    const uint32_t gaussianId = hit.instanceIndex;
    if (params.train.dLdImage != nullptr)
    {
        accumulateGaussianTrainGradient(params, pixel, gaussianId, origin, direction);
        return;
    }

    const glm::vec3 color = params.train.colorRgb[gaussianId]
        / static_cast<float>(params.train.samplesPerPixel);
    params.train.outputColor[pixel] += color;
}
