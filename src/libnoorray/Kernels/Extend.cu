#ifdef NR_OPTIX_PTX_BUILD

#include <optix_device.h>

#include "Raytracing/RayTraversal.h"
#include "Raytracing/SceneData.h"

extern "C"
{
__constant__ KernelParams params;
}

extern "C" __global__ void __raygen__extend()
{
    const uint32_t index = NR_GPU_OPTIX_LAUNCH_ID;
    const uint32_t activeCount = params.queues.rayCounts[params.depth];
    if (index >= activeCount)
        return;

    const PathRayWorkItem ray = params.queues.rayQueues[params.depth & 1u][index];
    const bool gaussianEnabled = (params.frame.visibilityMask & 0x02) != 0;
    const bool meshVisibilityBoundEnabled = gaussianEnabled && params.scene.meshInstanceCount > 0;
    const RayHit hit = intersectRay(params.scene.tlasHandle, ray.origin, ray.direction,
        0.001f, 1000.0f, ray.sampleIndex, gaussianEnabled, meshVisibilityBoundEnabled);

    HitWorkItem item{};
    item.sampleIndex = ray.sampleIndex;
    item.rayOrigin = ray.origin;
    item.rayDirection = ray.direction;
    if (hit.instanceIndex == InvalidIndex)
    {
        // Miss
        item.positionOrDirection = ray.direction;
        item.instanceIndex = InvalidIndex;
    }
    else if (hit.primitiveIndex == InvalidIndex)
    {
        // Gaussian hit: precompute hit position so Shade doesn't need rayOrigin
        const glm::vec3 gaussianPos = ray.origin + hit.t * ray.direction;
        item.positionOrDirection = gaussianPos;
        item.attribute0    = hit.gaussianAlpha;   // density alpha (unused by Shade)
        item.instanceIndex = params.scene.meshInstanceCount + hit.instanceIndex;
    }
    else
    {
        // Mesh hit
        item.positionOrDirection = ray.direction;
        item.attribute0    = hit.u;               // baryU
        item.attribute1    = hit.v;               // baryV
        item.instanceIndex = hit.instanceIndex;
    }

    item.primitiveIndex = hit.primitiveIndex;
    params.queues.hitQueue[index] = item;
}

#include "Kernels/GaussianHit.cu"
#include "Kernels/GaussianProxyOverdraw.cu"
#include "Kernels/Connect.cu"

#endif
