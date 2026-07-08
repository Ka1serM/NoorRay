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
    const RayHit hit = intersectRay(params.scene.tlasHandle, ray.origin, ray.direction, 0.001f, 1000.0f, ray.sampleIndex, gaussianEnabled);

    HitWorkItem item{};
    item.rayOrigin     = ray.origin;
    item.rayDirection  = ray.direction;
    item.sampleIndex   = ray.sampleIndex;
    item.primitiveIndex = hit.hit ? hit.primitiveIndex : InvalidIndex;
    if (hit.isGaussianHit)
    {
        // Use offset instance index so that Shade can discriminate via
        // instanceIndex >= meshInstanceCount.
        item.instanceIndex = params.scene.meshInstanceCount + hit.instanceIndex;
        item.attribute0    = hit.t;               // world-space hit distance
        item.attribute1    = hit.gaussianAlpha;   // density alpha
    }
    else
    {
        item.instanceIndex = hit.instanceIndex;
        item.attribute0    = hit.u;               // baryU
        item.attribute1    = hit.v;               // baryV
    }
    params.queues.hitQueue[index] = item;
}

#include "Kernels/GaussianHit.cu"
#include "Kernels/GaussianProxyOverdraw.cu"
#include "Kernels/Connect.cu"

#endif
