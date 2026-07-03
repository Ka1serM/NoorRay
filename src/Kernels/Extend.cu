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
    const RayHit hit = intersectRay(params.scene.tlasHandle, ray.origin, ray.direction, 0.001f, 1000.0f);

    HitWorkItem item{};
    item.rayDirection = ray.direction;
    item.sampleIndex = ray.sampleIndex;
    item.baryU = hit.u;
    item.baryV = hit.v;
    item.instanceIndex = hit.instanceIndex;
    item.primitiveIndex = hit.hit ? hit.primitiveIndex : InvalidIndex;
    params.queues.hitQueue[index] = item;
}

#include "Kernels/Connect.cu"

#endif
