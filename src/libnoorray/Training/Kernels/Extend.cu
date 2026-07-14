#ifdef NR_OPTIX_PTX_BUILD

#include <optix_device.h>

#include "Raytracing/RayTraversal.h"
#include "Training/GaussianTrainData.h"

extern "C"
{
__constant__ GaussianTrainingKernelParams params;
}

extern "C" __global__ void __raygen__trainingExtend()
{
    const uint32_t index = NR_GPU_OPTIX_LAUNCH_ID;
    const uint32_t activeCount = params.queues.rayCounts[params.depth];
    if (index >= activeCount)
        return;

    const PathRayWorkItem ray = params.queues.rayQueues[params.depth & 1u][index];
    const RayHit hit = intersectRay(params.scene.tlasHandle, ray.origin, ray.direction,
        0.001f, 1000.0f, ray.sampleIndex, true, params.scene.meshInstanceCount > 0);

    HitWorkItem item{};
    item.sampleIndex = ray.sampleIndex;
    item.direction = ray.direction;
    item.gaussianData = ray.origin;
    if (hit.instanceIndex == InvalidIndex)
    {
        item.instanceIndex = InvalidIndex;
    }
    else if (hit.primitiveIndex == InvalidIndex)
    {
        item.instanceIndex = params.scene.meshInstanceCount + hit.instanceIndex;
    }
    else
    {
        item.attribute0 = hit.u;
        item.attribute1 = hit.v;
        item.instanceIndex = hit.instanceIndex;
    }
    item.primitiveIndex = hit.primitiveIndex;
    params.queues.hitQueue[index] = item;
}

#include "Training/Kernels/GaussianHit.cu"

#endif
