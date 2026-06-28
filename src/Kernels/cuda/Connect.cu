#ifdef NR_OPTIX_PTX_BUILD

#include <optix_device.h>

#include "GPU/RayTraversal.h"
#include "Kernels/OptixLaunchParams.h"
#include "Kernels/Types.h"

extern "C"
{
extern __constant__ OptixLaunchParams params;
}

extern "C" __global__ void __raygen__connect()
{
    const uint32_t index = NR_GPU_OPTIX_LAUNCH_ID;
    const uint32_t activeCount = params.queues.rayCounts[params.depth];
    if (index >= activeCount)
        return;

    const ShadowWorkItem shadow = params.queues.shadowQueue[index];
    if (shadow.tMax <= shadow.tMin)
        return;
    if (!testOcclusion(params.traversable, shadow.origin, shadow.direction, shadow.tMin, shadow.tMax))
        params.queues.pathStates[shadow.sampleIndex].radiance += shadow.contribution;
}

#endif
