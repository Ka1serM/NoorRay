#include <optix_device.h>

#include "Kernels/OptixLaunchParams.h"
#include "Kernels/Connect.h"
#include "Kernels/Extend.h"

extern "C"
{
__constant__ OptixLaunchParams params;
}

extern "C" __global__ void __raygen__extend()
{
    extendPass(params.traversable, params.queues, params.depth, NR_GPU_OPTIX_LAUNCH_ID);
}

extern "C" __global__ void __raygen__connect()
{
    connectPass(params.traversable, params.queues, params.depth, NR_GPU_OPTIX_LAUNCH_ID);
}
