#pragma once

#include "Log.h"

#if defined(NR_CUDA_ACTIVE)
#include <cuda_runtime.h>
#endif

namespace nr {

inline bool synchronizeBeforeManagedMutation(const char* reason)
{
#if defined(NR_CUDA_ACTIVE)
    const cudaError_t result = cudaDeviceSynchronize();
    if (result == cudaSuccess)
        return true;

    LOG_ERROR(reason << ": CUDA synchronize failed before managed memory mutation: "
                     << cudaGetErrorString(result));
    return false;
#else
    (void)reason;
    return true;
#endif
}

}
