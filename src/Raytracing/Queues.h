#pragma once

#include <cstdint>

#include "CUDA/Annotations.h"
#include "Raytracing/Types.h"

#if defined(NR_GPU_CODE)
NR_GPU inline uint32_t reserveWarpRange(uint32_t* count, const bool active)
{
    const unsigned int mask = __ballot_sync(0xffffffffu, active);
    const uint32_t lane = threadIdx.x & 31u;
    if (!active)
        return 0;
    const uint32_t leader = __ffs(mask) - 1u;
    uint32_t base = 0;
    if (mask != 0 && lane == leader)
        base = atomicAdd(count, __popc(mask));
    base = __shfl_sync(mask, base, static_cast<int>(leader));
    return base + __popc(mask & ((1u << lane) - 1u));
}

NR_GPU inline void appendRayWarp(
    WavefrontQueues queues,
    const uint32_t depth,
    const bool active,
    const PathRayWorkItem& item)
{
    const uint32_t index = reserveWarpRange(&queues.rayCounts[depth], active);
    if (!active)
        return;
#if !defined(NDEBUG)
    if (index >= queues.capacity)
        __trap();
#endif
    queues.rayQueues[depth & 1u][index] = item;
}
#endif
