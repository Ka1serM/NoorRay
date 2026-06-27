#pragma once

#include <cstdint>

#include "GPU/RayTraversal.h"
#include "Kernels/Types.h"

NR_GPU inline void extendPass(
    const SceneTraversable traversable,
    const WavefrontQueues queues,
    const uint32_t depth,
    const uint32_t index)
{
    const uint32_t activeCount = queues.rayCounts[depth];
    if (index >= activeCount)
        return;

    const PathRayWorkItem ray = queues.rayQueues[depth & 1u][index];
    const RayHit hit = intersectRay(traversable, ray.origin, ray.direction, 0.001f, 1000.0f);

    HitWorkItem item{};
    item.rayDirection = ray.direction;
    item.sampleIndex = ray.sampleIndex;
    item.baryU = hit.u;
    item.baryV = hit.v;
    item.instanceIndex = hit.instanceIndex;
    item.primitiveIndex = hit.hit ? hit.primitiveIndex : InvalidIndex;
    queues.hitQueue[index] = item;
}
