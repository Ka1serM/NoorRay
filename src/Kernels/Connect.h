#pragma once

#include <cstdint>

#include "GPU/RayTraversal.h"
#include "Kernels/Types.h"

NR_GPU inline void connectPass(
    const SceneTraversable traversable,
    const WavefrontQueues queues,
    const uint32_t depth,
    const uint32_t index)
{
    const uint32_t activeCount = queues.rayCounts[depth];
    if (index >= activeCount)
        return;

    const ShadowWorkItem shadow = queues.shadowQueue[index];
    if (shadow.tMax <= shadow.tMin)
        return;
    if (!testOcclusion(traversable, shadow.origin, shadow.direction, shadow.tMin, shadow.tMax))
        queues.pathStates[shadow.sampleIndex].radiance += shadow.contribution;
}
