#pragma once

#include <cstdint>

#include "GPU/RayTraversal.h"
#include "Kernels/Types.h"

struct OptixLaunchParams
{
    WavefrontQueues queues;
    SceneTraversable traversable{};
    uint32_t depth{};
};
