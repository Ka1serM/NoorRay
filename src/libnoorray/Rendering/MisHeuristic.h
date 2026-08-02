#pragma once

#include "Backend/CUDA/Annotations.h"

// Veach's multiple importance sampling weighting heuristics, used to combine
// BSDF and light sampling strategies.

NR_GPU inline float balanceHeuristic(const float a, const float b)
{
    return a / fmaxf(a + b, 1e-20f);
}

NR_GPU inline float powerHeuristic(const float a, const float b)
{
    const float a2 = a * a;
    const float b2 = b * b;
    return a2 / fmaxf(a2 + b2, 1e-20f);
}
