#pragma once

#include <cstdint>

#include "CUDA/Annotations.h"

NR_CPU_GPU inline uint32_t pcgHash(uint32_t value)
{
    value = value * 747796405u + 2891336453u;
    const uint32_t shift = (value >> 28u) + 4u;
    return ((value >> shift) ^ value) * 277803737u ^ value;
}

NR_CPU_GPU inline float randomFloat(uint32_t& state)
{
    state = pcgHash(state);
    return static_cast<float>(state >> 8u) * (1.0f / 16777216.0f);
}
