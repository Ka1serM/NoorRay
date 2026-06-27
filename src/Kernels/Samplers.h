#pragma once

#include <cstdint>

#include "GPU/Annotations.h"
#include "Kernels/Types.h"

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

NR_CPU_GPU inline float radicalInverse(uint32_t value, const uint32_t base)
{
    float result = 0.0f;
    float factor = 1.0f / static_cast<float>(base);
    while (value != 0)
    {
        result += static_cast<float>(value % base) * factor;
        value /= base;
        factor /= static_cast<float>(base);
    }
    return result;
}

NR_CPU_GPU inline glm::vec2 haltonSample(const uint32_t index)
{
    return {radicalInverse(index + 1, 2), radicalInverse(index + 1, 3)};
}

NR_CPU_GPU inline glm::vec2 r2Sample(const uint32_t index)
{
    constexpr float plastic = 1.324717957244746f;
    constexpr float a1 = 1.0f / plastic;
    constexpr float a2 = 1.0f / (plastic * plastic);
    return {fmodf(0.5f + a1 * static_cast<float>(index), 1.0f),
            fmodf(0.5f + a2 * static_cast<float>(index), 1.0f)};
}
