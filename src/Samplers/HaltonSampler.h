#pragma once

#include <cstdint>

#include <glm/vec2.hpp>

#include "CUDA/Annotations.h"

struct HaltonSampler
{
    NR_CPU_GPU static glm::vec2 sample(const uint32_t index)
    {
        float r2 = 0.0f;
        float factor = 0.5f;
        for (uint32_t v = index + 1; v != 0; v >>= 1)
        {
            r2 += static_cast<float>(v & 1u) * factor;
            factor *= 0.5f;
        }
        float r3 = 0.0f;
        factor = 1.0f / 3.0f;
        for (uint32_t v = index + 1; v != 0; v /= 3)
        {
            r3 += static_cast<float>(v % 3u) * factor;
            factor /= 3.0f;
        }
        return {r2, r3};
    }
};
