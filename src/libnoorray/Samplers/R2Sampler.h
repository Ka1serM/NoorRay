#pragma once

#include <cmath>
#include <cstdint>

#include <glm/vec2.hpp>

#include "CUDA/Annotations.h"

struct R2Sampler
{
    NR_CPU_GPU static glm::vec2 sample(const uint32_t index)
    {
        constexpr float plastic = 1.324717957244746f;
        constexpr float a1 = 1.0f / plastic;
        constexpr float a2 = 1.0f / (plastic * plastic);
        return {fmodf(0.5f + a1 * static_cast<float>(index), 1.0f),
                fmodf(0.5f + a2 * static_cast<float>(index), 1.0f)};
    }
};
