#pragma once

#include <cstdint>

#include "GPU/Annotations.h"
#include "GPU/TaggedPointer.h"
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

struct R2Sampler
{
    NR_CPU_GPU glm::vec2 sample(const uint32_t index) const
    {
        constexpr float plastic = 1.324717957244746f;
        constexpr float a1 = 1.0f / plastic;
        constexpr float a2 = 1.0f / (plastic * plastic);
        return {fmodf(0.5f + a1 * static_cast<float>(index), 1.0f),
                fmodf(0.5f + a2 * static_cast<float>(index), 1.0f)};
    }
};

struct HaltonSampler
{
    NR_CPU_GPU glm::vec2 sample(const uint32_t index) const
    {
        // Radical inverse base-2 (bit reversal) and base-3
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

class Sampler : public nr::TaggedPointer<R2Sampler, HaltonSampler>
{
public:
    using nr::TaggedPointer<R2Sampler, HaltonSampler>::TaggedPointer;

    NR_CPU_GPU glm::vec2 sample(const uint32_t index) const
    {
        return Dispatch([index](const auto* s) { return s->sample(index); });
    }
};
