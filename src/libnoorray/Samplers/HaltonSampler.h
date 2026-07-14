#pragma once

#include <cstdint>

#include <glm/vec2.hpp>

#include "Samplers/Sampler.h"

class HaltonSampler : public IndexedSampler<HaltonSampler>
{
public:
    NR_CPU_GPU explicit HaltonSampler(const SampleKey key)
        : IndexedSampler(key), point(basePoint(key.index))
    {
    }

    NR_CPU_GPU float evaluate1D(
        const uint32_t seed, const SampleDimension dimension) const
    {
        const uint32_t component = static_cast<uint32_t>(dimension) & 1u;
        return rotateUnitSample(component == 0 ? point.x : point.y, seed);
    }

    NR_CPU_GPU glm::vec2 evaluate2D(
        const uint32_t seedX, const uint32_t seedY, SampleDimensionPair) const
    {
        return {rotateUnitSample(point.x, seedX), rotateUnitSample(point.y, seedY)};
    }

private:
    glm::vec2 point;

    NR_CPU_GPU static glm::vec2 basePoint(const uint32_t index)
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
