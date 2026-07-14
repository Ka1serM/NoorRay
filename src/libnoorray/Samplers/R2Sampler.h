#pragma once

#include <cmath>
#include <cstdint>

#include <glm/vec2.hpp>

#include "Samplers/Sampler.h"

class R2Sampler : public IndexedSampler<R2Sampler>
{
public:
    NR_CPU_GPU explicit R2Sampler(const SampleKey key)
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
        // The plastic constant defines the additive-recurrence R2 sequence.
        constexpr float PlasticConstant = 1.324717957244746f;
        constexpr float FirstReciprocalPower = 1.0f / PlasticConstant;
        constexpr float SecondReciprocalPower =
            1.0f / (PlasticConstant * PlasticConstant);
        return {
            wrapUnitSample(0.5f + FirstReciprocalPower * static_cast<float>(index)),
            wrapUnitSample(0.5f + SecondReciprocalPower * static_cast<float>(index))};
    }
};
