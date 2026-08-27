#pragma once

#include <cstdint>
#include <cmath>

#include <glm/vec2.hpp>

#include "Backend/Host/Platform.h"
#include "Rendering/Sampling/RandomSampler.h"

enum class SampleDimension : uint32_t
{
    PixelX,
    PixelY,
    Wavelength,
    LensX,
    LensY,
    Opacity,
    Count,
};

struct SampleKey
{
    uint32_t index{};
    uint32_t scramble{};
};

struct SampleDimensionPair
{
    SampleDimension x;
    SampleDimension y;
};

inline constexpr SampleDimensionPair PixelSampleDimensions{
    SampleDimension::PixelX, SampleDimension::PixelY};
inline constexpr SampleDimensionPair LensSampleDimensions{
    SampleDimension::LensX, SampleDimension::LensY};

NR_CPU_GPU inline float wrapUnitSample(const float value)
{
    return value - floorf(value);
}

NR_CPU_GPU inline float rotateUnitSample(const float value, const uint32_t seed)
{
    return wrapUnitSample(value + hashFloat(seed));
}

template <typename Implementation>
class IndexedSampler
{
public:
    NR_CPU_GPU explicit IndexedSampler(const SampleKey key) : key(key) {}

    NR_CPU_GPU float sample1D(const SampleDimension dimension) const
    {
        return implementation().evaluate1D(
            sampleDimensionSeed(key.scramble, static_cast<uint32_t>(dimension)), dimension);
    }

    NR_CPU_GPU glm::vec2 sample2D(const SampleDimensionPair dimensions) const
    {
        return implementation().evaluate2D(
            sampleDimensionSeed(key.scramble, static_cast<uint32_t>(dimensions.x)),
            sampleDimensionSeed(key.scramble, static_cast<uint32_t>(dimensions.y)),
            dimensions);
    }

protected:
    SampleKey key;

private:
    NR_CPU_GPU const Implementation& implementation() const
    {
        return static_cast<const Implementation&>(*this);
    }
};
