#pragma once

#include <cstdint>

#include "CUDA/Annotations.h"

enum class SphericalHarmonicsOrder : uint32_t
{
    Degree0 = 0,
    Degree1 = 1,
    Degree2 = 2,
    Degree3 = 3,
};

NR_CPU_GPU constexpr uint32_t sphericalHarmonicsCoefficientCount(const SphericalHarmonicsOrder order)
{
    const uint32_t degree = static_cast<uint32_t>(order);
    return (degree + 1) * (degree + 1);
}

constexpr SphericalHarmonicsOrder clampSphericalHarmonicsOrder(const int degree)
{
    return degree <= 0 ? SphericalHarmonicsOrder::Degree0
        : degree == 1 ? SphericalHarmonicsOrder::Degree1
        : degree == 2 ? SphericalHarmonicsOrder::Degree2
                      : SphericalHarmonicsOrder::Degree3;
}
