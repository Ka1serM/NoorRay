#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "NoorRay/Core/Math.h"

namespace NoorRay::Scene {

inline constexpr std::uint32_t getShCoefficientCount(const std::uint32_t degree) {
    return (degree + 1) * (degree + 1);
}

//! Rotations are quaternions (x, y, z, w). Scales are linear, opacities are in [0, 1]. sh holds half RGB triples, getShCoefficientCount(shDegree) per Gaussian.
struct GaussianCloud {
    std::string name;
    Core::Transform transform;
    std::vector<Core::float3> positions;
    std::vector<Core::float3> scales;
    std::vector<Core::float4> rotations;
    std::vector<float> opacities;
    std::uint32_t shDegree = 0;
    std::vector<std::uint16_t> sh;

    std::uint32_t getCount() const { return static_cast<std::uint32_t>(positions.size()); }
    Core::Bounds computeBounds() const;
};

}
