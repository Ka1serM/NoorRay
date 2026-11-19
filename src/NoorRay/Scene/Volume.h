#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "NoorRay/Core/Math.h"

namespace NoorRay::Scene {

enum class VolumeKind : std::uint8_t { Homogeneous, Grid };

//! Coefficients are per metre in linear RGB. A grid stores half densities in x-fastest order inside the bounds.
struct Volume {
    std::string name;
    VolumeKind kind = VolumeKind::Homogeneous;
    Core::Transform transform;
    Core::Bounds bounds{{-1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, 1.0f}};
    Core::float3 absorption{0.1f, 0.1f, 0.1f};
    Core::float3 scattering{0.5f, 0.5f, 0.5f};
    Core::float3 emission;
    float anisotropy = 0.0f;
    float densityScale = 1.0f;
    Core::uint3 resolution;
    std::vector<std::uint16_t> density;
};

}
