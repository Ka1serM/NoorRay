#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "DiffRenderer/Core/Random.h"
#include "NoorRHI/Core/Types.h"

namespace DiffRenderer::Algorithms {

struct SdfCamera {
    NoorRHI::float3 origin;
    NoorRHI::float3 target{0.5f, 0.5f, 0.5f};
    NoorRHI::float3 up{0.0f, 1.0f, 0.0f};
    float fieldOfView = 40.0f;
};

struct SdfSphere {
    NoorRHI::float3 center;
    float radius;
};

//! A blend of three spheres sampled into a grid SDF that covers [0, 1]^3, seen from six views around it.
class SdfScene {
public:
    static constexpr std::uint32_t resolution = 32;
    static constexpr std::size_t gridSize = std::size_t{resolution} * resolution * resolution;
    static constexpr std::size_t viewCount = 6;

    explicit SdfScene(Core::Random& random);

    //! Samples the smooth union of the spheres at the texel centers, in [z][y][x] order.
    static std::vector<float> sampleSpheres(const std::vector<SdfSphere>& spheres);

    std::array<SdfCamera, viewCount> views;
    NoorRHI::float3 lightDirection;
    float ambient = 0.1f;
    std::array<float, 3> albedo{0.8f, 0.5f, 0.3f};
    std::vector<float> grid;
    std::vector<float> startGrid;
};

}
