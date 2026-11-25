#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "DiffRenderer/Core/Random.h"
#include "NoorRHI/Core/Types.h"

namespace DiffRenderer::Algorithms {

//! The test scene of the experiment: a unit cube with a procedural texture, seen through a fixed perspective camera.
struct TexturedCube {
    static constexpr std::uint32_t width = 256;
    static constexpr std::uint32_t height = 256;
    static constexpr std::uint32_t textureWidth = 64;
    static constexpr std::uint32_t textureHeight = 64;

    TexturedCube();

    std::size_t getVertexCount() const { return positions.size() / 3; }
    void project(std::span<const float> points, std::span<NoorRHI::float4> clip) const;
    void pullBack(std::span<const NoorRHI::float4> clipGradient, std::span<float> gradient) const;
    std::vector<float> perturb(Core::Random& random) const;

    std::vector<float> positions;
    std::vector<std::uint32_t> triangles;
    std::vector<std::int32_t> opposites;
    std::vector<NoorRHI::float2> uvs;
    std::vector<std::uint32_t> uvTriangles;
    std::vector<float> texture;
    NoorRHI::float4 background{0.2f, 0.3f, 0.45f, 0.0f};
    std::array<float, 16> transform{};
};

}
