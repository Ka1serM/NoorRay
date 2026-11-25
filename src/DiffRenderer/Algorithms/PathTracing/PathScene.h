#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "DiffRenderer/Core/Random.h"
#include "NoorRHI/Core/Types.h"

namespace DiffRenderer::Algorithms {

struct PathMesh {
    std::vector<NoorRHI::float3> positions;
    std::vector<std::uint32_t> indices;
    std::vector<NoorRHI::float3> normals;
};

//! The object albedo, the ground albedo and the object translation, in the order of DirectLighting::Parameters.
struct PathParameters {
    std::array<float, 6> albedos{};
    std::array<float, 3> translation{};
};

//! An icosphere above a ground plane, lit by one rectangular area light.
class PathScene {
public:
    explicit PathScene(Core::Random& random);

    PathParameters perturb(Core::Random& random) const;

    PathMesh ground;
    PathMesh object;
    std::vector<NoorRHI::uint4> edges;
    NoorRHI::float3 cameraOrigin{0.0f, 1.6f, 4.0f};
    NoorRHI::float3 cameraTarget{0.0f, 0.4f, 0.0f};
    NoorRHI::float3 cameraUp{0.0f, 1.0f, 0.0f};
    float fieldOfView = 40.0f;
    NoorRHI::float3 lightCorner{-1.4f, 2.5f, 0.1f};
    NoorRHI::float3 lightEdgeU{0.8f, 0.0f, 0.0f};
    NoorRHI::float3 lightEdgeV{0.0f, 0.0f, 0.8f};
    NoorRHI::float3 lightRadiance{30.0f, 30.0f, 30.0f};
    PathParameters truth;
};

}
