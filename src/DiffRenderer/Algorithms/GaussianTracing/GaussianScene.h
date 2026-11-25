#pragma once

#include <cstdint>
#include <vector>

#include "DiffRenderer/Algorithms/GaussianSplatting/Gaussians.h"
#include "DiffRenderer/Core/Random.h"
#include "NoorRHI/Core/Types.h"

namespace DiffRenderer::Algorithms {

//! Everything that is not optimized: the camera, a diffuse triangle mesh, a blackbody point light and a constant environment.
struct GaussianScene {
    void addTriangle(NoorRHI::float3 a, NoorRHI::float3 b, NoorRHI::float3 c, NoorRHI::float3 albedo);
    void addQuad(NoorRHI::float3 corner, NoorRHI::float3 edgeU, NoorRHI::float3 edgeV, NoorRHI::float3 albedo);
    void addBox(NoorRHI::float3 low, NoorRHI::float3 high, NoorRHI::float3 albedo);

    std::vector<NoorRHI::float3> positions;
    std::vector<std::uint32_t> indices;
    //! Linear sRGB, one for each triangle.
    std::vector<NoorRHI::float3> albedos;
    NoorRHI::float3 cameraOrigin{0.0f, 0.5f, 4.2f};
    NoorRHI::float3 cameraTarget{0.0f, -0.1f, 0.0f};
    NoorRHI::float3 cameraUp{0.0f, 1.0f, 0.0f};
    float fieldOfView = 40.0f;
    NoorRHI::float3 lightPosition{1.8f, 2.8f, 2.4f};
    float lightIntensity = 60.0f;
    float lightTemperature = 3200.0f;
    NoorRHI::float3 environment{0.25f, 0.4f, 0.7f};
};

//! A floor, and a box in front of some of the Gaussians.
GaussianScene getFloorAndBox();

}
