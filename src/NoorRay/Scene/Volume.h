#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "NoorRay/Core/Math.h"
#include "NoorRay/Scene/Material.h"

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
    //! The emission per metre that does not depend on the density.
    Core::float3 emissionRate;
    float anisotropy = 0.0f;
    //! The Draine alpha of the phase function of Cycles; 0 gives Henyey-Greenstein.
    float draineAlpha = 0.0f;
    float densityScale = 1.0f;
    Core::uint3 resolution;
    std::vector<std::uint16_t> density;
    //! The emission per metre of the voxels of resolution as RGB9E5, for a fire; empty for none.
    std::vector<std::uint32_t> emissionGrid;
    //! The share of each cell that the volume fills as halves, over the bounds, for a volume that is not a box; empty for a box.
    Core::uint3 maskResolution;
    std::vector<std::uint16_t> mask;
    //! A density node graph of a Grid volume: the device bakes its emission red at the voxel centres of resolution in place of density.
    std::shared_ptr<const Materials::Graph> densityGraph;
    std::vector<Core::float4> densityParameters;
    std::vector<MaterialTexture> densityTextures;
    //! The light set whose lights scatter in the volume.
    std::uint32_t lightSet = 0;
};

}
