#pragma once

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "NoorRay/Core/Handle.h"
#include "NoorRay/Core/Math.h"
#include "NoorRay/Scene/Material.h"

namespace NoorRay::Scene {

struct Texture;

//! An equirectangular map with +Y up; without a texture the environment is the constant color. The rotation turns the map about +Y.
struct Environment {
    Core::Handle<Texture> texture;
    Core::float3 color{0.0f, 0.0f, 0.0f};
    float strength = 1.0f;
    float rotation = 0.0f;
    //! Camera rays that miss the scene see black unless this is set; the environment lights the scene and shows in reflections either way.
    bool visibleToCamera = false;
    //! Where camera rays do not see the environment, they see opaque black and not a transparent film.
    bool opaque = false;
    //! Whether light sampling picks the environment; without it only scattered rays find it, as Cycles' world Sampling Method None.
    bool sampled = true;
    //! The rays that do not see the environment, as the lightHidden bits of the GPU records.
    std::uint32_t hidden = 0;
    TextureFilter filter = TextureFilter::Linear;
    //! A node graph in place of the texture: the device bakes its emission over a map. A graph that reads the ray type has a map of its own for camera rays.
    std::shared_ptr<const Materials::Graph> graph;
    std::vector<Core::float4> graphParameters;
    std::vector<MaterialTexture> graphTextures;
    //! The world node tree that the graph compiles from.
    std::shared_ptr<const Materials::NodeTree> nodeTree;

    bool operator==(const Environment&) const = default;
};

//! u = 0.5 looks down -Z and u grows towards +X; v = 0 is the top row (+Y). The shaders use the same mapping.
inline Core::float2 directionToEquirect(const Core::float3 direction) {
    const float u = std::atan2(direction.x, -direction.z) / (2.0f * Core::pi) + 0.5f;
    const float v = std::acos(std::clamp(direction.y, -1.0f, 1.0f)) / Core::pi;
    return {u, v};
}

inline Core::float3 equirectToDirection(const Core::float2 uv) {
    const float phi = (uv.x - 0.5f) * 2.0f * Core::pi;
    const float theta = uv.y * Core::pi;
    return {std::sin(theta) * std::sin(phi), std::cos(theta), -std::sin(theta) * std::cos(phi)};
}

}
