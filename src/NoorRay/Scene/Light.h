#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "NoorRay/Core/Math.h"
#include "NoorRay/Scene/Material.h"

namespace NoorRay::Scene {

enum class LightType : std::uint8_t { Point, Spot, Rect, Disk, Sphere, Directional };

//! Power is in watts for local lights and in watts per square metre for directional lights; the light faces its local -Z.
struct Light {
    std::string name;
    LightType type = LightType::Point;
    Core::Transform transform;
    Core::float3 color{1.0f, 1.0f, 1.0f};
    float power = 100.0f;
    float radius = 0.0f;
    float width = 1.0f;
    float height = 1.0f;
    float spotAngle = Core::pi / 4.0f;
    float spotBlend = 0.15f;
    float angle = 0.0093f;
    //! The Light Falloff output of the light shader, as Cycles' Quadratic (0), Linear (1) or Constant (2): radiance scales by distance to this power.
    std::uint32_t falloffPower = 0;
    //! The Smooth of the Light Falloff, which scales radiance by d^2 / (smooth + d^2) at distance d; 0 is off.
    float falloffSmooth = 0.0f;
    //! Whether objects occlude the light, as Cycles' Cast Shadow.
    bool castsShadow = true;
    //! The last bounce that the light lights, as Cycles' per-light Max Bounces.
    std::uint32_t maxBounces = 1024;
    //! The rays that do not see the light, as the lightHidden bits of the GPU records, and whether camera rays see it.
    std::uint32_t hidden = 0;
    bool visibleToCamera = false;
    //! The cone angle of an area light, as Cycles' Spread; pi is a Lambertian light.
    float spread = Core::pi;
    //! The light node tree: its emission multiplies color and power over the light UV, as the light shader of Cycles does.
    std::shared_ptr<const Materials::Graph> graph;
    std::vector<Core::float4> graphParameters;
    std::vector<MaterialTexture> graphTextures;
    //! The light node tree that the graph compiles from.
    std::shared_ptr<const Materials::NodeTree> nodeTree;
    //! The light sets the light reaches; bit n is set n.
    std::uint64_t lightMask = ~std::uint64_t{0};

    //! The graph compares by pointer; Scene::setLight compares the programs.
    bool operator==(const Light&) const = default;
};

}
