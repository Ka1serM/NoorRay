#pragma once

#include <cstdint>
#include <string>

#include "NoorRay/Core/Math.h"

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
};

}
