#pragma once

#include <cstdint>

#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Import {

enum class Primitive : std::uint8_t { Cube, Plane, Sphere };

//! Adds a unit primitive at the origin with its own default material.
class Primitives {
public:
    static Scene::InstanceHandle add(Scene::Scene& scene, Primitive primitive);
};

}
