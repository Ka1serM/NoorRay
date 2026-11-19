#pragma once

#include <cstdint>

#include "NoorRay/Core/Handle.h"
#include "NoorRay/Core/Math.h"

namespace NoorRay::Scene {

struct Mesh;
struct Material;

enum class InstanceFlags : std::uint32_t {
    None = 0,
    Hidden = 1u << 0,
    NoShadow = 1u << 1,
    NoCameraRays = 1u << 2,
    DoubleSided = 1u << 3,
};

constexpr InstanceFlags operator|(const InstanceFlags a, const InstanceFlags b) {
    return static_cast<InstanceFlags>(static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b));
}
constexpr bool hasFlag(const InstanceFlags flags, const InstanceFlags flag) {
    return (static_cast<std::uint32_t>(flags) & static_cast<std::uint32_t>(flag)) != 0;
}

struct Instance {
    Core::Transform transform;
    Core::Transform previousTransform;
    Core::Handle<Mesh> mesh;
    InstanceFlags flags = InstanceFlags::None;
    std::uint32_t id = 0;
    Core::Handle<Material> material;
};

}
