#pragma once

#include <cstdint>
#include <vector>

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
    NoDiffuseRays = 1u << 4,
    NoGlossyRays = 1u << 5,
    NoTransmissionRays = 1u << 6,
    NoVolumeScatterRays = 1u << 7,
    Holdout = 1u << 8,
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
    //! Light linking as in Cycles: the light set whose lights reach the surface, and the sets its emission reaches.
    std::uint32_t lightSet = 0;
    std::uint64_t lightMask = ~std::uint64_t{0};
    //! The Object Info node of Cycles: the object colour, pass index and random number.
    Core::float4 color{1.0f, 1.0f, 1.0f, 1.0f};
    std::uint32_t passIndex = 0;
    float random = 0.0f;
    //! The values that the materials read by slot; the device pads the instances to one length.
    std::vector<Core::float4> attributes;
};

}
