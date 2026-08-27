#pragma once

#include <cstddef>
#include <cstdint>

// Compact, scalar Vulkan ABI for analytic scene lights.  The renderer uploads
// one immutable array per scene revision; Slang reads it with ByteAddressBuffer
// loads so no glm/std430 layout assumptions cross the host/device boundary.
//
// Layout (bytes):
//   0..11   position (or unused for directional)
//   12      LightInstance type: 0 point, 1 spot, 2 rectangle, 3 directional
//   16..27  direction
//   28      flags (bit 0 = rectangle two-sided)
//   32..43  RGB colour
//   44      intensity
//   48..59  rectangle tangent
//   60      reserved
//   64..95  type-specific params:
//           point/spot: radius, inner angle, outer angle
//           rectangle: width, height, two-sided (also flags), barn-door angle,
//                     barn-door length
//           directional: soft angular diameter
struct VulkanLightRecord
{
    float position[3]{};
    uint32_t type{};
    float direction[3]{};
    uint32_t flags{};
    float color[3]{1.0f, 1.0f, 1.0f};
    float intensity{1.0f};
    float tangent[3]{1.0f, 0.0f, 0.0f};
    uint32_t reserved{};
    float params[8]{};
};

struct VulkanLightHeader
{
    uint32_t count{};
    float finiteWeight{};
    uint32_t reserved[2]{};
};

static_assert(sizeof(VulkanLightRecord) == 96);
static_assert(offsetof(VulkanLightRecord, position) == 0);
static_assert(offsetof(VulkanLightRecord, direction) == 16);
static_assert(offsetof(VulkanLightRecord, color) == 32);
static_assert(offsetof(VulkanLightRecord, tangent) == 48);
static_assert(offsetof(VulkanLightRecord, params) == 64);
static_assert(sizeof(VulkanLightHeader) == 16);
