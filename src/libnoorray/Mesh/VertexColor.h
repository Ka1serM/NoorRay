#pragma once

#include <algorithm>
#include <cstdint>

#include <glm/vec4.hpp>

// Vertex colors are Unreal's FColor, read as linear UNORM8 as Unreal's
// vertex factory does: B | G << 8 | R << 16 | A << 24.
namespace nr::vertex_color
{

inline constexpr uint32_t White = 0xffffffffu;

inline uint32_t quantize(const float value)
{
    return static_cast<uint32_t>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
}

inline uint32_t pack(const glm::vec4 value)
{
    return quantize(value.b) | (quantize(value.g) << 8u)
        | (quantize(value.r) << 16u) | (quantize(value.a) << 24u);
}

inline glm::vec4 unpack(const uint32_t packed)
{
    constexpr float ByteToUnit = 1.0f / 255.0f;
    return glm::vec4(static_cast<float>((packed >> 16u) & 0xffu),
        static_cast<float>((packed >> 8u) & 0xffu),
        static_cast<float>(packed & 0xffu),
        static_cast<float>(packed >> 24u)) * ByteToUnit;
}

}
