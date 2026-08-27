#pragma once

#include <cmath>
#include <cstdint>

#include <glm/vec4.hpp>

#include "Backend/Host/Platform.h"

namespace nr::vertex_color
{

inline constexpr uint32_t White = 0xffffffffu;

NR_CPU_GPU inline float clampUnit(const float value)
{
    return fminf(fmaxf(value, 0.0f), 1.0f);
}

// RGB bytes are sRGB-encoded; alpha is ordinary linear UNORM8.
NR_CPU_GPU inline float linearToSrgb(const float value)
{
    const float linear = clampUnit(value);
    return linear <= 0.0031308f
        ? linear * 12.92f
        : 1.055f * powf(linear, 1.0f / 2.4f) - 0.055f;
}

NR_CPU_GPU inline float srgbToLinear(const float value)
{
    const float srgb = clampUnit(value);
    return srgb <= 0.04045f
        ? srgb / 12.92f
        : powf((srgb + 0.055f) / 1.055f, 2.4f);
}

inline uint8_t quantize(const float value)
{
    return static_cast<uint8_t>(clampUnit(value) * 255.0f + 0.5f);
}

// Packed channel layout: R | G<<8 | B<<16 | A<<24.
inline uint32_t packSrgb(const glm::vec4 value)
{
    return static_cast<uint32_t>(quantize(value.r))
        | (static_cast<uint32_t>(quantize(value.g)) << 8u)
        | (static_cast<uint32_t>(quantize(value.b)) << 16u)
        | (static_cast<uint32_t>(quantize(value.a)) << 24u);
}

inline uint32_t packLinear(const glm::vec4 value)
{
    return packSrgb(glm::vec4(
        linearToSrgb(value.r),
        linearToSrgb(value.g),
        linearToSrgb(value.b),
        clampUnit(value.a)));
}

NR_CPU_GPU inline glm::vec4 unpackLinear(const uint32_t packed)
{
    constexpr float ByteToUnit = 1.0f / 255.0f;
    const float r = static_cast<float>(packed & 0xffu) * ByteToUnit;
    const float g = static_cast<float>((packed >> 8u) & 0xffu) * ByteToUnit;
    const float b = static_cast<float>((packed >> 16u) & 0xffu) * ByteToUnit;
    const float a = static_cast<float>((packed >> 24u) & 0xffu) * ByteToUnit;
    return glm::vec4(srgbToLinear(r), srgbToLinear(g), srgbToLinear(b), a);
}

}
