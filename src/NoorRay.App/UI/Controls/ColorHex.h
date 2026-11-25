#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <format>
#include <optional>

#include <RmlUi/Core/Types.h>

#include "NoorRay/Core/Math.h"

namespace NoorRay::App {

//! The #rrggbb text of the colour pickers; channels above one show as one.
inline Rml::String hexOf(const Core::float3 color) {
    const auto byte = [](const float channel) { return static_cast<int>(std::lround(std::clamp(channel, 0.0f, 1.0f) * 255.0f)); };
    return std::format("#{:02x}{:02x}{:02x}", byte(color.x), byte(color.y), byte(color.z));
}

inline std::optional<Core::float3> colorOfHex(const Rml::String& text) {
    unsigned int packed = 0;
    if (text.size() != 7 || std::sscanf(text.c_str(), "#%6x", &packed) != 1)
        return std::nullopt;
    return Core::float3{static_cast<float>((packed >> 16) & 0xff) / 255.0f, static_cast<float>((packed >> 8) & 0xff) / 255.0f,
                        static_cast<float>(packed & 0xff) / 255.0f};
}

}
