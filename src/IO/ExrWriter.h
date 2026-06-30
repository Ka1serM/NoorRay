#pragma once

#include <cstdint>
#include <string>

// Writes unclamped, linear RGBA data as 32-bit floating-point OpenEXR.
bool writeFloatExr(
    const std::string& path, const float* rgba, uint32_t width, uint32_t height,
    std::string* errorMessage = nullptr);
