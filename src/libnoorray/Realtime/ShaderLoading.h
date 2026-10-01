#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <noorrhi/noorrhi.hpp>

#include "Realtime/EmbeddedShaders.h"

inline noorrhi::Shader loadShader(noorrhi::Device& device, const std::string& path,
    const std::string_view entryPoint = "main")
{
    const std::vector<std::uint32_t> words = embeddedSpirv(path);
    return device.create_shader(std::as_bytes(std::span(words)), entryPoint);
}

constexpr uint32_t divideRoundingUp(const uint32_t value, const uint32_t divisor)
{
    return (value + divisor - 1u) / divisor;
}
