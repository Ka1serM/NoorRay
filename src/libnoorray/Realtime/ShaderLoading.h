#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include <noorrhi/noorrhi.hpp>

// Creates a shader from SPIR-V embedded with #embed into a byte array.
template<std::size_t Size>
noorrhi::Shader loadShader(noorrhi::Device& device, const unsigned char (&bytes)[Size])
{
    return device.create_shader(std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(bytes), Size), "main");
}

constexpr uint32_t divideRoundingUp(const uint32_t value, const uint32_t divisor)
{
    return (value + divisor - 1u) / divisor;
}
