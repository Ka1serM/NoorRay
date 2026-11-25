#pragma once

#include <bit>
#include <cstdint>
#include <string_view>

//! The hashes of Cycles, so that random values match its renders.
namespace NoorRay::Python::CyclesHash {

inline std::uint32_t finish(std::uint32_t a, std::uint32_t b, std::uint32_t c) {
    const auto mix = [](std::uint32_t& target, const std::uint32_t source, const int turn) { target = (target ^ source) - std::rotl(source, turn); };
    mix(c, b, 14);
    mix(a, c, 11);
    mix(b, a, 25);
    mix(c, b, 16);
    mix(a, c, 4);
    mix(b, a, 14);
    mix(c, b, 24);
    return c;
}

inline std::uint32_t hash(const std::uint32_t x) {
    const std::uint32_t start = 0xdeadbeefu + (1u << 2) + 13u;
    return finish(start + x, start, start);
}

inline std::uint32_t hash(const std::uint32_t x, const std::uint32_t y) {
    const std::uint32_t start = 0xdeadbeefu + (2u << 2) + 13u;
    return finish(start + x, start + y, start);
}

inline std::uint32_t hash(const std::string_view text) {
    std::uint32_t result = 0;
    for (const char character : text)
        result = result * 37u + static_cast<std::uint32_t>(character);
    return result;
}

inline float toFloat(const std::uint32_t value) {
    return static_cast<float>(value) / static_cast<float>(0xffffffffu);
}

}
