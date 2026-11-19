#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <type_traits>

namespace NoorRay::Core {

inline constexpr std::uint64_t hashSeed = 0xcbf29ce484222325ull;

constexpr std::uint64_t hashBytes(const std::span<const std::byte> bytes, std::uint64_t hash = hashSeed) {
    for (const std::byte byte : bytes)
        hash = (hash ^ static_cast<std::uint64_t>(byte)) * 0x100000001b3ull;
    return hash;
}

constexpr std::uint64_t hashString(const std::string_view text, std::uint64_t hash = hashSeed) {
    for (const char character : text)
        hash = (hash ^ static_cast<unsigned char>(character)) * 0x100000001b3ull;
    return hash;
}

template <class T>
    requires std::is_trivially_copyable_v<T>
std::uint64_t hashValue(const T& value, const std::uint64_t hash = hashSeed) {
    return hashBytes(std::as_bytes(std::span(&value, 1)), hash);
}

constexpr std::uint64_t combineHash(const std::uint64_t hash, const std::uint64_t value) {
    return hash ^ (value + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2));
}

}
