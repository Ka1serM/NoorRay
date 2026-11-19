#pragma once

#include <cstdint>
#include <limits>

namespace NoorRay::Core {

template <class T>
struct Handle {
    static constexpr std::uint32_t invalidIndex = std::numeric_limits<std::uint32_t>::max();

    std::uint32_t index = invalidIndex;
    std::uint32_t generation = 0;

    constexpr bool isValid() const { return index != invalidIndex; }
    constexpr explicit operator bool() const { return isValid(); }
    constexpr bool operator==(const Handle&) const = default;
};

}
