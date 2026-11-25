#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <span>
#include <string_view>
#include <type_traits>

namespace NoorRay::Format {

static_assert(std::endian::native == std::endian::little);

template <class T>
concept Plain = std::is_trivially_copyable_v<T> && !std::is_pointer_v<T>;

inline constexpr std::uint64_t arrayAlignment = 16;

//! Writes little-endian data: plain values as their bytes, strings and arrays after a 64-bit count, arrays at 16-byte offsets.
class BinaryWriter {
public:
    explicit BinaryWriter(std::ostream& stream) : stream(stream) {}

    template <Plain T>
    void write(const T& value) {
        put(std::as_bytes(std::span(&value, 1)));
    }
    void write(std::string_view text);
    template <Plain T>
    void writeArray(const std::span<const T> values) {
        write(std::uint64_t{values.size()});
        align();
        put(std::as_bytes(values));
    }
    void align();
    std::uint64_t getPosition() const { return position; }

private:
    void put(std::span<const std::byte> bytes);

    std::ostream& stream;
    std::uint64_t position = 0;
};

}
