#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "NoorRay/Core/Error.h"
#include "NoorRay/Format/BinaryWriter.h"

namespace NoorRay::Format {

//! Reads what BinaryWriter writes from bytes whose start is 16-byte aligned in the file. A read past the end invalidates the reader and gives zeros.
class BinaryReader {
public:
    explicit BinaryReader(const std::span<const std::byte> bytes) : bytes(bytes) {}

    template <Plain T>
    void read(T& value) {
        const std::span<const std::byte> source = take(sizeof(T));
        if (source.empty())
            return;
        if constexpr (std::is_same_v<T, bool>)
            value = std::to_integer<std::uint8_t>(source[0]) != 0;
        else
            std::memcpy(&value, source.data(), sizeof(T));
    }
    template <Plain T>
    T read() {
        T value{};
        read(value);
        return value;
    }
    void read(std::string& text) {
        const std::span<const std::byte> source = take(read<std::uint64_t>());
        text.assign(reinterpret_cast<const char*>(source.data()), source.size());
    }
    template <Plain T>
    void readArray(std::vector<T>& values) {
        const std::span<const std::byte> source = takeArray(sizeof(T));
        values.resize(source.size() / sizeof(T));
        if (!source.empty())
            std::memcpy(values.data(), source.data(), source.size());
    }
    //! The bytes of an array of elements of elementSize, in place.
    std::span<const std::byte> takeArray(const std::size_t elementSize) {
        const std::uint64_t count = read<std::uint64_t>();
        align();
        if (count > getRemaining() / elementSize) {
            invalidate();
            return {};
        }
        return take(count * elementSize);
    }
    //! A count of elements that take at least one byte each, so that a damaged count fails here and not in an allocation.
    std::uint64_t readCount() {
        const std::uint64_t count = read<std::uint64_t>();
        if (count <= getRemaining())
            return count;
        invalidate();
        return 0;
    }
    std::span<const std::byte> take(const std::uint64_t size) {
        if (!valid || size > getRemaining()) {
            invalidate();
            return {};
        }
        const std::span<const std::byte> taken = bytes.subspan(position, size);
        position += size;
        return taken;
    }
    void align() {
        const std::uint64_t aligned = (position + arrayAlignment - 1) / arrayAlignment * arrayAlignment;
        if (aligned > bytes.size()) {
            invalidate();
            return;
        }
        position = aligned;
    }
    void invalidate() {
        valid = false;
        position = bytes.size();
    }
    bool isValid() const { return valid; }
    [[nodiscard]] std::expected<void, Core::Error> check(const std::string_view part) const {
        if (valid)
            return {};
        return Core::fail(Core::ErrorCode::InvalidFile, std::format("the {} runs past the end of its part of the scene file", part));
    }

private:
    std::uint64_t getRemaining() const { return bytes.size() - position; }

    std::span<const std::byte> bytes;
    std::uint64_t position = 0;
    bool valid = true;
};

}
