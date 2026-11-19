#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "NoorRay/Core/Error.h"

namespace NoorRay::Import {

enum class PlyType : std::uint8_t { Int8, Uint8, Int16, Uint16, Int32, Uint32, Float32, Float64 };

struct PlyProperty {
    std::string name;
    PlyType type = PlyType::Float32;
    bool isList = false;
    PlyType countType = PlyType::Uint8;
};

struct PlyElement {
    std::string name;
    std::size_t count = 0;
    std::vector<PlyProperty> properties;
    //! Byte offset of the element data; with no list properties also the fixed record size in stride.
    std::size_t offset = 0;
    std::uint32_t stride = 0;
    bool hasLists = false;

    std::uint32_t findProperty(std::string_view name) const;
    std::uint32_t getPropertyOffset(std::uint32_t property) const;
};

//! A binary little endian PLY file read in place.
class PlyFile {
public:
    static constexpr std::uint32_t missing = UINT32_MAX;

    [[nodiscard]] std::expected<void, Core::Error> open(std::span<const std::byte> bytes);
    const PlyElement* findElement(std::string_view name) const;
    //! Reads fixed-size properties of an element as floats, one column per name; missing names read as fallback.
    std::vector<float> readFloats(const PlyElement& element, std::span<const std::string_view> names, float fallback = 0.0f) const;
    //! The same for the records [first, first + count), into result, so a large element can be read in batches.
    void readFloats(const PlyElement& element, std::span<const std::string_view> names, std::size_t first, std::size_t count, std::vector<float>& result,
                    float fallback = 0.0f) const;
    //! Triangulates the first list property of an element as fans.
    [[nodiscard]] std::expected<std::vector<std::uint32_t>, Core::Error> readTriangles(const PlyElement& element) const;

    static std::uint32_t getTypeSize(PlyType type);
    static double readValue(const std::byte* data, PlyType type);

private:
    std::span<const std::byte> bytes;
    std::vector<PlyElement> elements;
};

}
