#include "NoorRay/Import/PlyFile.h"

#include <algorithm>
#include <charconv>
#include <cstring>
#include <format>
#include <optional>

namespace NoorRay::Import {

namespace {

std::optional<PlyType> parseType(const std::string_view text) {
    if (text == "char" || text == "int8")
        return PlyType::Int8;
    if (text == "uchar" || text == "uint8")
        return PlyType::Uint8;
    if (text == "short" || text == "int16")
        return PlyType::Int16;
    if (text == "ushort" || text == "uint16")
        return PlyType::Uint16;
    if (text == "int" || text == "int32")
        return PlyType::Int32;
    if (text == "uint" || text == "uint32")
        return PlyType::Uint32;
    if (text == "float" || text == "float32")
        return PlyType::Float32;
    if (text == "double" || text == "float64")
        return PlyType::Float64;
    return std::nullopt;
}

std::vector<std::string_view> splitWords(const std::string_view line) {
    std::vector<std::string_view> words;
    for (std::size_t start = 0; start < line.size();) {
        const std::size_t space = std::min(line.find_first_of(" \t\r", start), line.size());
        if (space > start)
            words.push_back(line.substr(start, space - start));
        start = space + 1;
    }
    return words;
}

}

std::uint32_t PlyElement::findProperty(const std::string_view name) const {
    for (std::uint32_t index = 0; index < properties.size(); ++index)
        if (properties[index].name == name)
            return index;
    return PlyFile::missing;
}

std::uint32_t PlyElement::getPropertyOffset(const std::uint32_t property) const {
    std::uint32_t offset = 0;
    for (std::uint32_t index = 0; index < property; ++index)
        offset += PlyFile::getTypeSize(properties[index].type);
    return offset;
}

std::uint32_t PlyFile::getTypeSize(const PlyType type) {
    switch (type) {
    case PlyType::Int8:
    case PlyType::Uint8:
        return 1;
    case PlyType::Int16:
    case PlyType::Uint16:
        return 2;
    case PlyType::Int32:
    case PlyType::Uint32:
    case PlyType::Float32:
        return 4;
    case PlyType::Float64:
        return 8;
    }
    return 0;
}

double PlyFile::readValue(const std::byte* data, const PlyType type) {
    const auto load = [data]<class T>(T) {
        T value{};
        std::memcpy(&value, data, sizeof(T));
        return static_cast<double>(value);
    };
    switch (type) {
    case PlyType::Int8:
        return load(std::int8_t{});
    case PlyType::Uint8:
        return load(std::uint8_t{});
    case PlyType::Int16:
        return load(std::int16_t{});
    case PlyType::Uint16:
        return load(std::uint16_t{});
    case PlyType::Int32:
        return load(std::int32_t{});
    case PlyType::Uint32:
        return load(std::uint32_t{});
    case PlyType::Float32:
        return load(float{});
    case PlyType::Float64:
        return load(double{});
    }
    return 0.0;
}

std::expected<void, Core::Error> PlyFile::open(const std::span<const std::byte> data) {
    bytes = data;
    elements.clear();
    const std::string_view text(reinterpret_cast<const char*>(data.data()), std::min<std::size_t>(data.size(), 1 << 20));
    const std::size_t end = text.find("end_header");
    if (!text.starts_with("ply") || end == std::string_view::npos)
        return Core::fail(Core::ErrorCode::InvalidFile, "the file is not a PLY file");
    const std::size_t dataStart = text.find('\n', end);
    if (dataStart == std::string_view::npos)
        return Core::fail(Core::ErrorCode::InvalidFile, "the PLY header has no end");
    bool binary = false;
    std::size_t lineStart = 0;
    while (lineStart < end) {
        const std::size_t lineEnd = text.find('\n', lineStart);
        const std::vector<std::string_view> words = splitWords(text.substr(lineStart, lineEnd - lineStart));
        lineStart = lineEnd + 1;
        if (words.empty())
            continue;
        if (words[0] == "format")
            binary = words.size() > 1 && words[1] == "binary_little_endian";
        if (words[0] == "element" && words.size() == 3) {
            PlyElement element;
            element.name = std::string(words[1]);
            std::from_chars(words[2].data(), words[2].data() + words[2].size(), element.count);
            elements.push_back(std::move(element));
        }
        if (words[0] == "property" && !elements.empty()) {
            PlyElement& element = elements.back();
            if (words.size() == 5 && words[1] == "list") {
                const std::optional<PlyType> countType = parseType(words[2]);
                const std::optional<PlyType> type = parseType(words[3]);
                if (!countType || !type)
                    return Core::fail(Core::ErrorCode::InvalidFile, "the PLY file has an unknown property type");
                element.properties.push_back({std::string(words[4]), *type, true, *countType});
                element.hasLists = true;
            } else if (words.size() == 3) {
                const std::optional<PlyType> type = parseType(words[1]);
                if (!type)
                    return Core::fail(Core::ErrorCode::InvalidFile, "the PLY file has an unknown property type");
                element.properties.push_back({std::string(words[2]), *type});
                element.stride += getTypeSize(*type);
            }
        }
    }
    if (!binary)
        return Core::fail(Core::ErrorCode::Unsupported, "only binary little endian PLY files are supported");
    std::size_t offset = dataStart + 1;
    for (PlyElement& element : elements) {
        element.offset = offset;
        if (!element.hasLists) {
            offset += element.count * element.stride;
            continue;
        }
        for (std::size_t record = 0; record < element.count; ++record) {
            for (const PlyProperty& property : element.properties) {
                if (offset + 8 > bytes.size())
                    return Core::fail(Core::ErrorCode::InvalidFile, "the PLY file is shorter than its header says");
                if (!property.isList) {
                    offset += getTypeSize(property.type);
                    continue;
                }
                const std::size_t count = static_cast<std::size_t>(readValue(bytes.data() + offset, property.countType));
                offset += getTypeSize(property.countType) + count * getTypeSize(property.type);
            }
        }
    }
    if (offset > bytes.size())
        return Core::fail(Core::ErrorCode::InvalidFile, "the PLY file is shorter than its header says");
    return {};
}

const PlyElement* PlyFile::findElement(const std::string_view name) const {
    for (const PlyElement& element : elements)
        if (element.name == name)
            return &element;
    return nullptr;
}

std::vector<float> PlyFile::readFloats(const PlyElement& element, const std::span<const std::string_view> names, const float fallback) const {
    std::vector<float> result;
    readFloats(element, names, 0, element.count, result, fallback);
    return result;
}

void PlyFile::readFloats(const PlyElement& element, const std::span<const std::string_view> names, const std::size_t first, const std::size_t count,
                         std::vector<float>& result, const float fallback) const {
    result.assign(count * names.size(), fallback);
    if (element.hasLists || first + count > element.count)
        return;
    for (std::size_t column = 0; column < names.size(); ++column) {
        const std::uint32_t property = element.findProperty(names[column]);
        if (property == missing)
            continue;
        const std::uint32_t offset = element.getPropertyOffset(property);
        const PlyType type = element.properties[property].type;
        const std::byte* base = bytes.data() + element.offset + offset + first * element.stride;
        if (type == PlyType::Float32) {
            for (std::size_t record = 0; record < count; ++record)
                std::memcpy(&result[record * names.size() + column], base + record * element.stride, sizeof(float));
            continue;
        }
        for (std::size_t record = 0; record < count; ++record)
            result[record * names.size() + column] = static_cast<float>(readValue(base + record * element.stride, type));
    }
}

std::expected<std::vector<std::uint32_t>, Core::Error> PlyFile::readTriangles(const PlyElement& element) const {
    std::vector<std::uint32_t> triangles;
    std::size_t offset = element.offset;
    std::vector<std::uint32_t> polygon;
    for (std::size_t record = 0; record < element.count; ++record) {
        bool first = true;
        for (const PlyProperty& property : element.properties) {
            if (!property.isList) {
                offset += getTypeSize(property.type);
                continue;
            }
            const std::size_t count = static_cast<std::size_t>(readValue(bytes.data() + offset, property.countType));
            offset += getTypeSize(property.countType);
            if (first) {
                polygon.resize(count);
                for (std::size_t corner = 0; corner < count; ++corner)
                    polygon[corner] = static_cast<std::uint32_t>(readValue(bytes.data() + offset + corner * getTypeSize(property.type), property.type));
                for (std::size_t corner = 2; corner < count; ++corner)
                    triangles.insert(triangles.end(), {polygon[0], polygon[corner - 1], polygon[corner]});
                first = false;
            }
            offset += count * getTypeSize(property.type);
        }
    }
    return triangles;
}

}
