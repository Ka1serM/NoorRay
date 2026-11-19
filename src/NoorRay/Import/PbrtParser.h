#pragma once

#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "NoorRay/Core/Error.h"

namespace NoorRay::Import::Pbrt {

struct Parameter {
    std::string type;
    std::string name;
    std::vector<std::string> values;

    std::optional<float> getFloat() const;
    std::optional<int> getInt() const;
    std::optional<std::string> getString() const;
    std::vector<float> getFloats() const;
    std::vector<int> getInts() const;
};

struct Command {
    std::string name;
    std::vector<std::string> arguments;
    std::vector<Parameter> parameters;
    std::filesystem::path source;
    int line = 1;

    const Parameter* find(std::string_view name) const;
    float getFloat(std::string_view name, float fallback) const;
    std::string getString(std::string_view name, std::string fallback = {}) const;
};

//! Parses the PBRT-v4 scene grammar. Include and Import are expanded in place, relative to the including file.
class Parser {
public:
    [[nodiscard]] static std::expected<std::vector<Command>, Core::Error> parseFile(const std::filesystem::path& path);
    [[nodiscard]] static std::expected<std::vector<Command>, Core::Error> parseString(std::string_view text, const std::filesystem::path& source = "<string>");
};

}
