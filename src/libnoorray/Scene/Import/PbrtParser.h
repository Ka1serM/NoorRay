#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace nr::pbrt {

struct Parameter {
    std::string type;
    std::string name;
    std::vector<std::string> values;

    std::optional<float> floatValue() const;
    std::optional<int> intValue() const;
    std::optional<std::string> stringValue() const;
    std::vector<float> floatValues() const;
    std::vector<int> intValues() const;
};

struct Command {
    std::string name;
    std::vector<std::string> arguments;
    std::vector<Parameter> parameters;
    std::filesystem::path source;
    int line{1};

    const Parameter* find(std::string_view name) const;
};

// Parses the PBRT-v4 scene-description grammar. Include directives are expanded
// in place and paths are resolved relative to the file containing the directive.
std::vector<Command> parseFile(const std::filesystem::path& path);
std::vector<Command> parseString(std::string_view text,
    const std::filesystem::path& source = "<string>");

} // namespace nr::pbrt
