#include "NoorRay/Import/PbrtParser.h"

#include <algorithm>
#include <charconv>
#include <format>
#include <fstream>
#include <iterator>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace NoorRay::Import::Pbrt {

namespace {

struct Token {
    std::string text;
    bool quoted = false;
    int line = 1;
};

std::unexpected<Core::Error> failAt(const std::filesystem::path& source, const int line, const std::string_view message) {
    return Core::fail(Core::ErrorCode::InvalidFile, std::format("{}:{}: {}", source.string(), line, message));
}

std::expected<std::vector<Token>, Core::Error> tokenize(const std::string_view input, const std::filesystem::path& source) {
    std::vector<Token> result;
    std::size_t position = 0;
    int line = 1;
    while (position < input.size()) {
        const char character = input[position];
        if (character == '\n') {
            ++line;
            ++position;
            continue;
        }
        if (static_cast<unsigned char>(character) <= ' ') {
            ++position;
            continue;
        }
        if (character == '#') {
            while (position < input.size() && input[position] != '\n')
                ++position;
            continue;
        }
        if (character == '[' || character == ']') {
            result.push_back({std::string(1, character), false, line});
            ++position;
            continue;
        }
        if (character == '"') {
            const int tokenLine = line;
            ++position;
            std::string value;
            bool closed = false;
            while (position < input.size()) {
                const char next = input[position++];
                if (next == '"') {
                    closed = true;
                    break;
                }
                if (next == '\n')
                    ++line;
                if (next != '\\' || position >= input.size()) {
                    value.push_back(next);
                    continue;
                }
                const char escaped = input[position++];
                value.push_back(escaped == 'n' ? '\n' : escaped == 't' ? '\t' : escaped == 'r' ? '\r' : escaped);
            }
            if (!closed)
                return failAt(source, tokenLine, "unterminated string");
            result.push_back({std::move(value), true, tokenLine});
            continue;
        }
        const std::size_t begin = position;
        while (position < input.size() && static_cast<unsigned char>(input[position]) > ' ' && input[position] != '[' && input[position] != ']'
               && input[position] != '#')
            ++position;
        result.push_back({std::string(input.substr(begin, position - begin)), false, line});
    }
    return result;
}

bool isParameterDeclaration(const Token& token) {
    if (!token.quoted)
        return false;
    const std::size_t space = token.text.find_first_of(" \t");
    return space != std::string::npos && token.text.find_first_not_of(" \t", space) != std::string::npos;
}

const std::unordered_map<std::string_view, int> argumentCounts{
    {"Accelerator", 1},
    {"ActiveTransform", 1},
    {"AreaLightSource", 1},
    {"Attribute", 1},
    {"Camera", 1},
    {"ColorSpace", 1},
    {"CoordinateSystem", 1},
    {"CoordSysTransform", 1},
    {"Film", 1},
    {"Include", 1},
    {"Import", 1},
    {"Integrator", 1},
    {"LightSource", 1},
    {"MakeNamedMaterial", 1},
    {"MakeNamedMedium", 1},
    {"Material", 1},
    {"NamedMaterial", 1},
    {"ObjectBegin", 1},
    {"ObjectInstance", 1},
    {"PixelFilter", 1},
    {"Sampler", 1},
    {"Shape", 1},
    {"Texture", 3},
    {"Option", 2},
    {"MediumInterface", 2},
    {"Translate", 3},
    {"Scale", 3},
    {"Rotate", 4},
    {"LookAt", 9},
    {"TransformTimes", 2},
};

const std::unordered_set<std::string_view> noArgumentCommands{"AttributeBegin", "AttributeEnd", "Identity",   "ObjectEnd", "ReverseOrientation",
                                                              "TransformBegin", "TransformEnd", "WorldBegin", "WorldEnd"};

//! Directives that only configure pbrt's own renderer are parsed for correct tokenization and then dropped.
const std::unordered_set<std::string_view> supportedCommands{"AreaLightSource",
                                                             "AttributeBegin",
                                                             "AttributeEnd",
                                                             "Camera",
                                                             "ConcatTransform",
                                                             "CoordinateSystem",
                                                             "CoordSysTransform",
                                                             "Film",
                                                             "Identity",
                                                             "Include",
                                                             "Import",
                                                             "LightSource",
                                                             "LookAt",
                                                             "MakeNamedMaterial",
                                                             "Material",
                                                             "NamedMaterial",
                                                             "ObjectBegin",
                                                             "ObjectEnd",
                                                             "ObjectInstance",
                                                             "ReverseOrientation",
                                                             "Rotate",
                                                             "Scale",
                                                             "Shape",
                                                             "Texture",
                                                             "Transform",
                                                             "TransformBegin",
                                                             "TransformEnd",
                                                             "Translate",
                                                             "WorldBegin",
                                                             "WorldEnd"};

std::expected<std::vector<Command>, Core::Error> parseTokens(const std::vector<Token>& tokens, const std::filesystem::path& source) {
    std::vector<Command> commands;
    std::size_t position = 0;
    while (position < tokens.size()) {
        const Token& directive = tokens[position++];
        Command command{directive.text, {}, {}, source, directive.line};
        if (directive.quoted || directive.text == "[" || directive.text == "]")
            return failAt(source, directive.line, std::format("expected a directive, got '{}'", directive.text));
        const auto take = [&](std::string& out) {
            if (position >= tokens.size())
                return false;
            out = tokens[position++].text;
            return true;
        };
        std::string value;
        if (directive.text == "Transform" || directive.text == "ConcatTransform") {
            if (!take(value) || value != "[")
                return failAt(source, directive.line, std::format("expected '[' after {}", directive.text));
            for (int index = 0; index < 16; ++index) {
                if (!take(value))
                    return failAt(source, directive.line, "premature end of file");
                command.arguments.push_back(value);
            }
            if (!take(value) || value != "]")
                return failAt(source, directive.line, "expected 16 matrix values");
        } else if (const auto found = argumentCounts.find(directive.text); found != argumentCounts.end()) {
            int count = found->second;
            if (directive.text == "MediumInterface" && (position + 1 >= tokens.size() || !tokens[position + 1].quoted))
                count = 1;
            for (int index = 0; index < count; ++index) {
                if (!take(value))
                    return failAt(source, directive.line, "premature end of file");
                command.arguments.push_back(value);
            }
        } else if (!noArgumentCommands.contains(directive.text)) {
            return failAt(source, directive.line, std::format("unknown directive '{}'", directive.text));
        }
        while (position < tokens.size() && isParameterDeclaration(tokens[position])) {
            const Token& declaration = tokens[position++];
            const std::size_t space = declaration.text.find_first_of(" \t");
            const std::size_t nameStart = declaration.text.find_first_not_of(" \t", space);
            Parameter parameter{declaration.text.substr(0, space), declaration.text.substr(nameStart), {}};
            if (!take(value))
                return failAt(source, declaration.line, "a parameter has no value");
            if (value == "[") {
                while (true) {
                    if (!take(value))
                        return failAt(source, declaration.line, "unterminated parameter list");
                    if (value == "]")
                        break;
                    parameter.values.push_back(value);
                }
            } else {
                parameter.values.push_back(value);
            }
            command.parameters.push_back(std::move(parameter));
        }
        if (supportedCommands.contains(command.name))
            commands.push_back(std::move(command));
    }
    return commands;
}

std::expected<std::vector<Command>, Core::Error> parseRecursive(const std::filesystem::path& path, std::vector<std::filesystem::path>& stack) {
    const std::filesystem::path canonical = std::filesystem::weakly_canonical(path);
    if (std::ranges::find(stack, canonical) != stack.end())
        return Core::fail(Core::ErrorCode::InvalidFile, std::format("cyclic pbrt Include: {}", canonical.string()));
    std::ifstream file(canonical, std::ios::binary);
    if (!file)
        return Core::fail(Core::ErrorCode::Io, std::format("cannot open pbrt scene '{}'", path.string()));
    std::ostringstream contents;
    contents << file.rdbuf();
    const std::string text = contents.str();
    std::expected<std::vector<Token>, Core::Error> tokens = tokenize(text, canonical);
    if (!tokens)
        return std::unexpected(std::move(tokens.error()));
    std::expected<std::vector<Command>, Core::Error> parsed = parseTokens(*tokens, canonical);
    if (!parsed)
        return parsed;
    stack.push_back(canonical);
    std::vector<Command> expanded;
    for (Command& command : *parsed) {
        if (command.name != "Include" && command.name != "Import") {
            expanded.push_back(std::move(command));
            continue;
        }
        std::expected<std::vector<Command>, Core::Error> nested = parseRecursive(canonical.parent_path() / command.arguments.front(), stack);
        if (!nested)
            return nested;
        expanded.insert(expanded.end(), std::make_move_iterator(nested->begin()), std::make_move_iterator(nested->end()));
    }
    stack.pop_back();
    return expanded;
}

template <class T>
std::optional<T> parseNumber(const std::string& text) {
    T result{};
    const char* end = text.data() + text.size();
    const auto parsed = std::from_chars(text.data(), end, result);
    if (parsed.ec != std::errc{} || parsed.ptr != end)
        return std::nullopt;
    return result;
}

template <class T>
std::vector<T> parseNumbers(const std::vector<std::string>& values) {
    std::vector<T> result;
    result.reserve(values.size());
    for (const std::string& value : values) {
        const std::optional<T> number = parseNumber<T>(value);
        if (!number)
            return {};
        result.push_back(*number);
    }
    return result;
}

}

std::optional<float> Parameter::getFloat() const {
    return values.empty() ? std::nullopt : parseNumber<float>(values.front());
}
std::optional<int> Parameter::getInt() const {
    return values.empty() ? std::nullopt : parseNumber<int>(values.front());
}
std::optional<std::string> Parameter::getString() const {
    return values.empty() ? std::nullopt : std::optional(values.front());
}
std::vector<float> Parameter::getFloats() const {
    return parseNumbers<float>(values);
}
std::vector<int> Parameter::getInts() const {
    return parseNumbers<int>(values);
}

const Parameter* Command::find(const std::string_view name) const {
    for (const Parameter& parameter : parameters)
        if (parameter.name == name)
            return &parameter;
    return nullptr;
}

float Command::getFloat(const std::string_view name, const float fallback) const {
    const Parameter* parameter = find(name);
    return parameter ? parameter->getFloat().value_or(fallback) : fallback;
}

std::string Command::getString(const std::string_view name, std::string fallback) const {
    const Parameter* parameter = find(name);
    return parameter ? parameter->getString().value_or(std::move(fallback)) : std::move(fallback);
}

std::expected<std::vector<Command>, Core::Error> Parser::parseFile(const std::filesystem::path& path) {
    std::vector<std::filesystem::path> stack;
    return parseRecursive(path, stack);
}

std::expected<std::vector<Command>, Core::Error> Parser::parseString(const std::string_view text, const std::filesystem::path& source) {
    std::expected<std::vector<Token>, Core::Error> tokens = tokenize(text, source);
    if (!tokens)
        return std::unexpected(std::move(tokens.error()));
    return parseTokens(*tokens, source);
}

}
