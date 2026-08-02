#include "Scene/Import/PbrtParser.h"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace nr::pbrt {
namespace {

struct Token {
    std::string text;
    bool quoted{};
    int line{1};
};

std::vector<Token> tokenize(std::string_view input, const std::filesystem::path& source)
{
    std::vector<Token> result;
    size_t i = 0;
    int line = 1;
    while (i < input.size()) {
        const char c = input[i];
        if (c == '\n') { ++line; ++i; continue; }
        if (static_cast<unsigned char>(c) <= ' ') { ++i; continue; }
        if (c == '#') {
            while (i < input.size() && input[i] != '\n') ++i;
            continue;
        }
        if (c == '[' || c == ']') {
            result.push_back({std::string(1, c), false, line});
            ++i;
            continue;
        }
        if (c == '"') {
            const int tokenLine = line;
            ++i;
            std::string value;
            bool closed = false;
            while (i < input.size()) {
                const char ch = input[i++];
                if (ch == '"') { closed = true; break; }
                if (ch == '\n') ++line;
                if (ch == '\\' && i < input.size()) {
                    const char escaped = input[i++];
                    switch (escaped) {
                    case 'n': value.push_back('\n'); break;
                    case 'r': value.push_back('\r'); break;
                    case 't': value.push_back('\t'); break;
                    default: value.push_back(escaped); break;
                    }
                } else {
                    value.push_back(ch);
                }
            }
            if (!closed)
                throw std::runtime_error(source.string() + ":" + std::to_string(tokenLine)
                    + ": unterminated string");
            result.push_back({std::move(value), true, tokenLine});
            continue;
        }
        const int tokenLine = line;
        const size_t begin = i;
        while (i < input.size() && static_cast<unsigned char>(input[i]) > ' '
            && input[i] != '[' && input[i] != ']' && input[i] != '#')
            ++i;
        result.push_back({std::string(input.substr(begin, i - begin)), false, tokenLine});
    }
    return result;
}

bool isParameterDeclaration(const Token& token)
{
    if (!token.quoted) return false;
    const size_t space = token.text.find_first_of(" \t");
    return space != std::string::npos && token.text.find_first_not_of(" \t", space) != std::string::npos;
}

std::pair<std::string, std::string> splitDeclaration(const Token& token)
{
    const size_t space = token.text.find_first_of(" \t");
    const size_t name = token.text.find_first_not_of(" \t", space);
    return {token.text.substr(0, space), token.text.substr(name)};
}

const std::unordered_map<std::string, int>& argumentCounts()
{
    static const std::unordered_map<std::string, int> counts{
        {"Accelerator", 1}, {"ActiveTransform", 1}, {"AreaLightSource", 1},
        {"Attribute", 1}, {"Camera", 1}, {"ColorSpace", 1},
        {"CoordinateSystem", 1}, {"CoordSysTransform", 1}, {"Film", 1},
        {"Include", 1}, {"Import", 1}, {"Integrator", 1}, {"LightSource", 1},
        {"MakeNamedMaterial", 1}, {"MakeNamedMedium", 1}, {"Material", 1},
        {"NamedMaterial", 1}, {"ObjectBegin", 1}, {"ObjectInstance", 1},
        {"PixelFilter", 1}, {"Sampler", 1}, {"Shape", 1},
        {"Texture", 3}, {"Option", 2}, {"MediumInterface", 2},
        {"Translate", 3}, {"Scale", 3}, {"Rotate", 4}, {"LookAt", 9},
        {"TransformTimes", 2},
    };
    return counts;
}

const std::unordered_set<std::string>& noArgumentCommands()
{
    static const std::unordered_set<std::string> names{
        "AttributeBegin", "AttributeEnd", "Identity", "ObjectEnd",
        "ReverseOrientation", "TransformBegin", "TransformEnd", "WorldBegin", "WorldEnd"
    };
    return names;
}

bool isSupportedCommand(const std::string& name)
{
    static const std::unordered_set<std::string> supported{
        "AreaLightSource", "AttributeBegin", "AttributeEnd", "Camera",
        "ConcatTransform", "CoordinateSystem", "CoordSysTransform", "Film",
        "Identity", "Include", "Import", "LightSource", "LookAt",
        "MakeNamedMaterial", "Material", "NamedMaterial", "ObjectBegin",
        "ObjectEnd", "ObjectInstance", "ReverseOrientation", "Rotate", "Scale",
        "Shape", "Texture", "Transform", "TransformBegin", "TransformEnd",
        "Translate", "WorldBegin", "WorldEnd"
    };
    return supported.contains(name);
}

std::vector<Command> parseTokens(const std::vector<Token>& tokens,
    const std::filesystem::path& source)
{
    std::vector<Command> commands;
    size_t pos = 0;
    auto require = [&]() -> const Token& {
        if (pos >= tokens.size())
            throw std::runtime_error(source.string() + ": premature end of file");
        return tokens[pos++];
    };
    while (pos < tokens.size()) {
        const Token directive = require();
        Command command{directive.text, {}, {}, source, directive.line};
        if (directive.quoted || directive.text == "[" || directive.text == "]")
            throw std::runtime_error(source.string() + ":" + std::to_string(directive.line)
                + ": expected a directive, got '" + directive.text + "'");

        if (directive.text == "Transform" || directive.text == "ConcatTransform") {
            if (require().text != "[")
                throw std::runtime_error(source.string() + ":" + std::to_string(directive.line)
                    + ": expected '[' after " + directive.text);
            for (int i = 0; i < 16; ++i) command.arguments.push_back(require().text);
            if (require().text != "]")
                throw std::runtime_error(source.string() + ":" + std::to_string(directive.line)
                    + ": expected 16 matrix values");
        } else if (const auto it = argumentCounts().find(directive.text); it != argumentCounts().end()) {
            int count = it->second;
            // PBRT allows MediumInterface with one name; don't consume the next directive.
            if (directive.text == "MediumInterface" && (pos + 1 >= tokens.size() || !tokens[pos + 1].quoted))
                count = 1;
            for (int i = 0; i < count; ++i) command.arguments.push_back(require().text);
        } else if (!noArgumentCommands().contains(directive.text)) {
            throw std::runtime_error(source.string() + ":" + std::to_string(directive.line)
                + ": unknown directive '" + directive.text + "'");
        }

        while (pos < tokens.size() && isParameterDeclaration(tokens[pos])) {
            const Token declaration = require();
            auto [type, name] = splitDeclaration(declaration);
            Parameter parameter{std::move(type), std::move(name), {}};
            const Token value = require();
            if (value.text == "[") {
                while (true) {
                    const Token item = require();
                    if (item.text == "]") break;
                    parameter.values.push_back(item.text);
                }
            } else {
                parameter.values.push_back(value.text);
            }
            command.parameters.push_back(std::move(parameter));
        }
        // Keep the parse result aligned with NoorRay's feature set. Directives
        // that only configure PBRT's own renderer (integrators, samplers,
        // filters, media, etc.) are consumed for correct tokenization but do
        // not become importer commands.
        if (isSupportedCommand(command.name))
            commands.push_back(std::move(command));
    }
    return commands;
}

std::vector<Command> parseFileImpl(const std::filesystem::path& path,
    std::vector<std::filesystem::path>& includeStack)
{
    const std::filesystem::path canonical = std::filesystem::weakly_canonical(path);
    if (std::ranges::find(includeStack, canonical) != includeStack.end())
        throw std::runtime_error("cyclic PBRT Include: " + canonical.string());
    std::ifstream file(canonical, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open PBRT scene: " + path.string());
    std::ostringstream contents;
    contents << file.rdbuf();
    includeStack.push_back(canonical);
    auto parsed = parseTokens(tokenize(contents.str(), canonical), canonical);
    std::vector<Command> expanded;
    for (Command& command : parsed) {
        if (command.name != "Include" && command.name != "Import") {
            expanded.push_back(std::move(command));
            continue;
        }
        const std::filesystem::path included = canonical.parent_path() / command.arguments.front();
        auto nested = parseFileImpl(included, includeStack);
        expanded.insert(expanded.end(), std::make_move_iterator(nested.begin()),
            std::make_move_iterator(nested.end()));
    }
    includeStack.pop_back();
    return expanded;
}

template <typename T>
std::optional<T> parseNumber(const std::string& text)
{
    T result{};
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto parsed = std::from_chars(begin, end, result);
    if (parsed.ec != std::errc{} || parsed.ptr != end) return std::nullopt;
    return result;
}

} // namespace

std::optional<float> Parameter::floatValue() const
{
    return values.empty() ? std::nullopt : parseNumber<float>(values.front());
}

std::optional<int> Parameter::intValue() const
{
    return values.empty() ? std::nullopt : parseNumber<int>(values.front());
}

std::optional<std::string> Parameter::stringValue() const
{
    return values.empty() ? std::nullopt : std::optional(values.front());
}

std::vector<float> Parameter::floatValues() const
{
    std::vector<float> result;
    result.reserve(values.size());
    for (const std::string& value : values) {
        const auto number = parseNumber<float>(value);
        if (!number) return {};
        result.push_back(*number);
    }
    return result;
}

std::vector<int> Parameter::intValues() const
{
    std::vector<int> result;
    result.reserve(values.size());
    for (const std::string& value : values) {
        const auto number = parseNumber<int>(value);
        if (!number) return {};
        result.push_back(*number);
    }
    return result;
}

const Parameter* Command::find(const std::string_view name) const
{
    for (const Parameter& parameter : parameters)
        if (parameter.name == name) return &parameter;
    return nullptr;
}

std::vector<Command> parseFile(const std::filesystem::path& path)
{
    std::vector<std::filesystem::path> stack;
    return parseFileImpl(path, stack);
}

std::vector<Command> parseString(const std::string_view text, const std::filesystem::path& source)
{
    return parseTokens(tokenize(text, source), source);
}

} // namespace nr::pbrt
