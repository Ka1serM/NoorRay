#include "DiffRenderer/Training/Json.h"

#include <cmath>
#include <format>

namespace DiffRenderer::Training {

std::string toJson(const double value) {
    if (!std::isfinite(value))
        return "null";
    return std::format("{:.9g}", value);
}

std::string quote(const std::string_view text) {
    std::string result = "\"";
    for (const char character : text) {
        if (character == '"' || character == '\\')
            result += '\\';
        result += character;
    }
    return result + "\"";
}

}
