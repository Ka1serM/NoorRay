#pragma once

#include <string>
#include <string_view>

namespace DiffRenderer::Training {

//! A JSON number, or null for a value that is not finite.
std::string toJson(double value);
std::string quote(std::string_view text);

}
