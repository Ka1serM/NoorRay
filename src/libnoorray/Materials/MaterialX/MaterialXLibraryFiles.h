#pragma once

#include <string_view>

namespace nr::materialx
{

// The content of a file of MaterialX's libraries tree, embedded into the
// library, by its path relative to the "libraries" folder. Throws when the
// file is not embedded.
std::string_view embeddedLibraryFile(std::string_view path);

} // namespace nr::materialx
