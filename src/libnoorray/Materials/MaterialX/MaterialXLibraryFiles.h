#pragma once

#include <span>
#include <string_view>

namespace nr::materialx
{

// One file of MaterialX's libraries tree, relative to its "libraries" folder.
struct EmbeddedLibraryFile
{
    std::string_view path;
    std::span<const unsigned char> content;
};

// Generated at configure time from external/materialx/libraries.
extern const std::span<const EmbeddedLibraryFile> embeddedLibraryFiles;

// The content of an embedded file, by its path relative to "libraries".
// Throws when the file is not embedded.
std::string_view embeddedLibraryFile(std::string_view path);

} // namespace nr::materialx
