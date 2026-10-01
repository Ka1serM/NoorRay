#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cmrc/cmrc.hpp>

CMRC_DECLARE(noorray_shaders);

// Shaders and shader sources compiled into the library, by their path under
// the shader build directory.

// CMakeRC stores files as unaligned chars while Vulkan reads SPIR-V as 32-bit
// words, so the words are copied out.
inline std::vector<std::uint32_t> embeddedSpirv(const std::string& path)
{
    const cmrc::file file = cmrc::noorray_shaders::get_filesystem().open(path);
    if (file.size() % sizeof(std::uint32_t) != 0)
        throw std::runtime_error("embedded SPIR-V is not a whole number of words: " + path);
    std::vector<std::uint32_t> words(file.size() / sizeof(std::uint32_t));
    std::memcpy(words.data(), file.begin(), file.size());
    return words;
}

// CMakeRC terminates every file with a NUL, so sources are C strings.
inline const char* embeddedShaderSource(const std::string& path)
{
    return cmrc::noorray_shaders::get_filesystem().open(path).begin();
}
