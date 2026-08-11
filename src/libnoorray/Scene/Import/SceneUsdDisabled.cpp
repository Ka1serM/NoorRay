// USD-free implementation of the SceneUsd entry points, compiled in place of
// SceneUsd.cpp when NR_ENABLE_USD=OFF.
//
// Recognising a USD path stays exact so callers keep reporting "this is a USD
// file this build cannot read" rather than the misleading "unknown format".
// Only the reader and writer are unavailable, and they say so.

#include "Scene/Import/SceneUsd.h"

#include <algorithm>
#include <filesystem>
#include <stdexcept>

namespace nr::sceneio {

bool isUsdFile(const std::string& filepath)
{
    std::string extension = std::filesystem::path(filepath).extension().string();
    std::ranges::transform(extension, extension.begin(), [](const unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return extension == ".usd" || extension == ".usda" || extension == ".usdc"
        || extension == ".usdz";
}

void readUsd(Scene&, const std::string& filepath)
{
    throw std::runtime_error(
        "Cannot read '" + filepath + "': this build of NoorRay was configured with "
        "NR_ENABLE_USD=OFF, so USD scene import is unavailable.");
}

void writeUsd(const Scene&, const std::string& filepath)
{
    throw std::runtime_error(
        "Cannot write '" + filepath + "': this build of NoorRay was configured with "
        "NR_ENABLE_USD=OFF, so USD scene export is unavailable.");
}

}  // namespace nr::sceneio
