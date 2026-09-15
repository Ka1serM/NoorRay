#include "Scene/Import/AssetPath.h"

#include <utility>

namespace noorray
{
namespace
{
std::filesystem::path& assetDirectoryStorage()
{
    static std::filesystem::path directory;
    return directory;
}
}

void setAssetDirectory(std::filesystem::path directory)
{
    assetDirectoryStorage() = std::move(directory);
}

const std::filesystem::path& assetDirectory()
{
    return assetDirectoryStorage();
}

std::filesystem::path resolveAssetPath(const std::string& path)
{
    const std::filesystem::path direct(path);
    if (std::filesystem::exists(direct))
        return direct;
    const std::filesystem::path& directory = assetDirectory();
    if (!directory.empty()) {
        std::filesystem::path fallback = directory / path;
        if (std::filesystem::exists(fallback))
            return fallback;
    }
    return direct;
}

} // namespace noorray
