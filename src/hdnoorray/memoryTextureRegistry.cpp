#include "memoryTextureRegistry.h"

#include <limits>
#include <mutex>
#include <unordered_map>

namespace
{
std::mutex registryMutex;
std::unordered_map<std::string, std::shared_ptr<const hdnoorray::MemoryTexturePixels>> registry;

bool validImageSize(const size_t count, const int width, const int height)
{
    return width > 0 && height > 0
        && static_cast<size_t>(width) <= std::numeric_limits<size_t>::max()
            / static_cast<size_t>(height) / 4
        && count == static_cast<size_t>(width) * static_cast<size_t>(height) * 4;
}
} // namespace

extern "C" int HdNoorRayRegisterMemoryTextureF32(const char* uri,
    const float* rgba, const size_t floatCount, const int width, const int height)
{
    if (!uri || !*uri || !rgba || !validImageSize(floatCount, width, height))
        return 0;
    try {
        auto pixels = std::make_shared<std::vector<float>>(rgba, rgba + floatCount);
        auto entry = std::make_shared<hdnoorray::MemoryTexturePixels>();
        entry->rgba = std::move(pixels);
        entry->width = width;
        entry->height = height;
        std::scoped_lock lock(registryMutex);
        registry.insert_or_assign(uri, std::move(entry));
        return 1;
    }
    catch (...) {
        return 0;
    }
}

extern "C" void HdNoorRayUnregisterMemoryTexture(const char* uri)
{
    if (!uri)
        return;
    std::scoped_lock lock(registryMutex);
    registry.erase(uri);
}

namespace hdnoorray
{
std::shared_ptr<const MemoryTexturePixels> findMemoryTexture(
    const std::string& uri)
{
    std::scoped_lock lock(registryMutex);
    const auto found = registry.find(uri);
    return found == registry.end() ? nullptr : found->second;
}
} // namespace hdnoorray
