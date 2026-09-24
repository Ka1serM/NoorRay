#include "IO/TextureReader.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <stb_image.h>

#include "IO/BitmapReader.h"

namespace
{
struct StbiImageDeleter
{
    void operator()(void* pixels) const noexcept { stbi_image_free(pixels); }
};

Texture decode(const std::string& path, const std::string& name, const TextureEncoding encoding)
{
    std::string extension = std::filesystem::path(path).extension().string();
    std::ranges::transform(extension, extension.begin(), [](const unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    if (extension == ".exr") {
        const Bitmap bitmap = BitmapReader::read(path);
        const int width = static_cast<int>(bitmap.width());
        const int height = static_cast<int>(bitmap.height());
        return Texture(name, std::vector<float>(bitmap.rgba(),
            bitmap.rgba() + static_cast<size_t>(width) * height * 4), width, height);
    }

    int width{};
    int height{};
    int channels{};
    if (stbi_is_hdr(path.c_str())) {
        std::unique_ptr<float, StbiImageDeleter> source(
            stbi_loadf(path.c_str(), &width, &height, &channels, 4));
        if (!source || width <= 0 || height <= 0)
            throw std::runtime_error("Failed to load texture: " + path);
        return Texture(name, std::vector<float>(source.get(),
            source.get() + static_cast<size_t>(width) * height * 4), width, height);
    }
    std::unique_ptr<uint8_t, StbiImageDeleter> source(
        stbi_load(path.c_str(), &width, &height, &channels, 4));
    if (!source || width <= 0 || height <= 0)
        throw std::runtime_error("Failed to load texture: " + path);
    return Texture(name, std::vector<uint8_t>(source.get(),
        source.get() + static_cast<size_t>(width) * height * 4), width, height, encoding);
}
} // namespace

Texture TextureReader::read(const std::string& path, const TextureEncoding encoding)
{
    Texture texture = decode(path, std::filesystem::path(path).stem().string(), encoding);
    // Scene deduplicates file textures by their source path, not their name.
    texture.path = path;
    return texture;
}
