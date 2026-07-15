#include "Scene/Texture.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <utility>

#include <stb_image.h>

#include "IO/BitmapReader.h"
#include "Scene/SceneImporter.h"

namespace
{
struct StbiImageDeleter
{
    void operator()(void* pixels) const noexcept { stbi_image_free(pixels); }
};

float decodeByte(const uint8_t value, const TextureEncoding encoding)
{
    const float normalized = static_cast<float>(value) / 255.0f;
    if (encoding != TextureEncoding::Srgb8)
        return normalized;
    return normalized <= 0.04045f ? normalized / 12.92f
        : std::pow((normalized + 0.055f) / 1.055f, 2.4f);
}
}

Texture::Texture(const std::string& filepath, const TextureEncoding encoding)
{
    int channels{};
    std::string extension = std::filesystem::path(filepath).extension().string();
    std::ranges::transform(extension, extension.begin(), [](const unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    if (extension == ".exr") {
        const Bitmap bitmap = BitmapReader::read(filepath);
        width = static_cast<int>(bitmap.width());
        height = static_cast<int>(bitmap.height());
        pixels.assign(bitmap.rgba(), bitmap.rgba() + static_cast<size_t>(width) * height * 4);
    } else if (stbi_is_hdr(filepath.c_str())) {
        std::unique_ptr<float, StbiImageDeleter> source(
            stbi_loadf(filepath.c_str(), &width, &height, &channels, 4));
        if (!source || width <= 0 || height <= 0)
            throw std::runtime_error("Failed to load texture: " + filepath);
        pixels.assign(source.get(), source.get() + static_cast<size_t>(width) * height * 4);
    } else {
        std::unique_ptr<uint8_t, StbiImageDeleter> source(
            stbi_load(filepath.c_str(), &width, &height, &channels, 4));
        if (!source || width <= 0 || height <= 0)
            throw std::runtime_error("Failed to load texture: " + filepath);
        pixels.resize(static_cast<size_t>(width) * height * 4);
        for (size_t index = 0; index < pixels.size(); ++index)
            pixels[index] = decodeByte(source.get()[index], encoding);
    }
    name = SceneImporter::nameFromPath(filepath);
}

Texture::Texture(std::string textureName, const void* data,
    const int textureWidth, const int textureHeight, const TextureEncoding encoding)
    : name(std::move(textureName)), width(textureWidth), height(textureHeight)
{
    if (width <= 0 || height <= 0)
        throw std::runtime_error("Texture has invalid dimensions: " + name);
    pixels.resize(static_cast<size_t>(width) * height * 4);
    if (encoding == TextureEncoding::Float32)
        std::memcpy(pixels.data(), data, pixels.size() * sizeof(float));
    else {
        const auto* bytes = static_cast<const uint8_t*>(data);
        for (size_t index = 0; index < pixels.size(); ++index)
            pixels[index] = decodeByte(bytes[index], encoding);
    }
}
