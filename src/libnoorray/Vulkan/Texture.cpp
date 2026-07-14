#include "Texture.h"
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <stb_image.h>
#include <string>
#include <cstring>

#include "IO/BitmapReader.h"
#include "Scene/SceneImporter.h"

namespace
{
struct StbiImageDeleter
{
    void operator()(void* pixels) const noexcept { stbi_image_free(pixels); }
};
}

Texture::Texture(Context& context, const std::string& filepath, vk::Format format)
{
    (void)context;
    int channels = 0;
    std::string extension = std::filesystem::path(filepath).extension().string();
    std::ranges::transform(extension, extension.begin(), [](const unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    if (extension == ".exr")
    {
        const Bitmap bitmap = BitmapReader::read(filepath);
        width = static_cast<int>(bitmap.width());
        height = static_cast<int>(bitmap.height());
        pixels.assign(bitmap.rgba(), bitmap.rgba() + static_cast<size_t>(width) * height * 4);
    }
    else if (stbi_is_hdr(filepath.c_str()))
    {
        std::unique_ptr<float, StbiImageDeleter> rawPixels(
            stbi_loadf(filepath.c_str(), &width, &height, &channels, 4));
        if (!rawPixels || width <= 0 || height <= 0)
            throw std::runtime_error("Failed to load texture: " + filepath);
        pixels.assign(rawPixels.get(), rawPixels.get() + static_cast<size_t>(width) * height * 4);
    }
    else
    {
        std::unique_ptr<uint8_t, StbiImageDeleter> rawPixels(
            stbi_load(filepath.c_str(), &width, &height, &channels, 4));
        if (!rawPixels || width <= 0 || height <= 0)
            throw std::runtime_error("Failed to load texture: " + filepath);
        const size_t pixelCount = static_cast<size_t>(width) * height;
        pixels.resize(pixelCount * 4);
        for (size_t index = 0; index < pixels.size(); ++index)
        {
            const float c = static_cast<float>(rawPixels.get()[index]) / 255.0f;
            if (format == vk::Format::eR8G8B8A8Srgb)
                pixels[index] = c <= 0.04045f ? c / 12.92f
                    : powf((c + 0.055f) / 1.055f, 2.4f);
            else
                pixels[index] = c;
        }
    }
    name = SceneImporter::nameFromPath(filepath);
}

Texture::Texture(Context& context, const std::string& textureName, const void* data, const int textureWidth, const int textureHeight, const vk::Format format)
    : name(textureName), width(textureWidth), height(textureHeight)
{
    (void)context;
    if (width <= 0 || height <= 0)
        throw std::runtime_error("Texture constructor (raw data): Invalid dimensions provided (W=" + std::to_string(width) + ", H=" + std::to_string(height) + "). Name: " + name);
    const size_t pixelCount = static_cast<size_t>(width) * height;
    pixels.resize(pixelCount * 4);
    if (format == vk::Format::eR32G32B32A32Sfloat)
        std::memcpy(pixels.data(), data, pixels.size() * sizeof(float));
    else if (format == vk::Format::eR8G8B8A8Unorm || format == vk::Format::eR8G8B8A8Srgb)
    {
        const auto* bytes = static_cast<const uint8_t*>(data);
        for (size_t index = 0; index < pixels.size(); ++index)
        {
            const float c = static_cast<float>(bytes[index]) / 255.0f;
            if (format == vk::Format::eR8G8B8A8Srgb)
                pixels[index] = c <= 0.04045f ? c / 12.92f
                    : powf((c + 0.055f) / 1.055f, 2.4f);
            else
                pixels[index] = c;
        }
    }
    else
        throw std::runtime_error("Unsupported CPU texture format");
}
