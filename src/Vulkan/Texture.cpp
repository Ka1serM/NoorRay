#include "Texture.h"
#include <stdexcept>
#include <stb_image.h>
#include <string>
#include <cstring>

#include "Scene/SceneImporter.h"

Texture::Texture(Context& context, const std::string& filepath)
{
    (void)context;
    int channels = 0;
    float* rawPixels = stbi_loadf(filepath.c_str(), &width, &height, &channels, 4);
    if (rawPixels == nullptr || width <= 0 || height <= 0)
    {
        stbi_image_free(rawPixels);
        throw std::runtime_error("Failed to load texture: " + filepath);
    }
    pixels.assign(rawPixels, rawPixels + static_cast<size_t>(width) * height * 4);
    stbi_image_free(rawPixels);
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
            pixels[index] = static_cast<float>(bytes[index]) / 255.0f;
    }
    else
        throw std::runtime_error("Unsupported CPU texture format");
}
