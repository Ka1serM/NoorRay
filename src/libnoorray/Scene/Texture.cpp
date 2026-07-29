#include "Scene/Texture.h"

#include <algorithm>
#include <bit>
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

float decodeHalf(const uint16_t value)
{
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x03ffu;

    if (exponent == 0) {
        if (mantissa == 0)
            return std::bit_cast<float>(sign);
        // Normalize a binary16 subnormal before rebiasing its exponent.
        uint32_t shift = 0;
        while ((mantissa & 0x0400u) == 0) {
            mantissa <<= 1;
            ++shift;
        }
        mantissa &= 0x03ffu;
        exponent = 127u - 15u + 1u - shift;
    } else if (exponent == 0x1fu) {
        exponent = 0xffu;
    } else {
        exponent += 127u - 15u;
    }

    return std::bit_cast<float>(
        sign | (exponent << 23) | (mantissa << 13));
}

template<class T>
const std::vector<T>& emptyStorage()
{
    static const std::vector<T> empty;
    return empty;
}

}

Texture::Texture(
    const std::string& filepath, const TextureEncoding requestedEncoding)
    : encoding(requestedEncoding)
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
        floatPixels = std::make_shared<const std::vector<float>>(
            bitmap.rgba(),
            bitmap.rgba() + static_cast<size_t>(width) * height * 4);
        encoding = TextureEncoding::Float32;
    } else if (stbi_is_hdr(filepath.c_str())) {
        std::unique_ptr<float, StbiImageDeleter> source(
            stbi_loadf(filepath.c_str(), &width, &height, &channels, 4));
        if (!source || width <= 0 || height <= 0)
            throw std::runtime_error("Failed to load texture: " + filepath);
        floatPixels = std::make_shared<const std::vector<float>>(
            source.get(),
            source.get() + static_cast<size_t>(width) * height * 4);
        encoding = TextureEncoding::Float32;
    } else {
        std::unique_ptr<uint8_t, StbiImageDeleter> source(
            stbi_load(filepath.c_str(), &width, &height, &channels, 4));
        if (!source || width <= 0 || height <= 0)
            throw std::runtime_error("Failed to load texture: " + filepath);
        const size_t valueCount = static_cast<size_t>(width) * height * 4;
        bytePixels = std::make_shared<const std::vector<uint8_t>>(
            source.get(), source.get() + valueCount);
    }
    name = SceneImporter::nameFromPath(filepath);
}

const std::vector<uint8_t>& Texture::getBytePixels() const
{
    return bytePixels ? *bytePixels : emptyStorage<uint8_t>();
}

const std::vector<uint16_t>& Texture::getHalfPixels() const
{
    return halfPixels ? *halfPixels : emptyStorage<uint16_t>();
}

const std::vector<float>& Texture::getPixels() const
{
    if (floatPixels)
        return *floatPixels;
    if (expandedPixels.empty() && usesByteStorage()) {
        expandedPixels.resize(bytePixels->size());
        for (size_t index = 0; index < bytePixels->size(); ++index) {
            expandedPixels[index] = index % 4 == 3
                ? static_cast<float>((*bytePixels)[index]) / 255.0f
                : decodeByte((*bytePixels)[index], encoding);
        }
    } else if (expandedPixels.empty() && usesHalfStorage()) {
        expandedPixels.resize(halfPixels->size());
        std::ranges::transform(
            *halfPixels, expandedPixels.begin(), decodeHalf);
    }
    return expandedPixels;
}

void Texture::releaseResources()
{
    name.clear();
    width = 0;
    height = 0;
    sceneIndex = -1;
    encoding = TextureEncoding::Linear8;
    bytePixels.reset();
    halfPixels.reset();
    floatPixels.reset();
    expandedPixels = std::vector<float>{};
}

Texture::Texture(std::string textureName, const void* data,
    const int textureWidth, const int textureHeight, const TextureEncoding encoding)
    : name(std::move(textureName))
    , width(textureWidth)
    , height(textureHeight)
    , encoding(encoding)
{
    if (width <= 0 || height <= 0)
        throw std::runtime_error("Texture has invalid dimensions: " + name);
    if (!data)
        throw std::runtime_error("Texture has null pixel storage: " + name);
    const size_t valueCount = static_cast<size_t>(width) * height * 4;
    if (encoding == TextureEncoding::Float32) {
        auto storage = std::make_shared<std::vector<float>>(valueCount);
        std::memcpy(
            storage->data(), data, storage->size() * sizeof(float));
        floatPixels = std::move(storage);
    } else if (encoding == TextureEncoding::Float16) {
        auto storage = std::make_shared<std::vector<uint16_t>>(valueCount);
        std::memcpy(
            storage->data(), data, storage->size() * sizeof(uint16_t));
        halfPixels = std::move(storage);
    } else {
        const auto* bytes = static_cast<const uint8_t*>(data);
        bytePixels = std::make_shared<const std::vector<uint8_t>>(
            bytes, bytes + valueCount);
    }
}

Texture::Texture(std::string textureName, std::vector<uint8_t>&& data,
    const int textureWidth, const int textureHeight,
    const TextureEncoding textureEncoding)
    : Texture(std::move(textureName),
        std::make_shared<const std::vector<uint8_t>>(std::move(data)),
        textureWidth, textureHeight, textureEncoding)
{
}

Texture::Texture(std::string textureName, std::vector<uint16_t>&& data,
    const int textureWidth, const int textureHeight,
    const TextureEncoding textureEncoding)
    : Texture(std::move(textureName),
        std::make_shared<const std::vector<uint16_t>>(std::move(data)),
        textureWidth, textureHeight, textureEncoding)
{
}

Texture::Texture(std::string textureName, std::vector<float>&& data,
    const int textureWidth, const int textureHeight,
    const TextureEncoding textureEncoding)
    : Texture(std::move(textureName),
        std::make_shared<const std::vector<float>>(std::move(data)),
        textureWidth, textureHeight, textureEncoding)
{
}

Texture::Texture(std::string textureName,
    std::shared_ptr<const std::vector<uint8_t>> data,
    const int textureWidth, const int textureHeight,
    const TextureEncoding textureEncoding)
    : name(std::move(textureName))
    , width(textureWidth)
    , height(textureHeight)
    , encoding(textureEncoding)
    , bytePixels(std::move(data))
{
    if (encoding != TextureEncoding::Linear8
        && encoding != TextureEncoding::Srgb8) {
        throw std::runtime_error(
            "Byte texture has a non-byte encoding: " + name);
    }
    validateStorageSize(bytePixels ? bytePixels->size() : 0);
}

Texture::Texture(std::string textureName,
    std::shared_ptr<const std::vector<uint16_t>> data,
    const int textureWidth, const int textureHeight,
    const TextureEncoding textureEncoding)
    : name(std::move(textureName))
    , width(textureWidth)
    , height(textureHeight)
    , encoding(textureEncoding)
    , halfPixels(std::move(data))
{
    if (encoding != TextureEncoding::Float16)
        throw std::runtime_error(
            "Half texture has a non-half encoding: " + name);
    validateStorageSize(halfPixels ? halfPixels->size() : 0);
}

Texture::Texture(std::string textureName,
    std::shared_ptr<const std::vector<float>> data,
    const int textureWidth, const int textureHeight,
    const TextureEncoding textureEncoding)
    : name(std::move(textureName))
    , width(textureWidth)
    , height(textureHeight)
    , encoding(textureEncoding)
    , floatPixels(std::move(data))
{
    if (encoding != TextureEncoding::Float32)
        throw std::runtime_error(
            "Float texture has a non-float encoding: " + name);
    validateStorageSize(floatPixels ? floatPixels->size() : 0);
}

void Texture::validateStorageSize(const size_t valueCount) const
{
    if (width <= 0 || height <= 0)
        throw std::runtime_error("Texture has invalid dimensions: " + name);
    const size_t expected =
        static_cast<size_t>(width) * static_cast<size_t>(height) * 4;
    if (valueCount != expected)
        throw std::runtime_error("Texture has invalid pixel storage: " + name);
}
