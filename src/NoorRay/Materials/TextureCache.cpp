#include "NoorRay/Materials/TextureCache.h"

#include <cstdlib>
#include <cstring>
#include <format>
#include <fstream>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#include <stb_image.h>
#include <tinyexr.h>

namespace NoorRay::Materials {

Scene::Texture TextureCache::fromRgba8(std::string name, const std::uint32_t width, const std::uint32_t height, const std::span<const std::uint8_t> pixels,
                                       const bool srgb) {
    Scene::Texture texture{std::move(name), srgb ? NoorRHI::ImageFormat::Rgba8Srgb : NoorRHI::ImageFormat::Rgba8Unorm, width, height, {}, {}};
    texture.data.resize(pixels.size());
    std::memcpy(texture.data.data(), pixels.data(), pixels.size());
    texture.levels.push_back({0, texture.data.size(), width, height});
    return texture;
}

Scene::Texture TextureCache::fromRgbaFloat(std::string name, const std::uint32_t width, const std::uint32_t height, const std::span<const float> pixels) {
    Scene::Texture texture{std::move(name), NoorRHI::ImageFormat::Rgba16Float, width, height, {}, {}};
    texture.data.resize(pixels.size() * sizeof(std::uint16_t));
    for (std::size_t index = 0; index < pixels.size(); ++index) {
        const std::uint16_t bits = Core::toHalf(pixels[index]).bits;
        std::memcpy(texture.data.data() + index * sizeof(bits), &bits, sizeof(bits));
    }
    texture.levels.push_back({0, texture.data.size(), width, height});
    return texture;
}

std::expected<Scene::Texture, Core::Error> TextureCache::decode(std::string name, const std::span<const std::byte> encoded, const ColorSpace colorSpace) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(encoded.data());
    const int size = static_cast<int>(encoded.size());
    if (IsEXRFromMemory(bytes, encoded.size()) == TINYEXR_SUCCESS) {
        float* pixels = nullptr;
        int width = 0, height = 0;
        const char* message = nullptr;
        if (LoadEXRFromMemory(&pixels, &width, &height, bytes, encoded.size(), &message) != TINYEXR_SUCCESS) {
            const std::string reason = message ? message : "unknown error";
            FreeEXRErrorMessage(message);
            return Core::fail(Core::ErrorCode::InvalidFile, std::format("{}: {}", name, reason));
        }
        Scene::Texture texture = fromRgbaFloat(std::move(name), static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height),
                                               std::span(pixels, std::size_t(width) * height * 4));
        std::free(pixels);
        return texture;
    }
    int width = 0, height = 0, channels = 0;
    if (stbi_is_hdr_from_memory(bytes, size)) {
        float* pixels = stbi_loadf_from_memory(bytes, size, &width, &height, &channels, 4);
        if (!pixels)
            return Core::fail(Core::ErrorCode::InvalidFile, std::format("{}: {}", name, stbi_failure_reason()));
        Scene::Texture texture = fromRgbaFloat(std::move(name), static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height),
                                               std::span(pixels, std::size_t(width) * height * 4));
        stbi_image_free(pixels);
        return texture;
    }
    stbi_uc* pixels = stbi_load_from_memory(bytes, size, &width, &height, &channels, 4);
    if (!pixels)
        return Core::fail(Core::ErrorCode::InvalidFile, std::format("{}: {}", name, stbi_failure_reason()));
    Scene::Texture texture = fromRgba8(std::move(name), static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height),
                                       std::span(pixels, std::size_t(width) * height * 4), colorSpace == ColorSpace::Srgb);
    stbi_image_free(pixels);
    return texture;
}

std::expected<Scene::TextureHandle, Core::Error> TextureCache::load(const std::string_view name, const std::span<const std::byte> encoded,
                                                                    const ColorSpace colorSpace) {
    const std::pair<std::string, ColorSpace> key{std::string(name), colorSpace};
    if (const auto found = loaded.find(key); found != loaded.end())
        return found->second;
    std::expected<Scene::Texture, Core::Error> texture = decode(std::string(name), encoded, colorSpace);
    if (!texture)
        return std::unexpected(std::move(texture.error()));
    const Scene::TextureHandle handle = scene.addTexture(std::move(*texture));
    loaded.emplace(key, handle);
    return handle;
}

std::expected<Scene::TextureHandle, Core::Error> TextureCache::load(const std::filesystem::path& path, const ColorSpace colorSpace) {
    const std::pair<std::string, ColorSpace> key{path.string(), colorSpace};
    if (const auto found = loaded.find(key); found != loaded.end())
        return found->second;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        return Core::fail(Core::ErrorCode::Io, std::format("cannot open the texture {}", path.string()));
    std::vector<std::byte> encoded(static_cast<std::size_t>(file.tellg()));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(encoded.data()), static_cast<std::streamsize>(encoded.size()));
    return load(path.string(), encoded, colorSpace);
}

}
