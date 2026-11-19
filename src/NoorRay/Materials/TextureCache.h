#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <map>
#include <span>
#include <string>
#include <utility>

#include "NoorRay/Core/Error.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Materials {

enum class ColorSpace : std::uint8_t { Srgb, Linear, NonColor };

//! Decodes image files into scene textures with a full mip chain, once per path and colour space.
class TextureCache {
public:
    explicit TextureCache(Scene::Scene& scene) : scene(scene) {}

    [[nodiscard]] std::expected<Scene::TextureHandle, Core::Error> load(const std::filesystem::path& path, ColorSpace colorSpace);
    [[nodiscard]] std::expected<Scene::TextureHandle, Core::Error> load(std::string_view name, std::span<const std::byte> encoded, ColorSpace colorSpace);

    [[nodiscard]] static std::expected<Scene::Texture, Core::Error> decode(std::string name, std::span<const std::byte> encoded, ColorSpace colorSpace);
    static Scene::Texture fromRgba8(std::string name, std::uint32_t width, std::uint32_t height, std::span<const std::uint8_t> pixels, bool srgb);
    static Scene::Texture fromRgbaFloat(std::string name, std::uint32_t width, std::uint32_t height, std::span<const float> pixels);

private:
    Scene::Scene& scene;
    std::map<std::pair<std::string, ColorSpace>, Scene::TextureHandle> loaded;
};

}
