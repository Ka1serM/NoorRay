#include "NoorRay/Gpu/EnvironmentBuffers.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "NoorRay/Gpu/AliasTable.h"
#include "NoorRay/Gpu/BufferUpload.h"
#include "NoorRay/Gpu/TextureTable.h"

namespace NoorRay::Gpu {

namespace {

float srgbToLinear(const float value) {
    return value <= 0.04045f ? value / 12.92f : std::pow((value + 0.055f) / 1.055f, 2.4f);
}

template <class T>
T readValue(const std::byte* data) {
    T value;
    std::memcpy(&value, data, sizeof(T));
    return value;
}

}

Core::float4 EnvironmentBuffers::readTexel(const Scene::Texture& texture, const std::uint32_t x, const std::uint32_t y) {
    const std::size_t index = std::size_t{y} * texture.width + x;
    const std::byte* base = texture.data.data();
    switch (texture.format) {
    case NoorRHI::ImageFormat::Rgba32Float: {
        const std::byte* texel = base + index * 16;
        return {readValue<float>(texel), readValue<float>(texel + 4), readValue<float>(texel + 8), readValue<float>(texel + 12)};
    }
    case NoorRHI::ImageFormat::Rgba16Float: {
        const std::byte* texel = base + index * 8;
        const auto channel = [&](const int i) { return Core::fromHalf(NoorRHI::half{readValue<std::uint16_t>(texel + i * 2)}); };
        return {channel(0), channel(1), channel(2), channel(3)};
    }
    case NoorRHI::ImageFormat::Rgba8Unorm:
    case NoorRHI::ImageFormat::Rgba8Srgb: {
        const std::byte* texel = base + index * 4;
        const bool srgb = texture.format == NoorRHI::ImageFormat::Rgba8Srgb;
        const auto channel = [&](const int i) {
            const float value = static_cast<float>(std::to_integer<std::uint8_t>(texel[i])) / 255.0f;
            return srgb && i < 3 ? srgbToLinear(value) : value;
        };
        return {channel(0), channel(1), channel(2), channel(3)};
    }
    default:
        return {1.0f, 1.0f, 1.0f, 1.0f};
    }
}

std::expected<void, Core::Error> EnvironmentBuffers::update(NoorRHI::Device& device, const Scene::Scene& scene, const TextureTable& textures) {
    const Scene::Environment& environment = scene.getEnvironment();
    color = environment.color;
    strength = environment.strength;
    rotation = environment.rotation;
    visibleToCamera = environment.visibleToCamera;
    const Scene::Texture* map = scene.find(environment.texture);
    texture = map ? textures.getTexture(environment.texture) : NoorRHI::TextureHandle{};
    sampler = textures.getSampler(Scene::TextureWrap::Repeat, Scene::TextureFilter::Linear);
    width = 0;
    height = 0;
    std::vector<float> weights;
    if (map && texture) {
        std::uint32_t shift = 0;
        while ((map->width >> shift) > maxWidth)
            ++shift;
        width = std::max(map->width >> shift, 1u);
        height = std::max(map->height >> shift, 1u);
        std::vector<float> cells(std::size_t{width} * height);
        for (std::uint32_t y = 0; y < map->height; ++y)
            for (std::uint32_t x = 0; x < map->width; ++x) {
                const Core::float4 texel = readTexel(*map, x, y);
                cells[std::size_t{std::min(y >> shift, height - 1)} * width + std::min(x >> shift, width - 1)] +=
                    std::max(Core::luminance({texel.x, texel.y, texel.z}), 0.0f);
            }
        const float cellArea = static_cast<float>(1u << shift) * static_cast<float>(1u << shift);
        weights.resize(cells.size());
        double weighted = 0.0;
        double solidAngle = 0.0;
        for (std::uint32_t y = 0; y < height; ++y) {
            const float sinTheta = std::sin(Core::pi * (static_cast<float>(y) + 0.5f) / static_cast<float>(height));
            for (std::uint32_t x = 0; x < width; ++x) {
                const float luminance = cells[std::size_t{y} * width + x] / cellArea;
                weights[std::size_t{y} * width + x] = luminance * sinTheta;
                weighted += luminance * sinTheta;
                solidAngle += sinTheta;
            }
        }
        averageLuminance = solidAngle > 0.0 ? static_cast<float>(weighted / solidAngle) * strength : 0.0f;
    } else
        averageLuminance = std::max(Core::luminance(color), 0.0f) * strength;
    const std::vector<Shaders::AliasEntry> table = buildAliasTable(weights);
    return uploadValues<Shaders::AliasEntry>(device, alias, table);
}

void EnvironmentBuffers::fill(Shaders::SceneRecord& record) const {
    record.environmentAlias = alias.getPtr();
    record.environment = {texture.value, sampler.value, width, height};
    record.environmentColor = {color.x, color.y, color.z, strength};
    record.environmentParams.x = rotation;
    record.sceneCenter.w = visibleToCamera ? 1.0f : 0.0f;
}

}
