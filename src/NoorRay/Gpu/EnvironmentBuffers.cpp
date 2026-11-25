#include "NoorRay/Gpu/EnvironmentBuffers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

#include "NoorRay/Gpu/AliasTable.h"
#include "NoorRay/Gpu/BufferUpload.h"
#include "NoorRay/Gpu/GraphBakes.h"
#include "NoorRay/Gpu/TextureTable.h"
#include "NoorRay/Materials/Graph.h"
#include "NoorRay/Shaders/Materials/GraphBakeRecords.h"

namespace NoorRay::Gpu {

namespace {

float srgbToLinear(const float value) {
    return value <= 0.04045f ? value / 12.92f : std::pow((value + 0.055f) / 1.055f, 2.4f);
}

constexpr std::uint32_t minBakeWidth = 2048;
constexpr std::uint32_t maxBakeWidth = 8192;

constexpr std::array rayInputs{Materials::GraphInput::IsCameraRay,   Materials::GraphInput::IsDiffuseRay, Materials::GraphInput::RayDepth,
                               Materials::GraphInput::DiffuseDepth, Materials::GraphInput::IsGlossyRay,  Materials::GraphInput::IsTransmissionRay};

Core::float3 unpackRgb9e5(const std::uint32_t packed) {
    const float scale = std::exp2(static_cast<float>(packed >> 27) - 24.0f);
    return {float(packed & 511u) * scale, float((packed >> 9) & 511u) * scale, float((packed >> 18) & 511u) * scale};
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

std::expected<void, Core::Error> EnvironmentBuffers::bake(NoorRHI::Device& device, const Scene::Scene& scene, const TextureTable& textures,
                                                           GraphBakes& bakes) {
    const Scene::Environment& environment = scene.getEnvironment();
    if (!environment.graph) {
        baked = {};
        image = {};
        cameraImage = {};
        bakedKey = 0;
        return {};
    }
    const std::uint64_t key = hashGraphValues(*environment.graph, environment.graphParameters, environment.graphTextures);
    if (key == bakedKey)
        return {};
    std::uint32_t side = minBakeWidth;
    for (const Scene::MaterialTexture& item : environment.graphTextures)
        if (const Scene::Texture* source = scene.find(item.texture))
            side = std::clamp(source->width, side, maxBakeWidth);
    side &= ~1u;
    const auto bakeMap = [&](const std::uint32_t kind, NoorRHI::Image& target) -> std::expected<void, Core::Error> {
        std::expected<std::vector<std::uint32_t>, Core::Error> texels = bakes.bakeWorld(device, environment, textures, kind, side);
        if (!texels)
            return std::unexpected(std::move(texels.error()));
        baked = {.name = "World", .format = NoorRHI::ImageFormat::Rgba16Float, .width = side, .height = side / 2, .levels = {}, .data = {}};
        baked.data.resize(texels->size() * sizeof(std::array<Core::half, 4>));
        for (std::size_t index = 0; index < texels->size(); ++index) {
            const Core::float3 value = unpackRgb9e5((*texels)[index]);
            const std::array texel{Core::toHalf(value.x), Core::toHalf(value.y), Core::toHalf(value.z), Core::toHalf(1.0f)};
            std::memcpy(baked.data.data() + index * sizeof(texel), &texel, sizeof(texel));
        }
        target = {};
        if (auto opened = Core::check(target.open(device, {.width = side, .height = side / 2, .format = baked.format, .name = "World"})); !opened)
            return opened;
        return Core::check(target.upload(baked.data));
    };
    const auto readsRay = [&](const Materials::GraphInput input) { return environment.graph->reads(Materials::GraphOutput::Emission, input); };
    cameraImage = {};
    if (std::ranges::any_of(rayInputs, readsRay))
        if (auto result = bakeMap(Shaders::graphBakeWorldCamera, cameraImage); !result)
            return result;
    if (auto result = bakeMap(Shaders::graphBakeWorld, image); !result)
        return result;
    bakedKey = key;
    return {};
}

std::expected<void, Core::Error> EnvironmentBuffers::update(NoorRHI::Device& device, const Scene::Scene& scene, const TextureTable& textures,
                                                             GraphBakes& bakes) {
    const Scene::Environment& environment = scene.getEnvironment();
    color = environment.color;
    strength = environment.strength;
    rotation = environment.rotation;
    hidden = environment.hidden;
    cameraView = environment.visibleToCamera ? Shaders::environmentVisible : environment.opaque ? Shaders::environmentBlack : Shaders::environmentTransparent;
    if (auto result = bake(device, scene, textures, bakes); !result)
        return result;
    const Scene::Texture* map = environment.graph ? &baked : scene.find(environment.texture);
    texture = environment.graph ? image.getSampledHandle() : map ? textures.getTexture(environment.texture) : NoorRHI::TextureHandle{};
    sampler = textures.getSampler(Scene::TextureWrap::Repeat, environment.filter);
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
    record.environmentCamera = {cameraImage ? cameraImage.getSampledHandle().value : 0, sampler.value};
    record.environmentColor = {color.x, color.y, color.z, strength};
    record.environmentParams.x = rotation;
    record.sceneCenter.w = cameraView;
}

}
