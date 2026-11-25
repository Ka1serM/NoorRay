#include "NoorRay/Materials/Nodes/NodeImage.h"

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <utility>

#include "NoorRay/Materials/Nodes/NodeAccess.h"
#include "NoorRay/Materials/Nodes/NodeTextures.h"
#include "NoorRay/Materials/TextureCache.h"

namespace NoorRay::Materials {

namespace {

enum class Component : std::uint8_t { Unorm8, Unorm16, Half, Float };

struct Layout {
    Component component;
    std::uint32_t count;
    bool srgb = false;
    bool swapped = false;
    bool gray = false;
};

std::optional<Layout> layoutOf(const NoorRHI::ImageFormat format) {
    using enum NoorRHI::ImageFormat;
    switch (format) {
    case Rgba8Unorm:
        return Layout{Component::Unorm8, 4};
    case Rgba8Srgb:
        return Layout{Component::Unorm8, 4, true};
    case Bgra8Unorm:
        return Layout{Component::Unorm8, 4, false, true};
    case Bgra8Srgb:
        return Layout{Component::Unorm8, 4, true, true};
    case R8Unorm:
        return Layout{Component::Unorm8, 1};
    case L8Unorm:
        return Layout{Component::Unorm8, 1, false, false, true};
    case L8Srgb:
        return Layout{Component::Unorm8, 1, true, false, true};
    case Rg8Unorm:
        return Layout{Component::Unorm8, 2};
    case R16Unorm:
        return Layout{Component::Unorm16, 1};
    case Rg16Unorm:
        return Layout{Component::Unorm16, 2};
    case Rgba16Unorm:
        return Layout{Component::Unorm16, 4};
    case R16Float:
        return Layout{Component::Half, 1};
    case Rg16Float:
        return Layout{Component::Half, 2};
    case Rgba16Float:
        return Layout{Component::Half, 4};
    case R32Float:
        return Layout{Component::Float, 1};
    case Rg32Float:
        return Layout{Component::Float, 2};
    case Rgba32Float:
        return Layout{Component::Float, 4};
    default:
        return std::nullopt;
    }
}

std::size_t sizeOf(const Component component) {
    if (component == Component::Unorm8)
        return 1;
    return component == Component::Float ? 4 : 2;
}

float readComponent(const std::byte* data, const Component component) {
    if (component == Component::Unorm8)
        return static_cast<float>(std::to_integer<std::uint32_t>(*data)) / 255.0f;
    if (component == Component::Float) {
        float value = 0.0f;
        std::memcpy(&value, data, sizeof(value));
        return value;
    }
    std::uint16_t bits = 0;
    std::memcpy(&bits, data, sizeof(bits));
    return component == Component::Half ? Core::fromHalf(std::bit_cast<Core::half>(bits)) : static_cast<float>(bits) / 65535.0f;
}

Core::float4 readTexel(const std::byte* data, const Layout& layout) {
    std::array<float, 4> values{0.0f, 0.0f, 0.0f, 1.0f};
    for (std::uint32_t index = 0; index < layout.count; ++index) {
        values[index] = readComponent(data + index * sizeOf(layout.component), layout.component);
        if (layout.srgb && index < 3)
            values[index] = TextureCache::srgbToLinear(values[index]);
    }
    if (layout.gray)
        values[1] = values[2] = values[0];
    if (layout.swapped)
        std::swap(values[0], values[2]);
    return {values[0], values[1], values[2], values[3]};
}

}

std::vector<Core::float4> sampleImage(const Scene::Texture& texture) {
    const std::optional<Layout> layout = layoutOf(texture.format);
    if (!layout || texture.levels.empty())
        return {};
    const Scene::TextureLevel& level = texture.levels.front();
    const std::size_t texelSize = sizeOf(layout->component) * layout->count;
    if (level.width == 0 || level.height == 0 || texture.data.size() < level.offset + texelSize * level.width * level.height)
        return {};
    const std::byte* texels = texture.data.data() + level.offset;
    std::vector<Core::float4> samples;
    samples.reserve(foldSide * foldSide);
    for (std::size_t row = 0; row < foldSide; ++row) {
        const std::size_t y = level.height - 1 - row * (level.height - 1) / (foldSide - 1);
        for (std::size_t column = 0; column < foldSide; ++column) {
            const std::size_t x = column * (level.width - 1) / (foldSide - 1);
            samples.push_back(readTexel(texels + (y * level.width + x) * texelSize, *layout));
        }
    }
    return samples;
}

void setNodeImage(NodeTree& tree, const std::uint32_t node, const std::string& image, const Core::Handle<Scene::Texture> texture) {
    constexpr std::array imageUses{NodeTextureUse::Premultiplied, NodeTextureUse::Straight, NodeTextureUse::Color, NodeTextureUse::Data};
    constexpr std::array environmentUses{NodeTextureUse::Premultiplied, NodeTextureUse::Straight, NodeTextureUse::Background};
    Node& target = tree.nodes[node];
    NodeProperty name;
    name.name = "image";
    name.kind = NodePropertyKind::Text;
    name.text = image.empty() ? "Image" : image;
    NodeProperty source;
    source.name = "image_source";
    source.kind = NodePropertyKind::Text;
    source.text = "FILE";
    target.properties.push_back(std::move(name));
    target.properties.push_back(std::move(source));
    const bool environment = nodeType(target.type) == "TEX_ENVIRONMENT";
    for (const NodeTextureUse use : environment ? std::span<const NodeTextureUse>(environmentUses) : std::span<const NodeTextureUse>(imageUses))
        addTextureRecord(target, textureRecord(use, {texture}));
}

}
