#include "NoorRay/Materials/Nodes/NodeTextures.h"

#include <algorithm>

namespace NoorRay::Materials {

namespace {

constexpr std::string_view textureName = "texture";
constexpr std::string_view samplesName = "image_samples";

}

NodeProperty textureRecord(const NodeTextureUse use, const NodeTexture& texture) {
    NodeProperty record;
    record.name = textureName;
    record.kind = NodePropertyKind::Texture;
    record.integer = static_cast<std::int64_t>(use);
    record.texture = texture.texture;
    record.number = {float(texture.columns), float(texture.rows), 0.0f, 0.0f};
    return record;
}

NodeProperty samplesRecord(const std::span<const Core::float4> samples) {
    NodeProperty record;
    record.name = samplesName;
    record.kind = NodePropertyKind::Numbers;
    for (const Core::float4 texel : samples)
        record.numbers.insert(record.numbers.end(), {texel.x, texel.y, texel.z, texel.w});
    return record;
}

bool isTextureRecord(const NodeProperty& property) {
    return property.name == textureName || property.name == samplesName;
}

void addTextureRecord(Node& node, NodeProperty record) {
    const bool recorded = std::ranges::any_of(node.properties, [&](const NodeProperty& property) {
        return property.name == record.name && (record.name == samplesName || property.integer == record.integer);
    });
    if (!recorded)
        node.properties.push_back(std::move(record));
}

std::expected<NodeTexture, std::string> RecordedTextures::load(const NodeTree& tree, const std::uint32_t node, const NodeTextureUse use) {
    for (const NodeProperty& record : tree.nodes[node].properties)
        if (record.name == textureName && record.integer == static_cast<std::int64_t>(use))
            return NodeTexture{record.texture, static_cast<std::uint32_t>(record.number.x), static_cast<std::uint32_t>(record.number.y)};
    if (use == NodeTextureUse::Ies)
        return std::unexpected("has no IES table");
    return std::unexpected("has no texture");
}

std::span<const Core::float4> RecordedTextures::samples(const NodeTree& tree, const std::uint32_t node) {
    for (const NodeProperty& record : tree.nodes[node].properties) {
        if (record.name != samplesName)
            continue;
        std::vector<Core::float4>& values = texels.emplace_back();
        for (std::size_t index = 0; index + 3 < record.numbers.size(); index += 4)
            values.push_back({record.numbers[index], record.numbers[index + 1], record.numbers[index + 2], record.numbers[index + 3]});
        return values;
    }
    return {};
}

}
