#include <cmath>
#include <format>
#include <limits>
#include <optional>
#include <ranges>
#include <string_view>
#include <utility>
#include <vector>

#include "NoorRay/Materials/NodeCatalog.h"
#include "NoorRay/Materials/Nodes/NodeAccess.h"
#include "NoorRay/Materials/Nodes/NodeImage.h"

namespace NoorRay::Materials {

namespace {

using Scene::TextureSlot;

constexpr float columnWidth = 300.0f;
constexpr float rowHeight = 300.0f;

struct Output {
    std::uint32_t node = noNode;
    std::uint32_t socket = 0;
};

Core::float4 toFloat4(const Core::float3 value) {
    return {value.x, value.y, value.z, 1.0f};
}

std::string_view toExtension(const Scene::TextureWrap wrap) {
    if (wrap == Scene::TextureWrap::Extend)
        return "EXTEND";
    if (wrap == Scene::TextureWrap::Clip)
        return "CLIP";
    return wrap == Scene::TextureWrap::Mirror ? "MIRROR" : "REPEAT";
}

std::string_view toInterpolation(const Scene::TextureFilter filter) {
    if (filter == Scene::TextureFilter::Closest)
        return "Closest";
    return filter == Scene::TextureFilter::Cubic ? "Cubic" : "Linear";
}

std::string_view toSampling(const Scene::EmissionSampling sampling) {
    if (sampling == Scene::EmissionSampling::Disabled)
        return "NONE";
    if (sampling == Scene::EmissionSampling::Front)
        return "FRONT";
    if (sampling == Scene::EmissionSampling::Back)
        return "BACK";
    return sampling == Scene::EmissionSampling::FrontAndBack ? "FRONT_AND_BACK" : "AUTO";
}

//! Builds the nodes of a fixed material around the Principled BSDF and Material Output.
class MaterialNodes {
public:
    MaterialNodes(NodeTree& tree, const Scene::Material& material) : tree(tree), material(material) {}

    void build();

private:
    struct Image {
        Scene::MaterialTexture texture;
        std::uint32_t node = noNode;
        std::uint32_t separate = noNode;
    };

    std::uint32_t add(std::string_view type, float column);
    void setText(std::uint32_t node, std::string_view name, std::string_view value);
    void setInput(std::uint32_t node, std::string_view identifier, Core::float4 value);
    void setInput(const std::uint32_t node, const std::string_view identifier, const float value) {
        setInput(node, identifier, Core::float4{value, 0.0f, 0.0f, 0.0f});
    }
    void link(Output from, std::uint32_t node, std::string_view identifier);
    Output output(std::uint32_t node, std::string_view identifier);

    Image& image(const Scene::MaterialTexture& texture, std::string_view name);
    void coordinates(const Scene::MaterialTexture& texture, std::uint32_t node);
    Output channel(const Scene::MaterialTexture& texture, std::string_view name);
    Output math(std::string_view operation, std::optional<Output> first, float a, std::optional<Output> second, float b);
    Output multiply(Output color, Core::float3 factor, std::optional<Output> second);
    std::uint32_t vertexColor();

    void color(TextureSlot slot, std::string_view identifier, Core::float3 factor);
    void scalar(TextureSlot slot, std::string_view identifier, float factor);
    void roughness();
    void alpha();
    void normal(TextureSlot slot, std::string_view identifier, float strength);
    void settings();

    NodeTree& tree;
    const Scene::Material& material;
    std::uint32_t principled = noNode;
    std::uint32_t colorNode = noNode;
    std::vector<Image> images;
    float row = 0.0f;
};

std::uint32_t MaterialNodes::add(const std::string_view type, const float column) {
    const Core::float2 location{-200.0f - column * columnWidth, 100.0f - row * rowHeight};
    row += 0.5f;
    return NodeCatalog::addNode(tree, type, location).value_or(noNode);
}

void MaterialNodes::setText(const std::uint32_t node, const std::string_view name, const std::string_view value) {
    NodeProperty property;
    property.name = name;
    property.kind = NodePropertyKind::Text;
    property.text = value;
    static_cast<void>(NodeCatalog::setProperty(tree, node, std::move(property)));
}

void MaterialNodes::setInput(const std::uint32_t node, const std::string_view identifier, const Core::float4 value) {
    const auto socket = std::ranges::find(tree.nodes[node].inputs, identifier, &NodeSocket::identifier);
    if (socket != tree.nodes[node].inputs.end())
        socket->value = value;
}

void MaterialNodes::link(const Output from, const std::uint32_t node, const std::string_view identifier) {
    const SocketRef socket = input({&tree, node}, identifier);
    if (socket)
        static_cast<void>(NodeCatalog::connect(tree, {from.node, from.socket, node, socket.index}));
}

Output MaterialNodes::output(const std::uint32_t node, const std::string_view identifier) {
    return {node, Materials::output({&tree, node}, identifier).index};
}

MaterialNodes::Image& MaterialNodes::image(const Scene::MaterialTexture& texture, const std::string_view name) {
    const auto sameImage = [&](const Image& image) {
        Scene::MaterialTexture other = image.texture;
        other.channel = texture.channel;
        return other == texture;
    };
    if (const auto found = std::ranges::find_if(images, sameImage); found != images.end())
        return *found;
    const std::uint32_t node = add("ShaderNodeTexImage", 4.0f);
    setText(node, "extension", toExtension(texture.wrap));
    setText(node, "interpolation", toInterpolation(texture.filter));
    setNodeImage(tree, node, std::string(name), texture.texture);
    coordinates(texture, node);
    return images.emplace_back(texture, node);
}

void MaterialNodes::coordinates(const Scene::MaterialTexture& texture, const std::uint32_t node) {
    const bool transformed = texture.transform != Scene::UvTransform();
    if (!transformed && texture.uvSet == 0)
        return;
    Output coordinate;
    if (texture.uvSet == 0) {
        coordinate = output(add("ShaderNodeTexCoord", 6.0f), "UV");
    } else {
        const std::string map = std::format("UVMap.{:03}", texture.uvSet);
        const std::uint32_t uv = add("ShaderNodeUVMap", 6.0f);
        setText(uv, "uv_map", map);
        coordinate = output(uv, "UV");
        if (std::ranges::none_of(tree.properties, [&](const NodeProperty& record) { return record.name == "uv_stream" && record.text == map; })) {
            NodeProperty& record = tree.properties.emplace_back();
            record.name = "uv_stream";
            record.kind = NodePropertyKind::Text;
            record.text = map;
            record.integer = texture.uvSet;
        }
    }
    if (!transformed) {
        link(coordinate, node, "Vector");
        return;
    }
    const Core::float3 x = texture.transform.row0;
    const Core::float3 y = texture.transform.row1;
    const bool flat = x.x == 0.0f && y.x == 0.0f;
    const float angle = flat ? std::atan2(-x.y, y.y) : std::atan2(y.x, x.x);
    const std::uint32_t mapping = add("ShaderNodeMapping", 5.0f);
    setInput(mapping, "Location", Core::float4{x.z, y.z, 0.0f, 0.0f});
    setInput(mapping, "Rotation", Core::float4{0.0f, 0.0f, angle, 0.0f});
    setInput(mapping, "Scale", Core::float4{std::hypot(x.x, y.x), y.y * std::cos(angle) - x.y * std::sin(angle), 1.0f, 0.0f});
    link(coordinate, mapping, "Vector");
    link(output(mapping, "Vector"), node, "Vector");
}

Output MaterialNodes::channel(const Scene::MaterialTexture& texture, const std::string_view name) {
    Image& source = image(texture, name);
    if (texture.channel == 3)
        return output(source.node, "Alpha");
    if (source.separate == noNode) {
        source.separate = add("ShaderNodeSeparateColor", 3.0f);
        link(output(source.node, "Color"), source.separate, "Color");
    }
    return {source.separate, texture.channel};
}

Output MaterialNodes::math(const std::string_view operation, const std::optional<Output> first, const float a, const std::optional<Output> second,
                           const float b) {
    const std::uint32_t node = add("ShaderNodeMath", 1.0f);
    setText(node, "operation", operation);
    tree.nodes[node].inputs[0].value = {a, 0.0f, 0.0f, 0.0f};
    tree.nodes[node].inputs[1].value = {b, 0.0f, 0.0f, 0.0f};
    if (first)
        link(*first, node, tree.nodes[node].inputs[0].identifier);
    if (second)
        link(*second, node, tree.nodes[node].inputs[1].identifier);
    return {node, 0};
}

Output MaterialNodes::multiply(const Output color, const Core::float3 factor, const std::optional<Output> second) {
    const std::uint32_t node = add("ShaderNodeMix", 1.0f);
    setText(node, "data_type", "RGBA");
    setText(node, "blend_type", "MULTIPLY");
    setInput(node, "Factor_Float", 1.0f);
    setInput(node, "B_Color", toFloat4(factor));
    link(color, node, "A_Color");
    if (second)
        link(*second, node, "B_Color");
    return output(node, "Result_Color");
}

std::uint32_t MaterialNodes::vertexColor() {
    if (colorNode == noNode)
        colorNode = add("ShaderNodeVertexColor", 4.0f);
    return colorNode;
}

void MaterialNodes::color(const TextureSlot slot, const std::string_view identifier, const Core::float3 factor) {
    setInput(principled, identifier, toFloat4(factor));
    std::optional<Output> source;
    if (const std::optional<Scene::MaterialTexture>& texture = material.texture(slot))
        source = output(image(*texture, identifier).node, "Color");
    if (slot == TextureSlot::BaseColor && material.useVertexColor) {
        const Output vertex = output(vertexColor(), "Color");
        source = source ? multiply(*source, {1.0f, 1.0f, 1.0f}, vertex) : vertex;
    }
    if (!source)
        return;
    if (factor != Core::float3{1.0f, 1.0f, 1.0f})
        source = multiply(*source, factor, std::nullopt);
    link(*source, principled, identifier);
}

void MaterialNodes::scalar(const TextureSlot slot, const std::string_view identifier, const float factor) {
    setInput(principled, identifier, factor);
    const std::optional<Scene::MaterialTexture>& texture = material.texture(slot);
    if (!texture)
        return;
    Output source = channel(*texture, identifier);
    if (factor != 1.0f)
        source = math("MULTIPLY", source, 0.0f, std::nullopt, factor);
    link(source, principled, identifier);
}

void MaterialNodes::roughness() {
    const std::optional<Scene::MaterialTexture>& texture = material.texture(TextureSlot::Roughness);
    if (!material.roughnessIsGlossiness) {
        scalar(TextureSlot::Roughness, "Roughness", material.roughness);
        return;
    }
    setInput(principled, "Roughness", 1.0f - material.roughness);
    if (!texture)
        return;
    Output source = channel(*texture, "Roughness");
    if (material.roughness != 1.0f)
        source = math("MULTIPLY", source, 0.0f, std::nullopt, material.roughness);
    link(math("SUBTRACT", std::nullopt, 1.0f, source, 0.0f), principled, "Roughness");
}

void MaterialNodes::alpha() {
    if (material.alphaMode == Scene::AlphaMode::Opaque)
        return;
    const bool mask = material.alphaMode == Scene::AlphaMode::Mask;
    const std::optional<Scene::MaterialTexture>& texture = material.texture(TextureSlot::Alpha);
    std::optional<Output> source;
    if (texture)
        source = channel(*texture, "Alpha");
    if (material.useVertexColor) {
        const Output vertex = output(vertexColor(), "Alpha");
        source = source ? math("MULTIPLY", source, 0.0f, vertex, 0.0f) : vertex;
    }
    if (!source) {
        setInput(principled, "Alpha", mask ? (material.alpha < material.alphaCutoff ? 0.0f : 1.0f) : material.alpha);
        return;
    }
    if (material.alpha != 1.0f)
        source = math("MULTIPLY", source, 0.0f, std::nullopt, material.alpha);
    if (mask)
        source = math("GREATER_THAN", source, 0.0f, std::nullopt, std::nextafter(material.alphaCutoff, -std::numeric_limits<float>::max()));
    //! The compiler tests the alpha of a linked input only when its value is below one.
    setInput(principled, "Alpha", std::min(material.alpha, 0.5f));
    link(*source, principled, "Alpha");
}

void MaterialNodes::normal(const TextureSlot slot, const std::string_view identifier, const float strength) {
    const std::optional<Scene::MaterialTexture>& texture = material.texture(slot);
    if (!texture)
        return;
    const std::uint32_t source = image(*texture, identifier).node;
    const std::uint32_t node = add("ShaderNodeNormalMap", 1.0f);
    setInput(node, "Strength", strength);
    link(output(source, "Color"), node, "Color");
    link(output(node, "Normal"), principled, identifier);
}

void MaterialNodes::settings() {
    const auto find = [&](const std::string_view name) { return std::ranges::find(tree.properties, name, &NodeProperty::name); };
    find("emission_sampling")->text = toSampling(material.emissionSampling);
    find("diffuse_color")->numbers = {material.baseColor.x, material.baseColor.y, material.baseColor.z, material.alpha};
    find("metallic")->number.x = material.metallic;
    find("roughness")->number.x = material.roughnessIsGlossiness ? 1.0f - material.roughness : material.roughness;
}

void MaterialNodes::build() {
    NodeCatalog::addDefaultNodes(tree, NodeTreeKind::Material);
    principled = 0;
    settings();
    setInput(principled, "IOR", material.ior);
    setInput(principled, "Specular Tint", toFloat4(material.specularTint));
    setInput(principled, "Anisotropic", material.anisotropic);
    setInput(principled, "Anisotropic Rotation", material.anisotropicRotation);
    setInput(principled, "Sheen Roughness", material.sheenRoughness);
    setInput(principled, "Emission Strength", material.emissionStrength);
    setInput(principled, "Thin Film Thickness", material.thinFilmThickness);
    setInput(principled, "Thin Film IOR", material.thinFilmIor);
    if (Core::maxComponent(material.sheenColor) > 0.0f) {
        setInput(principled, "Sheen Weight", 1.0f);
        setInput(principled, "Sheen Tint", toFloat4(material.sheenColor));
    }
    color(TextureSlot::BaseColor, "Base Color", material.baseColor);
    scalar(TextureSlot::Metallic, "Metallic", material.metallic);
    roughness();
    alpha();
    scalar(TextureSlot::Specular, "Specular IOR Level", material.specularIorLevel);
    scalar(TextureSlot::Transmission, "Transmission Weight", material.transmission);
    scalar(TextureSlot::Coat, "Coat Weight", material.coat);
    scalar(TextureSlot::CoatRoughness, "Coat Roughness", material.coatRoughness);
    color(TextureSlot::Emission, "Emission Color", material.emissionColor);
    normal(TextureSlot::Normal, "Normal", material.normalScale);
    normal(TextureSlot::CoatNormal, "Coat Normal", material.coatNormalScale);
}

}

void NodeCatalog::addMaterialNodes(NodeTree& tree, const Scene::Material& material) {
    MaterialNodes(tree, material).build();
}

}
