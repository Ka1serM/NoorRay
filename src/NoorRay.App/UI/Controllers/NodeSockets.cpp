#include "NoorRay.App/UI/Controllers/NodeSockets.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>

#include <RmlUi/Core/Element.h>

#include "NoorRay.App/UI/Controls/ColorHex.h"
#include "NoorRay/Materials/NodeCatalog.h"

namespace NoorRay::App {

namespace {

using Materials::NodeSocketType;

constexpr int maxVectorComponents = 3;

//! Blender's socket colours by type.
const std::array<Rml::Colourb, 12> socketColours{
    Rml::Colourb{161, 161, 161}, Rml::Colourb{89, 140, 92},   Rml::Colourb{204, 166, 214}, Rml::Colourb{99, 99, 199},
    Rml::Colourb{199, 199, 41},  Rml::Colourb{99, 199, 99},   Rml::Colourb{112, 178, 255}, Rml::Colourb{166, 99, 199},
    Rml::Colourb{184, 51, 138},  Rml::Colourb{102, 102, 102}, Rml::Colourb{140, 140, 140}, Rml::Colourb{51, 51, 51},
};

float srgbOf(const float linear) {
    const float clamped = std::clamp(linear, 0.0f, 1.0f);
    return clamped <= 0.0031308f ? clamped * 12.92f : 1.055f * std::pow(clamped, 1.0f / 2.4f) - 0.055f;
}

float linearOf(const float srgb) {
    return srgb <= 0.04045f ? srgb / 12.92f : std::pow((srgb + 0.055f) / 1.055f, 2.4f);
}

Rml::String editorOf(const Materials::NodeSocket& socket) {
    switch (socket.type) {
    case NodeSocketType::Float:
        return "number";
    case NodeSocketType::Int:
        return "integer";
    case NodeSocketType::Bool:
        return "toggle";
    case NodeSocketType::Vector:
    case NodeSocketType::Rotation:
        return "vector";
    case NodeSocketType::Color:
        return "color";
    case NodeSocketType::String:
    case NodeSocketType::Menu:
        return "text";
    case NodeSocketType::Shader:
    case NodeSocketType::Matrix:
    case NodeSocketType::Closure:
    case NodeSocketType::Custom:
        break;
    }
    return "";
}

}

Rml::Colourb socketColour(const NodeSocketType type) {
    return socketColours[static_cast<std::size_t>(type)];
}

Rml::String cssOf(const Rml::Colourb colour) {
    return std::format("#{:02x}{:02x}{:02x}", colour.red, colour.green, colour.blue);
}

Rml::String swatchOf(const Core::float4 linear) {
    return hexOf({srgbOf(linear.x), srgbOf(linear.y), srgbOf(linear.z)});
}

std::optional<Core::float3> linearOfSwatch(const Rml::String& swatch) {
    const std::optional<Core::float3> srgb = colorOfHex(swatch);
    if (!srgb)
        return std::nullopt;
    return Core::float3{linearOf(srgb->x), linearOf(srgb->y), linearOf(srgb->z)};
}

NodeGraphSocketRow socketRow(const Materials::NodeSocket& socket, const std::uint32_t index, const bool editable) {
    NodeGraphSocketRow row;
    row.name = socket.name;
    row.socket = index;
    row.color = cssOf(socketColour(socket.type));
    if (!editable)
        return row;
    row.editor = editorOf(socket);
    row.components = std::clamp<int>(socket.components, 1, maxVectorComponents);
    row.x = socket.value.x;
    row.y = socket.value.y;
    row.z = socket.value.z;
    row.checked = row.editor == "toggle" && socket.value.x != 0.0f;
    row.swatch = swatchOf(socket.value);
    row.text = socket.text;
    return row;
}

bool isEditorEvent(const NodeGraphSocketRow& row, const Rml::Event& event) {
    const Rml::Element* target = event.GetTargetElement();
    if (!target || target->GetTagName() != "input")
        return false;
    const Rml::String type = target->GetAttribute<Rml::String>("type", "text");
    if (row.editor == "toggle")
        return type == "checkbox";
    return type == "text" && (row.editor == "number" || row.editor == "integer" || row.editor == "vector");
}

Core::float4 valueOf(const NodeGraphSocketRow& row, const Core::float4 old) {
    if (row.editor == "toggle")
        return {row.checked ? 1.0f : 0.0f, old.y, old.z, old.w};
    if (row.editor == "integer")
        return {std::round(float(row.x)), old.y, old.z, old.w};
    if (row.editor == "number")
        return {row.x, old.y, old.z, old.w};
    return {row.x, row.y, row.z, old.w};
}

bool canLink(const NodeSocketType from, const NodeSocketType to) {
    return from != NodeSocketType::Shader || to == NodeSocketType::Shader;
}

bool hidesValue(const Materials::Node& node, const std::uint32_t socket) {
    const auto hidden = std::ranges::find(node.properties, std::string_view("hide_value"), &Materials::NodeProperty::name);
    if (hidden != node.properties.end() && std::ranges::contains(hidden->numbers, static_cast<float>(socket)))
        return true;
    const Materials::NodeType* type = Materials::NodeCatalog::find(node.type);
    if (!type || socket >= node.inputs.size())
        return false;
    const std::string& identifier = node.inputs[socket].identifier;
    return std::ranges::any_of(type->inputs,
                               [&](const Materials::NodeSocketTemplate& entry) { return entry.hideValue && entry.socket.identifier == identifier; });
}

}
