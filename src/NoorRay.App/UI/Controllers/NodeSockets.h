#pragma once

#include <cstdint>
#include <optional>

#include <RmlUi/Core/Event.h>
#include <RmlUi/Core/Types.h>

#include "NoorRay.App/UI/Models/NodeGraphModel.h"
#include "NoorRay/Materials/NodeTree.h"

namespace NoorRay::App {

Rml::Colourb socketColour(Materials::NodeSocketType type);
Rml::String cssOf(Rml::Colourb colour);
//! The #rrggbb swatch of a scene linear colour.
Rml::String swatchOf(Core::float4 linear);
//! The scene linear colour of a #rrggbb swatch.
std::optional<Core::float3> linearOfSwatch(const Rml::String& swatch);
//! An input that is not editable, such as a linked one, shows only its name.
NodeGraphSocketRow socketRow(const Materials::NodeSocket& socket, std::uint32_t index, bool editable);
//! A row keeps the hidden fields of every editor, and data bindings send change events from them; only the field of the row's editor edits the value.
bool isEditorEvent(const NodeGraphSocketRow& row, const Rml::Event& event);
//! The value that the fields of an edited socket row give, from the old value of the socket.
Core::float4 valueOf(const NodeGraphSocketRow& row, Core::float4 old);
//! The node or its type hides the value of the input, as Blender does for a normal input; hide_value lists the hidden input indices.
bool hidesValue(const Materials::Node& node, std::uint32_t socket);
//! Blender marks a link from a shader output to another socket type as invalid.
bool canLink(Materials::NodeSocketType from, Materials::NodeSocketType to);

}
