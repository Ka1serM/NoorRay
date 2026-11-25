#pragma once

#include <cstdint>
#include <span>
#include <string_view>

#include "NoorRay/Core/Math.h"
#include "NoorRay/Materials/NodeCatalog.h"

namespace NoorRay::Materials {

struct NodeSocketData {
    std::string_view identifier;
    std::string_view name;
    NodeSocketType type;
    std::uint8_t components;
    bool enabled;
    bool hidden;
    bool hideValue;
    Core::float4 value;
    std::string_view text;
    float minimum;
    float maximum;
    std::string_view subtype;
};

struct NodeEnumData {
    std::string_view identifier;
    std::string_view name;
};

//! shown is false for a setting that the editor does not edit; stored is false for a setting that a new node does not have, such as its image.
struct NodePropertyData {
    std::string_view name;
    std::string_view label;
    NodePropertyKind kind;
    float number;
    std::int64_t integer;
    std::string_view text;
    std::span<const float> numbers;
    std::span<const NodeEnumData> items;
    float minimum;
    float maximum;
    bool shown;
    bool stored;
};

struct NodeSocketName {
    bool output;
    std::uint32_t index;
    std::string_view name;
};

//! The enabled sockets, the inputs without a value and the changed socket names that a combination of settings gives; bit n is socket n.
struct NodeSocketState {
    std::uint64_t enabledInputs;
    std::uint64_t enabledOutputs;
    std::uint64_t hiddenValues;
    std::span<const NodeSocketName> names;
};

//! The inputs that a muted node can pass to an output, and the one that it passes without links; -1 for none.
struct NodeMuting {
    std::uint64_t inputs;
    std::int32_t unlinked;
};

//! ruleStates indexes states by the choices of the ruleSettings, the last setting first.
struct NodeTypeData {
    std::string_view type;
    std::string_view name;
    std::string_view category;
    float width;
    std::span<const NodeSocketData> inputs;
    std::span<const NodeSocketData> outputs;
    std::span<const NodePropertyData> properties;
    std::span<const NodeTreeKind> trees;
    std::span<const std::string_view> ruleSettings;
    std::span<const std::uint8_t> ruleStates;
    std::span<const NodeSocketState> states;
    std::span<const NodeMuting> muting;
};

struct NodeDefault {
    std::string_view type;
    Core::float2 location;
    float width;
};

struct NodeDefaultValue {
    std::uint32_t node;
    std::uint32_t input;
    Core::float4 value;
};

struct NodeTreeDefault {
    NodeTreeKind kind;
    std::string_view name;
    std::span<const NodeDefault> nodes;
    std::span<const NodeLink> links;
    std::span<const NodeDefaultValue> values;
    std::span<const NodePropertyData> properties;
};

//! The node types, sorted by type.
std::span<const NodeTypeData> nodeTypeData();
std::span<const NodeTreeDefault> nodeTreeDefaults();
//! Null for a type that the catalog does not hold.
const NodeTypeData* findNodeTypeData(std::string_view type);

}
