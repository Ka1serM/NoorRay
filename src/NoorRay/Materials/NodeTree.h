#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "NoorRay/Core/Handle.h"
#include "NoorRay/Core/Math.h"

namespace NoorRay::Scene {
struct Texture;
}

namespace NoorRay::Materials {

enum class NodeSocketType : std::uint8_t { Float, Int, Bool, Vector, Color, Shader, String, Rotation, Matrix, Menu, Closure, Custom };

//! value holds a number, vector or colour socket value; text holds a string or menu value.
struct NodeSocket {
    std::string identifier;
    std::string name;
    NodeSocketType type = NodeSocketType::Float;
    std::uint8_t components = 1;
    bool enabled = true;
    bool hidden = false;
    Core::float4 value;
    std::string text;

    bool operator==(const NodeSocket&) const = default;
};

enum class NodePropertyKind : std::uint8_t { Number, Integer, Boolean, Text, Numbers, Texture };

//! A setting of a node or tree by its Blender property name. Numbers holds an array, such as the flattened elements of a colour ramp or the points of a curve.
struct NodeProperty {
    std::string name;
    NodePropertyKind kind = NodePropertyKind::Number;
    Core::float4 number;
    std::int64_t integer = 0;
    std::string text;
    std::vector<float> numbers;
    Core::Handle<Scene::Texture> texture;

    bool operator==(const NodeProperty&) const = default;
};

inline constexpr std::uint32_t noNode = UINT32_MAX;

//! type is the Blender bl_idname, such as ShaderNodeBsdfPrincipled. parent is the frame node that holds the node; group indexes NodeTree::groups.
struct Node {
    std::string type;
    std::string name;
    std::string label;
    Core::float2 location;
    float width = 140.0f;
    bool muted = false;
    bool hidden = false;
    std::uint32_t parent = noNode;
    std::uint32_t group = noNode;
    std::vector<NodeSocket> inputs;
    std::vector<NodeSocket> outputs;
    std::vector<NodeProperty> properties;

    bool operator==(const Node&) const = default;
};

//! Sockets are indices into the outputs of fromNode and the inputs of toNode.
struct NodeLink {
    std::uint32_t fromNode = 0;
    std::uint32_t fromSocket = 0;
    std::uint32_t toNode = 0;
    std::uint32_t toSocket = 0;
    bool muted = false;

    bool operator==(const NodeLink&) const = default;
};

//! A Blender node tree as data. properties holds the settings of the owner that the compiler reads, such as the blend or displacement method of a material.
struct NodeTree {
    std::string name;
    std::vector<Node> nodes;
    std::vector<NodeLink> links;
    std::vector<NodeProperty> properties;
    std::vector<std::shared_ptr<const NodeTree>> groups;
};

}
