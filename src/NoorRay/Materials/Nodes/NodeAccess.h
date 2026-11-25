#pragma once

#include <array>
#include <initializer_list>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "NoorRay/Materials/NodeTree.h"

namespace NoorRay::Materials {

struct NodeRef {
    const NodeTree* tree = nullptr;
    std::uint32_t index = noNode;

    explicit operator bool() const { return tree && index != noNode; }
    const Node& get() const { return tree->nodes[index]; }
    const Node* operator->() const { return &tree->nodes[index]; }
    bool operator==(const NodeRef&) const = default;
};

struct SocketRef {
    const NodeTree* tree = nullptr;
    std::uint32_t node = noNode;
    std::uint32_t index = 0;
    bool output = false;

    explicit operator bool() const { return tree && node != noNode; }
    const NodeSocket& get() const { return output ? tree->nodes[node].outputs[index] : tree->nodes[node].inputs[index]; }
    const NodeSocket* operator->() const { return &get(); }
    NodeRef owner() const { return {tree, node}; }
    bool operator==(const SocketRef&) const = default;
};

//! A node output that drives an input.
struct NodeLinkEnd {
    NodeRef node;
    SocketRef socket;
};

//! A socket value or a property array: one to four floats.
struct NodeValues {
    std::array<float, 4> values{};
    std::size_t count = 0;
};

//! The Blender node type of a bl_idname, such as BSDF_PRINCIPLED for ShaderNodeBsdfPrincipled; CUSTOM for a node that Blender does not ship.
std::string_view nodeType(std::string_view idname);
inline std::string_view nodeType(const NodeRef node) {
    return nodeType(node->type);
}
//! The Blender name of a socket type, such as VALUE or RGBA.
std::string_view socketTypeName(NodeSocketType type);

SocketRef input(NodeRef node, std::string_view identifier);
SocketRef output(NodeRef node, std::string_view identifier);
SocketRef inputAt(NodeRef node, std::size_t index);
SocketRef outputAt(NodeRef node, std::size_t index);
//! The first enabled input or output with the name, as collection.get(name) in Blender.
SocketRef inputNamed(NodeRef node, std::string_view name);
SocketRef outputNamed(NodeRef node, std::string_view name);
std::vector<SocketRef> inputs(NodeRef node);
std::vector<SocketRef> outputs(NodeRef node);

NodeValues socketValues(SocketRef socket);
float socketNumber(SocketRef socket);
//! The default value of a socket as up to three components, as the fixed material takes it.
NodeValues socketValue(SocketRef socket);

const NodeProperty* findProperty(std::span<const NodeProperty> properties, std::string_view name);
inline const NodeProperty* findProperty(const NodeRef node, const std::string_view name) {
    return findProperty(node->properties, name);
}
//! The text of a property, empty when the node has no such property.
std::string text(std::span<const NodeProperty> properties, std::string_view name);
float number(std::span<const NodeProperty> properties, std::string_view name, float fallback = 0.0f);
NodeValues values(std::span<const NodeProperty> properties, std::string_view name);
std::int64_t integer(std::span<const NodeProperty> properties, std::string_view name);
bool flag(std::span<const NodeProperty> properties, std::string_view name);
std::span<const float> numbers(std::span<const NodeProperty> properties, std::string_view name);
inline std::string text(const NodeRef node, const std::string_view name) {
    return text(node->properties, name);
}
inline float number(const NodeRef node, const std::string_view name) {
    return number(node->properties, name);
}
inline std::int64_t integer(const NodeRef node, const std::string_view name) {
    return integer(node->properties, name);
}
inline bool flag(const NodeRef node, const std::string_view name) {
    return flag(node->properties, name);
}
inline std::span<const float> numbers(const NodeRef node, const std::string_view name) {
    return numbers(node->properties, name);
}

//! The links of the trees that a compile reads, indexed once per tree.
class NodeLinks {
public:
    //! True when any link, muted or not, ends at the input or starts at the output.
    bool isLinked(SocketRef socket);
    //! The node and output socket that drive the input, through reroutes; empty when the input is unlinked or its link is muted.
    NodeLinkEnd linkedNode(SocketRef input);

private:
    struct Table {
        //! The first link into each input, by node and socket.
        std::unordered_map<std::uint64_t, std::uint32_t> inputs;
        std::unordered_map<std::uint64_t, std::uint32_t> outputs;
    };
    const Table& table(const NodeTree& tree);

    std::unordered_map<const NodeTree*, Table> tables;
};

//! A colour ramp element in color_ramp.elements: its position, then its colour.
inline constexpr std::size_t rampElementSize = 5;

//! A piece of a sampled colour ramp, which blends to the colour at start + step.
struct RampPiece {
    float start = 0.0f;
    float step = 0.0f;
};
//! The pieces of a spline or hue ramp: each span between its elements, 0 and 1 cut in eight. The ramp evaluates at first, then at the end of each piece.
struct RampPieces {
    float first = 0.0f;
    std::vector<RampPiece> pieces;
};
RampPieces rampPieces(std::span<const float> elements);

//! The points of the piecewise linear stand-in for a curve mapping.
inline constexpr std::size_t curveSamples = 17;
//! The range of the x of the points of the curves, each a list of x and y; 0 to 1 without points.
std::pair<float, float> curveRange(std::initializer_list<std::span<const float>> curves);
//! Where a curve evaluates: low, then curveSamples - 1 steps to high; only low for an empty range.
std::vector<float> curveSamplePositions(float low, float high);

//! Whether two trees hold the same nodes, links and properties, and their groups the same, whichever objects hold the groups.
bool isSameTree(const NodeTree& a, const NodeTree& b);

//! The group tree of a group node; null when it has none.
const NodeTree* groupTree(NodeRef node);
//! The active output node of the bl_idname that Cycles renders, or the first one.
NodeRef findOutput(const NodeTree& tree, std::string_view idname);
//! The first Group Output of a group tree, or the last active one after it.
NodeRef findGroupOutput(const NodeTree& tree);

}
