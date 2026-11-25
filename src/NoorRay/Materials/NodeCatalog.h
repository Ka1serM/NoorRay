#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "NoorRay/Core/Error.h"
#include "NoorRay/Materials/NodeTree.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Materials {

enum class NodeTreeKind : std::uint8_t { Material, Light, World };

//! The range and display of a socket; a socket without a range has minimum greater than maximum.
struct NodeSocketTemplate {
    NodeSocket socket;
    float minimum = 1.0f;
    float maximum = 0.0f;
    //! The Blender subtype, such as FACTOR, PERCENTAGE, ANGLE or DISTANCE.
    std::string subtype;
    bool hideValue = false;
};

struct NodeEnumItem {
    std::string identifier;
    std::string name;
};

//! A node setting that the editor shows; items lists the choices of a Text enum.
struct NodePropertyType {
    NodeProperty value;
    std::string label;
    std::vector<NodeEnumItem> items;
    float minimum = 1.0f;
    float maximum = 0.0f;
};

//! A node type that the compiler supports, with the sockets and settings of a new node, as Blender 5.2 creates it.
struct NodeType {
    std::string type;
    std::string name;
    //! The Blender add menu category, such as Input, Output, Shader, Displacement, Color, Texture, Math or Vector; empty outside of the menu.
    std::string category;
    float width = 140.0f;
    std::vector<NodeSocketTemplate> inputs;
    std::vector<NodeSocketTemplate> outputs;
    std::vector<NodePropertyType> properties;
    //! The tree kinds whose add menu shows the type.
    std::vector<NodeTreeKind> trees;
};

//! The node types of the compiler, generated from Blender, and the edits of a node tree that keep it the same as Blender would make it.
class NodeCatalog {
public:
    static std::span<const NodeType> getTypes();
    static const NodeType* find(std::string_view type);
    //! The types of a tree kind that match the text by name, sorted with the best match first; an empty text gives every type of the kind.
    static std::vector<const NodeType*> search(std::string_view text, NodeTreeKind kind);

    //! Adds a new node of the type with a unique name, and returns its index.
    static std::expected<std::uint32_t, Core::Error> addNode(NodeTree& tree, std::string_view type, Core::float2 location);
    //! Removes the nodes and their links, and keeps the indices of the other links, parents and groups correct.
    static void removeNodes(NodeTree& tree, std::span<const std::uint32_t> nodes);
    //! Copies the nodes and the links between them with new unique names, moved by offset, and returns the indices of the copies.
    static std::vector<std::uint32_t> duplicateNodes(NodeTree& tree, std::span<const std::uint32_t> nodes, Core::float2 offset);
    //! Links an output to an input as Blender does: an input takes one link, so an older link to it goes away. Fails for a link that makes a cycle.
    static std::expected<void, Core::Error> connect(NodeTree& tree, NodeLink link);
    static void disconnect(NodeTree& tree, std::uint32_t toNode, std::uint32_t toSocket);
    //! Fails when another node of the tree has the name.
    static std::expected<void, Core::Error> renameNode(NodeTree& tree, std::uint32_t node, std::string name);
    //! Sets a node setting, then updates the enabled sockets and the derived data, such as the samples of a colour ramp or curve.
    static std::expected<void, Core::Error> setProperty(NodeTree& tree, std::uint32_t node, NodeProperty property);
    //! Sets the image of an image or environment texture node, and writes the texture records that the compiler reads.
    static std::expected<void, Core::Error> setTexture(NodeTree& tree, std::uint32_t node, const Scene::Scene& scene, Scene::TextureHandle texture);

    //! Adds the nodes that Blender gives a new material (Principled BSDF to Material Output), light or world to an empty tree.
    static void addDefaultNodes(NodeTree& tree, NodeTreeKind kind);
    //! Adds nodes to an empty tree that give the same result as the fixed inputs and textures of a material without nodes.
    static void addMaterialNodes(NodeTree& tree, const Scene::Material& material);

    //! Scales the energy and colour settings of a light tree so that it compiles to the power and colour; compiled is the light that the tree gives now.
    static void setLightOutput(NodeTree& tree, const Scene::Light& compiled, float power, Core::float3 color);
    //! Writes the world values of wanted into a tree that compiles to compiled; fails without a change when the tree cannot hold a changed value.
    static std::expected<void, Core::Error> setEnvironmentOutput(NodeTree& tree, const Scene::Environment& compiled, const Scene::Environment& wanted);
};

}
