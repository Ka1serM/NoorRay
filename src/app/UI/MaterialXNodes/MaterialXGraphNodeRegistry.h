#pragma once

#include "UI/MaterialXNodes/MaterialXGraphNode.h"

#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_map>

// Builds a node class instance and adds it to the flow at a grid position.
using MaterialXGraphNodeCreator = std::function<std::shared_ptr<MaterialXGraphNode>(
    ImFlow::ImNodeFlow&, const ImVec2&, MaterialX::NodePtr)>;

// Maps MaterialX categories, and output types for the shading roots, to the
// node class that draws them.
//
// Node classes register themselves from their own files rather than being
// listed here, so a new node type is one new file: define the class, add a
// MaterialXGraphNodeRegistrar for it, done. A category with no entry falls back
// to the generic MaterialXGraphNode.
class MaterialXGraphNodeRegistry
{
public:
    static MaterialXGraphNodeRegistry& instance();

    void registerCategory(const std::string& category, MaterialXGraphNodeCreator creator);
    void registerOutputType(const std::string& outputType, MaterialXGraphNodeCreator creator);

    // Creator registered for a node, or nullptr when it should use the generic
    // class. Category wins over output type.
    const MaterialXGraphNodeCreator* find(
        const std::string& category, const std::string& outputType) const;

private:
    std::unordered_map<std::string, MaterialXGraphNodeCreator> byCategory_;
    std::unordered_map<std::string, MaterialXGraphNodeCreator> byOutputType_;
};

// Registers node class T at start-up. Declare one at namespace scope in the
// file that defines T:
//
//   const MaterialXGraphNodeRegistrar<MyNode> registration{
//       {"mycategory"}, {}, ImFlow::NodeStyle::cyan()};
template <typename T>
struct MaterialXGraphNodeRegistrar
{
    MaterialXGraphNodeRegistrar(
        const std::initializer_list<const char*> categories,
        const std::initializer_list<const char*> outputTypes,
        std::shared_ptr<ImFlow::NodeStyle> style = nullptr)
    {
        MaterialXGraphNodeCreator creator =
            [style = std::move(style)](ImFlow::ImNodeFlow& flow, const ImVec2& position,
                MaterialX::NodePtr node) -> std::shared_ptr<MaterialXGraphNode> {
                std::shared_ptr<T> created = flow.addNode<T>(position, std::move(node));
                if (style)
                    created->setStyle(style);
                return created;
            };
        MaterialXGraphNodeRegistry& registry = MaterialXGraphNodeRegistry::instance();
        for (const char* category : categories)
            registry.registerCategory(category, creator);
        for (const char* outputType : outputTypes)
            registry.registerOutputType(outputType, creator);
    }
};

// Constructs the node class registered for this MaterialX node, adds it to the
// flow at the given grid position and builds its pins.
std::shared_ptr<MaterialXGraphNode> addMaterialXGraphNode(
    ImFlow::ImNodeFlow& flow, const ImVec2& position, MaterialX::NodePtr node);
