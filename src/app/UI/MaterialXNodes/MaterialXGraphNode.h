#pragma once

#include <ImNodeFlow.h>
#include <MaterialXCore/Node.h>

#include <string>

// Base class for every node drawn in the MaterialX node editor, and the node
// class used for any MaterialX category without a more specific one.
//
// One file per node class: a class registers itself for the categories it draws
// (see MaterialXGraphNodeRegistry.h), so adding a node type means adding one
// file and nothing else. MaterialX ships hundreds of definitions and most of
// them differ only in data, which is why the generic rendering here has to
// stand in for the long tail; see MaterialXNodeCatalog for that data.
class MaterialXGraphNode : public ImFlow::BaseNode
{
public:
    explicit MaterialXGraphNode(MaterialX::NodePtr node);

    void draw() final;

    const MaterialX::NodePtr& materialNode() const { return node_; }
    // Adds one input pin per exposed MaterialX input plus the generic output.
    // Called after construction so the subclass overrides are in effect.
    void buildPins();
    // Rebinds this node to the matching node of a freshly loaded document.
    // Node objects outlive the documents they were built from, which is what
    // keeps the position, size and selection the user set from being thrown
    // away every time the document is reparsed.
    void setMaterialNode(MaterialX::NodePtr node);
    // Drops the pins and rebuilds them, for after a rebind changed the inputs.
    void rebuildPins();
    // Returns the matching input pin, or null for inputs rendered inline in
    // the node body. ImNodeFlow's inPin() is not a nullable lookup.
    ImFlow::Pin* findInputPin(const std::string& inputName);

protected:
    // Body drawn under the title bar. The default shows the node category.
    virtual void drawBody();
    // True for inputs the body renders itself, which then get no pin unless
    // something is connected to them.
    virtual bool isInlineInput(const std::string& inputName) const;

    // Value of an input as authored, or an empty string when it is connected
    // or unset.
    std::string inputValue(const std::string& inputName) const;

    MaterialX::NodePtr node_;
};
