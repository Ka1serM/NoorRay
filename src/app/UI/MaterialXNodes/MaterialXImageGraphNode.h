#pragma once

#include "UI/MaterialXNodes/MaterialXGraphNode.h"

// Texture lookups. Shows the resolved file name so a graph full of images stays
// readable without selecting each node, and keeps the file input off the pin
// list while it holds a plain value.
class MaterialXImageGraphNode final : public MaterialXGraphNode
{
public:
    using MaterialXGraphNode::MaterialXGraphNode;

protected:
    void drawBody() override;
    bool isInlineInput(const std::string& inputName) const override;
};
