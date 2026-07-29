#pragma once

#include "UI/MaterialXNodes/MaterialXGraphNode.h"

// Literal values. Draws a colour swatch for colour and vector types and the
// number otherwise, so a constant reads without opening the parameter pane.
class MaterialXConstantGraphNode final : public MaterialXGraphNode
{
public:
    using MaterialXGraphNode::MaterialXGraphNode;

protected:
    void drawBody() override;
    bool isInlineInput(const std::string& inputName) const override;
};
