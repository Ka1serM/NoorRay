#pragma once

#include "UI/MaterialXNodes/MaterialXGraphNode.h"

// Arithmetic and vector maths. Drawn with the operator symbol where there is
// one, so short maths chains read at a glance.
class MaterialXMathGraphNode final : public MaterialXGraphNode
{
public:
    using MaterialXGraphNode::MaterialXGraphNode;

protected:
    void drawBody() override;
};
