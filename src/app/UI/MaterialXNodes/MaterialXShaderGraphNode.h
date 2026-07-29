#pragma once

#include "UI/MaterialXNodes/MaterialXGraphNode.h"

// Shading roots: surface, displacement, volume and light shaders, and the
// material node itself. Registered by output type rather than by category,
// because any node producing a shader closure belongs here.
class MaterialXShaderGraphNode final : public MaterialXGraphNode
{
public:
    using MaterialXGraphNode::MaterialXGraphNode;

protected:
    void drawBody() override;
};
