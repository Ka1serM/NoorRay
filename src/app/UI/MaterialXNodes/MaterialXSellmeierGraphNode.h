#pragma once

#include "UI/MaterialXNodes/MaterialXGraphNode.h"

// NoorRay's spectral IOR extension. The inputs remain editable in the
// MaterialX parameter pane, while the graph node identifies the custom
// renderer extension at a glance.
class MaterialXSellmeierGraphNode final : public MaterialXGraphNode
{
public:
    using MaterialXGraphNode::MaterialXGraphNode;

protected:
    void drawBody() override;
};
