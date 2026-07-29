#include "UI/MaterialXNodes/MaterialXShaderGraphNode.h"

#include "UI/MaterialXNodes/MaterialXGraphNodeRegistry.h"

#include <imgui.h>

namespace
{
const MaterialXGraphNodeRegistrar<MaterialXShaderGraphNode> registration{
    {},
    {"surfaceshader", "displacementshader", "volumeshader", "lightshader", "material"},
    ImFlow::NodeStyle::green()};
}

void MaterialXShaderGraphNode::drawBody()
{
    ImGui::TextDisabled("%s", node_->getCategory().c_str());
    ImGui::TextDisabled("-> %s", node_->getType().c_str());
}
