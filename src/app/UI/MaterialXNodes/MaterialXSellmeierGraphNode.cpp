#include "UI/MaterialXNodes/MaterialXSellmeierGraphNode.h"

#include "UI/MaterialXNodes/MaterialXGraphNodeRegistry.h"

#include <imgui.h>

namespace
{
const MaterialXGraphNodeRegistrar<MaterialXSellmeierGraphNode> registration{
    {"noorray_sellmeier_ior"}, {}, ImFlow::NodeStyle::cyan()};
}

void MaterialXSellmeierGraphNode::drawBody()
{
    ImGui::TextDisabled("spectral IOR");
    ImGui::Text("Sellmeier");
    ImGui::TextDisabled("n(546.074 nm) derived");
}
