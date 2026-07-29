#include "UI/MaterialXNodes/MaterialXMathGraphNode.h"

#include "UI/MaterialXNodes/MaterialXGraphNodeRegistry.h"

#include <imgui.h>

namespace
{
const MaterialXGraphNodeRegistrar<MaterialXMathGraphNode> registration{
    {"add", "subtract", "multiply", "divide", "modulo", "power",
     "dotproduct", "crossproduct"},
    {}, ImFlow::NodeStyle::red()};

const char* mathSymbol(const std::string& category)
{
    if (category == "add") return "+";
    if (category == "subtract") return "-";
    if (category == "multiply") return "x";
    if (category == "divide") return "/";
    if (category == "modulo") return "%";
    if (category == "power") return "^";
    return nullptr;
}
}

void MaterialXMathGraphNode::drawBody()
{
    if (const char* symbol = mathSymbol(node_->getCategory())) {
        ImGui::Text("%s", symbol);
        ImGui::SameLine();
    }
    ImGui::TextDisabled("%s", node_->getCategory().c_str());
}
