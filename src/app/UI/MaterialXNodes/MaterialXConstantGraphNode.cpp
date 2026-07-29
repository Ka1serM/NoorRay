#include "UI/MaterialXNodes/MaterialXConstantGraphNode.h"

#include "UI/MaterialXNodes/MaterialXGraphNodeRegistry.h"

#include <imgui.h>

#include <array>
#include <cstdio>

namespace
{
const MaterialXGraphNodeRegistrar<MaterialXConstantGraphNode> registration{
    {"constant"}, {}, ImFlow::NodeStyle::cyan()};
}

void MaterialXConstantGraphNode::drawBody()
{
    const std::string value = inputValue("value");
    const std::string& type = node_->getType();
    if (value.empty()) {
        ImGui::TextDisabled("%s", type.c_str());
        return;
    }
    if (type == "color3" || type == "color4" || type == "vector3" || type == "vector4") {
        std::array<float, 4> components{0.0f, 0.0f, 0.0f, 1.0f};
        std::sscanf(value.c_str(), "%f, %f, %f, %f",
            &components[0], &components[1], &components[2], &components[3]);
        ImGui::ColorButton("##constant",
            ImVec4(components[0], components[1], components[2], components[3]),
            ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoDragDrop,
            ImVec2(48.0f, 16.0f));
        return;
    }
    ImGui::Text("%s", value.c_str());
}

bool MaterialXConstantGraphNode::isInlineInput(const std::string& inputName) const
{
    return inputName == "value";
}
