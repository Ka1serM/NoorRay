#include "UI/MaterialXNodes/MaterialXImageGraphNode.h"

#include "UI/MaterialXNodes/MaterialXGraphNodeRegistry.h"

#include <imgui.h>

#include <filesystem>

namespace
{
const MaterialXGraphNodeRegistrar<MaterialXImageGraphNode> registration{
    {"image", "tiledimage", "triplanarprojection"}, {}, ImFlow::NodeStyle::brown()};
}

void MaterialXImageGraphNode::drawBody()
{
    ImGui::TextDisabled("%s", node_->getCategory().c_str());
    const std::string file = inputValue("file");
    if (file.empty()) {
        ImGui::TextDisabled("(no file)");
        return;
    }
    ImGui::Text("%s", std::filesystem::path(file).filename().string().c_str());
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s", file.c_str());
}

bool MaterialXImageGraphNode::isInlineInput(const std::string& inputName) const
{
    return inputName == "file";
}
