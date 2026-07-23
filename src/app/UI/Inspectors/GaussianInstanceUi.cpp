#include <imgui.h>

#include "Mesh/Assets/GaussianAsset.h"
#include "Scene/GaussianInstance.h"
#include "UI/ImGuiManager.h"
#include "UI/ObjectUi.h"

namespace
{
bool renderGaussianInstance(GaussianInstance& instance)
{
    const GaussianAsset& asset = instance.getGaussianAsset();
    ImGuiManager::tableRowLabel("Gaussian Count");
    ImGui::Text("%u", asset.getGaussianCount());
    ImGuiManager::tableRowLabel("Source");
    ImGui::TextUnformatted(asset.getPath().c_str());
    return false;
}

}

void ObjectUiVisitor::visit(GaussianInstance& instance)
{
    changed |= renderGaussianInstance(instance);
}
