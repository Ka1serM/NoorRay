#include <imgui.h>

#include "Geometry/Mesh/Assets/GaussianAsset.h"
#include "Scene/Objects/GaussianInstance.h"
#include "UI/ImGuiManager.h"
#include "UI/ObjectUi.h"

namespace
{
bool renderGaussianInstance(GaussianInstance& instance)
{
    if (!instance.hasGaussianAsset())
        return false;
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
