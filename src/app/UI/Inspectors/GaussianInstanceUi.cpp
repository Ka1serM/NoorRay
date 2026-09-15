#include <imgui.h>

#include "Mesh/Assets/Gaussian.h"
#include "Scene/GaussianInstance.h"
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

bool object_ui::render(GaussianInstance& instance)
{
    return renderGaussianInstance(instance);
}
