#include "GaussianInstance.h"
#include <imgui.h>
#include "UI/ImGuiManager.h"

GaussianInstance::GaussianInstance(Scene& scene, const std::string& name, const uint32_t gaussianAssetIndex, const Transform& transf)
    : SceneObject(scene, name, transf), gaussianAssetIndex(gaussianAssetIndex) {}

GaussianInstance::GaussianInstance(const GaussianInstance& other)
    : SceneObject(other), gaussianAssetIndex(other.gaussianAssetIndex) {}

std::unique_ptr<SceneObject> GaussianInstance::clone() const
{
    return std::make_unique<GaussianInstance>(*this);
}

void GaussianInstance::onTransformUpdated()
{
    SceneObject::onTransformUpdated();
    scene.setDirtyFlag(TLAS);
}

bool GaussianInstance::renderUi()
{
    bool changed = SceneObject::renderUi();
    ImGuiManager::tableRowLabel("Gaussians");
    if (!ImGui::TreeNodeEx("Gaussian Asset###GaussianProperties", ImGuiTreeNodeFlags_Framed))
        return changed;
    if (ImGui::BeginTable("GaussianTable", 2, ImGuiTableFlags_SizingStretchProp)) {
        changed |= scene.getGaussianAsset(gaussianAssetIndex).renderUi();
        ImGui::EndTable();
    }
    ImGui::TreePop();
    return changed;
}
