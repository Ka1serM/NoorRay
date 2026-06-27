#include "MeshInstance.h"
#include <imgui.h>
#include <utility>

#include "../UI/ImGuiManager.h"

MeshInstance::MeshInstance(Scene& scene, const std::string& name, std::shared_ptr<MeshAsset> asset, const Transform& transf)
    : SceneObject(scene, name, transf), meshAsset(std::move(asset))
{}

MeshInstance::MeshInstance(const MeshInstance& other)
    : SceneObject(other),
      meshAsset(other.meshAsset)
{}

std::unique_ptr<SceneObject> MeshInstance::clone() const {
    return std::make_unique<MeshInstance>(*this);
}

void MeshInstance::onTransformUpdated() {
    SceneObject::onTransformUpdated();
    scene.setDirtyFlag(TLAS);
}

void MeshInstance::renderUi() {
    SceneObject::renderUi();

    ImGui::SeparatorText("Mesh Asset");
    
    ImGui::TableNextColumn();
    if (meshAsset)
        meshAsset->renderUi();  // render mesh details inside the value column
    else
        ImGui::TextUnformatted("No Mesh Assigned.");
}
