#include "MeshInstance.h"
#include <imgui.h>
#include <utility>

#include "../UI/ImGuiManager.h"

MeshInstance::MeshInstance(Scene& scene, const std::string& name, const uint32_t meshIndex, const Transform& transf)
    : SceneObject(scene, name, transf), meshIndex(meshIndex)
{}

MeshInstance::MeshInstance(const MeshInstance& other)
    : SceneObject(other),
      meshIndex(other.meshIndex)
{}

std::unique_ptr<SceneObject> MeshInstance::clone() const {
    return std::make_unique<MeshInstance>(*this);
}

void MeshInstance::onTransformUpdated() {
    SceneObject::onTransformUpdated();
    scene.setDirtyFlag(TLAS);
    scene.markMeshInstanceTransformDirty(scene.getMeshInstanceIndex(this));
}

bool MeshInstance::renderUi() {
    bool changed = SceneObject::renderUi();

    ImGuiManager::tableRowLabel("Mesh");
    if (!ImGui::TreeNodeEx("Mesh Asset###MeshProperties", ImGuiTreeNodeFlags_Framed))
        return changed;

    if (ImGui::BeginTable("MeshTable", 2, ImGuiTableFlags_SizingStretchProp)) {
        changed |= scene.getMeshAsset(meshIndex).renderUi();
        ImGui::EndTable();
    }
    ImGui::TreePop();
    return changed;
}
