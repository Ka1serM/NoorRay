#include "MeshInstance.h"
#include <imgui.h>
#include <utility>

#include "../UI/ImGuiManager.h"

MeshInstance::MeshInstance(Scene& scene, const std::string& name, std::shared_ptr<MeshAsset> asset, const Transform& transf) : SceneObject(scene, name, transf), meshAsset(std::move(asset)) {

    instanceData = vk::AccelerationStructureInstanceKHR{};
    instanceData.setTransform(getWorldTransform().getVkTransformMatrix());
    instanceData.setInstanceCustomIndex(meshAsset->getMeshIndex());
    instanceData.setMask(0xFF);
    instanceData.setInstanceShaderBindingTableRecordOffset(0);
    instanceData.setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);

    instanceData.setAccelerationStructureReference(meshAsset->getBlasAddress());
}

MeshInstance::MeshInstance(const MeshInstance& other)
    : SceneObject(other),
      meshAsset(other.meshAsset)
{
    if (meshAsset) {
        instanceData = vk::AccelerationStructureInstanceKHR{};
        instanceData.setTransform(getWorldTransform().getVkTransformMatrix());
        instanceData.setInstanceCustomIndex(meshAsset->getMeshIndex());
        instanceData.setMask(0xFF);
        instanceData.setInstanceShaderBindingTableRecordOffset(0);
        instanceData.setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);
        instanceData.setAccelerationStructureReference(meshAsset->getBlasAddress());
    }
}

std::unique_ptr<SceneObject> MeshInstance::clone() const {
    return std::make_unique<MeshInstance>(*this);
}

void MeshInstance::onTransformUpdated() {
    SceneObject::onTransformUpdated();
    instanceData.setTransform(getWorldTransform().getVkTransformMatrix());
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
