#include "MeshInstance.h"

#include <imgui.h>

#include <utility>
#include "../UI/ImGuiManager.h"

MeshInstance::MeshInstance(const std::shared_ptr<Scene>& scene, const std::string& name, std::shared_ptr<MeshAsset> asset, const Transform& transf) : SceneObject(scene, name, transf), meshAsset(std::move(asset)) {

    instanceData = vk::AccelerationStructureInstanceKHR{};
    instanceData.setTransform(transform.getVkTransformMatrix());
    instanceData.setInstanceCustomIndex(meshAsset->getMeshIndex()); // Use in Shader to access correct Buffer for Mesh
    instanceData.setMask(0xFF);
    instanceData.setInstanceShaderBindingTableRecordOffset(0);
    instanceData.setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);

    instanceData.setAccelerationStructureReference(meshAsset->getBlasAddress());
}

void MeshInstance::updateInstanceTransform() {
    instanceData.setTransform(transform.getVkTransformMatrix());
    //TODO update scene
}

void MeshInstance::renderUi() {
    SceneObject::renderUi();

    ImGui::SeparatorText("Mesh Asset");

    ImGuiManager::tableRowLabel("Source");

    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
    ImGui::Text("%s", meshAsset->name.c_str());
    ImGui::PopStyleColor();
}

void MeshInstance::setPosition(const glm::vec3& pos)
{
    SceneObject::setPosition(pos);
    updateInstanceTransform();
}

void MeshInstance::setRotationEuler(const glm::vec3& rot)
{
    SceneObject::setRotationEuler(rot);
    updateInstanceTransform();
}

void MeshInstance::setScale(const glm::vec3& scale)
{
    SceneObject::setScale(scale);
    updateInstanceTransform();
}
