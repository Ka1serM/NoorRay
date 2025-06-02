#include "MeshInstance.h"

#include <imgui.h>

#include <utility>
#include "../UI/ImGuiManager.h"

MeshInstance::MeshInstance(Renderer& renderer, const std::string& name, std::shared_ptr<MeshAsset> asset, const Transform& transf) : SceneObject(renderer, name, transf), meshAsset(std::move(asset)) {

    instanceData = vk::AccelerationStructureInstanceKHR{};
    instanceData.setTransform(transform.getVkTransformMatrix());
    instanceData.setInstanceCustomIndex(meshAsset->getMeshIndex()); // Use in Shader to access correct Buffer for Mesh
    instanceData.setMask(0xFF);
    instanceData.setInstanceShaderBindingTableRecordOffset(0);
    instanceData.setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);

    instanceData.setAccelerationStructureReference(meshAsset->getBlasAddress());

    renderer.add(instanceData);
}

void MeshInstance::updateInstanceTransform() {
    instanceData.setTransform(transform.getVkTransformMatrix());
    renderer.buildTLAS();
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
    updateInstanceTransform();
    SceneObject::setPosition(pos);
}

void MeshInstance::setRotationEuler(const glm::vec3& rot)
{
    updateInstanceTransform();
    SceneObject::setRotationEuler(rot);
}

void MeshInstance::setScale(const glm::vec3& scale)
{
    updateInstanceTransform();
    SceneObject::setScale(scale);
}
