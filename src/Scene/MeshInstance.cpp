#include "MeshInstance.h"

#include <imgui.h>

#include <utility>
#include "../UI/ImGuiManager.h"
#include "Vulkan/Renderer.h"

MeshInstance::MeshInstance(Renderer& renderer, const std::string& name, std::shared_ptr<MeshAsset> asset, const Transform& transf) : SceneObject(renderer, name, transf), meshAsset(std::move(asset)) {

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
    renderer.rebuildTLAS();
}

void MeshInstance::renderUi() {
    SceneObject::renderUi();

    ImGui::SeparatorText("Mesh Asset");
    meshAsset.get()->renderUi();
}

void MeshInstance::setPosition(const glm::vec3& pos)
{
    SceneObject::setPosition(pos);
    updateInstanceTransform();
}

void MeshInstance::setRotation(const glm::quat& rot)
{
    SceneObject::setRotation(rot);
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
