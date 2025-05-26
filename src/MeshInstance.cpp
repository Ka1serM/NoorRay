#include "MeshInstance.h"

#include <imgui.h>
#include <glm/gtc/type_ptr.hpp>

#include "ImGuiOverlay.h"

MeshInstance::MeshInstance(const std::string& name, std::shared_ptr<MeshAsset> asset, const Transform& transform) : SceneObject(name, transform), meshAsset(asset) {

    glm::mat4 mat = getTransformMatrix();
    vk::TransformMatrixKHR vkTransform{};
    vkTransform.matrix[0][0] = mat[0][0]; vkTransform.matrix[0][1] = mat[1][0]; vkTransform.matrix[0][2] = mat[2][0]; vkTransform.matrix[0][3] = mat[3][0];
    vkTransform.matrix[1][0] = mat[0][1]; vkTransform.matrix[1][1] = mat[1][1]; vkTransform.matrix[1][2] = mat[2][1]; vkTransform.matrix[1][3] = mat[3][1];
    vkTransform.matrix[2][0] = mat[0][2]; vkTransform.matrix[2][1] = mat[1][2]; vkTransform.matrix[2][2] = mat[2][2]; vkTransform.matrix[2][3] = mat[3][2];

    instanceData = vk::AccelerationStructureInstanceKHR{};
    instanceData.setTransform(vkTransform);
    instanceData.setInstanceCustomIndex(meshAsset->getMeshIndex()); // Use in Shader to access correct Buffer for Mesh
    instanceData.setMask(0xFF);
    instanceData.setInstanceShaderBindingTableRecordOffset(0);
    instanceData.setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);

    instanceData.setAccelerationStructureReference(meshAsset->getBlasAddress());
}


void MeshInstance::renderUiOverride() {
    ImGuiOverlay::tableRowLabel("Mesh Asset Name");
    ImGui::TextUnformatted(meshAsset->name.c_str());
}
