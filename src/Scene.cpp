#include "Scene.h"

#include <imgui.h>

#include "MeshInstance.h"  // Include the full definition here
#include "MeshAsset.h"     // If needed

Scene::Scene(std::shared_ptr<Context> context) : context(context)
{
    PointLight pointLight{};
    pointLights.push_back(pointLight);
}

void Scene::buildBuffersGPU() {

    std::vector<MeshAddresses> meshAddresses;
    for (const auto& mesh : meshAssets)
        meshAddresses.push_back(mesh.get()->getBufferAddresses());

    meshBuffer = Buffer{context, Buffer::Type::Storage, sizeof(MeshAddresses) * meshAddresses.size(), meshAddresses.data()};

    materialBuffer = Buffer{context, Buffer::Type::Storage, sizeof(Material) * materials.size(), materials.data()};
    pointLightBuffer = Buffer{context, Buffer::Type::Storage, sizeof(PointLight) * pointLights.size(), pointLights.data()};

    for (const auto& texture : textures)
        textureImageInfos.push_back(texture.getDescriptorInfo());

    textureBuffer = Buffer{context, Buffer::Type::Storage, sizeof(vk::DescriptorImageInfo) * textureImageInfos.size(), textureImageInfos.data()};
}

void Scene::addMeshInstance(std::unique_ptr<MeshInstance> meshInstance) {
    instances.push_back(meshInstance->instanceData);
    meshInstances.push_back(std::move(meshInstance));
}

void Scene::addMeshAsset(const std::shared_ptr<MeshAsset> &meshAsset) {
    meshAsset->setBufferIndex(meshAssets.size());
    meshAssets.push_back(meshAsset);
}

void Scene::buildTLAS() {
    instancesBuffer = Buffer{context, Buffer::Type::AccelInput, sizeof(vk::AccelerationStructureInstanceKHR) * instances.size(), instances.data()};

    vk::AccelerationStructureGeometryInstancesDataKHR instancesData;
    instancesData.setArrayOfPointers(false);
    instancesData.setData(instancesBuffer.getDeviceAddress());

    vk::AccelerationStructureGeometryKHR instanceGeometry;
    instanceGeometry.setGeometryType(vk::GeometryTypeKHR::eInstances);
    instanceGeometry.setGeometry({instancesData});
    instanceGeometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

    tlas = std::make_unique<Accel>(context, instanceGeometry, static_cast<uint32_t>(instances.size()), vk::AccelerationStructureTypeKHR::eTopLevel);
}

void Scene::renderUI() {
    // --- OUTLINER ---
    ImGui::Begin("Outliner");

    for (size_t i = 0; i < meshInstances.size(); ++i) {

        // Create a label like "Cube###0", "Cube###1" – visible label is "Cube", ID is unique
        std::string label = meshInstances[i]->name + "###" + std::to_string(i);

        if (ImGui::Selectable(label.c_str(), selectedInstanceIndex == i))
            selectedInstanceIndex = i;
    }

    ImGui::End();

    // --- DETAILS ---
    ImGui::Begin("Details");

    if (selectedInstanceIndex != INVALID_INDEX)
        if (MeshInstance* selected = meshInstances.at(selectedInstanceIndex).get())
            selected->renderUi();
        else
            ImGui::Text("No mesh instance selected.");

    ImGui::End();
}