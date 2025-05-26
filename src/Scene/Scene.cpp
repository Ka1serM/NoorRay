#include "Scene.h"

#include "MeshInstance.h"
#include "../Mesh/MeshAsset.h"

Scene::Scene(const std::shared_ptr<Context>& context) : dirty(false), context(context)
{
    PointLight pointLight{};
    pointLights.push_back(pointLight);
}

void Scene::buildBuffersGPU() {

    std::vector<MeshAddresses> meshAddresses;
    for (const auto& mesh : meshAssets)
        meshAddresses.push_back(mesh->getBufferAddresses());

    meshBuffer = Buffer{context, Buffer::Type::Storage, sizeof(MeshAddresses) * meshAddresses.size(), meshAddresses.data()};

    materialBuffer = Buffer{context, Buffer::Type::Storage, sizeof(Material) * materials.size(), materials.data()};
    pointLightBuffer = Buffer{context, Buffer::Type::Storage, sizeof(PointLight) * pointLights.size(), pointLights.data()};

    for (const auto& texture : textures)
        textureImageInfos.push_back(texture.getDescriptorInfo());

    textureBuffer = Buffer{context, Buffer::Type::Storage, sizeof(vk::DescriptorImageInfo) * textureImageInfos.size(), textureImageInfos.data()};
}

void Scene::addSceneObject(std::unique_ptr<SceneObject> sceneObject) {
    // If the object is a PerspectiveCamera, save a raw pointer to activeCamera
    if (auto camera = dynamic_cast<PerspectiveCamera*>(sceneObject.get()))
        activeCamera = camera;

    sceneObjects.push_back(std::move(sceneObject));
}


void Scene::addMeshInstance(std::unique_ptr<MeshInstance> meshInstance) {
    instances.push_back(meshInstance->instanceData);
    sceneObjects.push_back(std::unique_ptr<SceneObject>(std::move(meshInstance).release()));
}

void Scene::addMeshAsset(const std::shared_ptr<MeshAsset> &meshAsset) {
    meshAsset->setMeshIndex(meshAssets.size());
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

void Scene::markDirty()
{
    dirty = true;
}

void Scene::resetDirty()
{
    dirty = false;
}