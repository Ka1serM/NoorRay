#pragma once
#include <memory>
#include <vector>

#include "SceneObject.h"
#include "Camera/PerspectiveCamera.h"
#include "Vulkan/Accel.h"
#include "Shaders/SharedStructs.h"
#include "Vulkan/Context.h"
#include "Vulkan/Texture.h"

//forward declare
class MeshInstance;
class MeshAsset;

class Scene {

    static constexpr size_t INVALID_INDEX = static_cast<size_t>(-1);

    size_t selectedInstanceIndex = INVALID_INDEX;


    std::vector<std::shared_ptr<MeshAsset>> meshAssets;

    bool dirty;

public:
    PerspectiveCamera* activeCamera = nullptr;  // non-owning pointer
    std::vector<std::unique_ptr<SceneObject>> sceneObjects;

    Scene(const std::shared_ptr<Context>& context);

    std::shared_ptr<Context> context;

    std::vector<Material> materials;
    std::vector<PointLight> pointLights;
    std::vector<Texture> textures;

    std::vector<vk::DescriptorImageInfo> textureImageInfos;

    Buffer materialBuffer;
    Buffer pointLightBuffer;
    Buffer textureBuffer;

    Buffer meshBuffer;
    Buffer instancesBuffer;

    std::vector<vk::AccelerationStructureInstanceKHR> instances;

    std::unique_ptr<Accel> tlas;

    void setActiveCamera(PerspectiveCamera* camera) {
        activeCamera = camera;
    }

    PerspectiveCamera* getActiveCamera() const {
        return activeCamera;
    }

    void buildBuffersGPU();

    void addSceneObject(std::unique_ptr<SceneObject> sceneObject);

    void addMeshInstance(std::unique_ptr<MeshInstance> meshInstance);

    void addMeshAsset(const std::shared_ptr<MeshAsset>& meshInstance);

    void buildTLAS();

    void markDirty();
    void resetDirty();

    bool getDirty() const { return dirty; }
};
