#pragma once
#include <memory>
#include <vector>

#include "Accel.h"
#include "Buffer.h"
#include "Texture.h"
#include "shaders/SharedStructs.h"
#include "Context.h"

//forward declare
class MeshInstance;
class MeshAsset;

class Scene {

    static constexpr size_t INVALID_INDEX = static_cast<size_t>(-1);

    size_t selectedInstanceIndex = INVALID_INDEX;

    std::vector<std::unique_ptr<MeshInstance>> meshInstances;

    std::vector<std::shared_ptr<MeshAsset>> meshAssets;

public:
    Scene(std::shared_ptr<Context> context);

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


    void buildBuffersGPU();

    void addMeshInstance(std::unique_ptr<MeshInstance> meshInstance);

    void addMeshAsset(const std::shared_ptr<MeshAsset>& meshInstance);

    void buildTLAS();

    void renderUI();
};
