#pragma once

#include <memory>

#include "Scene.h"
#include "Mesh/MeshAsset.h"
#include "SceneObject.h"

class Scene;
class MeshAsset;

class MeshInstance : public SceneObject {
    const std::shared_ptr<MeshAsset> meshAsset;
    vk::AccelerationStructureInstanceKHR instanceData;

public:
    MeshInstance(Scene& scene, const std::string& name, std::shared_ptr<MeshAsset> asset, const Transform& transf);
    MeshInstance(const MeshInstance& other);
    
    std::unique_ptr<SceneObject> clone() const override;

    const MeshAsset& getMeshAsset() const { return *meshAsset.get(); }
    const vk::AccelerationStructureInstanceKHR& getInstanceData() const { return instanceData; }
    void onTransformUpdated() override;
};