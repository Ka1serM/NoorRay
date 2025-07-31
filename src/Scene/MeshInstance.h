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
    void updateInstanceTransform();
    void renderUi() override;

    const MeshAsset& getMeshAsset() const { return *meshAsset.get(); }
    const vk::AccelerationStructureInstanceKHR& getInstanceData() const { return instanceData; }
    void setPosition(const glm::vec3& pos) override;
    void setRotation(const glm::quat& rot) override;
    void setRotationEuler(const glm::vec3& rot) override;
    void setScale(const glm::vec3& scale) override;
};