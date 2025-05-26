#pragma once

#include <memory>

#include "../Mesh/MeshAsset.h"
#include "SceneObject.h"

class MeshInstance : public SceneObject {
public:
    const std::shared_ptr<MeshAsset> meshAsset;
    vk::AccelerationStructureInstanceKHR instanceData;


    MeshInstance(const std::shared_ptr<Scene>& scene, const std::string& name, std::shared_ptr<MeshAsset> asset, const Transform& transform);
    void updateInstanceTransform();
    void renderUi() override;

    void setPosition(const glm::vec3& pos) override;
    void setRotationEuler(const glm::vec3& rot) override;
    void setScale(const glm::vec3& scale) override;
};