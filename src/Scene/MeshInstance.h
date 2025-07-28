#pragma once

#include <memory>

#include "Scene.h"
#include "Mesh/MeshAsset.h"
#include "SceneObject.h"

class Scene;
class MeshAsset;

class MeshInstance : public SceneObject {
public:
    const std::shared_ptr<MeshAsset> meshAsset;
    vk::AccelerationStructureInstanceKHR instanceData;

    MeshInstance(Scene& scene, const std::string& name, std::shared_ptr<MeshAsset> asset, const Transform& transf);
    void updateInstanceTransform();
    void renderUi() override;

    void setPosition(const glm::vec3& pos) override;
    void setRotation(const glm::quat& rot) override;
    void setRotationEuler(const glm::vec3& rot) override;
    void setScale(const glm::vec3& scale) override;
};