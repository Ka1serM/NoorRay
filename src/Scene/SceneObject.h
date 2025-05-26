#pragma once

#include <memory>
#include <string>
#include "../Mesh/Transform.h"
#include "UI/ImGuiComponent.h"

class Scene;

class SceneObject : public ImGuiComponent {

protected:
    Transform transform;
    std::shared_ptr<Scene> scene;

public:
    std::string getType() const override { return "Scene Object"; }

    std::string name;

    SceneObject(const std::shared_ptr<Scene>& scene, std::string  name, const Transform& transform);

    Transform getTransform() const {
        return transform;
    }

    glm::vec3 getPosition() const {
        return transform.getPosition();
    }

    glm::vec3 getRotationEuler() const {
        return transform.getRotationEuler();
    }

    glm::vec3 getScale() const {
        return transform.getScale();
    }

    void renderUi() override;
    virtual void setPosition(const glm::vec3& pos);
    virtual void setRotationEuler(const glm::vec3& rot);
    virtual void setScale(const glm::vec3& scale);
};