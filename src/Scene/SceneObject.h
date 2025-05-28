#pragma once

#include <string>

#include "../Mesh/Transform.h"
#include "UI/ImGuiComponent.h"
#include "Vulkan/Renderer.h"

class SceneObject : public ImGuiComponent {

protected:
    Transform transform;
    Renderer& renderer;

public:

    SceneObject(Renderer& renderer, std::string name, const Transform& transform);

    std::string getType() const override { return "Scene Object"; }

    std::string name;


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
