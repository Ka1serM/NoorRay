#pragma once

#include <string>

#include "Scene.h"
#include "../Mesh/Transform.h"
#include "UI/ImGuiComponent.h"

//forward declaration to avoid circular dependency
class Scene;

class SceneObject : public ImGuiComponent {

public:
    Transform transform;
    Scene& scene;
    
    SceneObject(Scene& scene, std::string name, const Transform& transform);

    std::string getType() const override { return "Scene Object"; }

    std::string name;


    Transform getTransform() const {
        return transform;
    }

    glm::vec3 getPosition() const {
        return transform.getPosition();
    }

    glm::quat getRotation() const {
        return transform.getRotation();
    }

    glm::vec3 getRotationEuler() const {
        return transform.getRotationEuler();
    }

    glm::vec3 getScale() const {
        return transform.getScale();
    }

    void renderUi() override;
    virtual void setPosition(const glm::vec3& pos);
    virtual void setRotation(const glm::quat& rot);
    virtual void setRotationEuler(const glm::vec3& rot);
    virtual void setScale(const glm::vec3& scale);

    virtual void setTransform(const Transform& transf);
    virtual void setTransformMatrix(const glm::mat4& transf);
};
