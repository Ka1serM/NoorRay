#pragma once

#include <string>
#include <vector>
#include "Scene.h"
#include "../Mesh/Transform.h"
#include "UI/ImGuiComponent.h"

class Scene;

class SceneObject : public ImGuiComponent {

protected:
    Transform transform;
    Scene& scene;

    SceneObject* parent = nullptr;
    std::vector<SceneObject*> children;

public:
    SceneObject(Scene& scene, const std::string& name, const Transform& transform);

    std::string getType() const override { return "Scene Object"; }
    void renderUi() override;

    SceneObject* getParent() const { return parent; }
    const std::vector<SceneObject*>& getChildren() const { return children; }

    void setParent(SceneObject* parent) { this->parent = parent; }
    void addChild(SceneObject* child) { children.push_back(child); }
    void removeChild(SceneObject* child) { std::erase(children, child); }
    
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

    Transform getTransform() const { return transform; }

    virtual void onTransformUpdated();
    
    virtual void setPosition(const glm::vec3& pos);
    virtual void setRotation(const glm::quat& rot);
    virtual void setRotationEuler(const glm::vec3& rot);
    virtual void setScale(const glm::vec3& scale);

    virtual void setLocalTransform(const Transform& transf);
    virtual void setWorldTransformFromMatrix(const glm::mat4& transf);

    Transform getWorldTransform() const;
};