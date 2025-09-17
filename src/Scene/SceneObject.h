#pragma once

#include <string>
#include <vector>
#include "Scene.h"
#include "../Mesh/Transform.h"
#include "../UI/ImGui/ImGuiComponent.h"

class Scene;

class SceneObject : public ImGuiComponent {

protected:
    Transform transform;
    Scene& scene;

    SceneObject* parent = nullptr;
    std::vector<SceneObject*> children;

    bool visible;

public:
    SceneObject(Scene& scene, const std::string& name, const Transform& transform);
    SceneObject(const SceneObject& other);
    virtual std::unique_ptr<SceneObject> clone() const;

    std::string getType() const override { return "Scene Object"; }
    void renderUi() override;

    bool isVisible() const { return visible; }
    void setVisible(const bool v) { visible = v; }

    SceneObject* getParent() const { return parent; }
    const std::vector<SceneObject*>& getChildren() const { return children; }

    void setParent(SceneObject* parent) { this->parent = parent; }
    
    void addChild(SceneObject* child) {
        children.push_back(child);
        child->setParent(this);
    }
    
    void removeChild(SceneObject* child) { std::erase(children, child); }
    
    vec3 getPosition() const {
        return transform.getPosition();
    }

    quat getRotation() const {
        return transform.getRotation();
    }

    vec3 getRotationEuler() const {
        return transform.getRotationEuler();
    }

    vec3 getScale() const {
        return transform.getScale();
    }

    Transform getTransform() const { return transform; }

    virtual void onTransformUpdated();
    
    virtual void setPosition(const vec3& pos);
    virtual void setRotation(const quat& rot);
    virtual void setRotationEuler(const vec3& rot);
    virtual void setScale(const vec3& scale);

    virtual void setLocalTransform(const Transform& transf);
    virtual void setWorldTransformFromMatrix(const mat4& transf);

    Transform getWorldTransform() const;
};