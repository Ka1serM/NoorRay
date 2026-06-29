#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include "Scene.h"
#include "Inspectable.h"
#include "../Mesh/Transform.h"

class Scene;

class SceneObject : public Inspectable, public std::enable_shared_from_this<SceneObject> {
    friend class Scene;

protected:
    uint64_t id = 0;
    std::string name;
    Transform transform;
    Scene& scene;

    std::weak_ptr<SceneObject> parent;
    std::vector<std::weak_ptr<SceneObject>> children;

    bool visible;

public:
    SceneObject(Scene& scene, const std::string& name, const Transform& transform);
    SceneObject(const SceneObject& other);
    virtual std::unique_ptr<SceneObject> clone() const;

    uint64_t getId() const { return id; }
    const std::string& getName() const override { return name; }
    std::string getType() const override { return "Scene Object"; }
    bool renderUi() override;

    bool isVisible() const { return visible; }
    void setVisible(const bool v) { visible = v; }

    SceneObject* getParent() const { return parent.lock().get(); }
    std::shared_ptr<SceneObject> getParentPtr() const { return parent.lock(); }
    std::vector<std::shared_ptr<SceneObject>> getChildren() const;

    void setParent(const std::shared_ptr<SceneObject>& parent) { this->parent = parent; }
    void clearParent() { parent.reset(); }
    
    void addChild(const std::shared_ptr<SceneObject>& child) {
        children.push_back(child);
        child->setParent(shared_from_this());
    }
    
    void removeChild(const SceneObject* child);
    
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

private:
    void setId(const uint64_t newId) { id = newId; }
};
