#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include "Scene.h"
#include "Inspectable.h"
#include "Geometry/Mesh/Transform.h"

class Scene;
class SceneObjectVisitor;

class SceneObject : public Inspectable, public std::enable_shared_from_this<SceneObject> {
    friend class Scene;

protected:
    SceneObjectHandle handle;
    std::string name;
    Transform transform;
    Scene* scene{};

    std::weak_ptr<SceneObject> parent;
    std::vector<std::weak_ptr<SceneObject>> children;

    bool visible;
    std::string sourceType;
    std::string sourcePath;

public:
    SceneObject(const std::string& name, const Transform& transform);
    SceneObject(Scene& scene, const std::string& name, const Transform& transform);
    SceneObject(const SceneObject& other);
    virtual std::unique_ptr<SceneObject> clone() const;
    virtual void accept(SceneObjectVisitor& visitor);

    SceneObjectHandle getHandle() const { return handle; }
    Scene* getScene() const { return scene; }
    const std::string& getName() const override { return name; }
    std::string getType() const override { return "Scene Object"; }

    bool isVisible() const { return visible; }
    void setVisible(const bool v) { visible = v; }
    void setSource(const std::string& type, const std::string& path) { sourceType = type; sourcePath = path; }
    const std::string& getSourceType() const { return sourceType; }
    const std::string& getSourcePath() const { return sourcePath; }

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
    void setHandle(const SceneObjectHandle newHandle) { handle = newHandle; }
};
