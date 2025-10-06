#include "SceneObject.h"
#include <glm/gtc/type_ptr.hpp>
#include "MeshInstance.h"
#include "Scene.h"

SceneObject::SceneObject(Scene& scene, const std::string& name, const Transform& transform) 
    : transform(transform), scene(scene), visible(true), name(name)
{
    // ID is initialized to -1 in the header. It will be set by Scene::add().
}

// Copy constructor for cloning.
SceneObject::SceneObject(const SceneObject& other)
    :name( other.name + " Copy"),
      scene(other.scene), 
      visible(other.visible),
      transform(other.transform)
{
    // The new, cloned object has no valid ID yet. 
    // It will get a new unique ID when it's added to the scene.
    // Do NOT copy the ID from 'other'.
}

std::unique_ptr<SceneObject> SceneObject::clone() const {
    // Use std::make_unique with the copy constructor.
    return std::make_unique<SceneObject>(*this);
}

void SceneObject::setPosition(const vec3& position) {
    transform.setPosition(position);
    onTransformUpdated();
}

void SceneObject::setRotation(const quat& rotation) {
    transform.setRotation(rotation);
    onTransformUpdated();
}

void SceneObject::setRotationEuler(const vec3& rotation) {
    transform.setRotationEuler(rotation);
    onTransformUpdated();
}

void SceneObject::setScale(const vec3& scale) {
    transform.setScale(scale);
    onTransformUpdated();
}

void SceneObject::setLocalTransform(const Transform& transf) {
    transform = transf;
    onTransformUpdated();
}

void SceneObject::setWorldTransformFromMatrix(const mat4& worldMatrix) {
    if (parent) {
        const mat4 parentWorld = parent->getWorldTransform().getMatrix();
        const mat4 localMatrix = inverse(parentWorld) * worldMatrix;
        transform.setFromMatrix(localMatrix);
    } else {
        transform.setFromMatrix(worldMatrix);
    }

    onTransformUpdated();
}

Transform SceneObject::getWorldTransform() const {
    mat4 worldMatrix = transform.getMatrix();

    const SceneObject* currentParent = parent;
    while (currentParent != nullptr) {
        worldMatrix = currentParent->transform.getMatrix() * worldMatrix;
        currentParent = currentParent->parent;
    }

    Transform worldTransform;
    worldTransform.setFromMatrix(worldMatrix);
    return worldTransform;
}

void SceneObject::onTransformUpdated() {
    scene.setDirtyFlag(Accumulation);

    // Also mark the TLAS dirty if this object is a mesh instance,
    // as its bounding box will have moved.
    if (dynamic_cast<const MeshInstance*>(this)) {
        scene.setDirtyFlag(TLAS);
    }

    for (SceneObject* child : children)
        child->onTransformUpdated();
}
