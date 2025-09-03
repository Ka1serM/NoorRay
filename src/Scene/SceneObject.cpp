#include "SceneObject.h"
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>

#include "MeshInstance.h"
#include "Scene.h"
#include "../UI/ImGuiManager.h"

SceneObject::SceneObject(Scene& scene, const std::string& name, const Transform& transform) : transform(transform), scene(scene), ImGuiComponent(name) {
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
    } else
        transform.setFromMatrix(worldMatrix);

    onTransformUpdated();
}


void SceneObject::renderUi() {
    bool anyChanged = false;

    // Name
    ImGuiManager::tableRowLabel("Name");
    ImGui::TextUnformatted(name.c_str());
    
    // Position
    ImGuiManager::dragFloat3Row("Position", transform.getPosition(), 0.01f, [&](const glm::vec3 v) {
        setPosition(v); anyChanged = true;
    });

    // Rotation
    ImGuiManager::dragFloat3Row("Rotation", transform.getRotationEuler(), 0.1f, [&](const glm::vec3 v) {
        setRotationEuler(v); anyChanged = true;
    });

    // Scale
    ImGuiManager::dragFloat3Row("Scale", transform.getScale(), 0.01f, [&](const glm::vec3 v) {
        setScale(v); anyChanged = true;
    });

    if (anyChanged)
       scene.setDirtyFlag(Accumulation);
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
    for (SceneObject* child : children)
        child->onTransformUpdated();
}