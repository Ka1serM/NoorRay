#include "SceneObject.h"
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>
#include "MeshInstance.h"
#include "Scene.h"
#include "UI/ImGuiManager.h"

SceneObject::SceneObject(const std::string& name, const Transform& transform)
    : name(name), transform(transform), visible(true)
{
}

SceneObject::SceneObject(Scene& scene, const std::string& name, const Transform& transform)
    : SceneObject(name, transform)
{
    this->scene = &scene;
}

SceneObject::SceneObject(const SceneObject& other)
    : id(0),
      name(other.name + " Copy"),
      scene(other.scene), visible(other.visible),
      transform(other.transform)
{}

std::unique_ptr<SceneObject> SceneObject::clone() const {
    return std::make_unique<SceneObject>(*this);
}

std::vector<std::shared_ptr<SceneObject>> SceneObject::getChildren() const {
    std::vector<std::shared_ptr<SceneObject>> result;
    result.reserve(children.size());
    for (const auto& child : children)
        if (auto locked = child.lock())
            result.push_back(std::move(locked));
    return result;
}

void SceneObject::removeChild(const SceneObject* child) {
    std::erase_if(children, [child](const std::weak_ptr<SceneObject>& candidate) {
        const auto locked = candidate.lock();
        return !locked || locked.get() == child;
    });
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
    if (const auto parentPtr = parent.lock()) {
        const mat4 parentWorld = parentPtr->getWorldTransform().getMatrix();
        const mat4 localMatrix = inverse(parentWorld) * worldMatrix;
        transform.setFromMatrix(localMatrix);
    } else
        transform.setFromMatrix(worldMatrix);

    onTransformUpdated();
}


bool SceneObject::renderUi() {
    bool anyChanged = false;

    // Name
    ImGuiManager::tableRowLabel("Name");
    ImGui::TextUnformatted(name.c_str());
    
    // Position
    ImGuiManager::dragFloat3Row("Position", transform.getPosition(), 0.01f, [&](const vec3 v) {
        setPosition(v); anyChanged = true;
    });

    // Rotation
    ImGuiManager::dragFloat3Row("Rotation", transform.getRotationEuler(), 0.1f, [&](const vec3 v) {
          setRotationEuler(v);
        anyChanged = true;
    });

    // Scale
    ImGuiManager::dragFloat3Row("Scale", transform.getScale(), 0.01f, [&](const vec3 v) {
        setScale(v); anyChanged = true;
    });

    if (anyChanged)
    {
        onTransformUpdated();
       if (scene) scene->setDirtyFlag(Accumulation);
    }
    return anyChanged;
}

Transform SceneObject::getWorldTransform() const {
    mat4 worldMatrix = transform.getMatrix();

    auto currentParent = parent.lock();
    while (currentParent != nullptr) {
        worldMatrix = currentParent->transform.getMatrix() * worldMatrix;
        currentParent = currentParent->parent.lock();
    }

    Transform worldTransform;
    worldTransform.setFromMatrix(worldMatrix);
    return worldTransform;
}

void SceneObject::onTransformUpdated() {
    if (scene) scene->setDirtyFlag(Accumulation);
    for (const auto& child : getChildren())
        child->onTransformUpdated();
}
