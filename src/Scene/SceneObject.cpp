#include "SceneObject.h"
#include <glm/gtc/type_ptr.hpp>
#include <utility>
#include <imgui.h>

#include "Scene.h"
#include "../UI/ImGuiManager.h"

SceneObject::SceneObject(Scene& scene, std::string  name, const Transform& transform) : transform(transform), scene(scene), name(std::move(name)) {

}

void SceneObject::setPosition(const glm::vec3& position) {
    transform.setPosition(position);
    scene.setAccumulationDirty();
}

void SceneObject::setRotation(const glm::quat& rotation) {
    transform.setRotation(rotation);
    scene.setAccumulationDirty();
}

void SceneObject::setRotationEuler(const glm::vec3& rotation) {
    transform.setRotationEuler(rotation);
    scene.setAccumulationDirty();
}

void SceneObject::setScale(const glm::vec3& scale) {
    transform.setScale(scale);
    scene.setAccumulationDirty();
}

void SceneObject::setTransform(const Transform& transf) {
    transform = transf;
    scene.setAccumulationDirty();
}

void SceneObject::setTransformMatrix(const glm::mat4& transf) {
    transform.setFromMatrix(transf);
    scene.setAccumulationDirty();
}

void SceneObject::renderUi() {
    bool anyChanged = false;

    // Name
    ImGui::TableNextRow();
    ImGuiManager::tableRowLabel("Name");
    ImGui::TextUnformatted(name.c_str());
    
    // Position
    ImGui::TableNextRow();
    ImGuiManager::dragFloat3Row("Position", transform.getPosition(), 0.01f, [&](const glm::vec3 v) {
        setPosition(v); anyChanged = true;
    });

    // Rotation
    ImGui::TableNextRow();
    ImGuiManager::dragFloat3Row("Rotation", transform.getRotationEuler(), 0.1f, [&](const glm::vec3 v) {
        setRotationEuler(v); anyChanged = true;
    });

    // Scale
    ImGui::TableNextRow();
    ImGuiManager::dragFloat3Row("Scale", transform.getScale(), 0.01f, [&](const glm::vec3 v) {
        setScale(v); anyChanged = true;
    });

    if (anyChanged)
        scene.setAccumulationDirty();
}