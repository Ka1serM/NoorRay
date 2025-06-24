#include "SceneObject.h"
#include <glm/gtc/type_ptr.hpp>
#include <utility>
#include <imgui.h>
#include "../UI/ImGuiManager.h"
#include "Vulkan/Renderer.h"

SceneObject::SceneObject(Renderer& renderer, std::string  name, const Transform& transform) : transform(transform), renderer(renderer), name(std::move(name)) {

}

void SceneObject::setPosition(const glm::vec3& position) {
    transform.setPosition(position);
    renderer.markDirty();
}

void SceneObject::setRotation(const glm::quat& rotation) {
    transform.setRotation(rotation);
    renderer.markDirty();
}

void SceneObject::setRotationEuler(const glm::vec3& rotation) {
    transform.setRotationEuler(rotation);
    renderer.markDirty();
}

void SceneObject::setScale(const glm::vec3& scale) {
    transform.setScale(scale);
    renderer.markDirty();
}

void SceneObject::renderUi() {

    // Name
    ImGuiManager::tableRowLabel("Name");
    ImGui::TextUnformatted(name.c_str());

    ImGui::SeparatorText("Transform");

    // Position
    ImGuiManager::dragFloat3Row("Position", transform.getPosition(), 0.01f, [&](const glm::vec3 v) {
        setPosition(v);
    });

    // Rotation
    ImGuiManager::dragFloat3Row("Rotation", transform.getRotationEuler(), 0.1f, [&](const glm::vec3 v) {
        setRotationEuler(v);
    });

    // Scale
    ImGuiManager::dragFloat3Row("Scale", transform.getScale(), 0.01f, [&](const glm::vec3 v) {
        setScale(v);
    });
}