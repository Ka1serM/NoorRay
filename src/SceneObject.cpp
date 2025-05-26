#include "SceneObject.h"
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>

#include "ImGuiOverlay.h"

SceneObject::SceneObject(const std::string& name, const Transform& transform) : name(name), transform(transform) {
}

glm::mat4 SceneObject::getTransformMatrix() const {
    return transform.getMatrix();
}

void SceneObject::renderUi() {
    renderUiBegin();

    // Name (read-only)
    ImGuiOverlay::tableRowLabel("Name");
    ImGui::TextUnformatted(name.c_str());

    // Position
    ImGuiOverlay::dragFloat3Row("Position", transform.getPosition(), 0.1f, [&](glm::vec3 v) {
        transform.setPosition(v);
    });

    // Rotation
    ImGuiOverlay::dragFloat3Row("Rotation", transform.getRotationEuler(), 1.0f, [&](glm::vec3 v) {
        transform.setRotationEuler(v);
    });

    // Scale
    ImGuiOverlay::dragFloat3Row("Scale", transform.getScale(), 0.01f, [&](glm::vec3 v) {
        transform.setScale(v);
    });

    renderUiOverride();

    renderUiEnd();
}

void SceneObject::renderUiBegin() {
    if (!ImGui::BeginTable("ObjectDetails", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
        return;

    ImGui::TableSetupColumn("Label");
    ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
}

void SceneObject::renderUiEnd() {
    ImGui::EndTable();
}

void SceneObject::renderUiOverride() {
}
