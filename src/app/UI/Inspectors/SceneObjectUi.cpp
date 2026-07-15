#include "Scene/SceneObject.h"
#include <imgui.h>
#include "UI/ImGuiManager.h"
#include "UI/ObjectUi.h"

namespace
{
bool renderSceneObject(SceneObject& object)
{
    bool changed = false;
    ImGuiManager::tableRowLabel("Name");
    ImGui::TextUnformatted(object.getName().c_str());
    ImGuiManager::dragFloat3Row("Position", object.getPosition(), 0.01f,
        [&](const glm::vec3 value) { object.setPosition(value); changed = true; });
    ImGuiManager::dragFloat3Row("Rotation", object.getRotationEuler(), 0.1f,
        [&](const glm::vec3 value) { object.setRotationEuler(value); changed = true; });
    ImGuiManager::dragFloat3Row("Scale", object.getScale(), 0.01f,
        [&](const glm::vec3 value) { object.setScale(value); changed = true; });
    return changed;
}

}

void ObjectUiVisitor::visit(SceneObject& object)
{
    changed |= renderSceneObject(object);
}
