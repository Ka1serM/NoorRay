#include "UI/ImGuiManager.h"

#include <imgui.h>
#include <imgui_internal.h>
#include <glm/gtc/type_ptr.hpp>

#include "UI/MathInput.h"

void ImGuiManager::tableRowLabel(const char* label) {
    if (ImGui::GetCurrentTable()) {
        ImGui::TableNextRow(); ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted(label);
        ImGui::TableSetColumnIndex(1); ImGui::SetNextItemWidth(-FLT_MIN);
    } else { ImGui::TextUnformatted(label); ImGui::SameLine(); }
}

bool ImGuiManager::accordionRow(const char* label) {
    if (ImGui::GetCurrentTable()) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
    }

    // Do not push an indent when the accordion opens. Its contents are still
    // rendered conditionally beneath it, but continue using the full table.
    return ImGui::TreeNodeEx(label,
        ImGuiTreeNodeFlags_Framed
        | ImGuiTreeNodeFlags_SpanAllColumns
        | ImGuiTreeNodeFlags_LabelSpanAllColumns
        | ImGuiTreeNodeFlags_NoTreePushOnOpen);
}

void ImGuiManager::checkboxRow(const char* label, bool value, const std::function<void(bool)>& setter) {
    tableRowLabel(label);
    if (ImGui::Checkbox((std::string("##") + label).c_str(), &value)) setter(value);
}

void ImGuiManager::dragFloatRow(const char* label, float value, const float speed,
    const float min, const float max, const std::function<void(float)>& setter) {
    tableRowLabel(label);
    if (MathInput::DragFloat((std::string("##") + label).c_str(), &value, speed, min, max,
        "%.3f", ImGuiSliderFlags_AlwaysClamp)) setter(value);
}

void ImGuiManager::dragFloat3Row(const char* label, glm::vec3 value, const float speed,
    const std::function<void(glm::vec3)>& setter) {
    tableRowLabel(label);
    if (MathInput::DragFloat3((std::string("##") + label).c_str(), glm::value_ptr(value), speed)) setter(value);
}

void ImGuiManager::colorEdit3Row(const char* label, const glm::vec3 value,
    const std::function<void(glm::vec3)>& setter) {
    tableRowLabel(label);
    if (glm::vec3 temp = value; ImGui::ColorEdit3((std::string("##") + label).c_str(), glm::value_ptr(temp))) setter(temp);
}

void ImGuiManager::colorEdit4Row(const char* label, const glm::vec4 value,
    const std::function<void(glm::vec4)>& setter) {
    tableRowLabel(label);
    if (glm::vec4 temp = value; ImGui::ColorEdit4((std::string("##") + label).c_str(), glm::value_ptr(temp))) setter(temp);
}
