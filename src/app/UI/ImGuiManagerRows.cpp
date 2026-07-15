#include "UI/ImGuiManager.h"

#include <imgui.h>
#include <imgui_internal.h>
#include <glm/gtc/type_ptr.hpp>

void ImGuiManager::tableRowLabel(const char* label) {
    if (ImGui::GetCurrentTable()) {
        ImGui::TableNextRow(); ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted(label);
        ImGui::TableSetColumnIndex(1); ImGui::SetNextItemWidth(-FLT_MIN);
    } else { ImGui::TextUnformatted(label); ImGui::SameLine(); }
}

void ImGuiManager::checkboxRow(const char* label, bool value, const std::function<void(bool)>& setter) {
    tableRowLabel(label);
    if (ImGui::Checkbox((std::string("##") + label).c_str(), &value)) setter(value);
}

void ImGuiManager::dragFloatRow(const char* label, float value, const float speed,
    const float min, const float max, const std::function<void(float)>& setter) {
    tableRowLabel(label);
    if (ImGui::DragFloat((std::string("##") + label).c_str(), &value, speed, min, max,
        "%.3f", ImGuiSliderFlags_AlwaysClamp)) setter(value);
}

void ImGuiManager::dragFloat3Row(const char* label, glm::vec3 value, const float speed,
    const std::function<void(glm::vec3)>& setter) {
    tableRowLabel(label);
    if (ImGui::DragFloat3((std::string("##") + label).c_str(), glm::value_ptr(value), speed)) setter(value);
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
