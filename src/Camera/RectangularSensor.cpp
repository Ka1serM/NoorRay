#include "Camera/Camera.h"

#include <algorithm>
#include <imgui.h>
#include "Camera/CameraInstance.h"
#include "UI/ImGuiManager.h"

void Sensor::renderUi(CameraInstance& inst)
{
    bool changed = false;
    ImGuiManager::dragFloatRow("Sensor Width", widthMm, 0.1f, 0.1f, 500.f, [&](float v) {
        widthMm = std::max(v, 0.1f);
        changed = true;
    });
    ImGuiManager::dragFloatRow("Sensor Height", heightMm, 0.1f, 0.1f, 500.f, [&](float v) {
        heightMm = std::max(v, 0.1f);
        changed = true;
    });

    ImGuiManager::tableRowLabel("Resolution");
    int w = static_cast<int>(resolutionWidth);
    int h = static_cast<int>(resolutionHeight);
    ImGui::PushItemWidth((ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize("x").x
                          - ImGui::GetStyle().ItemSpacing.x * 2.f) * 0.5f);
    if (ImGui::DragInt("##ResW", &w, 1.f, 1, 16384))
        changed = true;
    ImGui::SameLine();
    ImGui::TextUnformatted("x");
    ImGui::SameLine();
    if (ImGui::DragInt("##ResH", &h, 1.f, 1, 16384))
        changed = true;
    ImGui::PopItemWidth();
    resolutionWidth  = static_cast<uint32_t>(std::max(w, 1));
    resolutionHeight = static_cast<uint32_t>(std::max(h, 1));

    if (changed)
        inst.markDirty();
}
