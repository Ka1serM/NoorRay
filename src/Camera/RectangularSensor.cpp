#include "Camera/Sensor.h"

#include <algorithm>
#include <imgui.h>
#include "UI/ImGuiManager.h"

bool Sensor::renderUi()
{
    ImGuiManager::tableRowLabel("Sensor");
    if (!ImGui::TreeNodeEx("Rectangular###SensorProperties", ImGuiTreeNodeFlags_Framed))
        return false;

    bool changed = false;
    if (ImGui::BeginTable("SensorTable", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGuiManager::dragFloatRow("Width (mm)", widthMm, 0.1f, 0.1f, 500.f, [&](float v) {
            widthMm = std::max(v, 0.1f);
            changed = true;
        });
        ImGuiManager::dragFloatRow("Height (mm)", heightMm, 0.1f, 0.1f, 500.f, [&](float v) {
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
        ImGui::EndTable();
    }
    ImGui::TreePop();

    return changed;
}
