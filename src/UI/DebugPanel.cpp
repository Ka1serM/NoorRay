#include "DebugPanel.h"
#include <imgui.h>

DebugPanel::DebugPanel() {}

void DebugPanel::renderUi() {
    ImGui::Begin(getType().c_str());

    // Info section
    ImGui::SeparatorText("Info");
    ImGui::Text("FPS: %.2f", fps);
    
    // Settings section
    ImGui::SeparatorText("Settings");
    bool oldIsRayTracing = isRayTracing;

    if (ImGui::RadioButton("Path Tracing", !isRayTracing))
        isRayTracing = false;

    ImGui::SameLine();

    if (ImGui::RadioButton("Ray Tracing", isRayTracing))
        isRayTracing = true;

    if (oldIsRayTracing != isRayTracing && modeChangedCallback)
        modeChangedCallback();

    ImGui::End();
}