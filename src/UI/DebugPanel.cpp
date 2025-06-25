#include "DebugPanel.h"
#include <imgui.h>

DebugPanel::DebugPanel() {}

void DebugPanel::renderUi() {
    ImGui::Begin(getType().c_str());

    // Info section
    ImGui::SeparatorText("Info");
    ImGui::Text("FPS: %.2f", fps);
    
    ImGui::End();
}