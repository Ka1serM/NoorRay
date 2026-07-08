#include "DebugPanel.h"
#include <imgui.h>
#include "ImGuiManager.h"

DebugPanel::DebugPanel(std::string name) 
    : ImGuiComponent(std::move(name)) {}

void DebugPanel::setSampleInfo(const int current, const int max) {
    m_currentSample = current;
    m_maxSamples = max;
}

void DebugPanel::onComputeFinished(const float raytraceMs) {
    m_raytraceMs = raytraceMs;
    fps = raytraceMs > 0.0f ? 1000.0f / raytraceMs : 0.0f;
}

void DebugPanel::renderUi() {
    ImGui::Begin(name.c_str());

    if (ImGui::BeginTable("Debug Table", 2, ImGuiTableFlags_SizingStretchProp)) {

        ImGuiManager::tableRowLabel("FPS");
        ImGui::Text("%.2f", fps);

        ImGuiManager::tableRowLabel("Raytrace");
        ImGui::Text("%.2f ms", m_raytraceMs);

        ImGuiManager::tableRowLabel("Samples");
        ImGui::Text("%d / %d", m_currentSample, m_maxSamples);

        //ImGuiManager::tableRowLabel("Show BVH");
        //ImGui::Combo("##BVHMode", &visualizeBVH, modes, IM_ARRAYSIZE(modes));
        
        ImGui::EndTable();
    }

    ImGui::End();
}
