#include "DebugPanel.h"
#include <imgui.h>
#include "ImGuiManager.h"

DebugPanel::DebugPanel(std::string name) 
    : ImGuiComponent(std::move(name)), lastTime(std::chrono::high_resolution_clock::now()) {}

void DebugPanel::onComputeFinished(const float raytraceMs) {
    const auto now = std::chrono::high_resolution_clock::now();
    const float deltaTime = std::chrono::duration<float>(now - lastTime).count();
    lastTime = now;

    m_raytraceMs = raytraceMs;

    timeAccumulator += deltaTime;
    frameCounter++;

    if (timeAccumulator >= 1.0f) {
        fps = static_cast<float>(frameCounter) / timeAccumulator;
        timeAccumulator = 0.0f;
        frameCounter = 0;
    }
}

void DebugPanel::renderUi() {
    ImGui::Begin(name.c_str());

    if (ImGui::BeginTable("Debug Table", 2, ImGuiTableFlags_SizingStretchProp)) {

        ImGuiManager::tableRowLabel("FPS");
        ImGui::Text("%.2f", fps);

        ImGuiManager::tableRowLabel("Raytrace");
        ImGui::Text("%.2f ms", m_raytraceMs);

        //ImGuiManager::tableRowLabel("Show BVH");
        //ImGui::Combo("##BVHMode", &visualizeBVH, modes, IM_ARRAYSIZE(modes));
        
        ImGui::EndTable();
    }

    ImGui::End();
}