#include "DebugPanel.h"
#include <imgui.h>
#include "ImGuiManager.h"

DebugPanel::DebugPanel(std::string name) 
    : ImGuiComponent(std::move(name)) {}

void DebugPanel::setSampleInfo(const int current, const int max) {
    m_currentSample = current;
    m_maxSamples = max;
}

void DebugPanel::resetRenderTimer() {
    m_renderTimeSeconds = 0.0;
}

void DebugPanel::onComputeFinished(const float raytraceMs) {
    m_renderTimeSeconds += static_cast<double>(raytraceMs) / 1000.0;
    m_accumMs += raytraceMs;
    m_accumFps += raytraceMs > 0.0f ? 1000.0f / raytraceMs : 0.0f;
    m_frameCount++;

    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration<double>(now - m_lastResetTime).count();
    if (elapsed >= 1.0) {
        m_avgMs = static_cast<float>(m_accumMs / m_frameCount);
        m_avgFps = static_cast<float>(m_accumFps / m_frameCount);
        m_accumMs = 0.0;
        m_accumFps = 0.0;
        m_frameCount = 0;
        m_lastResetTime = now;
    }
}

void DebugPanel::renderUi() {
    ImGui::Begin(name.c_str());

    if (ImGui::BeginTable("Timings Table", 2, ImGuiTableFlags_SizingStretchProp)) {

        ImGuiManager::tableRowLabel("FPS");
        ImGui::Text("%.2f", m_avgFps);

        ImGuiManager::tableRowLabel("Raytrace");
        ImGui::Text("%.2f ms", m_avgMs);

        ImGuiManager::tableRowLabel("Samples");
        ImGui::Text("%d / %d", m_currentSample, m_maxSamples);

        ImGuiManager::tableRowLabel("Render Time");
        ImGui::Text("%.2f s", m_renderTimeSeconds);

        //ImGuiManager::tableRowLabel("Show BVH");
        //ImGui::Combo("##BVHMode", &visualizeBVH, modes, IM_ARRAYSIZE(modes));
        
        ImGui::EndTable();
    }

    ImGui::End();
}
