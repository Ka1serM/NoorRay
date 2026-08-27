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

void DebugPanel::onFrameCompleted(const double frameSeconds, const float raytraceMs,
    const int samplesThisFrame) {
    if (samplesThisFrame > 0) {
        m_accumMs += raytraceMs;
        m_dispatchCount++;
        // Wall clock spent converging, so the timer stops on its own once the
        // sample budget is reached rather than counting idle frames.
        m_renderTimeSeconds += frameSeconds;
    }

    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration<double>(now - m_lastResetTime).count();
    const bool windowComplete = elapsed >= 1.0;
    // Until the first window closes, publish the growing average every frame.
    // Waiting the full second instead left the panel reading a flat 0.00 on
    // startup, which is exactly what a broken readout looks like.
    if (!windowComplete && m_published)
        return;
    // Hold the last measurement while the renderer idles; a completed render
    // has no new dispatches to average and zeroing it would just read as broken.
    if (m_dispatchCount > 0) {
        m_avgMs = static_cast<float>(m_accumMs / m_dispatchCount);
        // Raytracer throughput, not application frame rate: this is how many
        // samples per second the path-tracing dispatch itself sustains, with
        // UI, composite and present cost excluded.
        m_avgFps = m_avgMs > 0.0f ? 1000.0f / m_avgMs : 0.0f;
    }
    if (!windowComplete)
        return;
    m_accumMs = 0.0;
    m_dispatchCount = 0;
    m_lastResetTime = now;
    m_published = true;
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
