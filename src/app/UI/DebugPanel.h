#pragma once
#include "ImGuiComponent.h"
#include <chrono>
#include <string>

class DebugPanel : public ImGuiComponent {
public:
    explicit DebugPanel(std::string name);
    
    void onComputeFinished(float raytraceMs);
    void setSampleInfo(int current, int max);
    void resetRenderTimer();
    void renderUi() override;

    int getBvhMode() const { return visualizeBVH; }

private:
    float m_avgFps = 0.0f;
    float m_avgMs = 0.0f;
    int m_currentSample = 0;
    int m_maxSamples = 0;

    double m_accumMs = 0.0;
    int m_frameCount = 0;
    std::chrono::steady_clock::time_point m_lastResetTime = std::chrono::steady_clock::now();

    double m_renderTimeSeconds = 0.0;

    int visualizeBVH = 0; // 0 = Disabled, 1-3 = Modes

    static inline const char* modes[4] = { "Disabled", "Opaque", "Wireframe", "Heatmap" };
};
