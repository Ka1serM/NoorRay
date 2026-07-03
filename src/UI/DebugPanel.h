#pragma once
#include "ImGuiComponent.h"
#include <chrono>
#include <string>

class DebugPanel : public ImGuiComponent {
public:
    explicit DebugPanel(std::string name);
    
    void onComputeFinished(float raytraceMs);
    void setSampleInfo(int current, int max);
    void renderUi() override;

    int getBvhMode() const { return visualizeBVH; }

private:
    std::chrono::high_resolution_clock::time_point lastTime;
    float fps = 0.0f;
    float timeAccumulator = 0.0f;
    int frameCounter = 0;
    float m_raytraceMs = 0.0f;
    int m_currentSample = 0;
    int m_maxSamples = 0;

    int visualizeBVH = 0; // 0 = Disabled, 1-3 = Modes

    static inline const char* modes[4] = { "Disabled", "Opaque", "Wireframe", "Heatmap" };
};