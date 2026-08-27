#pragma once
#include "ImGuiComponent.h"
#include <chrono>
#include <string>

class DebugPanel : public ImGuiComponent {
public:
    explicit DebugPanel(std::string name);

    // Called once per presented frame by the UI loop.
    //   frameSeconds     wall-clock duration of the loop iteration, which only
    //                    advances the render timer. FPS is raytracer throughput
    //                    (1000 / raytraceMs), not application frame rate.
    //   raytraceMs       GPU time of a single path-tracing dispatch.
    //   samplesThisFrame dispatches recorded this frame; zero once the sample
    //                    budget is reached and the renderer idles, which is what
    //                    freezes the render timer and the raytrace average.
    void onFrameCompleted(double frameSeconds, float raytraceMs, int samplesThisFrame);
    void setSampleInfo(int current, int max);
    void resetRenderTimer();
    void renderUi() override;

    int getBvhMode() const { return visualizeBVH; }

private:
    float m_avgFps = 0.0f;
    float m_avgMs = 0.0f;
    int m_currentSample = 0;
    int m_maxSamples = 0;

    // Averaged over a one-second window so the readout stays legible instead of
    // flickering with per-frame jitter.
    double m_accumMs = 0.0;
    int m_dispatchCount = 0;
    bool m_published = false;
    std::chrono::steady_clock::time_point m_lastResetTime = std::chrono::steady_clock::now();

    double m_renderTimeSeconds = 0.0;

    int visualizeBVH = 0; // 0 = Disabled, 1-3 = Modes

    static inline const char* modes[4] = { "Disabled", "Opaque", "Wireframe", "Heatmap" };
};
