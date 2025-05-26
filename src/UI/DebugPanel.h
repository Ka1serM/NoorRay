#pragma once

#include "ImGuiComponent.h"
#include <functional>
#include <string>

class DebugPanel : public ImGuiComponent {
public:
    DebugPanel();

    void renderUi() override;

    std::string getType() const override { return "Debug"; }

    void setFps(float newFps) {
        fps = newFps;
    }

    void setModeChangedCallback(std::function<void()> callback) {
        modeChangedCallback = std::move(callback);
    }

    void setSceneStats(int triangles, int instances, int lights) {
        triangleCount = triangles;
        instanceCount = instances;
        lightCount = lights;
    }
    
private:
    float fps = 0.0f;
    int triangleCount = 0;
    int instanceCount = 0;
    int lightCount = 0;
    bool isRayTracing = false;
    std::function<void()> modeChangedCallback;
};