#pragma once

#include "ImGuiComponent.h"
#include <functional>
#include <string>

class DebugPanel : public ImGuiComponent {
public:
    DebugPanel();

    void renderUi() override;

    std::string getType() const override { return "Debug"; }

    void setFps(const float newFps) {
        fps = newFps;
    }

    void setModeChangedCallback(std::function<void()> callback) {
        modeChangedCallback = std::move(callback);
    }
    
private:
    float fps = 0.0f;
    bool isRayTracing = false;
    std::function<void()> modeChangedCallback;
};