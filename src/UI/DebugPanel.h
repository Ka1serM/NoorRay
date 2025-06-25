#pragma once

#include "ImGuiComponent.h"
#include <string>

class DebugPanel : public ImGuiComponent {
public:
    DebugPanel();

    void renderUi() override;

    std::string getType() const override { return "Debug"; }

    void setFps(const float newFps) {
        fps = newFps;
    }
    
private:
    float fps = 0.0f;
};