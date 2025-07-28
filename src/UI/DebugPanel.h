#pragma once

#include "ImGuiComponent.h"
#include <string>

class DebugPanel : public ImGuiComponent {
private:
    float fps = 0.0f;
    
public:
    DebugPanel();

    void renderUi() override;

    std::string getType() const override { return "Debug"; }

    void setFps(const float newFps) {
        fps = newFps;
    }
};