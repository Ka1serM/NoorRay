#pragma once

#include "ImGuiComponent.h"
#include <string>

#include "Vulkan/Renderer.h"

class EnvironmentPanel : public ImGuiComponent {
public:
    EnvironmentPanel(Renderer& renderer);

    void renderUi() override;

    std::string getType() const override { return "Environment"; }

    int getHdriTexture() const { return hdriTexture; }
    
private:
    Renderer& renderer;
    int hdriTexture;
};
