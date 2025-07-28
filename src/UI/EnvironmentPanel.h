#pragma once

#include "ImGuiComponent.h"
#include <string>

#include "Scene/Scene.h"
#include "Vulkan/Renderer.h"

class EnvironmentPanel : public ImGuiComponent {
private:
    Scene& scene;
    int hdriTexture;
    
public:
    EnvironmentPanel(Scene& scene);

    void renderUi() override;

    std::string getType() const override { return "Environment"; }

    int getHdriTexture() const { return hdriTexture; }
};
