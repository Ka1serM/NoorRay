#pragma once

#include "ImGuiComponent.h"
#include <string>

#include "Scene/Scene.h"
#include "Shaders/SharedStructs.h"

class EnvironmentPanel : public ImGuiComponent {
    Scene& scene;
    EnvironmentData enviromentData{};

public:
    EnvironmentPanel(Scene& scene);
    
    void renderUi() override;
    
    std::string getType() const override { return "Environment"; }
    
    const EnvironmentData& getEnvironmentData() const { return enviromentData; }
};
