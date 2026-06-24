#pragma once

#include "ImGuiComponent.h"
#include <string>
#include "Scene/Scene.h"
#include "Shaders/Shared.h"

class EnvironmentPanel : public ImGuiComponent {
    Scene& scene;
    EnvironmentSettings enviromentData{};

public:
    EnvironmentPanel(std::string name, Scene& scene);
    void renderUi() override;
    
    const EnvironmentSettings& getEnvironmentData() const { return enviromentData; }
};