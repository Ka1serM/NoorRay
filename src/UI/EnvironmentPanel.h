#pragma once

#include "ImGuiComponent.h"
#include <string>
#include "Scene/Scene.h"

class EnvironmentPanel : public ImGuiComponent {
    Scene& scene;

public:
    EnvironmentPanel(std::string name, Scene& scene);
    void renderUi() override;
};
