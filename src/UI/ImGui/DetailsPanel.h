#pragma once

#include "ImGuiComponent.h"
#include "Scene/Scene.h"

class DetailsPanel : public ImGuiComponent {
public:
    DetailsPanel(std::string name, Scene& scene);
    void renderUi() override;

private:
    Scene& scene;
};