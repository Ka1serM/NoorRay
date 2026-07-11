#pragma once

#include <string>

#include "UI/ImGuiComponent.h"

class Scene;

class RenderSettingsPanel : public ImGuiComponent
{
public:
    RenderSettingsPanel(std::string name, Scene& scene);

    void renderUi() override;

private:
    Scene& scene;
};
