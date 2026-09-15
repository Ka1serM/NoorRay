#pragma once

#include <optional>
#include <string>

#include "UI/ImGuiComponent.h"
#include "Shared/RenderSettings.h"

class Scene;

class RenderSettingsPanel : public ImGuiComponent
{
public:
    RenderSettingsPanel(std::string name, Scene& scene);

    void renderUi() override;

private:
    Scene& scene;
    std::optional<BufferVisualization> previousProxyOverdrawBuffer;
};
