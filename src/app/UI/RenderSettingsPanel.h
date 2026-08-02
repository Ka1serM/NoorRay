#pragma once

#include <optional>
#include <string>

#include "UI/ImGuiComponent.h"

class Scene;
enum class BufferVisualization : int;

class RenderSettingsPanel : public ImGuiComponent
{
public:
    RenderSettingsPanel(std::string name, Scene& scene);

    void renderUi() override;

private:
    Scene& scene;
    std::optional<BufferVisualization> previousDenoiserBuffer;
    std::optional<BufferVisualization> previousProxyOverdrawBuffer;
};
