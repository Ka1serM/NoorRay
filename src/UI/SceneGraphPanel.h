#pragma once

#include <memory>
#include "UI/ImGuiComponent.h"
#include "Scene/Scene.h"

class SceneGraphPanel : public ImGuiComponent {
public:
    SceneGraphPanel(std::string name, Scene& scene);
    void renderUi() override;

private:
    void drawNode(const std::shared_ptr<SceneObject>& node);
    Scene& scene;
};
