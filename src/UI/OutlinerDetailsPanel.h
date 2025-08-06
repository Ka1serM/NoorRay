#pragma once

#include "Scene/MeshInstance.h"

class InputTracker;
class Scene;

class OutlinerDetailsPanel : public ImGuiComponent {
public:
    OutlinerDetailsPanel(Scene& scene);
    void renderUi() override;

    std::string getType() const override { return "Outliner Details"; }

private:
    Scene& scene;
};
