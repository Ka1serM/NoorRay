#pragma once

#include "Scene/MeshInstance.h"

class InputTracker;
class Scene;

class OutlinerDetailsPanel : public ImGuiComponent {
public:
    OutlinerDetailsPanel(Scene& scene, InputTracker& inputTracker);
    void renderUi() override;

    std::string getType() const override { return "Outliner Details"; }

    void setSelectedIndex(int index);

private:
    Scene& scene;
    InputTracker& inputTracker;
    
    int selectedObjectIndex;
};
