#pragma once

#include "Scene/MeshInstance.h"

class InputTracker;
class Scene;

class OutlinerDetailsPanel : public ImGuiComponent {
public:
    // Constructor accepts a const reference
    OutlinerDetailsPanel(Renderer& renderer, InputTracker& inputTracker);
    void renderUi() override;

    std::string getType() const override { return "Outliner Details"; }

    void setSelectedIndex(int index);

private:
    // Store the reference
    Renderer& renderer;
    InputTracker& inputTracker;
    
    int selectedObjectIndex;
};
