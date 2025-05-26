#pragma once

#include "Scene/MeshInstance.h"

class OutlinerDetailsPanel : public ImGuiComponent {
public:
    // Constructor accepts a const reference to the vector
    OutlinerDetailsPanel(const std::vector<std::unique_ptr<SceneObject>>& sceneObjects);

    void renderUi() override;

    std::string getType() const override { return "Outliner Details"; }

private:
    // Store the reference
    const std::vector<std::unique_ptr<SceneObject>>& sceneObjects;

    int selectedObjectIndex = -1;
};