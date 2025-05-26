#include "OutlinerDetailsPanel.h"
#include <imgui.h>

OutlinerDetailsPanel::OutlinerDetailsPanel(const std::vector<std::unique_ptr<SceneObject>>& sceneObj) : sceneObjects(sceneObj)
{

}


void OutlinerDetailsPanel::renderUi() {
    ImGui::Begin("Outliner");

    for (size_t i = 0; i < sceneObjects.size(); ++i)
        // Construct label with unique ID
        if (ImGui::Selectable((sceneObjects[i]->name + "###" + std::to_string(i)).c_str(), selectedObjectIndex == i))
            selectedObjectIndex = i;

    ImGui::End();

    ImGui::Begin("Details");

    ImGui::BeginTable("ObjectDetails", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV);

    ImGui::TableSetupColumn("Label");
    ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

    // Check valid selectedObjectIndex and non-empty sceneObjects before dereferencing
    if (selectedObjectIndex >= 0 && selectedObjectIndex < sceneObjects.size() && sceneObjects[selectedObjectIndex])
        sceneObjects[selectedObjectIndex]->renderUi();
    else
        ImGui::TextUnformatted("No Object Selected.");

    ImGui::EndTable();
    ImGui::End();
}