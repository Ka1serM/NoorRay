#include "DetailsPanel.h"
#include "Scene/SceneObject.h"
#include <imgui.h>

DetailsPanel::DetailsPanel(std::string name, Scene& scene)
    : ImGuiComponent(std::move(name)), scene(scene) {}

void DetailsPanel::renderUi() {
    ImGui::Begin(name.c_str());
    
    if (ImGui::BeginTable("ObjectDetails", 2, ImGuiTableFlags_SizingStretchProp)) {
        
        if (const auto activeObject = scene.getActiveObjectPtr())
            activeObject->renderUi(); 
        else {
            ImGui::TableNextColumn();
            ImGui::TextUnformatted("No Object Selected");
        }
        
        ImGui::EndTable();
    }
    
    ImGui::End();
}
