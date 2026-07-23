#include "DetailsPanel.h"
#include "Scene/SceneObject.h"
#include "UI/ObjectUi.h"
#include <imgui.h>

DetailsPanel::DetailsPanel(std::string name, Scene& scene)
    : ImGuiComponent(std::move(name)), scene(scene) {}

void DetailsPanel::renderUi() {
    ImGui::Begin(name.c_str());
    
    const ImGuiTableFlags tableFlags =
        ImGuiTableFlags_SizingStretchProp
        | ImGuiTableFlags_Resizable
        | ImGuiTableFlags_PadOuterX;
    if (ImGui::BeginTable("ObjectDetails", 2, tableFlags)) {
        ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthStretch, 0.42f);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch, 0.58f);
        
        if (const auto activeObject = scene.getActiveObjectPtr())
            domain_ui::render(*activeObject);
        else {
            ImGui::TableNextColumn();
            ImGui::TextUnformatted("No Object Selected");
        }
        
        ImGui::EndTable();
    }
    
    ImGui::End();
}
