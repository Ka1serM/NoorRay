#include "RenderPanel.h"
#include <imgui.h>
#include "portable-file-dialogs.h"

RenderPanel::RenderPanel() 
    : saveLocation(".") // default to current dir
{}

void RenderPanel::renderUi() {
    ImGui::Begin(getType().c_str());
    
    ImGui::SeparatorText("Save Location");

    // Figure out how wide the Browse button will be
    float buttonWidth = ImGui::CalcTextSize("Select").x + ImGui::GetStyle().FramePadding.x * 2.0f;
    // Make the input take the rest of the row
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - buttonWidth - ImGui::GetStyle().ItemSpacing.x);
    ImGui::InputText("##savePath", const_cast<char*>(saveLocation.c_str()), saveLocation.size() + 1, ImGuiInputTextFlags_ReadOnly);
    ImGui::PopItemWidth();
    ImGui::PopStyleColor();

    // Put Select button to the right
    ImGui::SameLine();
    if (ImGui::Button("Select", ImVec2(buttonWidth, 0))) {
        const auto selection = pfd::select_folder("Select Render Output Folder", saveLocation).result();
        if (!selection.empty())
            saveLocation = selection;
    }
        
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::Spacing();

    // Big render button
    ImGui::PushStyleColor(ImGuiCol_Button, IM_COL32(100, 180, 255, 255));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, IM_COL32(120, 200, 255, 255));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, IM_COL32(80, 160, 220, 255));

    if (ImGui::Button("Render", ImVec2(-FLT_MIN, 0))) {
        printf("Render button pressed! Saving to: %s/render.hdr\n", saveLocation.c_str());
    }

    ImGui::PopStyleColor(3);

    ImGui::End();
}