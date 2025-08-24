#include "EnvironmentPanel.h"
#include <imgui.h>

#include "ImGuiManager.h"
#include "Scene/Scene.h"

EnvironmentPanel::EnvironmentPanel(Scene& scene) : scene(scene) {}

void EnvironmentPanel::renderUi() {
    ImGui::Begin(getType().c_str());
    bool anyChanged = false;

    if (ImGui::BeginTable("Environment Table", 2, ImGuiTableFlags_SizingStretchProp))
    {
        ImGuiManager::colorEdit3Row("Color", enviromentData.color, [&](const vec3 v) { enviromentData.color = v; anyChanged = true; });
        
        ImGui::TableNextRow();
        ImGuiManager::dragFloatRow("Intensity", enviromentData.intensity, 0.01f, 0.0f, 1000000.0f, [&](const float v) { enviromentData.intensity = v; anyChanged = true; });
        
        ImGui::TableNextRow();
        ImGuiManager::tableRowLabel("HDRI Texture");
        
        const auto& textures = scene.getTextures();
        int oldHdriTexture = enviromentData.textureIndex;

        if (enviromentData.textureIndex >= static_cast<int>(textures.size()))
            enviromentData.textureIndex = -1;

        const char* comboPreview = "No Texture";
        if (enviromentData.textureIndex != -1)
            comboPreview = textures[enviromentData.textureIndex].getName().c_str();

        if (ImGui::BeginCombo("##hdriTextureCombo", comboPreview)) {
            // Add a selectable for the "No Texture" option
            bool isNoneSelected = (enviromentData.textureIndex == -1);
            if (ImGui::Selectable("No Texture", isNoneSelected))
                enviromentData.textureIndex = -1;
            if (isNoneSelected)
                ImGui::SetItemDefaultFocus();

            // Add all available textures from the scene
            for (int i = 0; i < static_cast<int>(textures.size()); ++i) {
                const bool isSelected = (enviromentData.textureIndex == i);
                if (ImGui::Selectable(textures[i].getName().c_str(), isSelected))
                    enviromentData.textureIndex = i;
                if (isSelected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        
        if (oldHdriTexture != enviromentData.textureIndex) {
            anyChanged = true;
        }
        
        if (enviromentData.textureIndex != -1) {
            ImGui::TableNextRow();
            ImGuiManager::dragFloatRow("Exposure", enviromentData.exposure, 0.01f, -100.f, 100.f, [&](const float v) { enviromentData.exposure = v; anyChanged = true; });
            
            ImGui::TableNextRow();
            ImGuiManager::dragFloatRow("Rotation", enviromentData.rotation, 0.1f, 0, 360, [&](const float v) { enviromentData.rotation = v; anyChanged = true; });
        }

        ImGui::TableNextRow();
        ImGuiManager::checkboxRow("Visible", enviromentData.visible, [&](const bool v) { enviromentData.visible = v; anyChanged = true; });
        
        ImGui::EndTable();
    }
    
    if (anyChanged) {
        scene.setAccumulationDirty();
    }
    
    ImGui::End();
}