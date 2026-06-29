#include "EnvironmentPanel.h"
#include <imgui.h>

#include "ImGuiManager.h"
#include "Scene/Scene.h"

EnvironmentPanel::EnvironmentPanel(std::string name, Scene& scene) : ImGuiComponent(std::move(name)), scene(scene) {}

void EnvironmentPanel::renderUi() {
    ImGui::Begin(name.c_str());
    bool anyChanged = false;
    Environment& environment = scene.getEnvironment();

    if (ImGui::BeginTable("Environment Table", 2, ImGuiTableFlags_SizingStretchProp))
    {
        ImGuiManager::dragFloatRow("Intensity", environment.lightingExposure, 0.01f, 0.0f, 1000000.0f, [&](const float v) { environment.lightingExposure = v; anyChanged = true; });
        
        ImGuiManager::tableRowLabel("HDRI Texture");
        
        const auto& textures = scene.getTextures();
        const int oldHdriTexture = environment.textureIndex;

        if (environment.textureIndex >= static_cast<int>(textures.size()))
            environment.textureIndex = -1;

        const char* comboPreview = "No Texture";
        if (environment.textureIndex != -1)
            comboPreview = textures[environment.textureIndex].getName().c_str();

        if (ImGui::BeginCombo("##hdriTextureCombo", comboPreview)) {
            // Add a selectable for the "No Texture" option
            bool isNoneSelected = (environment.textureIndex == -1);
            if (ImGui::Selectable("No Texture", isNoneSelected))
                environment.textureIndex = -1;
            if (isNoneSelected)
                ImGui::SetItemDefaultFocus();

            // Add all available textures from the scene
            for (int i = 0; i < static_cast<int>(textures.size()); ++i) {
                const bool isSelected = (environment.textureIndex == i);
                if (ImGui::Selectable(textures[i].getName().c_str(), isSelected))
                    environment.textureIndex = i;
                if (isSelected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        
        if (oldHdriTexture != environment.textureIndex) {
            anyChanged = true;
            scene.setDirtyFlag(EnvironmentCdf);
        }

        ImGuiManager::colorEdit3Row("HDRI Color", environment.color, [&](const vec3 v) { environment.color = v; anyChanged = true; });
        
        if (environment.textureIndex != -1) {
            ImGuiManager::dragFloatRow("Visible Exposure", environment.visibleExposure, 0.01f, -100.f, 100.f, [&](const float v) { environment.visibleExposure = v; anyChanged = true; });
            ImGuiManager::dragFloatRow("Rotation", environment.rotation, 0.1f, 0, 360, [&](const float v) { environment.rotation = v; anyChanged = true; });
        }

        ImGuiManager::checkboxRow("Visible", environment.visible, [&](const bool v) { environment.visible = v; anyChanged = true; });
        ImGui::EndTable();
    }
    
    if (anyChanged) {
        environment.updateDerivedSettings();
        scene.setDirtyFlag(Accumulation);
    }
    
    ImGui::End();
}
