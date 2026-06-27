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
        ImGuiManager::dragFloatRow("Lighting Intensity", environment.settings.lightingExposure, 0.01f, 0.0f, 1000000.0f, [&](const float v) { environment.settings.lightingExposure = v; anyChanged = true; });
        
        ImGuiManager::tableRowLabel("HDRI Texture");
        
        const auto& textures = scene.getTextures();
        const int oldHdriTexture = environment.settings.textureIndex;

        if (environment.settings.textureIndex >= static_cast<int>(textures.size()))
            environment.settings.textureIndex = -1;

        const char* comboPreview = "No Texture";
        if (environment.settings.textureIndex != -1)
            comboPreview = textures[environment.settings.textureIndex].getName().c_str();

        if (ImGui::BeginCombo("##hdriTextureCombo", comboPreview)) {
            // Add a selectable for the "No Texture" option
            bool isNoneSelected = (environment.settings.textureIndex == -1);
            if (ImGui::Selectable("No Texture", isNoneSelected))
                environment.settings.textureIndex = -1;
            if (isNoneSelected)
                ImGui::SetItemDefaultFocus();

            // Add all available textures from the scene
            for (int i = 0; i < static_cast<int>(textures.size()); ++i) {
                const bool isSelected = (environment.settings.textureIndex == i);
                if (ImGui::Selectable(textures[i].getName().c_str(), isSelected))
                    environment.settings.textureIndex = i;
                if (isSelected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        
        if (oldHdriTexture != environment.settings.textureIndex) {
            anyChanged = true;
        }
        
        if (environment.settings.textureIndex != -1) {
            ImGuiManager::dragFloatRow("Visible Exposure", environment.settings.visibleExposure, 0.01f, -100.f, 100.f, [&](const float v) { environment.settings.visibleExposure = v; anyChanged = true; });
            ImGuiManager::dragFloatRow("Rotation", environment.settings.rotation, 0.1f, 0, 360, [&](const float v) { environment.settings.rotation = v; anyChanged = true; });
        }

        ImGuiManager::checkboxRow("Visible", environment.settings.visible, [&](const bool v) { environment.settings.visible = v; anyChanged = true; });
        ImGuiManager::dragFloatRow("Sun Intensity", environment.settings.directionalIntensity, 0.01f, 0.0f, 1000000.0f, [&](const float v) { environment.settings.directionalIntensity = v; anyChanged = true; });
        ImGuiManager::dragFloatRow("Sun Soft Angle", environment.settings.directionalSoftAngle, 0.01f, 0.0f, 90.0f, [&](const float v) { environment.settings.directionalSoftAngle = v; anyChanged = true; });
        ImGuiManager::dragFloatRow("Sun Direction X", environment.settings.directionalDirection.x, 0.01f, -1.0f, 1.0f, [&](const float v) { environment.settings.directionalDirection.x = v; anyChanged = true; });
        ImGuiManager::dragFloatRow("Sun Direction Y", environment.settings.directionalDirection.y, 0.01f, -1.0f, 1.0f, [&](const float v) { environment.settings.directionalDirection.y = v; anyChanged = true; });
        ImGuiManager::dragFloatRow("Sun Direction Z", environment.settings.directionalDirection.z, 0.01f, -1.0f, 1.0f, [&](const float v) { environment.settings.directionalDirection.z = v; anyChanged = true; });
        
        ImGui::EndTable();
    }
    
    if (anyChanged)
        scene.setDirtyFlag(Accumulation);
    
    ImGui::End();
}
