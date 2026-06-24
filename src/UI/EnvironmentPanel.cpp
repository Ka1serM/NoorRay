#include "EnvironmentPanel.h"
#include <imgui.h>

#include "ImGuiManager.h"
#include "Scene/Scene.h"

EnvironmentPanel::EnvironmentPanel(std::string name, Scene& scene) : ImGuiComponent(std::move(name)), scene(scene) {
    enviromentData.textureIndex    = 0;  // HDRI loaded as texture 0
    enviromentData.cdfTextureIndex = 1;  // CDF loaded as texture 1
}

void EnvironmentPanel::renderUi() {
    ImGui::Begin(name.c_str());
    bool anyChanged = false;

    if (ImGui::BeginTable("Environment Table", 2, ImGuiTableFlags_SizingStretchProp))
    {
        ImGuiManager::dragFloatRow("Lighting Intensity", enviromentData.lightingExposure, 0.01f, 0.0f, 1000000.0f, [&](const float v) { enviromentData.lightingExposure = v; anyChanged = true; });
        
        ImGuiManager::tableRowLabel("HDRI Texture");
        
        const auto& textures = scene.getTextures();
        const int oldHdriTexture = enviromentData.textureIndex;

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
            ImGuiManager::dragFloatRow("Visible Exposure", enviromentData.visibleExposure, 0.01f, -100.f, 100.f, [&](const float v) { enviromentData.visibleExposure = v; anyChanged = true; });
            ImGuiManager::dragFloatRow("Rotation", enviromentData.rotation, 0.1f, 0, 360, [&](const float v) { enviromentData.rotation = v; anyChanged = true; });
        }

        ImGuiManager::checkboxRow("Visible", enviromentData.visible, [&](const bool v) { enviromentData.visible = v; anyChanged = true; });
        ImGuiManager::dragFloatRow("Sun Intensity", enviromentData.directionalIntensity, 0.01f, 0.0f, 1000000.0f, [&](const float v) { enviromentData.directionalIntensity = v; anyChanged = true; });
        ImGuiManager::dragFloatRow("Sun Soft Angle", enviromentData.directionalSoftAngle, 0.01f, 0.0f, 90.0f, [&](const float v) { enviromentData.directionalSoftAngle = v; anyChanged = true; });
        ImGuiManager::dragFloatRow("Sun Direction X", enviromentData.directionalDirection.x, 0.01f, -1.0f, 1.0f, [&](const float v) { enviromentData.directionalDirection.x = v; anyChanged = true; });
        ImGuiManager::dragFloatRow("Sun Direction Y", enviromentData.directionalDirection.y, 0.01f, -1.0f, 1.0f, [&](const float v) { enviromentData.directionalDirection.y = v; anyChanged = true; });
        ImGuiManager::dragFloatRow("Sun Direction Z", enviromentData.directionalDirection.z, 0.01f, -1.0f, 1.0f, [&](const float v) { enviromentData.directionalDirection.z = v; anyChanged = true; });
        
        ImGui::EndTable();
    }
    
    if (anyChanged)
        scene.setDirtyFlag(Accumulation);
    
    ImGui::End();
}
