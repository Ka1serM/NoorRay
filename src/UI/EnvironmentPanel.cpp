#include "EnvironmentPanel.h"
#include <imgui.h>

#include "ImGuiManager.h"
#include "Scene/Scene.h"

EnvironmentPanel::EnvironmentPanel(std::string name, Scene& scene) : ImGuiComponent(std::move(name)), scene(scene) {}

void EnvironmentPanel::renderUi() {
    ImGui::Begin(name.c_str());
    bool anyChanged = false;
    Environment& environment = scene.getEnvironment();
    EnvironmentSettings& settings = *environment.settings;

    if (ImGui::BeginTable("Environment Table", 2, ImGuiTableFlags_SizingStretchProp))
    {
        ImGuiManager::dragFloatRow("Intensity", settings.lightingExposure, 0.01f, 0.0f, 1000000.0f, [&](const float v) { settings.lightingExposure = v; anyChanged = true; });
        
        ImGuiManager::tableRowLabel("HDRI Texture");
        
        const auto& textures = scene.getTextures();
        const int oldHdriTexture = settings.textureIndex;

        if (settings.textureIndex >= static_cast<int>(textures.size()))
            settings.textureIndex = -1;

        const char* comboPreview = "No Texture";
        if (settings.textureIndex != -1)
            comboPreview = textures[settings.textureIndex].getName().c_str();

        if (ImGui::BeginCombo("##hdriTextureCombo", comboPreview)) {
            // Add a selectable for the "No Texture" option
            bool isNoneSelected = (settings.textureIndex == -1);
            if (ImGui::Selectable("No Texture", isNoneSelected))
                settings.textureIndex = -1;
            if (isNoneSelected)
                ImGui::SetItemDefaultFocus();

            // Add all available textures from the scene
            for (int i = 0; i < static_cast<int>(textures.size()); ++i) {
                const bool isSelected = (settings.textureIndex == i);
                if (ImGui::Selectable(textures[i].getName().c_str(), isSelected))
                    settings.textureIndex = i;
                if (isSelected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        
        if (oldHdriTexture != settings.textureIndex) {
            anyChanged = true;
            scene.setDirtyFlag(EnvironmentCdf);
        }

        ImGuiManager::colorEdit3Row("HDRI Color", settings.color, [&](const vec3 v) { settings.color = v; anyChanged = true; });
        
        if (settings.textureIndex != -1) {
            ImGuiManager::dragFloatRow("Visible Exposure", settings.visibleExposure, 0.01f, -100.f, 100.f, [&](const float v) { settings.visibleExposure = v; anyChanged = true; });
            ImGuiManager::dragFloatRow("Rotation", settings.rotation, 0.1f, 0, 360, [&](const float v) { settings.rotation = v; anyChanged = true; });
        }

        ImGuiManager::checkboxRow("Visible", settings.visible, [&](const bool v) { settings.visible = v; anyChanged = true; });
        ImGuiManager::dragFloatRow("Sun Intensity", settings.directionalIntensity, 0.01f, 0.0f, 1000000.0f, [&](const float v) { settings.directionalIntensity = v; anyChanged = true; });
        ImGuiManager::dragFloatRow("Sun Soft Angle", settings.directionalSoftAngle, 0.01f, 0.0f, 90.0f, [&](const float v) { settings.directionalSoftAngle = v; anyChanged = true; });
        ImGuiManager::dragFloatRow("Sun Direction X", settings.directionalDirection.x, 0.01f, -1.0f, 1.0f, [&](const float v) { settings.directionalDirection.x = v; anyChanged = true; });
        ImGuiManager::dragFloatRow("Sun Direction Y", settings.directionalDirection.y, 0.01f, -1.0f, 1.0f, [&](const float v) { settings.directionalDirection.y = v; anyChanged = true; });
        ImGuiManager::dragFloatRow("Sun Direction Z", settings.directionalDirection.z, 0.01f, -1.0f, 1.0f, [&](const float v) { settings.directionalDirection.z = v; anyChanged = true; });
        
        ImGui::EndTable();
    }
    
    if (anyChanged) {
        environment.updateDerivedSettings();
        scene.setDirtyFlag(Accumulation);
    }
    
    ImGui::End();
}
