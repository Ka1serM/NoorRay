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
        ImGuiManager::dragFloatRow("Intensity", environment.lightingExposure, 0.01f, 0.0f, 1000000.0f, [&](const float v) {
            scene.synchronizeBeforeMutation(); environment.lightingExposure = v; anyChanged = true;
        });
        
        ImGuiManager::tableRowLabel("HDRI Texture");
        
        const auto& textures = scene.getTextures();
        const auto& textureRegistry = scene.getTextureRegistry();
        const int oldHdriTexture = environment.textureIndex;
        int selectedHdriTexture = oldHdriTexture;

        if (selectedHdriTexture < 0
            || selectedHdriTexture >= static_cast<int>(textures.size())
            || !textureRegistry.isLiveSlot(static_cast<uint32_t>(selectedHdriTexture)))
            selectedHdriTexture = -1;

        const char* comboPreview = "No Texture";
        if (selectedHdriTexture != -1)
            comboPreview = textures[selectedHdriTexture].getName().c_str();

        if (ImGui::BeginCombo("##hdriTextureCombo", comboPreview)) {
            // Add a selectable for the "No Texture" option
            bool isNoneSelected = (selectedHdriTexture == -1);
            if (ImGui::Selectable("No Texture", isNoneSelected))
                selectedHdriTexture = -1;
            if (isNoneSelected)
                ImGui::SetItemDefaultFocus();

            // Add all available textures from the scene
            for (int i = 0; i < static_cast<int>(textures.size()); ++i) {
                if (!textureRegistry.isLiveSlot(static_cast<uint32_t>(i)))
                    continue;
                const bool isSelected = (selectedHdriTexture == i);
                const std::string label = textures[i].getName().empty()
                    ? "Texture " + std::to_string(i)
                    : textures[i].getName();
                const std::string selectableId = label + "##hdriTexture" + std::to_string(i);
                if (ImGui::Selectable(selectableId.c_str(), isSelected))
                    selectedHdriTexture = i;
                if (isSelected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        
        if (oldHdriTexture != selectedHdriTexture) {
            scene.synchronizeBeforeMutation();
            if (selectedHdriTexture == -1)
                scene.clearEnvironmentTexture();
            else
                scene.setEnvironmentTexture(scene.getTextureRef(
                    textureRegistry.handleAt(static_cast<uint32_t>(selectedHdriTexture))));
            anyChanged = true;
        }

        ImGuiManager::colorEdit3Row("HDRI Color", environment.color, [&](const vec3 v) {
            scene.synchronizeBeforeMutation(); environment.color = v; anyChanged = true;
        });
        
        if (environment.textureIndex != -1) {
            ImGuiManager::dragFloatRow("Visible Exposure", environment.visibleExposure, 0.01f, -100.f, 100.f, [&](const float v) {
                scene.synchronizeBeforeMutation(); environment.visibleExposure = v; anyChanged = true;
            });
            ImGuiManager::dragFloatRow("Rotation", environment.rotation, 0.1f, 0, 360, [&](const float v) {
                scene.synchronizeBeforeMutation(); environment.rotation = v; anyChanged = true;
            });
        }

        ImGui::EndTable();
    }
    
    if (anyChanged) {
        environment.updateDerivedSettings();
        scene.setDirtyFlag(Accumulation);
    }
    
    ImGui::End();
}
