#include "EnvironmentPanel.h"
#include <imgui.h>

#include "ImGuiManager.h"
#include "Scene/Scene.h"

using glm::vec3;

EnvironmentPanel::EnvironmentPanel(std::string name, Scene& scene) : ImGuiComponent(std::move(name)), scene(scene) {}

void EnvironmentPanel::renderUi() {
    ImGui::Begin(name.c_str());
    bool anyChanged = false;
    Environment& environment = scene.getEnvironment();

    if (ImGui::BeginTable("Environment Table", 2, ImGuiTableFlags_SizingStretchProp))
    {
        float lightingExposure = environment.getLightingExposure();
        ImGuiManager::dragFloatRow("Intensity", lightingExposure, 0.01f, 0.0f, 1000000.0f, [&](const float v) {
            scene.synchronizeBeforeMutation(); environment.setLightingExposure(v); anyChanged = true;
        });
        
        ImGuiManager::tableRowLabel("HDRI Texture");
        
        const auto& textures = scene.getTextures();
        const int oldHdriTexture = environment.getTextureIndex();
        int selectedHdriTexture = oldHdriTexture;

        if (selectedHdriTexture < 0
            || selectedHdriTexture >= static_cast<int>(textures.size())
            || scene.getTexture(static_cast<uint32_t>(selectedHdriTexture)) == nullptr)
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
                scene.setEnvironmentTexture(scene.getTexture(
                    static_cast<uint32_t>(selectedHdriTexture)));
            anyChanged = true;
        }

        vec3 color = environment.getColor();
        ImGuiManager::colorEdit3Row("HDRI Color", color, [&](const vec3 v) {
            scene.synchronizeBeforeMutation(); environment.setColor(v); anyChanged = true;
        });
        
        if (environment.getTextureIndex() != -1) {
            float visibleExposure = environment.getVisibleExposure();
            ImGuiManager::dragFloatRow("Visible Exposure", visibleExposure, 0.01f, -100.f, 100.f, [&](const float v) {
                scene.synchronizeBeforeMutation(); environment.setVisibleExposure(v); anyChanged = true;
            });
            float rotation = environment.getRotation();
            ImGuiManager::dragFloatRow("Rotation", rotation, 0.1f, 0, 360, [&](const float v) {
                scene.synchronizeBeforeMutation(); environment.setRotation(v); anyChanged = true;
            });
        }

        ImGui::EndTable();
    }
    
    if (anyChanged) {
        // The Vulkan renderer consumes an immutable environment record.
        // Every editor change must publish that record, not only reset the
        // accumulation buffer.
        scene.setDirtyFlag(EnvironmentCdf);
        scene.setDirtyFlag(Accumulation);
    }
    
    ImGui::End();
}
