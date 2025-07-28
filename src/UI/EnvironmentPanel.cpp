#include "EnvironmentPanel.h"
#include <imgui.h>

#include "ImGuiManager.h"
#include "Scene/Scene.h"

EnvironmentPanel::EnvironmentPanel(Scene& scene) : scene(scene), hdriTexture(0) {}
void EnvironmentPanel::renderUi() {
    ImGui::Begin(getType().c_str());

    const auto& textures = scene.getTextures();

    if (!textures.empty()) {
        std::vector<std::string> textureNames;
        textureNames.reserve(textures.size());
        for (const auto& tex : textures)
            textureNames.push_back(tex.getName());

        ImGuiManager::tableRowLabel("HDRI Texture");

        int oldHdriTexture = hdriTexture;  // Save old index to detect changes

        if (ImGui::BeginCombo("##hdriTextureCombo", textureNames[hdriTexture].c_str())) {
            for (int i = 0; i < static_cast<int>(textureNames.size()); ++i) {
                bool isSelected = hdriTexture == i;

                if (ImGui::Selectable(textureNames[i].c_str(), isSelected))
                    hdriTexture = i;

                if (isSelected)
                    ImGui::SetItemDefaultFocus();
            }

            ImGui::EndCombo();
        }

        if (oldHdriTexture != hdriTexture)
            scene.setAccumulationDirty();

    } else {
        ImGui::Text("No textures available.");
    }

    ImGui::End();
}