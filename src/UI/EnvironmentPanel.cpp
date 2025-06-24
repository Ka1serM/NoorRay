#include "EnvironmentPanel.h"
#include <imgui.h>

#include "ImGuiManager.h"
#include "Vulkan/Renderer.h"

EnvironmentPanel::EnvironmentPanel(Renderer& renderer) : renderer(renderer), hdriTexture(0) {}

void EnvironmentPanel::renderUi() {
    ImGui::Begin(getType().c_str());

    if (const auto& textureNames = renderer.getTextureNames(); !textureNames.empty()) {
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
            renderer.markDirty();
    } else
        ImGui::Text("No textures available.");

    ImGui::End();
}
