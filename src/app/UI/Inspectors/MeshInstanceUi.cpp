#include "Scene/MeshInstance.h"

#include <string>

#include <imgui.h>

#include "Mesh/Assets/MeshAsset.h"
#include "Shading/Sellmeier.h"
#include "UI/ImGuiManager.h"
#include "UI/ObjectUi.h"

namespace
{
bool renderMeshAsset(MeshAsset& asset)
{
    ImGuiManager::tableRowLabel("Source");
    ImGui::TextUnformatted(asset.getPath().c_str());
    if (asset.getMaterialCount() == 0)
        return false;

    if (!ImGuiManager::accordionRow("Materials###MeshMaterials"))
        return false;

    bool changed = false;
    const auto textureNames = asset.getScene().getTextureNames();
    auto textureCombo = [&](const char* label, int& textureIndex, const int materialIndex,
                            bool& materialChanged) {
        ImGuiManager::tableRowLabel(label);
        // Reclaimed textures keep their slot but lose their name, so an empty
        // entry is a hole in the list rather than a choice.
        const bool validIndex = textureIndex >= 0
            && textureIndex < static_cast<int>(textureNames.size())
            && !textureNames[textureIndex].empty();
        if (!validIndex)
            textureIndex = -1;
        const char* preview = validIndex ? textureNames[textureIndex].c_str() : "No Texture";
        const std::string id = "##" + std::string(label) + std::to_string(materialIndex);
        if (!ImGui::BeginCombo(id.c_str(), preview))
            return;
        if (ImGui::Selectable("No Texture", textureIndex < 0)) {
            textureIndex = -1;
            materialChanged = true;
        }
        for (int index = 0; index < static_cast<int>(textureNames.size()); ++index) {
            if (textureNames[index].empty())
                continue;
            if (ImGui::Selectable(textureNames[index].c_str(), textureIndex == index)) {
                textureIndex = index;
                materialChanged = true;
            }
        }
        ImGui::EndCombo();
    };

    for (size_t index = 0; index < asset.getMaterialCount(); ++index)
    {
        Material material = asset.getMaterial(static_cast<uint32_t>(index));
        bool materialChanged = false;
        const std::string label = "Material " + std::to_string(index);
        if (!ImGuiManager::accordionRow(label.c_str()))
            continue;
        textureCombo("Albedo Texture", material.albedoIndex, static_cast<int>(index), materialChanged);
        ImGuiManager::colorEdit3Row("Albedo Color", material.albedo,
            [&](glm::vec3 value) { material.albedo = value; materialChanged = true; });
        textureCombo("Specular Texture", material.specularIndex, static_cast<int>(index), materialChanged);
        ImGuiManager::dragFloatRow("Specular", material.specular, 0.01f, 0.f, 1.f,
            [&](float value) { material.specular = value; materialChanged = true; });
        textureCombo("Metallic Texture", material.metallicIndex, static_cast<int>(index), materialChanged);
        ImGuiManager::dragFloatRow("Metallic", material.metallic, 0.01f, 0.f, 1.f,
            [&](float value) { material.metallic = value; materialChanged = true; });
        textureCombo("Roughness Texture", material.roughnessIndex, static_cast<int>(index), materialChanged);
        ImGuiManager::dragFloatRow("Roughness", material.roughness, 0.01f, 0.f, 1.f,
            [&](float value) { material.roughness = value; materialChanged = true; });
        textureCombo("Normal Texture", material.normalIndex, static_cast<int>(index), materialChanged);

        glm::vec3 iors = sellmeierFraunhoferIors(material.sellmeier);
        auto updateIor = [&](const int channel, const float value) {
            iors[channel] = value;
            material.sellmeier = fitSellmeierFromFraunhofer(iors);
            materialChanged = true;
        };
        ImGuiManager::dragFloatRow("Red IOR (C 656.27 nm)", iors.x, 0.001f, 1.f, 3.f,
            [&](float value) { updateIor(0, value); });
        ImGuiManager::dragFloatRow("Green IOR (e 546.07 nm)", iors.y, 0.001f, 1.f, 3.f,
            [&](float value) { updateIor(1, value); });
        ImGuiManager::dragFloatRow("Blue IOR (F 486.13 nm)", iors.z, 0.001f, 1.f, 3.f,
            [&](float value) { updateIor(2, value); });
        ImGuiManager::dragFloatRow("Transmission Strength", material.transmission,
            0.01f, 0.f, 1.f, [&](float value) { material.transmission = value; materialChanged = true; });
        textureCombo("Transmission Texture", material.transmissionIndex, static_cast<int>(index), materialChanged);
        ImGuiManager::colorEdit3Row("Transmission Color", material.transmissionColor,
            [&](glm::vec3 value) { material.transmissionColor = value; materialChanged = true; });
        ImGuiManager::dragFloatRow("Emission Strength", material.emissionStrength,
            0.1f, 0.f, 100000.f, [&](float value) { material.emissionStrength = value; materialChanged = true; });
        textureCombo("Emission Texture", material.emissionIndex, static_cast<int>(index), materialChanged);
        ImGuiManager::colorEdit3Row("Emission Color", material.emission,
            [&](glm::vec3 value) { material.emission = value; materialChanged = true; });
        ImGuiManager::dragFloatRow("Opacity", material.opacity, 0.01f, 0.f, 1.f,
            [&](float value) { material.opacity = value; materialChanged = true; });
        textureCombo("Opacity Texture", material.opacityIndex, static_cast<int>(index), materialChanged);
        if (materialChanged) {
            asset.setMaterial(static_cast<uint32_t>(index), material);
            changed = true;
        }
    }
    return changed;
}
}

namespace
{
bool renderMeshInstance(MeshInstance& instance)
{
    if (!ImGuiManager::accordionRow("Mesh Asset###MeshProperties"))
        return false;

    return instance.hasMeshAsset() && renderMeshAsset(instance.getMeshAsset());
}

}

void ObjectUiVisitor::visit(MeshInstance& instance)
{
    changed |= renderMeshInstance(instance);
}
