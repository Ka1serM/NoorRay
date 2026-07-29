#include "Scene/MeshInstance.h"

#include <string>
#include <filesystem>

#include <imgui.h>

#include "Mesh/Assets/MeshAsset.h"
#include "Shading/Sellmeier.h"
#include "UI/ImGuiManager.h"
#include "UI/ObjectUi.h"

namespace
{
bool renderMeshAsset(MeshAsset& asset)
{
    Scene& scene = asset.getScene();
    ImGuiManager::tableRowLabel("Source");
    ImGui::TextUnformatted(asset.getPath().c_str());
    if (asset.getMaterialCount() == 0)
        return false;
    for (size_t index = 0; index < asset.getMaterialCount(); ++index) {
        const MaterialHandle current = asset.getMaterialHandle(static_cast<uint32_t>(index));
        const auto& sourcePaths = scene.getMaterialXSourcePaths();
        std::string preview = "Default";
        if (current.index() < sourcePaths.size() && !sourcePaths[current.index()].empty())
            preview = std::filesystem::path(sourcePaths[current.index()]).stem().string();
        const std::string comboId = "##MaterialSlot" + std::to_string(index);
        if (ImGui::BeginCombo(comboId.c_str(), preview.c_str())) {
            const auto& materials = scene.getMaterials();
            for (uint32_t materialSlot = 0; materialSlot < materials.size(); ++materialSlot) {
                const MaterialHandle candidate = scene.getMaterialRegistry().handleAt(materialSlot);
                if (!candidate.isValid())
                    continue;
                std::string label = "Default";
                if (materialSlot < sourcePaths.size() && !sourcePaths[materialSlot].empty())
                    label = std::filesystem::path(sourcePaths[materialSlot]).stem().string();
                label += "##GlobalMaterial" + std::to_string(materialSlot);
                const bool selected = candidate == current;
                if (ImGui::Selectable(label.c_str(), selected)) {
                    asset.setMaterial(static_cast<uint32_t>(index), scene.getMaterialRef(candidate));
                    scene.setSelectedMaterialSlot(static_cast<uint32_t>(index));
                }
                if (selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        if (ImGui::IsItemClicked())
            scene.setSelectedMaterialSlot(static_cast<uint32_t>(index));
    }
    return false;
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
