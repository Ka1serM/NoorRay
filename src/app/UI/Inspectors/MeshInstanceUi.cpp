#include "Scene/Objects/MeshInstance.h"

#include <string>
#include <filesystem>

#include <imgui.h>
// ImGui::GetCurrentTable, used to lay the slot rows out the same way
// ImGuiManager::tableRowLabel does.
#include <imgui_internal.h>

#include "Geometry/Mesh/Assets/MeshAsset.h"
#include "Materials/Shading/Sellmeier.h"
#include "UI/ImGuiManager.h"
#include "UI/ObjectUi.h"

namespace
{
// Every material a scene owns needs a name that tells it apart from the
// others in a slot's dropdown. Disk-backed materials get their file stem;
// everything else -- the synthetic open_pbr_surface fallback that unassigned
// geometry and freshly imported meshes share -- has no authored name at all,
// so fall back to the registry index rather than labelling all of them
// "Default" and leaving the list unusable.
std::string materialLabel(Scene& scene, const uint32_t materialIndex)
{
    const auto& sourcePaths = scene.getMaterialXSourcePaths();
    if (materialIndex < sourcePaths.size() && !sourcePaths[materialIndex].empty())
        return std::filesystem::path(sourcePaths[materialIndex]).stem().string();
    return "Material " + std::to_string(materialIndex);
}

bool renderMeshAsset(MeshAsset& asset)
{
    Scene& scene = asset.getScene();
    ImGuiManager::tableRowLabel("Source");
    ImGui::TextUnformatted(asset.getPath().c_str());
    if (asset.getMaterialCount() == 0)
        return false;

    // The selected slot is what the MaterialX node editor edits (see
    // MaterialXNodeEditorPanel::resolveTarget), so which one is active has to
    // be visible here -- otherwise the graph silently belongs to whichever
    // slot was touched last.
    const uint32_t selected = std::min(scene.getSelectedMaterialSlot(),
        static_cast<uint32_t>(asset.getMaterialCount() - 1));
    for (size_t index = 0; index < asset.getMaterialCount(); ++index) {
        const uint32_t slot = static_cast<uint32_t>(index);
        const MaterialHandle current = asset.getMaterialHandle(slot);
        ImGui::PushID(static_cast<int>(index));

        // Selecting the slot and reassigning its material are separate
        // actions: clicking the slot's name picks which slot the node editor
        // edits, the combo beside it changes what is bound to that slot. The
        // name goes in the label column so the two do not fight over width;
        // ImGuiManager::tableRowLabel's plain text cannot be clicked, so the
        // row is laid out here instead.
        const std::string slotLabel = "Slot " + std::to_string(index);
        if (ImGui::GetCurrentTable()) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            if (ImGui::Selectable(slotLabel.c_str(), slot == selected))
                scene.setSelectedMaterialSlot(slot);
            ImGui::TableSetColumnIndex(1);
        } else {
            if (ImGui::Selectable(slotLabel.c_str(), slot == selected))
                scene.setSelectedMaterialSlot(slot);
            ImGui::SameLine();
        }

        const std::string preview = materialLabel(scene, current.index());
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::BeginCombo("##MaterialSlot", preview.c_str())) {
            const auto& materials = scene.getMaterials();
            for (uint32_t materialIndex = 0; materialIndex < materials.size(); ++materialIndex) {
                const MaterialHandle candidate =
                    scene.getMaterialRegistry().handleAt(materialIndex);
                if (!candidate.isValid())
                    continue;
                const std::string label = materialLabel(scene, materialIndex)
                    + "##GlobalMaterial" + std::to_string(materialIndex);
                const bool isCurrent = candidate == current;
                if (ImGui::Selectable(label.c_str(), isCurrent)) {
                    asset.setMaterial(slot, scene.getMaterialRef(candidate));
                    scene.setSelectedMaterialSlot(slot);
                }
                if (isCurrent)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        // Opening the dropdown is itself a statement of intent about which
        // slot is being worked on.
        if (ImGui::IsItemClicked())
            scene.setSelectedMaterialSlot(slot);
        ImGui::PopID();
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
