#include "Rendering/Camera/CameraInstance.h"

#include <array>
#include <imgui.h>

#include "UI/ImGuiManager.h"
#include "UI/ObjectUi.h"

namespace
{
bool renderCameraInstance(CameraInstance& instance)
{
    if (!ImGuiManager::accordionRow("Camera###CameraProperties"))
        return false;

    bool changed = false;
    ImGuiManager::tableRowLabel("Type");
    static constexpr CameraProjectionType types[] = {
        CameraProjectionType::Perspective, CameraProjectionType::ThinLens,
        CameraProjectionType::Realistic,
        CameraProjectionType::Orthographic, CameraProjectionType::Fisheye,
    };
    static const char* names[] = {
        "Perspective", "Thin Lens", "Realistic", "Orthographic", "Fisheye"};
    int selected = 0;
    for (int index = 0; index < static_cast<int>(std::size(types)); ++index)
        if (types[index] == instance.getProjectionType()) {
            selected = index;
            break;
        }
    if (ImGui::Combo("##CameraProjection", &selected, names, std::size(types))) {
        instance.switchTo(types[selected]);
        changed = true;
    }

    Camera* camera = instance.getCamera();
    if (auto* concrete = camera->CastOrNullptr<PerspectiveCamera>()) changed |= concrete->renderUi();
    else if (auto* concrete = camera->CastOrNullptr<ThinLensCamera>()) changed |= concrete->renderUi();
    else if (auto* concrete = camera->CastOrNullptr<OrthographicCamera>()) changed |= concrete->renderUi();
    else if (auto* concrete = camera->CastOrNullptr<FisheyeCamera>()) changed |= concrete->renderUi();
    else if (auto* concrete = camera->CastOrNullptr<RealisticCamera>()) changed |= concrete->renderUi();
    if (changed)
        instance.markDirty();
    return changed;
}

}

void ObjectUiVisitor::visit(CameraInstance& instance)
{
    changed |= renderCameraInstance(instance);
}
