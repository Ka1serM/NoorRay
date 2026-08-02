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
        CameraProjectionType::Realistic, CameraProjectionType::HybridPsf,
        CameraProjectionType::Orthographic, CameraProjectionType::Fisheye,
    };
    static const char* names[] = {
        "Perspective", "Thin Lens", "Realistic", "Hybrid PSF", "Orthographic", "Fisheye"};
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
    changed |= camera->DispatchCPU([](auto* concrete) { return concrete->renderUi(); });
    SensorType requestedType{};
    if (camera->getSensor().consumeRequestedType(requestedType)) {
        const Sensor& current = camera->getSensor();
        if (requestedType == SensorType::ScatterPsf)
            camera->setSensor(std::make_unique<ScatterPsfSensor>(current));
        else if (requestedType == SensorType::GatherPsf)
            camera->setSensor(std::make_unique<GatherPsfSensor>(current));
        else
            camera->setSensor(std::make_unique<RectangularSensor>(current));
        changed = true;
    }
    if (changed)
        instance.markDirty();
    return changed;
}

}

void ObjectUiVisitor::visit(CameraInstance& instance)
{
    changed |= renderCameraInstance(instance);
}
