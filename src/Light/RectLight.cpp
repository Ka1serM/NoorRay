#include "RectLight.h"
#include "UI/ImGuiManager.h"

bool RectLight::renderUi()
{
    bool changed = false;
    ImGuiManager::colorEdit3Row("Color", color,
        [&](glm::vec3 v) { color = v; changed = true; });
    ImGuiManager::dragFloatRow("Intensity", intensity, 0.1f, 0.f, 1e6f,
        [&](float v) { intensity = v; changed = true; });
    ImGuiManager::dragFloatRow("Width", width, 0.1f, 0.f, 1e6f,
        [&](float v) { width = v; changed = true; });
    ImGuiManager::dragFloatRow("Height", height, 0.1f, 0.f, 1e6f,
        [&](float v) { height = v; changed = true; });
    ImGuiManager::checkboxRow("Two Sided", twoSided != 0,
        [&](bool v) { twoSided = v ? 1 : 0; changed = true; });
    ImGuiManager::dragFloatRow("Barn Door Angle", barnDoorAngle, 0.5f, 0.f, 90.f,
        [&](float v) { barnDoorAngle = v; changed = true; });
    ImGuiManager::dragFloatRow("Barn Door Length", barnDoorLength, 0.01f, 0.f, 1e6f,
        [&](float v) { barnDoorLength = v; changed = true; });
    return changed;
}
