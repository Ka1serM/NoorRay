#include "DirectionalLight.h"

#include "UI/ImGuiManager.h"

bool DirectionalLight::renderUi()
{
    bool changed = false;
    ImGuiManager::colorEdit3Row("Color", color,
        [&](glm::vec3 value) { color = value; changed = true; });
    ImGuiManager::dragFloatRow("Intensity", intensity, 0.1f, 0.f, 1e6f,
        [&](float value) { intensity = value; changed = true; });
    ImGuiManager::dragFloatRow("Soft Angle", softAngle, 0.01f, 0.f, 90.f,
        [&](float value) { softAngle = value; changed = true; });
    return changed;
}
