#include "PointLight.h"
#include "UI/ImGuiManager.h"

bool PointLight::renderUi()
{
    bool changed = false;
    ImGuiManager::colorEdit3Row("Color", color,
        [&](glm::vec3 v) { color = v; changed = true; });
    ImGuiManager::dragFloatRow("Intensity", intensity, 0.1f, 0.f, 1e6f,
        [&](float v) { intensity = v; changed = true; });
    return changed;
}
