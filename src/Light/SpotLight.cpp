#include "SpotLight.h"
#include "UI/ImGuiManager.h"

bool SpotLight::renderUi()
{
    bool changed = false;
    ImGuiManager::colorEdit3Row("Color", color,
        [&](glm::vec3 v) { color = v; changed = true; });
    ImGuiManager::dragFloatRow("Intensity", intensity, 0.1f, 0.f, 1e6f,
        [&](float v) { intensity = v; changed = true; });
    ImGuiManager::dragFloatRow("Inner Cone", innerConeAngle, 0.5f, 0.f, 90.f,
        [&](float v) { innerConeAngle = v; changed = true; });
    ImGuiManager::dragFloatRow("Outer Cone", outerConeAngle, 0.5f, 0.f, 90.f,
        [&](float v) { outerConeAngle = v; changed = true; });
    return changed;
}
