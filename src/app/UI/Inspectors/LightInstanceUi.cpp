#include "Scene/LightInstance.h"
#include "Lights/DirectionalLightInstance.h"
#include "Lights/PointLightInstance.h"
#include "Lights/RectLightInstance.h"
#include "Lights/SpotLightInstance.h"

#include "UI/ImGuiManager.h"
#include "UI/ObjectUi.h"

namespace
{
bool renderLight(PointLight& light)
{
    bool changed = false;
    ImGuiManager::colorEdit3Row("Color", light.color,
        [&](glm::vec3 value) { light.color = value; changed = true; });
    ImGuiManager::dragFloatRow("Intensity", light.intensity, 0.1f, 0.f, 1e6f,
        [&](float value) { light.intensity = value; changed = true; });
    ImGuiManager::dragFloatRow("Soft Radius", light.softRadius, 0.01f, 0.f, 1e6f,
        [&](float value) { light.softRadius = value; changed = true; });
    return changed;
}

bool renderLight(SpotLight& light)
{
    bool changed = false;
    ImGuiManager::colorEdit3Row("Color", light.color,
        [&](glm::vec3 value) { light.color = value; changed = true; });
    ImGuiManager::dragFloatRow("Intensity", light.intensity, 0.1f, 0.f, 1e6f,
        [&](float value) { light.intensity = value; changed = true; });
    ImGuiManager::dragFloatRow("Soft Radius", light.softRadius, 0.01f, 0.f, 1e6f,
        [&](float value) { light.softRadius = value; changed = true; });
    ImGuiManager::dragFloatRow("Inner Cone", light.innerConeAngle, 0.5f, 0.f, 90.f,
        [&](float value) { light.innerConeAngle = value; changed = true; });
    ImGuiManager::dragFloatRow("Outer Cone", light.outerConeAngle, 0.5f, 0.f, 90.f,
        [&](float value) { light.outerConeAngle = value; changed = true; });
    return changed;
}

bool renderLight(RectLight& light)
{
    bool changed = false;
    ImGuiManager::colorEdit3Row("Color", light.color,
        [&](glm::vec3 value) { light.color = value; changed = true; });
    ImGuiManager::dragFloatRow("Intensity", light.intensity, 0.1f, 0.f, 1e6f,
        [&](float value) { light.intensity = value; changed = true; });
    ImGuiManager::dragFloatRow("Width", light.width, 0.1f, 0.f, 1e6f,
        [&](float value) { light.width = value; changed = true; });
    ImGuiManager::dragFloatRow("Height", light.height, 0.1f, 0.f, 1e6f,
        [&](float value) { light.height = value; changed = true; });
    ImGuiManager::checkboxRow("Two Sided", light.twoSided != 0,
        [&](bool value) { light.twoSided = value ? 1 : 0; changed = true; });
    ImGuiManager::dragFloatRow("Barn Door Angle", light.barnDoorAngle, 0.5f, 0.f, 90.f,
        [&](float value) { light.barnDoorAngle = value; changed = true; });
    ImGuiManager::dragFloatRow("Barn Door Length", light.barnDoorLength, 0.01f, 0.f, 1e6f,
        [&](float value) { light.barnDoorLength = value; changed = true; });
    return changed;
}

bool renderLight(DirectionalLight& light)
{
    bool changed = false;
    ImGuiManager::colorEdit3Row("Color", light.color,
        [&](glm::vec3 value) { light.color = value; changed = true; });
    ImGuiManager::dragFloatRow("Intensity", light.intensity, 0.1f, 0.f, 1e6f,
        [&](float value) { light.intensity = value; changed = true; });
    ImGuiManager::dragFloatRow("Soft Angle", light.softAngle, 0.01f, 0.f, 90.f,
        [&](float value) { light.softAngle = value; changed = true; });
    return changed;
}
}

namespace
{
// The typed record is protected on the instance; edit a copy and publish it
// back only when a widget changed it.
template<class Instance>
bool renderTypedLight(LightInstance& instance)
{
    auto& typed = static_cast<Instance&>(instance);
    auto light = typed.getData();
    const bool changed = renderLight(light);
    if (changed)
        typed.setData(light);
    return changed;
}

bool renderLightInstance(LightInstance& instance)
{
    bool changed = false;
    switch (instance.getLightType()) {
    case LightInstance::TypePoint:
        changed = renderTypedLight<PointLightInstance>(instance);
        break;
    case LightInstance::TypeSpot:
        changed = renderTypedLight<SpotLightInstance>(instance);
        break;
    case LightInstance::TypeRect:
        changed = renderTypedLight<RectLightInstance>(instance);
        break;
    case LightInstance::TypeDirectional:
        changed = renderTypedLight<DirectionalLightInstance>(instance);
        break;
    default:
        break;
    }
    if (changed)
        instance.commitLightChanges();
    return changed;
}

}

bool object_ui::render(LightInstance& instance)
{
    return renderLightInstance(instance);
}
