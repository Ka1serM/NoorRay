#include "diskLight.h"

#include <pxr/imaging/hd/sceneDelegate.h>

#include <algorithm>
#include <numbers>
#include <variant>

#include "Light/RectLight.h"
#include "Scene/LightInstance.h"

PXR_NAMESPACE_OPEN_SCOPE

HdNoorRayDiskLight::HdNoorRayDiskLight(const SdfPath& id)
    : HdNoorRayAnalyticLight(id)
{
}

int HdNoorRayDiskLight::NoorRayLightType(HdSceneDelegate*) const
{
    return LightInstance::TypeRect;
}

void HdNoorRayDiskLight::Configure(
    HdSceneDelegate* delegate, const glm::mat4& transform,
    LightInstance& light, float& intensity) const
{
    const float radius = std::max(
        0.0f, FloatParam(
            delegate, GetId(), HdLightTokens->radius, 0.0f));
    const float scaleX = BasisScale(transform, 0);
    const float scaleY = BasisScale(transform, 1);
    auto& rectangle = std::get<RectLight>(light.getLightData());
    rectangle.width = radius * 2.0f * scaleX;
    rectangle.height = radius * 2.0f * scaleY;
    if (BoolParam(
            delegate, GetId(), HdLightTokens->normalize, false)) {
        const float sourceArea = std::numbers::pi_v<float>
            * radius * radius * scaleX * scaleY;
        intensity /= std::max(sourceArea, 1.0e-8f);
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
