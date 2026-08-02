#include "rectLight.h"

#include <pxr/imaging/hd/sceneDelegate.h>

#include <algorithm>
#include <variant>

#include "Rendering/Lighting/RectLight.h"
#include "Scene/Objects/LightInstance.h"

PXR_NAMESPACE_OPEN_SCOPE

HdNoorRayRectLight::HdNoorRayRectLight(const SdfPath& id)
    : HdNoorRayAnalyticLight(id)
{
}

int HdNoorRayRectLight::NoorRayLightType(HdSceneDelegate*) const
{
    return LightInstance::TypeRect;
}

void HdNoorRayRectLight::Configure(
    HdSceneDelegate* delegate, const glm::mat4& transform,
    LightInstance& light, float& intensity) const
{
    auto& rectangle = std::get<RectLight>(light.getLightData());
    rectangle.width = FloatParam(
        delegate, GetId(), HdLightTokens->width, 1.0f)
        * BasisScale(transform, 0);
    rectangle.height = FloatParam(
        delegate, GetId(), HdLightTokens->height, 1.0f)
        * BasisScale(transform, 1);
    if (BoolParam(
            delegate, GetId(), HdLightTokens->normalize, false))
        intensity /= std::max(
            rectangle.width * rectangle.height, 1.0e-8f);
}

PXR_NAMESPACE_CLOSE_SCOPE
