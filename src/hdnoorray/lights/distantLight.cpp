#include "distantLight.h"

#include <pxr/imaging/hd/sceneDelegate.h>

#include <algorithm>
#include <cmath>
#include <numbers>

#include <glm/trigonometric.hpp>

#include "Scene/LightInstance.h"

PXR_NAMESPACE_OPEN_SCOPE

HdNoorRayDistantLight::HdNoorRayDistantLight(const SdfPath& id)
    : HdNoorRayAnalyticLight(id)
{
}

int HdNoorRayDistantLight::NoorRayLightType(HdSceneDelegate*) const
{
    return LightInstance::TypeDirectional;
}

void HdNoorRayDistantLight::Configure(
    HdSceneDelegate* delegate, const glm::mat4&,
    LightInstance& light, float& intensity) const
{
    const float halfAngle = std::clamp(
        FloatParam(delegate, GetId(), HdLightTokens->angle, 0.265f),
        0.0f, 180.0f);
    light.setDirectionalSoftAngle(halfAngle * 2.0f);
    if (!BoolParam(
            delegate, GetId(), HdLightTokens->normalize, false)
        && halfAngle > 0.0f) {
        const float sine = std::sin(glm::radians(halfAngle));
        const float sizeFactor = halfAngle <= 90.0f
            ? sine * sine * std::numbers::pi_v<float>
            : (2.0f - sine * sine) * std::numbers::pi_v<float>;
        intensity *= sizeFactor;
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
