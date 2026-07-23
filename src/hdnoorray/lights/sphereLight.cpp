#include "sphereLight.h"

#include <pxr/imaging/hd/sceneDelegate.h>

#include <algorithm>
#include <numbers>

#include "Scene/LightInstance.h"

PXR_NAMESPACE_OPEN_SCOPE

HdNoorRaySphereLight::HdNoorRaySphereLight(const SdfPath& id)
    : HdNoorRayAnalyticLight(id)
{
}

int HdNoorRaySphereLight::NoorRayLightType(
    HdSceneDelegate* delegate) const
{
    const float coneAngle = FloatParam(
        delegate, GetId(), HdLightTokens->shapingConeAngle, -1.0f);
    return coneAngle >= 0.0f
        ? LightInstance::TypeSpot : LightInstance::TypePoint;
}

void HdNoorRaySphereLight::Configure(
    HdSceneDelegate* delegate, const glm::mat4& transform,
    LightInstance& light, float& intensity) const
{
    const float radius = std::max(
        0.0f, FloatParam(
            delegate, GetId(), HdLightTokens->radius, 0.0f))
        * std::max({
            BasisScale(transform, 0),
            BasisScale(transform, 1),
            BasisScale(transform, 2)});
    if (!BoolParam(
            delegate, GetId(), HdLightTokens->normalize, false)
        && radius > 0.0f)
        intensity *= 4.0f * std::numbers::pi_v<float> * radius * radius;

    light.setPointRadius(radius);
    light.setSpotRadius(radius);
    if (NoorRayLightType(delegate) == LightInstance::TypeSpot) {
        const float coneAngle = FloatParam(
            delegate, GetId(), HdLightTokens->shapingConeAngle, 45.0f);
        const float softness = std::clamp(FloatParam(
            delegate, GetId(), HdLightTokens->shapingConeSoftness, 0.0f),
            0.0f, 1.0f);
        light.setSpotAngles(
            coneAngle * (1.0f - softness), coneAngle);
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
