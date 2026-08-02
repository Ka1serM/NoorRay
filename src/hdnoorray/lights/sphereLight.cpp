#include "sphereLight.h"

#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/usd/usdLux/tokens.h>

#include <algorithm>
#include <numbers>

#include "Scene/Objects/LightInstance.h"

PXR_NAMESPACE_OPEN_SCOPE

HdNoorRaySphereLight::HdNoorRaySphereLight(const SdfPath& id)
    : HdNoorRayAnalyticLight(id)
{
}

int HdNoorRaySphereLight::NoorRayLightType(
    HdSceneDelegate* delegate) const
{
    // Blender's live Hydra scene index publishes these two shaping values
    // with the USD `inputs:` prefix, while UsdImaging delegates expose the
    // corresponding HdLightTokens without that prefix. Support both paths.
    const float coneAngle = FloatParam(
        delegate, GetId(), HdLightTokens->shapingConeAngle, -1.0f);
    const float usdInputConeAngle = FloatParam(
        delegate, GetId(), UsdLuxTokens->inputsShapingConeAngle, -1.0f);
    return coneAngle >= 0.0f
        || usdInputConeAngle >= 0.0f
        ? LightInstance::TypeSpot : LightInstance::TypePoint;
}

void HdNoorRaySphereLight::Configure(
    HdSceneDelegate* delegate, const glm::mat4& transform,
    LightInstance& light, float& intensity) const
{
    // Blender's USD/Hydra exporter writes non-sun energy as energy / pi.
    // Cycles' point and spherical emitters use energy / (4 pi) for their
    // zero-radius and normalized-sphere radiance, while NoorRay's light
    // primitives take the latter factor outside their sampling formulas.
    intensity *= 0.25f;

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
        const float hydraConeAngle = FloatParam(
            delegate, GetId(), HdLightTokens->shapingConeAngle, -1.0f);
        const float usdConeAngle = FloatParam(
            delegate, GetId(), UsdLuxTokens->inputsShapingConeAngle, -1.0f);
        const float coneAngle = hydraConeAngle >= 0.0f
            ? hydraConeAngle : usdConeAngle;
        const float softness = std::clamp(FloatParam(
            delegate, GetId(), UsdLuxTokens->inputsShapingConeSoftness,
            FloatParam(delegate, GetId(), HdLightTokens->shapingConeSoftness, 0.0f)),
            0.0f, 1.0f);
        light.setSpotAngles(
            std::max(coneAngle, 0.0f) * (1.0f - softness),
            std::max(coneAngle, 0.0f));
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
