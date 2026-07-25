#include "light.h"

#include "../renderParam.h"

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/usdLux/blackbody.h>

#include <algorithm>
#include <cmath>
#include <mutex>

#include <glm/gtc/matrix_transform.hpp>

#include "Mesh/Transform.h"
#include "Scene/LightInstance.h"
#include "Scene/Scene.h"

PXR_NAMESPACE_OPEN_SCOPE

HdNoorRayLight::HdNoorRayLight(const SdfPath& id)
    : HdLight(id)
{
}

float HdNoorRayLight::FloatParam(
    HdSceneDelegate* delegate, const SdfPath& id, const TfToken& token,
    const float fallback)
{
    const VtValue value = delegate->GetLightParamValue(id, token);
    if (value.IsHolding<float>())
        return value.UncheckedGet<float>();
    if (value.IsHolding<double>())
        return static_cast<float>(value.UncheckedGet<double>());
    return fallback;
}

bool HdNoorRayLight::BoolParam(
    HdSceneDelegate* delegate, const SdfPath& id, const TfToken& token,
    const bool fallback)
{
    const VtValue value = delegate->GetLightParamValue(id, token);
    return value.IsHolding<bool>() ? value.UncheckedGet<bool>() : fallback;
}

glm::vec3 HdNoorRayLight::ColorParam(
    HdSceneDelegate* delegate, const SdfPath& id, const TfToken& token,
    const glm::vec3& fallback)
{
    const VtValue value = delegate->GetLightParamValue(id, token);
    if (value.IsHolding<GfVec3f>()) {
        const GfVec3f& v = value.UncheckedGet<GfVec3f>();
        return {v[0], v[1], v[2]};
    }
    if (value.IsHolding<GfVec3d>()) {
        const GfVec3d& v = value.UncheckedGet<GfVec3d>();
        return {v[0], v[1], v[2]};
    }
    return fallback;
}

std::string HdNoorRayLight::AssetPathParam(
    HdSceneDelegate* delegate, const SdfPath& id, const TfToken& token)
{
    const VtValue value = delegate->GetLightParamValue(id, token);
    if (!value.IsHolding<SdfAssetPath>())
        return {};
    const SdfAssetPath& asset = value.UncheckedGet<SdfAssetPath>();
    return asset.GetResolvedPath().empty()
        ? asset.GetAssetPath() : asset.GetResolvedPath();
}

glm::vec3 HdNoorRayLight::LightColor(
    HdSceneDelegate* delegate, const SdfPath& id)
{
    glm::vec3 color = ColorParam(
        delegate, id, HdLightTokens->color, glm::vec3(1.0f));
    if (BoolParam(
            delegate, id, HdLightTokens->enableColorTemperature, false)) {
        const GfVec3f temperatureColor = UsdLuxBlackbodyTemperatureAsRgb(
            FloatParam(delegate, id, HdLightTokens->colorTemperature, 6500.0f));
        color *= glm::vec3(
            temperatureColor[0], temperatureColor[1], temperatureColor[2]);
    }
    return color;
}

float HdNoorRayLight::LightIntensity(
    HdSceneDelegate* delegate, const SdfPath& id)
{
    return FloatParam(delegate, id, HdLightTokens->intensity, 1.0f)
        * std::exp2(std::clamp(
            FloatParam(delegate, id, HdLightTokens->exposure, 0.0f),
            -50.0f, 50.0f));
}

glm::mat4 HdNoorRayLight::ToGlm(const GfMatrix4d& value)
{
    glm::mat4 result(1.0f);
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
            result[column][row] = static_cast<float>(value[column][row]);
    return result;
}

float HdNoorRayLight::BasisScale(
    const glm::mat4& transform, const int column)
{
    return glm::length(glm::vec3(transform[column]));
}

HdDirtyBits HdNoorRayLight::GetInitialDirtyBitsMask() const
{
    return AllDirty;
}

HdNoorRayAnalyticLight::HdNoorRayAnalyticLight(const SdfPath& id)
    : HdNoorRayLight(id)
{
}

void HdNoorRayAnalyticLight::Sync(
    HdSceneDelegate* delegate, HdRenderParam* renderParam,
    HdDirtyBits* dirtyBits)
{
    auto& param = *static_cast<HdNoorRayRenderParam*>(renderParam);
    std::scoped_lock lock(param.mutex);
    Scene& scene = param.session.scene;

    const int requestedType = NoorRayLightType(delegate);
    if (!scene.isValid(object_) || requestedType != lightType_) {
        scene.removeObject(object_);
        auto light = std::make_unique<LightInstance>(
            scene, GetId().GetString(), Transform(), requestedType);
        object_ = scene.add(std::move(light));
        lightType_ = requestedType;
    }

    auto* light = dynamic_cast<LightInstance*>(scene.getObject(object_));
    if (light == nullptr) {
        *dirtyBits = Clean;
        return;
    }

    const glm::mat4 hydraTransform = ToGlm(
        delegate->GetTransform(GetId()));
    const glm::mat4 axisCorrection = glm::rotate(
        glm::mat4(1.0f), glm::radians(90.0f),
        glm::vec3(1.0f, 0.0f, 0.0f));
    light->setWorldTransformFromMatrix(hydraTransform * axisCorrection);
    light->setVisible(delegate->GetVisible(GetId()));

    float intensity = LightIntensity(delegate, GetId());
    Configure(delegate, hydraTransform, *light, intensity);
    light->setPhotometry(LightColor(delegate, GetId()), intensity);
    light->commitLightChanges();
    param.MarkSceneDirty();
    *dirtyBits = Clean;
}

void HdNoorRayAnalyticLight::Finalize(HdRenderParam* renderParam)
{
    auto& param = *static_cast<HdNoorRayRenderParam*>(renderParam);
    std::scoped_lock lock(param.mutex);
    if (param.session.scene.removeObject(object_)) {
        object_ = {};
        lightType_ = -1;
        param.MarkSceneDirty();
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
