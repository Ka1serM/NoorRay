#include "light.h"

#include "renderParam.h"

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <variant>

#include <glm/gtc/matrix_transform.hpp>

#include "Light/RectLight.h"
#include "Mesh/Transform.h"
#include "Scene/LightInstance.h"
#include "Scene/Scene.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace
{
glm::mat4 ToGlm(const GfMatrix4d& value)
{
    glm::mat4 result(1.0f);
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
            result[column][row] = static_cast<float>(value[column][row]);
    return result;
}

float FloatParam(
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

glm::vec3 ColorParam(
    HdSceneDelegate* delegate, const SdfPath& id, const TfToken& token,
    const glm::vec3 fallback)
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
}

HdNoorRayLight::HdNoorRayLight(
    const SdfPath& id, const TfToken& typeId)
    : HdLight(id)
    , typeId_(typeId)
{
}

void HdNoorRayLight::Sync(
    HdSceneDelegate* delegate, HdRenderParam* renderParam,
    HdDirtyBits* dirtyBits)
{
    auto& param = *static_cast<HdNoorRayRenderParam*>(renderParam);
    std::scoped_lock lock(param.mutex);
    Scene& scene = param.session.scene;

    const float coneAngle = FloatParam(
        delegate, GetId(), HdLightTokens->shapingConeAngle, -1.0f);
    int lightType = LightInstance::TypePoint;
    if (typeId_ == HdPrimTypeTokens->rectLight
        || typeId_ == HdPrimTypeTokens->diskLight)
        lightType = LightInstance::TypeRect;
    else if (typeId_ == HdPrimTypeTokens->distantLight)
        lightType = LightInstance::TypeDirectional;
    else if (coneAngle >= 0.0f)
        lightType = LightInstance::TypeSpot;

    if (objectId_ == 0) {
        auto light = std::make_unique<LightInstance>(
            scene, GetId().GetString(), Transform(), lightType);
        objectId_ = scene.add(std::move(light));
    }
    auto* light = dynamic_cast<LightInstance*>(scene.getObject(objectId_));
    if (light == nullptr) {
        *dirtyBits = Clean;
        return;
    }

    const glm::mat4 axisCorrection = glm::rotate(
        glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    light->setWorldTransformFromMatrix(
        ToGlm(delegate->GetTransform(GetId())) * axisCorrection);
    light->setVisible(delegate->GetVisible(GetId()));

    const glm::vec3 color = ColorParam(
        delegate, GetId(), HdLightTokens->color, glm::vec3(1.0f));
    const float intensity = FloatParam(
        delegate, GetId(), HdLightTokens->intensity, 1.0f)
        * std::exp2(FloatParam(
            delegate, GetId(), HdLightTokens->exposure, 0.0f));
    light->setPhotometry(color, intensity);
    const float radius = std::max(
        0.0f, FloatParam(delegate, GetId(), HdLightTokens->radius, 0.0f));
    light->setPointRadius(radius);
    light->setSpotRadius(radius);
    if (lightType == LightInstance::TypeSpot) {
        const float softness = std::clamp(FloatParam(
            delegate, GetId(), HdLightTokens->shapingConeSoftness, 0.0f),
            0.0f, 1.0f);
        light->setSpotAngles(coneAngle * (1.0f - softness), coneAngle);
    } else if (lightType == LightInstance::TypeDirectional) {
        light->setDirectionalSoftAngle(
            FloatParam(delegate, GetId(), HdLightTokens->angle, 0.53f) * 2.0f);
    } else if (lightType == LightInstance::TypeRect) {
        auto& rectangle = std::get<RectLight>(light->getLightData());
        if (typeId_ == HdPrimTypeTokens->diskLight) {
            rectangle.width = rectangle.height = radius * 2.0f;
        } else {
            rectangle.width = FloatParam(
                delegate, GetId(), HdLightTokens->width, 1.0f);
            rectangle.height = FloatParam(
                delegate, GetId(), HdLightTokens->height, 1.0f);
        }
    }
    light->commitLightChanges();
    param.MarkSceneDirty();
    *dirtyBits = Clean;
}

void HdNoorRayLight::Finalize(HdRenderParam* renderParam)
{
    auto& param = *static_cast<HdNoorRayRenderParam*>(renderParam);
    std::scoped_lock lock(param.mutex);
    if (objectId_ != 0) {
        param.session.scene.removeObject(objectId_);
        objectId_ = 0;
        param.MarkSceneDirty();
    }
}

HdDirtyBits HdNoorRayLight::GetInitialDirtyBitsMask() const
{
    return AllDirty;
}

PXR_NAMESPACE_CLOSE_SCOPE
