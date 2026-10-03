#include "Lights/LightInstance.h"

#include "Lights/DirectionalLightInstance.h"
#include "Lights/PointLightInstance.h"
#include "Lights/RectLightInstance.h"
#include "Lights/SpotLightInstance.h"
#include "Scene/Scene.h"

#include <cmath>

#include <glm/matrix.hpp>

LightInstance::LightInstance(Scene& scene, const std::string& name,
                             const Transform& transform, const int type)
    : SceneObject(scene, name, transform)
    , lightType(type >= TypePoint && type <= TypeDirectional ? type : TypePoint)
{
    billboard = Billboard{static_cast<BillboardIcon>(lightType)};
}

void LightInstance::onTransformUpdated()
{
    SceneObject::onTransformUpdated();
    const Transform world = getWorldTransform();
    worldScale = std::cbrt(std::abs(glm::determinant(glm::mat3(world.getMatrix()))));
    updateTransformData(world);
    updateSceneRecord();
}

glm::vec3 LightInstance::worldDirection(const Transform& world, const glm::vec3 local)
{
    return glm::normalize(glm::mat3(world.getMatrix()) * local);
}

void LightInstance::updateSceneRecord()
{
    if (!scene || lightIndex == ~0u)
        return;

    billboard->color = getColor();
    notifyBillboardChanged();
    switch (lightType) {
    case TypePoint: {
        PointLight record = static_cast<PointLightInstance*>(this)->getData();
        record.softRadius *= worldScale;
        record.sourceLength *= worldScale;
        record.falloff.invRadius /= worldScale;
        record.controls.visible = visible ? 1u : 0u;
        scene->pointLights[lightIndex] = record;
        break;
    }
    case TypeSpot: {
        SpotLight record = static_cast<SpotLightInstance*>(this)->getData();
        record.softRadius *= worldScale;
        record.sourceLength *= worldScale;
        record.falloff.invRadius /= worldScale;
        record.controls.visible = visible ? 1u : 0u;
        scene->spotLights[lightIndex] = record;
        break;
    }
    case TypeRect: {
        RectLight record = static_cast<RectLightInstance*>(this)->getData();
        record.width *= worldScale;
        record.height *= worldScale;
        record.barnDoorLength *= worldScale;
        record.falloff.invRadius /= worldScale;
        record.controls.visible = visible ? 1u : 0u;
        scene->rectLights[lightIndex] = record;
        break;
    }
    case TypeDirectional: {
        DirectionalLight record = static_cast<DirectionalLightInstance*>(this)->getData();
        record.controls.visible = visible ? 1u : 0u;
        scene->directionalLights[lightIndex] = record;
        break;
    }
    default:
        return;
    }
    scene->markLightChanged(lightType, lightIndex);
    scene->setDirtyFlag(Lights);
    scene->setDirtyFlag(Accumulation);
}

void LightInstance::setPhotometry(const glm::vec3& color, const float intensity)
{
    setColor(color);
    setIntensity(intensity);
}

void LightInstance::commitLightChanges()
{
    if (!scene || lightIndex == ~0u)
        return;
    scene->synchronizeBeforeMutation();
    updateSceneRecord();
}

void LightInstance::copyCommonStateTo(LightInstance& target) const
{
    target.visible = visible;
    target.sourceType = sourceType;
    target.sourcePath = sourcePath;
}

std::string LightInstance::getType() const
{
    switch (lightType) {
    case TypePoint: return "Point Light";
    case TypeSpot: return "Spot Light";
    case TypeRect: return "Rect Light";
    case TypeDirectional: return "Directional Light";
    default: return "Light";
    }
}
