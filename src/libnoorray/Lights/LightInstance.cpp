#include "Lights/LightInstance.h"

#include "Lights/DirectionalLightInstance.h"
#include "Lights/PointLightInstance.h"
#include "Lights/RectLightInstance.h"
#include "Lights/SpotLightInstance.h"
#include "Scene/Scene.h"

LightInstance::LightInstance(Scene& scene, const std::string& name,
                             const Transform& transform, const int type)
    : SceneObject(scene, name, transform)
    , lightType(type >= TypePoint && type <= TypeDirectional ? type : TypePoint)
{
}

void LightInstance::onTransformUpdated()
{
    SceneObject::onTransformUpdated();
    updateTransformData(getWorldTransform());
    updateSceneRecord();
}

void LightInstance::updateSceneRecord()
{
    if (!scene || lightIndex == ~0u)
        return;

    switch (lightType) {
    case TypePoint:
        scene->pointLights[lightIndex] =
            static_cast<PointLightInstance*>(this)->getData();
        break;
    case TypeSpot:
        scene->spotLights[lightIndex] =
            static_cast<SpotLightInstance*>(this)->getData();
        break;
    case TypeRect:
        scene->rectLights[lightIndex] =
            static_cast<RectLightInstance*>(this)->getData();
        break;
    case TypeDirectional:
        scene->directionalLights[lightIndex] =
            static_cast<DirectionalLightInstance*>(this)->getData();
        break;
    default:
        return;
    }
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
