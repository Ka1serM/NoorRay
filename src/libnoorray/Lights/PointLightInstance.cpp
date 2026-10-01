#include "Lights/PointLightInstance.h"

PointLightInstance::PointLightInstance(Scene& scene, const std::string& name,
                                       const Transform& transform)
    : LightInstance(scene, name, transform, TypePoint)
{
    onTransformUpdated();
}

void PointLightInstance::updateTransformData(const Transform& world)
{
    data.position = world.getPosition();
    data.axis = worldDirection(world, glm::vec3(1.0f, 0.0f, 0.0f));
}

void PointLightInstance::setColor(const glm::vec3& color)
{
    data.color = color;
    updateSceneRecord();
}

void PointLightInstance::setIntensity(const float intensity)
{
    data.intensity = intensity;
    updateSceneRecord();
}

void PointLightInstance::setSoftRadius(const float radius)
{
    data.softRadius = radius;
    updateSceneRecord();
}

void PointLightInstance::setSourceLength(const float length)
{
    data.sourceLength = length;
    updateSceneRecord();
}

std::unique_ptr<SceneObject> PointLightInstance::clone() const
{
    auto copy = std::make_unique<PointLightInstance>(*scene, getName() + " (copy)", transform);
    copy->data = data;
    copyCommonStateTo(*copy);
    return copy;
}
