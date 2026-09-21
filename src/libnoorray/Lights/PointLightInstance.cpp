#include "Lights/PointLightInstance.h"

#include <glm/geometric.hpp>

PointLightInstance::PointLightInstance(Scene& scene, const std::string& name,
                                       const Transform& transform)
    : LightInstance(scene, name, transform, TypePoint)
{
    onTransformUpdated();
}

void PointLightInstance::updateTransformData(const Transform& world)
{
    data.position = world.getPosition();
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

std::unique_ptr<SceneObject> PointLightInstance::clone() const
{
    auto copy = std::make_unique<PointLightInstance>(*scene, getName() + " (copy)", transform);
    copy->data = data;
    copyCommonStateTo(*copy);
    return copy;
}
