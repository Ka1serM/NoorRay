#include "Lights/SpotLightInstance.h"

#include <glm/geometric.hpp>
#include <glm/gtc/quaternion.hpp>

SpotLightInstance::SpotLightInstance(Scene& scene, const std::string& name,
                                     const Transform& transform)
    : LightInstance(scene, name, transform, TypeSpot)
{
    onTransformUpdated();
}

void SpotLightInstance::updateTransformData(const Transform& world)
{
    data.position = world.getPosition();
    data.direction = glm::normalize(
        world.getRotation() * glm::vec3(0.0f, -1.0f, 0.0f));
}

void SpotLightInstance::setColor(const glm::vec3& color)
{
    data.color = color;
    updateSceneRecord();
}

void SpotLightInstance::setIntensity(const float intensity)
{
    data.intensity = intensity;
    updateSceneRecord();
}

void SpotLightInstance::setSoftRadius(const float radius)
{
    data.softRadius = radius;
    updateSceneRecord();
}

void SpotLightInstance::setSpotAngles(
    const float innerDegrees, const float outerDegrees)
{
    data.innerConeAngle = innerDegrees;
    data.outerConeAngle = outerDegrees;
    updateSceneRecord();
}

std::unique_ptr<SceneObject> SpotLightInstance::clone() const
{
    auto copy = std::make_unique<SpotLightInstance>(*scene, getName() + " (copy)", transform);
    copy->data = data;
    copyCommonStateTo(*copy);
    return copy;
}
