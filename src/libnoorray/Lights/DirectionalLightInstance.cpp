#include "Lights/DirectionalLightInstance.h"

#include <glm/geometric.hpp>
#include <glm/gtc/quaternion.hpp>

DirectionalLightInstance::DirectionalLightInstance(
    Scene& scene, const std::string& name, const Transform& transform)
    : LightInstance(scene, name, transform, TypeDirectional)
{
    onTransformUpdated();
}

void DirectionalLightInstance::updateTransformData(const Transform& world)
{
    data.direction = glm::normalize(
        world.getRotation() * glm::vec3(0.0f, -1.0f, 0.0f));
}

void DirectionalLightInstance::setColor(const glm::vec3& color)
{
    data.color = color;
    updateSceneRecord();
}

void DirectionalLightInstance::setIntensity(const float intensity)
{
    data.intensity = intensity;
    updateSceneRecord();
}

void DirectionalLightInstance::setDirectionalSoftAngle(const float degrees)
{
    data.softAngle = degrees;
    updateSceneRecord();
}

std::unique_ptr<SceneObject> DirectionalLightInstance::clone() const
{
    auto copy = std::make_unique<DirectionalLightInstance>(*scene, getName() + " (copy)", transform);
    copy->data = data;
    copyCommonStateTo(*copy);
    return copy;
}
