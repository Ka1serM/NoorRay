#include "Lights/RectLightInstance.h"

#include <glm/geometric.hpp>
#include <glm/gtc/quaternion.hpp>

RectLightInstance::RectLightInstance(Scene& scene, const std::string& name,
                                     const Transform& transform)
    : LightInstance(scene, name, transform, TypeRect)
{
    onTransformUpdated();
}

void RectLightInstance::updateTransformData(const Transform& world)
{
    data.position = world.getPosition();
    const glm::quat rotation = world.getRotation();
    data.direction = glm::normalize(
        rotation * glm::vec3(0.0f, -1.0f, 0.0f));
    data.tangent = glm::normalize(
        rotation * glm::vec3(1.0f, 0.0f, 0.0f));
}

void RectLightInstance::setColor(const glm::vec3& color)
{
    data.color = color;
    updateSceneRecord();
}

void RectLightInstance::setIntensity(const float intensity)
{
    data.intensity = intensity;
    updateSceneRecord();
}

void RectLightInstance::setRectSize(const float width, const float height)
{
    data.width = width;
    data.height = height;
    updateSceneRecord();
}

std::unique_ptr<SceneObject> RectLightInstance::clone() const
{
    auto copy = std::make_unique<RectLightInstance>(*scene, getName() + " (copy)", transform);
    copy->data = data;
    copyCommonStateTo(*copy);
    return copy;
}
