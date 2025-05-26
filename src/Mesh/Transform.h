#pragma once

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "Vulkan/Context.h"

class Transform {

    glm::vec3 position;
    glm::quat rotation;
    glm::vec3 scale;

public:

    Transform();
    Transform(glm::vec3 translation, glm::quat rotation, glm::vec3 scale);
    Transform(glm::vec3 translation, glm::vec3 rotationDegrees, glm::vec3 scale);

    glm::mat4 getMatrix() const;

    vk::TransformMatrixKHR getVkTransformMatrix() const;

    void setRotationEuler(const glm::vec3& eulerDegrees);
    glm::vec3 getRotationEuler() const;

    void setPosition(const glm::vec3& eulerDegrees);
    void setRotation(const glm::quat& eulerDegrees);
    void setScale(const glm::vec3& eulerDegrees);

    glm::vec3 getPosition() const;
    glm::quat getRotation() const;
    glm::vec3 getScale() const;
};
