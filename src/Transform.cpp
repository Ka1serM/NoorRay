#include "Transform.h"

#include "glm/gtx/quaternion.hpp"

Transform::Transform() : position(0.0f), rotation(1.0f, 0.0f, 0.0f, 0.0f), scale(1.0f) {}

Transform::Transform(const glm::vec3 translation, const glm::quat rotation, const glm::vec3 scale) : position(translation), rotation(rotation), scale(scale) {}

Transform::Transform(const glm::vec3 translation, const glm::vec3 rotationDegrees, const glm::vec3 scale) : position(translation), scale(scale) {
    rotation = glm::quat(radians(rotationDegrees));
}

glm::mat4 Transform::getMatrix() const {
    glm::mat4 mat = translate(glm::mat4(1.0f), position);
    mat *= toMat4(rotation);
    mat = glm::scale(mat, scale);
    return mat;
}

void Transform::setRotationEuler(const glm::vec3& eulerDegrees) {
    glm::vec3 radians = glm::radians(eulerDegrees);
    rotation = glm::quat(radians);
}

glm::vec3 Transform::getRotationEuler() const {
    return degrees(eulerAngles(rotation));
}


void Transform::setRotation(const glm::quat& rot) {
    rotation = rot;
}

void Transform::setPosition(const glm::vec3& pos) {
    position = pos;
}

void Transform::setScale(const glm::vec3& scal) {
    scale = scal;
}

glm::vec3 Transform::getPosition() const {
    return position;
}

glm::quat Transform::getRotation() const {
    return rotation;
}

glm::vec3 Transform::getScale() const {
    return scale;
}

