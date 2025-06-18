#include "Transform.h"
#include "glm/gtx/quaternion.hpp"

Transform::Transform(const glm::vec3 position) : position(position), rotation(), scale(1.0f) {}

Transform::Transform() : position(), rotation(), scale(1.0f) {}

Transform::Transform(const glm::vec3 position, const glm::quat rotation, const glm::vec3 scale) : position(position), rotation(rotation), scale(scale) {}

Transform::Transform(const glm::vec3 position, const glm::vec3 rotationDegrees, const glm::vec3 scale) : position(position), scale(scale), rotation(glm::quat(radians(rotationDegrees))) {}

glm::mat4 Transform::getMatrix() const {
    glm::mat4 mat = translate(glm::mat4(1.0f), position);
    mat *= toMat4(rotation);
    mat = glm::scale(mat, scale);
    return mat;
}

vk::TransformMatrixKHR Transform::getVkTransformMatrix() const {
    glm::mat4 mat = getMatrix();
    vk::TransformMatrixKHR vkTransform{};
    vkTransform.matrix[0][0] = mat[0][0]; vkTransform.matrix[0][1] = mat[1][0]; vkTransform.matrix[0][2] = mat[2][0]; vkTransform.matrix[0][3] = mat[3][0];
    vkTransform.matrix[1][0] = mat[0][1]; vkTransform.matrix[1][1] = mat[1][1]; vkTransform.matrix[1][2] = mat[2][1]; vkTransform.matrix[1][3] = mat[3][1];
    vkTransform.matrix[2][0] = mat[0][2]; vkTransform.matrix[2][1] = mat[1][2]; vkTransform.matrix[2][2] = mat[2][2]; vkTransform.matrix[2][3] = mat[3][2];
    return vkTransform;
}

void Transform::setRotationEuler(const glm::vec3& eulerDegrees) {
    rotation = glm::quat(glm::radians(eulerDegrees));
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

