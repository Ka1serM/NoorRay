#include "Transform.h"
#include "glm/gtx/quaternion.hpp"

Transform::Transform(const glm::vec3 position) : position(position), rotation(), scale(1.0f) {}

Transform::Transform() : position(), rotation(), scale(1.0f) {}

Transform::Transform(const glm::mat4& matrix)
{
    setFromMatrix(matrix);
}

Transform::Transform(const glm::vec3 position, const glm::quat rotation, const glm::vec3 scale)
    : position(position), rotation(rotation), rotationEulerDegrees(degrees(eulerAngles(rotation))), scale(scale) {}

Transform::Transform(const glm::vec3 position, const glm::vec3 rotationDegrees, const glm::vec3 scale)
    : position(position), rotation(glm::quat(radians(rotationDegrees))), rotationEulerDegrees(rotationDegrees), scale(scale) {}

glm::mat4 Transform::getMatrix() const {
    glm::mat4 mat = translate(glm::mat4(1.0f), position);
    mat *= toMat4(rotation);
    mat = glm::scale(mat, scale);
    return mat;
}

void Transform::setFromMatrix(const glm::mat4& mat) {
    // Extract translation (position)
    position = glm::vec3(mat[3]);

    // Extract scale
    // Basis vectors are the first 3 columns of the matrix
    auto col0 = glm::vec3(mat[0]);
    auto col1 = glm::vec3(mat[1]);
    auto col2 = glm::vec3(mat[2]);

    scale.x = glm::length(col0);
    scale.y = glm::length(col1);
    scale.z = glm::length(col2);

    // Remove scale from rotation matrix
    glm::mat3 rotationMat;
    if (scale.x != 0) rotationMat[0] = col0 / scale.x; else rotationMat[0] = col0;
    if (scale.y != 0) rotationMat[1] = col1 / scale.y; else rotationMat[1] = col1;
    if (scale.z != 0) rotationMat[2] = col2 / scale.z; else rotationMat[2] = col2;

    // Convert rotation matrix to quaternion
    rotation = glm::quat_cast(rotationMat);
    rotationEulerDegrees = degrees(eulerAngles(rotation));
}

gpu::float4x4 Transform::getGpuTransform() const {
    glm::mat4 mat = getMatrix();
    gpu::float4x4 transform{};
    // The GPU API uses explicit row-major records; GLM stores columns.
    for (std::size_t row = 0; row < 3; ++row)
        for (std::size_t column = 0; column < 4; ++column)
            transform.values[row][column] = mat[column][row];
    transform.values[3][3] = 1.0f;
    return transform;
}

void Transform::setRotationEuler(const glm::vec3& eulerDegrees) {
    rotationEulerDegrees = eulerDegrees;
    rotation = glm::quat(glm::radians(eulerDegrees));
}


glm::vec3 Transform::getRotationEuler() const {
    return rotationEulerDegrees;
}


void Transform::setRotation(const glm::quat& rot) {
    rotation = rot;
    rotationEulerDegrees = degrees(eulerAngles(rotation));
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
