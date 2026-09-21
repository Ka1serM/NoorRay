#pragma once

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <noorrhi/types.hpp>

class Transform {

    // GLM leaves default-constructed types uninitialized unless
    // GLM_FORCE_CTOR_INIT is enabled. Spell out the identity transform so a
    // default SceneObject is valid independently of the consumer's GLM flags.
    glm::vec3 position{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 rotationEulerDegrees{0.0f};
    glm::vec3 scale{1.0f};

public:

    Transform();
    Transform(const glm::mat4& matrix);
    Transform(glm::vec3 position);
    Transform(glm::vec3 position, glm::quat rotation, glm::vec3 scale);
    Transform(glm::vec3 position, glm::vec3 rotationDegrees, glm::vec3 scale);

    glm::mat4 getMatrix() const;

    noorrhi::float4x4 getGpuTransform() const;

    void setRotationEuler(const glm::vec3& eulerDegrees);
    glm::vec3 getRotationEuler() const;

    void setPosition(const glm::vec3& eulerDegrees);
    void setRotation(const glm::quat& eulerDegrees);
    void setScale(const glm::vec3& eulerDegrees);
    void setFromMatrix(const glm::mat4& matrix);

    glm::vec3 getPosition() const;
    glm::quat getRotation() const;
    glm::vec3 getScale() const;
};
