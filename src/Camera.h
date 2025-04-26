#pragma once

#include <glm/glm.hpp>
#include "Structs.h"
#include "GLFW/glfw3.h"


class PerspectiveCamera {
private:
    glm::vec3 origin;
    glm::vec3 lowerLeft;
    glm::vec3 horizontal;
    glm::vec3 vertical;
    glm::vec3 rightVector, upVector, viewVector;
    float lensRadius;

public:
    PerspectiveCamera(const glm::vec3& origin, const glm::vec3& lookAt, const glm::vec3& up, float aspect, float aperture, float focusDist, float focalLengthMM, float sensorWidthMM);
    PerspectiveCamera(const glm::vec3& origin, const glm::vec3& lookAt, const glm::vec3& up, float aspect, float fovDegrees);

    void update(GLFWwindow *window, double mouseX, double mouseY);

    CameraData getGpuCameraData() const;
};
