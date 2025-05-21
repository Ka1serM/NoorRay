#pragma once

#include <InputTracker.h>
#include <glm/glm.hpp>

#include "shaders/SharedStructs.h"

class PerspectiveCamera {
private:
    const glm::vec3 UP;
    glm::vec3 position;
    glm::vec3 direction{};
    glm::vec3 horizontal{};
    glm::vec3 vertical{};

    CameraData cameraData{};

public:
    PerspectiveCamera(const glm::vec3& origin, const glm::vec3& lookAt, const glm::vec3& up, float aspect, float fovDegrees);

    void update(InputTracker &inputTracker, float deltaTime);

    const CameraData& getCameraData() const;
};
