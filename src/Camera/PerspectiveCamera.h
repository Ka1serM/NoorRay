#pragma once

#include "CUDA/Annotations.h"
#include "Camera/Camera.h"

class PerspectiveCamera : public Camera {
public:
    NR_CPU_GPU bool generateRay(glm::vec3& origin, glm::vec3& direction, float& weight,
        float nx, float ny, uint32_t&, uint32_t) const
    {
        weight = 1.0f;
        const glm::vec2 scale{
            sensor.widthMm / (2.f * focalLengthMm),
            sensor.heightMm / (2.f * focalLengthMm)};
        origin = glm::vec3(0.f);
        direction = glm::normalize(glm::vec3(nx * scale.x, ny * scale.y, -1.f));
        transformRay(origin, direction);
        return true;
    }

    bool renderUi();
};
