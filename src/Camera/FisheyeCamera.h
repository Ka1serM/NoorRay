#pragma once

#include "GPU/Annotations.h"
#include "Camera/Camera.h"
#include "Scene/SceneTypes.h"

class CameraInstance;

class FisheyeCamera : public Camera {
public:
    NR_CPU_GPU bool generateRay(glm::vec3& origin, glm::vec3& direction,
        float nx, float ny, uint32_t&, uint32_t) const
    {
        origin = glm::vec3(0.f);
        const float radius = sqrtf(nx * nx + ny * ny);
        if (radius > 1.f)
            return false;
        const float theta = radius * glm::radians(fieldOfView) * 0.5f;
        const float scale = radius > 1e-6f ? sinf(theta) / radius : 0.f;
        direction = glm::vec3(nx * scale, ny * scale, -cosf(theta));
        transformRay(origin, direction);
        return true;
    }

    void renderUi(CameraInstance& inst);
};
