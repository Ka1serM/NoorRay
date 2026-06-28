#pragma once

#include "GPU/Annotations.h"
#include "Camera/Camera.h"
#include "Scene/SceneTypes.h"

class CameraInstance;

class OrthographicCamera : public Camera {
public:
    NR_CPU_GPU bool generateRay(glm::vec3& origin, glm::vec3& direction,
        float nx, float ny, uint32_t&, uint32_t) const
    {
        constexpr float referenceHeight = 10.f;
        constexpr float referenceFocalLengthMm = 21.f;
        const float height = referenceHeight * referenceFocalLengthMm / fmaxf(0.001f, focalLengthMm);
        origin = glm::vec3(nx * height * sensor.aspectRatio() * 0.5f, ny * height * 0.5f, 0.f);
        direction = glm::vec3(0.f, 0.f, -1.f);
        transformRay(origin, direction);
        return true;
    }

    void renderUi(CameraInstance& inst);
};
