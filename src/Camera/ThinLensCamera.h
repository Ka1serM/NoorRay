#pragma once

#include "GPU/Annotations.h"
#include "Camera/Camera.h"
#include "Kernels/Samplers.h"
#include "Scene/SceneTypes.h"

class CameraInstance;

class ThinLensCamera : public Camera {
public:
    NR_CPU_GPU bool generateRay(glm::vec3& origin, glm::vec3& direction,
        float nx, float ny, uint32_t& rng, uint32_t) const
    {
        const glm::vec2 scale{
            sensor.widthMm / (2.f * focalLengthMm),
            sensor.heightMm / (2.f * focalLengthMm)};
        direction = glm::normalize(glm::vec3(nx * scale.x, ny * scale.y, -1.f));
        const float aperture = fStop > 0.f ? (focalLengthMm / fStop) * 0.5f * 0.001f : 0.f;
        if (aperture > 0.f) {
            const float angle  = 6.28318530718f * randomFloat(rng);
            const float radius = aperture * sqrtf(randomFloat(rng));
            origin = glm::vec3(cosf(angle) * radius, sinf(angle) * radius, 0.f);
            const glm::vec3 focusPoint = direction * (focusDistance / fmaxf(-direction.z, 1e-5f));
            direction = glm::normalize(focusPoint - origin);
        } else {
            origin = glm::vec3(0.f);
        }
        transformRay(origin, direction);
        return true;
    }

    void renderUi(CameraInstance& inst);
};
