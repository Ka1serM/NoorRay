#pragma once

#include "GPU/Annotations.h"
#include "Camera/Camera.h"
#include "Kernels/Samplers.h"

class CameraInstance;

class FisheyeCamera : public Camera {
public:
    NR_CPU_GPU bool generateRay(glm::vec3& origin, glm::vec3& direction,
        float nx, float ny, uint32_t& rng, uint32_t) const
    {
        const float aspect = sensor.widthMm / sensor.heightMm;
        const float sx = nx * aspect;
        const float sy = ny;
        const float r = sqrtf(sx * sx + sy * sy);
        const float rMax = sqrtf(aspect * aspect + 1.0f);
        if (r < 1e-6f) {
            origin = glm::vec3(0.f);
            direction = glm::vec3(0.f, 0.f, -1.f);
            transformRay(origin, direction);
            return true;
        }
        const float theta = (r / rMax) * glm::radians(fieldOfView) * 0.5f;
        const float scale = sinf(theta) / r;
        direction = glm::vec3(sx * scale, sy * scale, -cosf(theta));
        const float aperture = fStop > 0.f ? (focalLengthMm / fStop) * 0.5f * 0.001f : 0.f;
        if (aperture > 0.f) {
            const float angle  = 6.28318530718f * randomFloat(rng);
            const float rAperture = aperture * sqrtf(randomFloat(rng));
            origin = glm::vec3(cosf(angle) * rAperture, sinf(angle) * rAperture, 0.f);
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
