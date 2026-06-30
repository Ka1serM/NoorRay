#pragma once

#include "CUDA/Annotations.h"
#include "Camera/Camera.h"
#include "Samplers/RandomSampler.h"

class ThinLensCamera : public Camera {
public:
    float fStop{0.f};
    float bokehBias{1.f};

    NR_CPU_GPU bool generateRay(glm::vec3& origin, glm::vec3& direction, float& weight,
        float nx, float ny, RandomState& rng, uint32_t, float) const
    {
        weight = 1.0f;
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

    bool renderUi();
};
