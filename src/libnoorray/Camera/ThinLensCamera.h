#pragma once

#include "CUDA/Annotations.h"
#include "Camera/Camera.h"
#include "Samplers/RandomSampler.h"

class ThinLensCamera : public Camera {
public:
#ifndef NR_GPU_CODE
    ThinLensCamera();
    explicit ThinLensCamera(std::unique_ptr<Sensor> sensor);
    ThinLensCamera(const ThinLensCamera& other);
    ~ThinLensCamera();
#endif
    float apertureDiameterMm{};
    float bokehBias{1.f};

    NR_CPU_GPU bool generateRay(glm::vec3& origin, glm::vec3& direction, float& weight,
        float nx, float ny, const glm::vec2 lensSample, uint32_t, SampledWavelengths&,
        bool centered = false) const
    {
        weight = 1.0f;
        const Sensor& sensor = getSensor();
        const glm::vec2 scale{
            sensor.width() / (2.f * focalLengthMm),
            sensor.height() / (2.f * focalLengthMm)};
        direction = glm::normalize(glm::vec3(nx * scale.x, ny * scale.y, -1.f));
        const float apertureRadiusM = centered ? 0.0f : apertureDiameterMm * 0.0005f;
        if (apertureRadiusM > 0.f) {
            const float angle  = 6.28318530718f * lensSample.x;
            const float radius = apertureRadiusM * sqrtf(lensSample.y);
            origin = glm::vec3(cosf(angle) * radius, sinf(angle) * radius, 0.f);
            const glm::vec3 focusPoint = direction *
                ((focusDistanceCm * 0.01f) / fmaxf(-direction.z, 1e-5f));
            direction = glm::normalize(focusPoint - origin);
        } else {
            origin = glm::vec3(0.f);
        }
        transformRay(origin, direction);
        return true;
    }

    bool renderUi();
};
