#pragma once

#include "CUDA/Annotations.h"
#include "Camera/Camera.h"
#include "Samplers/RandomSampler.h"

class FisheyeCamera : public Camera {
public:
#ifndef NR_GPU_CODE
    FisheyeCamera();
    explicit FisheyeCamera(std::unique_ptr<Sensor> sensor);
    FisheyeCamera(const FisheyeCamera& other);
    ~FisheyeCamera();
#endif
    float apertureDiameterMm{};
    float bokehBias{1.f};

    NR_CPU_GPU bool generateRay(glm::vec3& origin, glm::vec3& direction, float& weight,
        float nx, float ny, const glm::vec2 lensSample, uint32_t, SampledWavelengths&,
        bool centered = false) const
    {
        weight = 1.0f;
        const Sensor& sensor = getSensor();
        const float aspect = sensor.width() / sensor.height();
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
        const float theta = (r / rMax) * glm::radians(fieldOfViewDegrees) * 0.5f;
        const float scale = sinf(theta) / r;
        direction = glm::vec3(sx * scale, sy * scale, -cosf(theta));
        const float apertureRadiusM = centered ? 0.0f : apertureDiameterMm * 0.0005f;
        if (apertureRadiusM > 0.f) {
            const float angle  = 6.28318530718f * lensSample.x;
            const float rAperture = apertureRadiusM * sqrtf(lensSample.y);
            origin = glm::vec3(cosf(angle) * rAperture, sinf(angle) * rAperture, 0.f);
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
