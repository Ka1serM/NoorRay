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
    float fStop{0.f};
    float bokehBias{1.f};

    NR_CPU_GPU bool generateRay(glm::vec3& origin, glm::vec3& direction, float& weight,
        float nx, float ny, RandomState& rng, uint32_t, SampledWavelengths&, bool centered = false) const
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
        const float theta = (r / rMax) * glm::radians(fieldOfView) * 0.5f;
        const float scale = sinf(theta) / r;
        direction = glm::vec3(sx * scale, sy * scale, -cosf(theta));
        const float aperture = !centered && fStop > 0.f ? (focalLengthMm / fStop) * 0.5f * 0.001f : 0.f;
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

    bool renderUi();
};
