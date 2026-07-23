#pragma once

#include "CUDA/Annotations.h"
#include "Camera/Camera.h"
#include "Samplers/RandomSampler.h"

class FisheyeCamera : public Camera::Type<FisheyeCamera> {
public:
    FisheyeCamera();
    explicit FisheyeCamera(std::unique_ptr<Sensor> sensor);
    FisheyeCamera(const FisheyeCamera& other);
    ~FisheyeCamera();
    float apertureDiameterMm{};
    float bokehBias{1.f};

    NR_CPU_GPU nr::rstd::optional<CameraRay> generateRay(
        float nx, float ny, const glm::vec2 lensSample, uint32_t, SampledWavelengths&,
        bool centered = false) const
    {
        const Sensor& sensor = getSensor();
        const float aspect = sensor.width() / sensor.height();
        const float sx = nx * aspect;
        const float sy = ny;
        const float r = sqrtf(sx * sx + sy * sy);
        const float rMax = sqrtf(aspect * aspect + 1.0f);
        CameraRay result{};
        if (r < 1e-6f) {
            result.ray.direction = glm::vec3(0.f, 0.f, -1.f);
            transformRay(result.ray.origin, result.ray.direction);
            return result;
        }
        const float theta = (r / rMax) * glm::radians(fieldOfViewDegrees) * 0.5f;
        const float scale = sinf(theta) / r;
        result.ray.direction = glm::vec3(sx * scale, sy * scale, -cosf(theta));
        const float apertureRadiusM = centered ? 0.0f : apertureDiameterMm * 0.0005f;
        if (apertureRadiusM > 0.f) {
            const float angle  = 6.28318530718f * lensSample.x;
            const float rAperture = apertureRadiusM * sqrtf(lensSample.y);
            result.ray.origin = glm::vec3(cosf(angle) * rAperture, sinf(angle) * rAperture, 0.f);
            const glm::vec3 focusPoint = result.ray.direction *
                ((focusDistanceCm * 0.01f) / fmaxf(-result.ray.direction.z, 1e-5f));
            result.ray.direction = glm::normalize(focusPoint - result.ray.origin);
        } else {
            result.ray.origin = glm::vec3(0.f);
        }
        transformRay(result.ray.origin, result.ray.direction);
        return result;
    }

    bool renderUi();
};
