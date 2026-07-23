#pragma once

#include "CUDA/Annotations.h"
#include "Camera/Camera.h"
#include "Samplers/RandomSampler.h"

class ThinLensCamera : public Camera::Type<ThinLensCamera> {
public:
    ThinLensCamera();
    explicit ThinLensCamera(std::unique_ptr<Sensor> sensor);
    ThinLensCamera(const ThinLensCamera& other);
    ~ThinLensCamera();
    float apertureDiameterMm{};
    float bokehBias{1.f};

    NR_CPU_GPU nr::rstd::optional<CameraRay> generateRay(
        float nx, float ny, const glm::vec2 lensSample, uint32_t, SampledWavelengths&,
        bool centered = false) const
    {
        const Sensor& sensor = getSensor();
        const glm::vec2 scale{
            sensor.width() / (2.f * focalLengthMm),
            sensor.height() / (2.f * focalLengthMm)};
        CameraRay result{};
        result.ray.direction = glm::normalize(glm::vec3(nx * scale.x, ny * scale.y, -1.f));
        const float apertureRadiusM = centered ? 0.0f : apertureDiameterMm * 0.0005f;
        if (apertureRadiusM > 0.f) {
            const float angle  = 6.28318530718f * lensSample.x;
            const float radius = apertureRadiusM * sqrtf(lensSample.y);
            result.ray.origin = glm::vec3(cosf(angle) * radius, sinf(angle) * radius, 0.f);
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
