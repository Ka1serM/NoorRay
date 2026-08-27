#pragma once

#include "Backend/Host/Platform.h"
#include "Rendering/Camera/Camera.h"
#include "Rendering/Sampling/RandomSampler.h"

class ThinLensCamera : public Camera {
public:
    ThinLensCamera();
    explicit ThinLensCamera(std::unique_ptr<Sensor> sensor);
    ThinLensCamera(const ThinLensCamera& other);
    ~ThinLensCamera();
    float apertureDiameterMm{};
    float bokehBias{1.f};

    NR_CPU_GPU std::optional<CameraSample> generateRay(
        float nx, float ny, const glm::vec2 lensSample, uint32_t, SampledWavelengths&,
        bool centered = false) const
    {
        const Sensor& sensor = getSensor();
        const glm::vec2 scale{
            sensor.filmWidth() / (2.f * focalLengthMm),
            sensor.filmHeight() / (2.f * focalLengthMm)};
        CameraSample result{};
        glm::vec3 origin(0.0f);
        glm::vec3 direction = glm::normalize(
            glm::vec3(nx * scale.x, ny * scale.y, -1.0f));
        const float apertureRadiusM = centered ? 0.0f : apertureDiameterMm * 0.0005f;
        if (apertureRadiusM > 0.f) {
            const float angle  = 6.28318530718f * lensSample.x;
            const float radius = apertureRadiusM * sqrtf(lensSample.y);
            origin = glm::vec3(cosf(angle) * radius,
                sinf(angle) * radius, 0.0f);
            const glm::vec3 focusPoint = direction
                * ((focusDistanceCm * 0.01f) / fmaxf(-direction.z, 1e-5f));
            direction = glm::normalize(focusPoint - origin);
        }
        result.ray = transformRay(Ray(origin, direction));
        return result;
    }

    bool renderUi();
};
