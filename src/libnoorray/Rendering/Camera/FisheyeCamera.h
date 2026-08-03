#pragma once

#include "Backend/CUDA/Annotations.h"
#include "Rendering/Camera/Camera.h"
#include "Rendering/Sampling/RandomSampler.h"

class FisheyeCamera : public Camera::Type<FisheyeCamera> {
public:
    FisheyeCamera();
    explicit FisheyeCamera(std::unique_ptr<Sensor> sensor);
    FisheyeCamera(const FisheyeCamera& other);
    ~FisheyeCamera();
    float apertureDiameterMm{};
    float bokehBias{1.f};

    NR_CPU_GPU nr::rstd::optional<CameraSample> generateRay(
        float nx, float ny, const glm::vec2 lensSample, uint32_t, SampledWavelengths&,
        bool centered = false) const
    {
        const Sensor& sensor = getSensor();
        const float aspect = sensor.filmWidth() / sensor.filmHeight();
        const float sx = nx * aspect;
        const float sy = ny;
        const float r = sqrtf(sx * sx + sy * sy);
        const float rMax = sqrtf(aspect * aspect + 1.0f);
        CameraSample result{};
        if (r < 1e-6f) {
            result.ray = transformRay(Ray(
                glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, -1.0f)));
            return result;
        }
        // Fisheye angular extent follows the same physical focal length and
        // film width as the pinhole cameras; FOV is derived, never stored.
        const float halfFov = atanf(
            getSensor().filmWidth() / (2.f * fmaxf(0.001f, focalLengthMm)));
        const float theta = (r / rMax) * halfFov;
        const float scale = sinf(theta) / r;
        glm::vec3 origin(0.0f);
        glm::vec3 direction(sx * scale, sy * scale, -cosf(theta));
        const float apertureRadiusM = centered ? 0.0f : apertureDiameterMm * 0.0005f;
        if (apertureRadiusM > 0.f) {
            const float angle  = 6.28318530718f * lensSample.x;
            const float rAperture = apertureRadiusM * sqrtf(lensSample.y);
            origin = glm::vec3(cosf(angle) * rAperture,
                sinf(angle) * rAperture, 0.0f);
            const glm::vec3 focusPoint = direction
                * ((focusDistanceCm * 0.01f) / fmaxf(-direction.z, 1e-5f));
            direction = glm::normalize(focusPoint - origin);
        }
        result.ray = transformRay(Ray(origin, direction));
        return result;
    }

    bool renderUi();
};
