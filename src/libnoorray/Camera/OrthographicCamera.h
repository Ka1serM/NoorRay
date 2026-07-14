#pragma once

#include "CUDA/Annotations.h"
#include "Camera/Camera.h"

class OrthographicCamera : public Camera {
public:
#ifndef NR_GPU_CODE
    OrthographicCamera();
    explicit OrthographicCamera(std::unique_ptr<Sensor> sensor);
    OrthographicCamera(const OrthographicCamera& other);
    ~OrthographicCamera();
#endif
    NR_CPU_GPU bool generateRay(glm::vec3& origin, glm::vec3& direction, float& weight,
        float nx, float ny, const glm::vec2&, uint32_t, SampledWavelengths&,
        bool = false) const
    {
        weight = 1.0f;
        constexpr float referenceHeight = 10.f;
        constexpr float referenceFocalLengthMm = 21.f;
        const float height = referenceHeight * referenceFocalLengthMm / fmaxf(0.001f, focalLengthMm);
        const Sensor& sensor = getSensor();
        origin = glm::vec3(nx * height * sensor.aspectRatio() * 0.5f, ny * height * 0.5f, 0.f);
        direction = glm::vec3(0.f, 0.f, -1.f);
        transformRay(origin, direction);
        return true;
    }

    bool renderUi();
};
