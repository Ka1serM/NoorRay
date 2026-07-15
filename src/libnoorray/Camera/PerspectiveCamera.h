#pragma once

#include "CUDA/Annotations.h"
#include "Camera/Camera.h"

class PerspectiveCamera : public Camera::Type<PerspectiveCamera> {
public:
    PerspectiveCamera();
    explicit PerspectiveCamera(std::unique_ptr<Sensor> sensor);
    PerspectiveCamera(const PerspectiveCamera& other);
    ~PerspectiveCamera();
    NR_CPU_GPU bool generateRay(glm::vec3& origin, glm::vec3& direction, float& weight,
        float nx, float ny, const glm::vec2&, uint32_t, SampledWavelengths&,
        bool = false) const
    {
        weight = 1.0f;
        const Sensor& sensor = getSensor();
        const glm::vec2 scale{
            sensor.width() / (2.f * focalLengthMm),
            sensor.height() / (2.f * focalLengthMm)};
        origin = glm::vec3(0.f);
        direction = glm::normalize(glm::vec3(nx * scale.x, ny * scale.y, -1.f));
        transformRay(origin, direction);
        return true;
    }

    bool renderUi();
};
