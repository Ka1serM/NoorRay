#pragma once

#include "Backend/Host/Platform.h"
#include "Rendering/Camera/Camera.h"

class PerspectiveCamera : public Camera {
public:
    PerspectiveCamera();
    explicit PerspectiveCamera(std::unique_ptr<Sensor> sensor);
    PerspectiveCamera(const PerspectiveCamera& other);
    ~PerspectiveCamera();
    NR_CPU_GPU std::optional<CameraSample> generateRay(
        float nx, float ny, const glm::vec2&, uint32_t, SampledWavelengths&,
        bool = false) const
    {
        const Sensor& sensor = getSensor();
        const glm::vec2 scale{
            sensor.filmWidth() / (2.f * focalLengthMm),
            sensor.filmHeight() / (2.f * focalLengthMm)};
        CameraSample result{};
        result.ray = transformRay(Ray(glm::vec3(0.0f),
            glm::normalize(glm::vec3(nx * scale.x, ny * scale.y, -1.0f))));
        return result;
    }

    bool renderUi();
};
