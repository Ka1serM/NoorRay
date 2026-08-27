#pragma once

#include "Backend/Host/Platform.h"
#include "Rendering/Camera/Camera.h"

class OrthographicCamera : public Camera {
public:
    OrthographicCamera();
    explicit OrthographicCamera(std::unique_ptr<Sensor> sensor);
    OrthographicCamera(const OrthographicCamera& other);
    ~OrthographicCamera();
    NR_CPU_GPU std::optional<CameraSample> generateRay(
        float nx, float ny, const glm::vec2&, uint32_t, SampledWavelengths&,
        bool = false) const
    {
        constexpr float referenceHeight = 10.f;
        constexpr float referenceFocalLengthMm = 21.f;
        const float height = referenceHeight * referenceFocalLengthMm / fmaxf(0.001f, focalLengthMm);
        const Sensor& sensor = getSensor();
        CameraSample result{};
        result.ray = transformRay(Ray(
            glm::vec3(nx * height * sensor.filmWidth() / sensor.filmHeight() * 0.5f,
                ny * height * 0.5f, 0.0f),
            glm::vec3(0.0f, 0.0f, -1.0f)));
        return result;
    }

    bool renderUi();
};
