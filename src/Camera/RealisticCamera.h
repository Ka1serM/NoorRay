#pragma once

#include "CUDA/Annotations.h"
#include "Camera/Camera.h"
#if !defined(NR_OPTIX_PTX_BUILD)
#include "libross/imaging/cameralens/CameraLens.h"
#include "libross/imaging/cameralens/raytracing/sequential/FromFilmToWorldRaytracer.h"
#include "libross/imaging/cameralens/raytracing/exitpupil/ExitPupil.h"
#include "libross/imaging/imagesensor/ImageSensor.h"
#else
namespace ross {
struct CameraLens;
struct ImageSensor;
struct ExitPupil;
}
#endif
#include <glm/vec2.hpp>
using glm::vec2;

class RealisticCamera : public Camera {
public:
    ross::CameraLens* rossLens{};
    ross::ExitPupil* exitPupil{};
    float sensorWidthCm{};
    float sensorHeightCm{};
    float filmDiagonalCm{};

    NR_CPU_GPU bool generateRay(glm::vec3& origin, glm::vec3& direction, float& weight,
        float nx, float ny, RandomState& rng, uint32_t, const float wavelengthNm,
        bool centered = false) const
    {
#if defined(NR_OPTIX_PTX_BUILD)
        return false;
#else
        weight = 0.0f;
        if (rossLens == nullptr || exitPupil == nullptr || exitPupil->pupilBounds.empty())
            return false;

        const ross::Vector2f filmPos(-nx * sensorWidthCm * 0.5f, -ny * sensorHeightCm * 0.5f);

        const ross::Vector2f sample(
            centered ? 0.5f : randomFloat(rng),
            centered ? 0.5f : randomFloat(rng));
        const auto pupilSample = exitPupil->samplePupil(filmPos, filmDiagonalCm, sample);

        const ross::Vector3f filmPoint(filmPos.x, filmPos.y, 0.0f);
        const ross::Vector3f pupilPoint(
            pupilSample.point.x, pupilSample.point.y,
            rossLens->getLastSurface().center);
        const ross::Ray filmRay = ross::Ray::betweenPoints(filmPoint, pupilPoint);

        ross::FromFilmToWorldRaytracer raytracer(*rossLens);
        const auto traced = raytracer.trace(filmRay, wavelengthNm);
        if (!traced)
            return false;

        weight = 1.0f;

        origin = glm::vec3(traced->startPoint.x * 0.01f, traced->startPoint.y * 0.01f,
            -traced->startPoint.z * 0.01f);
        direction = glm::normalize(glm::vec3(
            traced->direction.x, traced->direction.y, -traced->direction.z));
        transformRay(origin, direction);
        return true;
#endif
    }

#ifndef NR_GPU_CODE
    ~RealisticCamera();
    bool renderUi();
    void load(std::string lensPath, std::string sensorPath, std::string glassCatalogPaths);
    float derivedFocalLengthMm() const { return effectiveFocalLengthM * 1000.0f; }
    float apertureDiameterMm{0.f};
    void loadLensAndSensor();

private:
    std::string lensPath;
    std::string sensorPath;
    std::string glassCatalogPaths;
    float effectiveFocalLengthM = 0.045f;
    std::string loadStatus;

    void freeRossLens();
#endif
};
