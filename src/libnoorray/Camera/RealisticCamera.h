#pragma once

#include "CUDA/Annotations.h"
#include "CUDA/rstd/UniquePtr.h"
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
#ifndef NR_GPU_CODE
#include <memory>
#include "portable-file-dialogs.h"
#endif
using glm::vec2;

class RealisticCamera : public Camera {
public:
    nr::rstd::unique_ptr<ross::CameraLens> rossLens;
    nr::rstd::unique_ptr<ross::CameraLens> sourceRossLens;
    nr::rstd::unique_ptr<ross::ExitPupil> exitPupil;
    float sensorWidthCm{};
    float sensorHeightCm{};
    float filmDiagonalCm{};

#if !defined(NR_OPTIX_PTX_BUILD)
    // Construct the libross film ray used by generateRay. Keeping this public lets diagnostic
    // views exercise exactly the same film mapping and exit-pupil sampling as rendered rays.
    NR_CPU_GPU bool makeFilmRay(ross::Ray& filmRay, float nx, float ny,
        const ross::Vector2f& pupilSample, float* sampleBoundsArea = nullptr) const
    {
        if (!rossLens || !exitPupil || exitPupil->pupilBounds.empty())
            return false;

        const ross::Vector2f filmPos(-nx * sensorWidthCm * 0.5f, -ny * sensorHeightCm * 0.5f);
        const auto pupil = exitPupil->samplePupil(filmPos, filmDiagonalCm, pupilSample);
        if (sampleBoundsArea != nullptr)
            *sampleBoundsArea = pupil.sampleBoundsArea;
        filmRay = ross::Ray::betweenPoints(
            ross::Vector3f(filmPos.x, filmPos.y, 0.0f),
            ross::Vector3f(pupil.point.x, pupil.point.y,
                rossLens->getLastSurface().center));
        return true;
    }
#endif

    NR_CPU_GPU bool generateRay(glm::vec3& origin, glm::vec3& direction, float& weight,
        float nx, float ny, RandomState& rng, uint32_t, SampledWavelengths& wavelengths,
        bool centered = false) const
    {
#if defined(NR_OPTIX_PTX_BUILD)
        return false;
#else
        wavelengths.terminateSecondary();
        weight = 0.0f;
        const ross::Vector2f sample(
            centered ? 0.5f : randomFloat(rng),
            centered ? 0.5f : randomFloat(rng));
        ross::Ray filmRay;
        float sampleBoundsArea;
        if (!makeFilmRay(filmRay, nx, ny, sample, &sampleBoundsArea))
            return false;

        ross::FromFilmToWorldRaytracer raytracer(*rossLens);
        const auto traced = raytracer.trace(filmRay, wavelengths[0]);
        if (!traced)
            return false;

        // Match ROSS's RossRealisticCamera importance weighting: the sampled exit-pupil
        // area, cosine-fourth falloff, and inverse-square pupil-plane distance.
        const float cosTheta = filmRay.direction.z;
        const float cosTheta2 = cosTheta * cosTheta;
        const float pupilPlaneDistance = rossLens->getLastSurface().center;
        weight = (cosTheta2 * cosTheta2) /
            ((1.0f / sampleBoundsArea) * (pupilPlaneDistance * pupilPlaneDistance));

        origin = glm::vec3(traced->startPoint.x * 0.01f, traced->startPoint.y * 0.01f,
            -traced->startPoint.z * 0.01f);
        direction = glm::normalize(glm::vec3(
            traced->direction.x, traced->direction.y, -traced->direction.z));
        transformRay(origin, direction);
        return true;
#endif
    }

#ifndef NR_GPU_CODE
    RealisticCamera();
    explicit RealisticCamera(std::unique_ptr<Sensor> sensor);
    RealisticCamera(const RealisticCamera& other);
    ~RealisticCamera();
    bool renderUi();
    void load(std::string lensPath, std::string glassCatalogPaths);
    void load(std::string lensPath, const std::vector<std::string>& glassCatalogPaths);
    void setApertureDiameterMm(float apertureDiameterMm);
    void setOpticalFocusDistanceCm(float focusDistanceCm);
    void prepareOptics();
    void setOpticsPaths(std::string lensPath, std::string glassCatalogPaths);
    const std::string& getLensPath() const { return lensPath; }
    const std::string& getGlassCatalogPaths() const { return glassCatalogPaths; }
    float derivedFocalLengthMm() const { return focalLengthMm; }
    float apertureDiameterMm{0.f};
    void loadLensAndSensor(bool resetLensSettings = false);
    bool consumeOpticsDirty()
    {
        const bool wasDirty = opticsDirty;
        opticsDirty = false;
        return wasDirty;
    }

private:
    std::string lensPath;
    std::string glassCatalogPaths;
    std::string loadStatus;
    bool opticsDirty = true;
    bool opticsUpdatePending = false;

    std::unique_ptr<pfd::open_file> lensDialog;
    std::unique_ptr<pfd::open_file> glassCatalogDialog;

    void freeRossLens();
    void updateLensSettings();
#endif
};
