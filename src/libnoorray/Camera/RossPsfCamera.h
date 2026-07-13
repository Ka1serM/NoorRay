#pragma once

#include "Camera/Camera.h"
#include "CUDA/Annotations.h"
#include "CUDA/rstd/UniquePtr.h"

#if !defined(NR_OPTIX_PTX_BUILD)
#include "libross/imaging/cameralens/CameraLens.h"
#include "libross/imaging/cameralens/raytracing/exitpupil/ExitPupil.h"
#include "libross/imaging/cameralens/raylut/RayLUT.h"
#include "libross/imaging/imagesensor/ImageSensor.h"
#else
namespace ross {
struct CameraLens;
struct ExitPupil;
struct ImageSensor;
class RayLUT;
}
#endif

#ifndef NR_GPU_CODE
#include <memory>
#include <string>
#include "portable-file-dialogs.h"
#endif

class RossPsfCamera : public Camera {
public:
    nr::rstd::unique_ptr<ross::CameraLens> rossLens;
    nr::rstd::unique_ptr<ross::CameraLens> sourceRossLens;
    nr::rstd::unique_ptr<ross::ExitPupil> exitPupil;
    nr::rstd::unique_ptr<ross::ImageSensor> rossSensor;
    nr::rstd::unique_ptr<ross::RayLUT> rayLut;

    float apertureDiameterMm{};
    int rayLutStepSize{32};
    int samplesPerDimension{8};

    NR_CPU_GPU bool generateRay(glm::vec3& origin, glm::vec3& direction, float& weight,
        float nx, float ny, RandomState&, uint32_t, SampledWavelengths& wavelengths,
        bool = false) const
    {
#if defined(NR_OPTIX_PTX_BUILD)
        return false;
#else
        if (!rayLut || !rossLens || !exitPupil
            || exitPupil->pupilBounds.empty())
            return false;

        wavelengths.terminateSecondary();
        const Sensor& sensor = getSensor();
        const float width = static_cast<float>(sensor.resolutionX());
        const float height = static_cast<float>(sensor.resolutionY());
        const float px = (nx * 0.5f + 0.5f) * width;
        const float py = (1.0f - (ny * 0.5f + 0.5f)) * height;

        const auto traced = rayLut->lookupInterpolated(ross::Vector2f(px, py), wavelengths[0]);
        if (!traced)
            return false;

        // Match RealisticCamera's complete importance weight. The Hybrid path
        // uses its chief ray for direction, while the centered exit-pupil sample
        // supplies the projected pupil area and corresponding film-ray angle.
        const float sensorWidthCm = sensor.width() * 0.1f;
        const float sensorHeightCm = sensor.height() * 0.1f;
        const float filmDiagonalCm = sqrtf(
            sensorWidthCm * sensorWidthCm + sensorHeightCm * sensorHeightCm);
        const ross::Vector2f filmPos(
            -nx * sensorWidthCm * 0.5f, -ny * sensorHeightCm * 0.5f);
        const auto pupil = exitPupil->samplePupil(
            filmPos, filmDiagonalCm, ross::Vector2f(0.5f, 0.5f));
        const ross::Ray filmRay = ross::Ray::betweenPoints(
            ross::Vector3f(filmPos.x, filmPos.y, 0.0f),
            ross::Vector3f(pupil.point.x, pupil.point.y,
                rossLens->getLastSurface().center));
        const float cosTheta = filmRay.direction.z;
        const float cosTheta2 = cosTheta * cosTheta;
        const float pupilPlaneDistance = rossLens->getLastSurface().center;
        weight = (cosTheta2 * cosTheta2) * pupil.sampleBoundsArea
            / (pupilPlaneDistance * pupilPlaneDistance);
        origin = glm::vec3(traced->startPoint.x * 0.01f, traced->startPoint.y * 0.01f,
            -traced->startPoint.z * 0.01f);
        direction = glm::normalize(glm::vec3(
            traced->direction.x, traced->direction.y, -traced->direction.z));
        transformRay(origin, direction);
        return true;
#endif
    }

#ifndef NR_GPU_CODE
    RossPsfCamera();
    explicit RossPsfCamera(std::unique_ptr<Sensor> sensor);
    RossPsfCamera(const RossPsfCamera& other);
    ~RossPsfCamera();
    bool renderUi();
    void load(std::string lensPath, std::string glassCatalogPaths, std::string rayLutPath);
    void load(std::string lensPath, const std::vector<std::string>& glassCatalogPaths,
        std::string rayLutPath = {});
    void setApertureDiameter(float millimeters);
    void setOpticalFocusDistance(float meters);
    void prepareOptics();
    void setOpticsPaths(std::string lensPath, std::string glassCatalogPaths);
    void loadLensSensorAndPsf(bool buildRayLut = true, bool resetLensSettings = false);
    const std::string& getLensPath() const { return lensPath; }
    const std::string& getGlassCatalogPaths() const { return glassCatalogPaths; }
    const std::string& getRayLutPath() const { return rayLutPath; }
    bool consumeOpticsDirty()
    {
        const bool wasDirty = opticsDirty;
        opticsDirty = false;
        return wasDirty;
    }

private:
    std::string lensPath;
    std::string glassCatalogPaths;
    std::string rayLutPath;
    std::string loadStatus;
    bool opticsDirty{true};
    bool opticsUpdatePending{false};

    std::unique_ptr<pfd::open_file> lensDialog;
    std::unique_ptr<pfd::open_file> glassCatalogDialog;
    std::unique_ptr<pfd::open_file> rayLutOpenDialog;
    std::unique_ptr<pfd::save_file> rayLutSaveDialog;

    void freeRossObjects();
    void updateLensSettings();
#endif
};
