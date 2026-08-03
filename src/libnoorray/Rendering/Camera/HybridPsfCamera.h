#pragma once

#include "Rendering/Camera/Camera.h"
#include "Backend/CUDA/Annotations.h"
#include "Backend/CUDA/rstd/UniquePtr.h"

#include "libross/imaging/cameralens/CameraLens.h"
#include "libross/imaging/cameralens/raytracing/exitpupil/ExitPupil.h"
#include "libross/imaging/cameralens/raylut/RayLUT.h"
#include "libross/imaging/imagesensor/ImageSensor.h"

#include <memory>
#include <string>
#include <vector>

class HybridPsfCamera : public Camera::Type<HybridPsfCamera> {
public:
    nr::rstd::unique_ptr<ross::CameraLens> rossLens;
    nr::rstd::unique_ptr<ross::CameraLens> sourceRossLens;
    nr::rstd::unique_ptr<ross::ExitPupil> exitPupil;
    nr::rstd::unique_ptr<ross::ImageSensor> rossSensor;
    nr::rstd::unique_ptr<ross::RayLUT> rayLut;

    float apertureDiameterMm{};
    int rayLutStepSize{32};
    int samplesPerDimension{8};

    NR_CPU_GPU bool invalidRayIsOpaque() const { return true; }

    NR_CPU_GPU nr::rstd::optional<CameraSample> generateRay(
        float nx, float ny, const glm::vec2&, uint32_t, SampledWavelengths& wavelengths,
        bool = false) const
    {
        if (!rayLut)
            return nr::rstd::nullopt;

        wavelengths.terminateSecondary();
        const Sensor& sensor = getSensor();
        const float width = static_cast<float>(sensor.resolutionX());
        const float height = static_cast<float>(sensor.resolutionY());
        // The LUT is built in physical sensor coordinates. Sensor-fit may
        // crop that film for the current render target, so map the fitted
        // film coordinates back into the native LUT extent before lookup.
        const float sensorNx = nx * sensor.filmWidth() / sensor.width();
        const float sensorNy = ny * sensor.filmHeight() / sensor.height();
        const float px = (sensorNx * 0.5f + 0.5f) * width;
        const float py = (1.0f - (sensorNy * 0.5f + 0.5f)) * height;

        const auto traced = rayLut->lookupInterpolated(ross::Vector2f(px, py), wavelengths[0]);
        if (!traced)
            return nr::rstd::nullopt;

        // ROSS's hybrid PSF camera uses a unit camera weight.
        CameraSample result{};
        result.ray = transformRay(Ray(
            glm::vec3(traced->startPoint.x * 0.01f,
                traced->startPoint.y * 0.01f, -traced->startPoint.z * 0.01f),
            glm::normalize(glm::vec3(
                traced->direction.x, traced->direction.y, -traced->direction.z))));
        return result;
    }

    HybridPsfCamera();
    explicit HybridPsfCamera(std::unique_ptr<Sensor> sensor);
    HybridPsfCamera(const HybridPsfCamera& other);
    ~HybridPsfCamera();
    bool renderUi();
    void load(std::string lensPath, std::string glassCatalogPaths, std::string rayLutPath);
    void load(std::string lensPath, const std::vector<std::string>& glassCatalogPaths,
        std::string rayLutPath = {});
    void setApertureDiameterMm(float apertureDiameterMm);
    void setOpticalFocusDistanceCm(float focusDistanceCm);
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
};
