#pragma once

#include "Camera/Camera.h"
#include "CUDA/Annotations.h"

#if !defined(NR_OPTIX_PTX_BUILD)
#include "libross/imaging/cameralens/CameraLens.h"
#include "libross/imaging/cameralens/raylut/RayLUT.h"
#include "libross/imaging/imagesensor/ImageSensor.h"
#else
namespace ross {
struct CameraLens;
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
    ross::CameraLens* rossLens{};
    ross::ImageSensor* rossSensor{};
    ross::RayLUT* rayLut{};

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
        if (rayLut == nullptr)
            return false;

        wavelengths.terminateSecondary();
        const float width = static_cast<float>(sensor.resolutionX());
        const float height = static_cast<float>(sensor.resolutionY());
        const float px = (nx * 0.5f + 0.5f) * width;
        const float py = (1.0f - (ny * 0.5f + 0.5f)) * height;

        const auto traced = rayLut->lookupInterpolated(ross::Vector2f(px, py), wavelengths[0]);
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
    ~RossPsfCamera();
    bool renderUi();
    void load(std::string lensPath, std::string glassCatalogPaths, std::string rayLutPath);
    void setOpticsPaths(std::string lensPath, std::string glassCatalogPaths);
    void loadLensSensorAndPsf(bool buildRayLut = true);
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

    std::unique_ptr<pfd::open_file> lensDialog;
    std::unique_ptr<pfd::open_file> glassCatalogDialog;
    std::unique_ptr<pfd::open_file> rayLutOpenDialog;
    std::unique_ptr<pfd::save_file> rayLutSaveDialog;

    void freeRossObjects();
#endif
};
