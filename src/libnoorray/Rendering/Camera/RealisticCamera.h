#pragma once

#include "Backend/Host/Platform.h"
#include "Rendering/Camera/Camera.h"
#include "Rendering/Optics/KolbLens.h"
#include <memory>
#include <string>
#include <vector>

class RealisticCamera : public Camera {
public:
    nr::optics::LensSnapshot optics{};
    float sensorWidthCm{}, sensorHeightCm{}, filmDiagonalCm{};
    float apertureDiameterMm{};
    NR_CPU_GPU bool invalidRayIsOpaque() const { return true; }
    NR_CPU_GPU std::optional<CameraSample> generateRay(float nx, float ny,
        glm::vec2 lensSample, uint32_t, SampledWavelengths& wavelengths, bool centered = false) const
    {
        wavelengths.terminateSecondary();
        if (!optics.surfaceCount || optics.rearPupilRadius <= 0.f) return std::nullopt;
        const float sx = centered ? .5f : lensSample.x, sy = centered ? .5f : lensSample.y;
        const float a = 2.f * sx - 1.f, b = 2.f * sy - 1.f;
        const float r = fabsf(a) > fabsf(b) ? a : b;
        const float phi = fabsf(a) > fabsf(b) ? .785398163f * b / (a == 0.f ? 1.f : a) : 1.570796327f - .785398163f * a / (b == 0.f ? 1.f : b);
        const glm::vec2 pupil = optics.rearPupilRadius * r * glm::vec2(cosf(phi), sinf(phi));
        const Sensor& sensor = getSensor();
        const glm::vec3 origin(-nx * sensor.filmWidth() * .5f, -ny * sensor.filmHeight() * .5f, 0.f);
        glm::vec3 tracedOrigin = origin;
        glm::vec3 direction = glm::normalize(glm::vec3(pupil, optics.rearPupilZ) - origin);
        if (!nr::optics::traceFilmToWorld(optics, tracedOrigin, direction, wavelengths[0])) return std::nullopt;
        const float area = 3.141592654f * optics.rearPupilRadius * optics.rearPupilRadius;
        CameraSample result{};
        result.weight = area * direction.z * direction.z * direction.z * direction.z / fmaxf(1e-6f, optics.rearPupilZ * optics.rearPupilZ);
        result.ray = transformRay(Ray(tracedOrigin * .001f, glm::normalize(glm::vec3(direction.x, direction.y, -direction.z))));
        return result;
    }
    RealisticCamera(); explicit RealisticCamera(std::unique_ptr<Sensor> sensor); RealisticCamera(const RealisticCamera& other); ~RealisticCamera();
    bool renderUi(); void load(std::string lensPath, std::string glassCatalogPaths); void load(std::string lensPath, const std::vector<std::string>& glassCatalogPaths);
    void setApertureDiameterMm(float apertureDiameterMm); void setOpticalFocusDistanceCm(float focusDistanceCm); void prepareOptics(); void setOpticsPaths(std::string lensPath, std::string glassCatalogPaths);
    const std::string& getLensPath() const { return lensPath; } const std::string& getGlassCatalogPaths() const { return glassCatalogPaths; } float derivedFocalLengthMm() const { return focalLengthMm; }
    bool loadLensAndSensor(bool resetLensSettings = false); bool consumeOpticsDirty() { const bool result = opticsDirty; opticsDirty = false; return result; }
private:
    std::string lensPath, glassCatalogPaths, loadStatus; bool opticsDirty{true}, opticsUpdatePending{}; nr::optics::LensSnapshot sourceOptics{};
    std::unique_ptr<pfd::open_file> lensDialog; std::unique_ptr<pfd::open_file> glassCatalogDialog;
    void updateLensSettings();
};
