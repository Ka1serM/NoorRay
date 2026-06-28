#pragma once

#include "GPU/Annotations.h"
#include "Camera/Camera.h"
#include "Scene/SceneTypes.h"

#ifndef __CUDACC__
#include <memory>
#include <string>
#include <vector>
#include "portable-file-dialogs.h"
#endif

class CameraInstance;

class RealisticCamera : public Camera {
public:
    RealisticLensElement elements[MaxRealisticLensElements]{};
    RealisticPupilBound exitPupilBounds[MaxRealisticExitPupilBounds]{};
    int elementCount{};
    int pupilBoundCount{};
    float sensorWidth{};
    float sensorHeight{};
    float apertureRadius{};
    float filmDiagonal{};
    float rearElementZ{};
    float surfaceOffset{};
    RayLutEntry* rayLut{};
    uint32_t rayLutSize{};

    NR_CPU_GPU bool generateRay(glm::vec3& origin, glm::vec3& direction,
        float, float, uint32_t&, uint32_t pixelIndex) const
    {
        if (rayLut == nullptr || pixelIndex >= rayLutSize || rayLut[pixelIndex].originValid == 0.f)
            return false;
        origin    = rayLut[pixelIndex].origin;
        direction = rayLut[pixelIndex].direction;
        transformRay(origin, direction);
        return true;
    }

#ifndef __CUDACC__
    ~RealisticCamera();
    void renderUi(CameraInstance& inst);
    void load(std::string lensPath, std::string sensorPath, std::string glassCatalogPaths);

private:
    std::string lensPath;
    std::string sensorPath;
    std::string glassCatalogPaths;
    std::string rayLutPath;
    std::vector<RealisticLensElement> lensElements;
    std::vector<RealisticLensElement> maximumLensElements;
    std::vector<RealisticPupilBound> pupilBounds;
    float effectiveFocalLengthM = 0.045f;
    float focusSurfaceOffsetM = 0.f;
    float firstPrincipalZ{};
    float secondPrincipalZ{};
    bool thickLensValid{};
    int apertureIndex = -1;
    std::string loadStatus;
    std::unique_ptr<pfd::open_file> lensDialog;
    std::unique_ptr<pfd::open_file> sensorDialog;
    std::unique_ptr<pfd::open_file> glassCatalogDialog;
    std::unique_ptr<pfd::open_file> rayLutDialog;

    bool loadLensAndSensor();
    bool parseLensFile(const std::string& path, std::vector<RealisticLensElement>& elements, std::string& error) const;
    bool parseOlioFile(const std::string& text, std::vector<RealisticLensElement>& elements, std::string& error) const;
    bool parseDatFile(const std::string& text, std::vector<RealisticLensElement>& elements, std::string& error) const;
    bool parseZmxFile(const std::string& text, std::vector<RealisticLensElement>& elements, std::string& error) const;
    bool parseSensorFile(const std::string& path, std::string& error);
    void finalizeElements(std::vector<RealisticLensElement>& elements) const;
    void rebuildExitPupilBounds();
    void rebuildLensMetadata();
    void applyAperture();
    void applyFocus();
    void updateGpuData();
    void rebuildRayLut();
    void releaseRayLut();
#endif
};
