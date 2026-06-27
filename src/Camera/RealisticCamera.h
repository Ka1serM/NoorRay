#pragma once

#include "GPU/Annotations.h"

#ifndef __CUDACC__
#include <string>
#include <vector>
#include "CameraBase.h"
#include "portable-file-dialogs.h"
#endif

#ifdef __CUDACC__
#include "Kernels/SceneData.h"
#endif

class RealisticCamera
#ifndef __CUDACC__
    final : public CameraBase
#endif
{
public:
#ifndef __CUDACC__
    RealisticCamera(Scene& scene, const std::string& name, Transform transform, CameraSettings settings = {});
    RealisticCamera(Scene& scene, const std::string& name, Transform transform, CameraSettings settings, std::string lensPath, std::string sensorPath, std::string glassCatalogPaths = {});
    RealisticCamera(const RealisticCamera& other);

    std::unique_ptr<SceneObject> clone() const override;
    CameraProjectionType getProjectionType() const override { return CameraProjectionType::Realistic; }
    const char* getProjectionName() const override { return "Realistic"; }
    bool getPreferredRenderSize(uint32_t& width, uint32_t& height) const override;
    bool supportsDOF() const override { return true; }
    mat4 getProjectionMatrix() const override;
    void renderUi() override;
    void populateRealisticCameraSettings(RealisticCameraSettings& realisticSettings) const override;
    std::string getRayLutCacheFolder() const override { return rayLutCacheFolder; }

protected:
    void computeProjectionData(const vec3& direction, const vec3& up, const vec3& right, float aspectRatio) override;

private:
    std::string lensPath;
    std::string sensorPath;
    std::string glassCatalogPaths;
    std::string rayLutCacheFolder;
    std::vector<RealisticLensElement> lensElements;
    std::vector<RealisticLensElement> maximumLensElements;
    std::vector<RealisticPupilBound> pupilBounds;
    float effectiveFocalLengthM = 0.045f;
    float focusSurfaceOffsetM = 0.0f;
    int apertureIndex = -1;
    std::string loadStatus;
    std::unique_ptr<pfd::open_file> lensDialog;
    std::unique_ptr<pfd::open_file> sensorDialog;
    std::unique_ptr<pfd::open_file> glassCatalogDialog;
    std::unique_ptr<pfd::select_folder> rayLutCacheDialog;

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
#endif

#ifdef __CUDACC__
    NR_GPU bool generateRay(glm::vec3& origin, glm::vec3& direction,
        float, float, float, uint32_t&,
        const RayLutEntry* rayLut, uint32_t pixelIndex, int rayLutEnabled) const
    {
        if (rayLutEnabled == 0 || rayLut == nullptr || rayLut[pixelIndex].originValid == 0.0f)
            return false;
        origin = rayLut[pixelIndex].origin;
        direction = rayLut[pixelIndex].direction;
        return true;
    }
#endif
};
