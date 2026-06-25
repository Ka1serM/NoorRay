#pragma once

#include <string>
#include <vector>
#include "CameraBase.h"
#include "portable-file-dialogs.h"

class RealisticCamera final : public CameraBase {
public:
    RealisticCamera(Scene& scene, const std::string& name, Transform transform, CameraSettings settings = {});
    RealisticCamera(Scene& scene, const std::string& name, Transform transform, CameraSettings settings, std::string lensPath, std::string sensorPath, std::string glassCatalogPaths = {});
    RealisticCamera(const RealisticCamera& other);

    std::unique_ptr<SceneObject> clone() const override;
    CameraProjectionType getProjectionType() const override { return CameraProjectionType::Realistic; }
    const char* getProjectionName() const override { return "Realistic"; }
    bool getPreferredRenderSize(uint32_t& width, uint32_t& height) const override;
    mat4 getProjectionMatrix() const override;
    void renderUi() override;
    void populateRealisticCameraSettings(RealisticCameraSettings& realisticSettings) const override;

protected:
    void computeProjectionData(const vec3& direction, const vec3& up, const vec3& right, float aspectRatio) override;

private:
    std::string lensPath;
    std::string sensorPath;
    std::string glassCatalogPaths;
    std::vector<RealisticLensElement> lensElements;
    std::vector<RealisticLensElement> maximumLensElements;
    std::vector<RealisticPupilBound> pupilBounds;
    float sensorWidthM = CameraSettings::SensorWidthMm * 0.001f;
    float sensorHeightM = 18.0f * 0.001f;
    uint32_t sensorResolutionWidth = 0;
    uint32_t sensorResolutionHeight = 0;
    float effectiveFocalLengthM = 0.045f;
    float focusSurfaceOffsetM = 0.0f;
    int apertureIndex = -1;
    std::string loadStatus;
    std::unique_ptr<pfd::open_file> lensDialog;
    std::unique_ptr<pfd::open_file> sensorDialog;
    std::unique_ptr<pfd::open_file> glassCatalogDialog;

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
};
