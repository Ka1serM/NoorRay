#pragma once

#include "Camera/Camera.h"
#include "Shared/Lens.h"
#include <memory>
#include <string>
#include <vector>

class RealisticCamera : public Camera {
public:
    nr::graphics::Lens optics{};
    float sensorWidthCm{}, sensorHeightCm{}, filmDiagonalCm{};
    float apertureDiameterMm{};
    RealisticCamera(); explicit RealisticCamera(std::unique_ptr<Sensor> sensor); RealisticCamera(const RealisticCamera& other); ~RealisticCamera();
    bool renderUi(); void load(std::string lensPath, std::string glassCatalogPaths); void load(std::string lensPath, const std::vector<std::string>& glassCatalogPaths);
    void setApertureDiameterMm(float apertureDiameterMm); void setOpticalFocusDistanceCm(float focusDistanceCm); void prepareOptics(); void setOpticsPaths(std::string lensPath, std::string glassCatalogPaths);
    const std::string& getLensPath() const { return lensPath; } const std::string& getGlassCatalogPaths() const { return glassCatalogPaths; } float derivedFocalLengthMm() const { return focalLengthMm; }
    bool loadLensAndSensor(bool resetLensSettings = false); bool consumeOpticsDirty() { const bool result = opticsDirty; opticsDirty = false; return result; }
private:
    std::string lensPath, glassCatalogPaths, loadStatus; bool opticsDirty{true}, opticsUpdatePending{}; nr::graphics::Lens sourceOptics{};
    std::unique_ptr<pfd::open_file> lensDialog; std::unique_ptr<pfd::open_file> glassCatalogDialog;
    void updateLensSettings();
};
