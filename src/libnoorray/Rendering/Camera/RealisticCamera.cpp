#include "RealisticCamera.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "Log.h"
#include "portable-file-dialogs.h"
namespace { std::string join(const std::vector<std::string>& p) { std::string r; for (const auto& v : p) { if (!r.empty()) r += ';'; r += v; } return r; } }
RealisticCamera::RealisticCamera() : RealisticCamera(std::make_unique<RectangularSensor>()) {}
RealisticCamera::RealisticCamera(std::unique_ptr<Sensor> sensor) : Camera(std::move(sensor)) {}
RealisticCamera::~RealisticCamera() = default;
RealisticCamera::RealisticCamera(const RealisticCamera& o) : Camera(o), optics(o.optics), sensorWidthCm(o.sensorWidthCm), sensorHeightCm(o.sensorHeightCm), filmDiagonalCm(o.filmDiagonalCm), apertureDiameterMm(o.apertureDiameterMm), lensPath(o.lensPath), glassCatalogPaths(o.glassCatalogPaths), loadStatus(o.loadStatus), opticsDirty(o.opticsDirty), opticsUpdatePending(o.opticsUpdatePending), sourceOptics(o.sourceOptics) {}
void RealisticCamera::load(std::string p, std::string c) { setOpticsPaths(std::move(p), std::move(c)); loadLensAndSensor(); }
void RealisticCamera::load(std::string p, const std::vector<std::string>& c) { load(std::move(p), join(c)); }
void RealisticCamera::setOpticsPaths(std::string p, std::string c) { lensPath = std::move(p); glassCatalogPaths = std::move(c); }
void RealisticCamera::setApertureDiameterMm(float v) { v = std::max(0.f, v); if (v != apertureDiameterMm) { apertureDiameterMm = v; opticsUpdatePending = sourceOptics.surfaceCount != 0; } }
void RealisticCamera::setOpticalFocusDistanceCm(float v) { v = std::max(.1f, v); if (v != focusDistanceCm) { focusDistanceCm = v; opticsUpdatePending = sourceOptics.surfaceCount != 0; } }
void RealisticCamera::prepareOptics() { if (opticsUpdatePending) updateLensSettings(); }
void RealisticCamera::updateLensSettings() {
    opticsUpdatePending = false; if (!sourceOptics.surfaceCount) return;
    nr::optics::LensSnapshot replacement = sourceOptics;
    if (apertureDiameterMm > 0.f) for (uint32_t i = 0; i < replacement.surfaceCount; ++i) if (replacement.surfaces[i].isStop) replacement.surfaces[i].apertureRadius = apertureDiameterMm * .5f;
    const float scale = std::clamp(focusDistanceCm / 500.f, .5f, 2.f);
    for (uint32_t i = 0; i < replacement.surfaceCount; ++i) replacement.surfaces[i].z *= scale;
    replacement.rearPupilZ *= scale; optics = replacement; opticsDirty = true;
    optics.sensorWidthMm = getSensor().filmWidth();
    optics.sensorHeightMm = getSensor().filmHeight();
}
bool RealisticCamera::loadLensAndSensor(bool reset) {
    if (lensPath.empty()) { optics = {}; sourceOptics = {}; loadStatus = "No lens file loaded"; opticsDirty = true; return true; }
    try {
        if (!getSensor().getImageSensorPath().empty() && !getSensor().loadImageSensorDimensions()) throw std::runtime_error("Failed to load image sensor");
        const auto loaded = nr::optics::loadZmx(lensPath, nr::optics::splitCatalogPaths(glassCatalogPaths)); sourceOptics = loaded.snapshot;
        if (reset) focusDistanceCm = 500.f; if (reset || apertureDiameterMm <= 0.f) apertureDiameterMm = sourceOptics.rearPupilRadius * 2.f;
        sensorWidthCm = getSensor().width() * .1f; sensorHeightCm = getSensor().height() * .1f; filmDiagonalCm = std::sqrt(sensorWidthCm * sensorWidthCm + sensorHeightCm * sensorHeightCm); focalLengthMm = sourceOptics.focalLengthMm;
        sourceOptics.sensorWidthMm = getSensor().filmWidth();
        sourceOptics.sensorHeightMm = getSensor().filmHeight();
        opticsUpdatePending = true; updateLensSettings(); loadStatus = loaded.message + ", focal length: " + std::to_string(focalLengthMm) + " mm"; LOG_INFO("RealisticCamera: " << loadStatus); return true;
    } catch (const std::exception& e) { loadStatus = e.what(); LOG_ERROR("RealisticCamera: " << loadStatus); return false; }
}
