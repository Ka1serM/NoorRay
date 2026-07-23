#include "RealisticCamera.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <stdexcept>
#include "CUDA/ManagedMemory.h"
#include "Shading/Sellmeier.h"
#include "Log.h"
#include "libross/foundation/gpu/types/Allocator.h"
#include "libross/imaging/cameralens/lenssystemio/CameraLensSystemReader.h"
#include "libross/imaging/cameralens/raytracing/exitpupil/ExitPupilCalculator.h"
#include "openlensfileio/glasscatalogs/glasscatalog/GlassCatalogLibrary.h"
#include "portable-file-dialogs.h"

namespace {
std::string joinWithSemicolons(const std::vector<std::string>& paths)
{
    std::string result;
    for (const std::string& path : paths) {
        if (!result.empty())
            result += ';';
        result += path;
    }
    return result;
}

}

RealisticCamera::RealisticCamera()
    : RealisticCamera(std::make_unique<RectangularSensor>())
{
}

RealisticCamera::RealisticCamera(std::unique_ptr<Sensor> ownedSensor)
    : TaggedBase(std::move(ownedSensor))
{
}

void RealisticCamera::load(std::string lensPath_, std::string glassCatalogPaths_)
{
    lensPath           = std::move(lensPath_);
    glassCatalogPaths  = std::move(glassCatalogPaths_);
    loadLensAndSensor();
}

void RealisticCamera::load(
    std::string lensPath_, const std::vector<std::string>& glassCatalogPaths_)
{
    load(std::move(lensPath_), joinWithSemicolons(glassCatalogPaths_));
}

void RealisticCamera::setApertureDiameterMm(const float requestedApertureDiameterMm)
{
    const float clampedApertureDiameterMm = std::max(0.0f, requestedApertureDiameterMm);
    if (apertureDiameterMm == clampedApertureDiameterMm)
        return;
    nr::synchronizeBeforeManagedMutation("RealisticCamera aperture");
    apertureDiameterMm = clampedApertureDiameterMm;
    opticsUpdatePending = static_cast<bool>(sourceRossLens);
}

void RealisticCamera::setOpticalFocusDistanceCm(const float requestedFocusDistanceCm)
{
    const float minimumFocusDistanceCm = sourceRossLens
        ? std::max(0.1f, sourceRossLens->metadata.closestFocalDistance)
        : 0.1f;
    const float clampedFocusDistanceCm =
        std::max(minimumFocusDistanceCm, requestedFocusDistanceCm);
    if (focusDistanceCm == clampedFocusDistanceCm)
        return;
    nr::synchronizeBeforeManagedMutation("RealisticCamera focus distance");
    focusDistanceCm = clampedFocusDistanceCm;
    opticsUpdatePending = static_cast<bool>(sourceRossLens);
}

void RealisticCamera::prepareOptics()
{
    if (opticsUpdatePending)
        updateLensSettings();
}

RealisticCamera::RealisticCamera(const RealisticCamera& other)
    : TaggedBase(other)
{
    lensPath = other.lensPath;
    glassCatalogPaths = other.glassCatalogPaths;
    loadStatus = other.loadStatus;
    opticsDirty = other.opticsDirty;
    opticsUpdatePending = other.opticsUpdatePending;
    apertureDiameterMm = other.apertureDiameterMm;
    sensorWidthCm = other.sensorWidthCm;
    sensorHeightCm = other.sensorHeightCm;
    filmDiagonalCm = other.filmDiagonalCm;

    if (other.rossLens)
        rossLens = nr::rstd::make_unique<ross::CameraLens>(*other.rossLens);
    if (other.sourceRossLens)
        sourceRossLens = nr::rstd::make_unique<ross::CameraLens>(*other.sourceRossLens);
    if (other.exitPupil)
        exitPupil = nr::rstd::make_unique<ross::ExitPupil>(*other.exitPupil);
}

void RealisticCamera::setOpticsPaths(std::string lensPath_, std::string glassCatalogPaths_)
{
    lensPath = std::move(lensPath_);
    glassCatalogPaths = std::move(glassCatalogPaths_);
}

RealisticCamera::~RealisticCamera()
{
    freeRossLens();
}

void RealisticCamera::freeRossLens()
{
    if (!rossLens && !sourceRossLens && !exitPupil)
        return;

    nr::synchronizeBeforeManagedMutation("RealisticCamera optics free");

    exitPupil.reset();
    rossLens.reset();
    sourceRossLens.reset();
}

void RealisticCamera::updateLensSettings()
{
    opticsUpdatePending = false;
    if (!sourceRossLens)
        return;

    nr::synchronizeBeforeManagedMutation("RealisticCamera optics update");
    try {
        ross::CameraLens updatedLens(*sourceRossLens);
        if (apertureDiameterMm > 0.0f)
            updatedLens.changeAperture_mm(apertureDiameterMm);
        updatedLens.focusLens(focusDistanceCm);

        ross::ExitPupilCalculator::CalculationSettings settings;
        ross::TaskReporter reporter;
        ross::ExitPupilCalculator calculator(
            updatedLens, filmDiagonalCm, settings, reporter);
        ross::ExitPupil updatedPupil = calculator.calculate();

        auto newLens = nr::rstd::make_unique<ross::CameraLens>(updatedLens);
        auto newPupil = nr::rstd::make_unique<ross::ExitPupil>(updatedPupil);

        rossLens = std::move(newLens);
        exitPupil = std::move(newPupil);
        opticsDirty = true;
    } catch (const std::exception& error) {
        loadStatus = error.what();
        LOG_ERROR("RealisticCamera: " << loadStatus);
    }
}

bool RealisticCamera::loadLensAndSensor(const bool resetLensSettings)
{
    opticsUpdatePending = false;
    Sensor& sensor = getSensor();

    if (lensPath.empty()) {
        freeRossLens();
        sensorWidthCm = 0.0f;
        sensorHeightCm = 0.0f;
        filmDiagonalCm = 0.0f;
        opticsDirty = true;
        loadStatus = "No lens file loaded";
        LOG_INFO("RealisticCamera: no lens loaded");
        return true;
    }

    try {
        olio::GlassCatalogLibrary catalogs;
        std::string catalogList = glassCatalogPaths;
        std::ranges::replace(catalogList, ';', ',');
        if (!catalogList.empty())
            catalogs.loadCatalogsFromCommaSeperatedString(catalogList);

        ross::CameraLens loaded =
            ross::CameraLensSystemReader::readCameraLens(
                lensPath, catalogs, ross::ReadOptions{1.0f, true});
        const ross::CameraLens loadedSource(loaded);
        float loadedFocusDistanceCm = focusDistanceCm;
        float loadedApertureDiameterMm = apertureDiameterMm;
        if (resetLensSettings)
            loadedFocusDistanceCm = 500.0f;
        loadedFocusDistanceCm = std::max(
            loadedFocusDistanceCm, loaded.metadata.closestFocalDistance);
        if (resetLensSettings || loadedApertureDiameterMm <= 0.0f) {
            loadedApertureDiameterMm =
                std::max(0.0f, loaded.getApertureRadius() * 20.0f);
        } else {
            loaded.changeAperture_mm(loadedApertureDiameterMm);
        }
        loaded.focusLens(loadedFocusDistanceCm);

        const bool usesSensorFile = !sensor.getImageSensorPath().empty();
        if (usesSensorFile && !sensor.loadImageSensorDimensions())
            throw std::runtime_error("Failed to load image sensor");

        const float loadedSensorWidthCm = sensor.width() * 0.1f;
        const float loadedSensorHeightCm = sensor.height() * 0.1f;
        const float loadedFilmDiagonalCm = std::sqrt(
            loadedSensorWidthCm * loadedSensorWidthCm
            + loadedSensorHeightCm * loadedSensorHeightCm);

        ross::ExitPupilCalculator::CalculationSettings calcSettings;
        ross::TaskReporter taskReporter;
        ross::ExitPupilCalculator exitPupilCalc(
            loaded, loadedFilmDiagonalCm, calcSettings, taskReporter);
        ross::ExitPupil computedPupil = exitPupilCalc.calculate();

        auto loadedSourceLens = nr::rstd::make_unique<ross::CameraLens>(loadedSource);
        auto loadedRossLens = nr::rstd::make_unique<ross::CameraLens>(loaded);
        auto loadedExitPupil = nr::rstd::make_unique<ross::ExitPupil>(computedPupil);

        nr::synchronizeBeforeManagedMutation("RealisticCamera optics replace");
        sourceRossLens = std::move(loadedSourceLens);
        rossLens = std::move(loadedRossLens);
        exitPupil = std::move(loadedExitPupil);
        focusDistanceCm = loadedFocusDistanceCm;
        apertureDiameterMm = loadedApertureDiameterMm;
        sensorWidthCm = loadedSensorWidthCm;
        sensorHeightCm = loadedSensorHeightCm;
        filmDiagonalCm = loadedFilmDiagonalCm;
        focalLengthMm = loaded.metadata.focalLength * 10.0f;
        fieldOfViewDegrees = fovDegreesForFocalLengthMm(focalLengthMm);
        opticsDirty = true;

        loadStatus = std::to_string(rossLens->surfaces.size()) + " surfaces"
            ", pupil bounds: " + std::to_string(exitPupil->pupilBounds.size())
            + ", focal length: " + std::to_string(focalLengthMm) + " mm"
            + (usesSensorFile ? ", sensor file" : ", manual sensor");
        LOG_INFO("RealisticCamera: " << loadStatus);
        return true;
    } catch (const std::exception& e) {
        loadStatus = e.what();
        LOG_ERROR("RealisticCamera: " << loadStatus);
        return false;
    }
}
