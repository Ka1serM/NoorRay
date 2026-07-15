#include "RealisticCamera.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <stdexcept>
#include "CUDA/ManagedMemory.h"
#include "Raytracing/Sellmeier.h"
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

    if (other.rossLens) {
        nr::rstd::allocator<ross::CameraLens> allocator;
        rossLens.reset(allocator.allocate(1));
        allocator.construct(rossLens.get(), *other.rossLens);
    }
    if (other.sourceRossLens) {
        nr::rstd::allocator<ross::CameraLens> allocator;
        sourceRossLens.reset(allocator.allocate(1));
        allocator.construct(sourceRossLens.get(), *other.sourceRossLens);
    }
    if (other.exitPupil) {
        nr::rstd::allocator<ross::ExitPupil> allocator;
        exitPupil.reset(allocator.allocate(1));
        allocator.construct(exitPupil.get(), *other.exitPupil);
    }
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

        nr::rstd::allocator<ross::CameraLens> lensAllocator;
        nr::rstd::unique_ptr<ross::CameraLens> newLens(lensAllocator.allocate(1));
        lensAllocator.construct(newLens.get(), updatedLens);
        nr::rstd::allocator<ross::ExitPupil> pupilAllocator;
        nr::rstd::unique_ptr<ross::ExitPupil> newPupil(pupilAllocator.allocate(1));
        pupilAllocator.construct(newPupil.get(), updatedPupil);

        rossLens = std::move(newLens);
        exitPupil = std::move(newPupil);
        opticsDirty = true;
    } catch (const std::exception& error) {
        loadStatus = error.what();
        LOG_ERROR("RealisticCamera: " << loadStatus);
    }
}

void RealisticCamera::loadLensAndSensor(const bool resetLensSettings)
{
    opticsUpdatePending = false;
    Sensor& sensor = getSensor();
    freeRossLens();
    opticsDirty = true;

    if (lensPath.empty()) {
        sensorWidthCm = 0.0f;
        sensorHeightCm = 0.0f;
        filmDiagonalCm = 0.0f;
        loadStatus = "No lens file loaded";
        LOG_INFO("RealisticCamera: no lens loaded");
        return;
    }

    try {
        olio::GlassCatalogLibrary catalogs;
        std::string catalogList = glassCatalogPaths;
        std::ranges::replace(catalogList, ';', ',');
        if (!catalogList.empty())
            catalogs.loadCatalogsFromCommaSeperatedString(catalogList);

        ross::CameraLens loaded =
            ross::CameraLensSystemReader::readCameraLens(
                lensPath, catalogs, ross::ReadOptions{1.0f, false});
        nr::rstd::allocator<ross::CameraLens> lensAlloc;
        sourceRossLens.reset(lensAlloc.allocate(1));
        lensAlloc.construct(sourceRossLens.get(), loaded);
        if (resetLensSettings)
            focusDistanceCm = 500.0f;
        focusDistanceCm = std::max(
            focusDistanceCm, loaded.metadata.closestFocalDistance);
        if (resetLensSettings || apertureDiameterMm <= 0.0f) {
            apertureDiameterMm = std::max(0.0f, loaded.getApertureRadius() * 20.0f);
        } else {
            loaded.changeAperture_mm(apertureDiameterMm);
        }
        loaded.focusLens(focusDistanceCm);

        rossLens.reset(lensAlloc.allocate(1));
        lensAlloc.construct(rossLens.get(), loaded);

        const bool usesSensorFile = !sensor.getImageSensorPath().empty();
        if (usesSensorFile && !sensor.loadImageSensorDimensions())
            throw std::runtime_error("Failed to load image sensor");

        sensorWidthCm = sensor.width() * 0.1f;
        sensorHeightCm = sensor.height() * 0.1f;
        filmDiagonalCm = std::sqrt(
            sensorWidthCm * sensorWidthCm + sensorHeightCm * sensorHeightCm);

        const float filmDiagonalCmVal = filmDiagonalCm;
        ross::ExitPupilCalculator::CalculationSettings calcSettings;
        ross::TaskReporter taskReporter;
        ross::ExitPupilCalculator exitPupilCalc(
            *rossLens, filmDiagonalCmVal, calcSettings, taskReporter);
        ross::ExitPupil computedPupil = exitPupilCalc.calculate();

        nr::rstd::allocator<ross::ExitPupil> pupilAlloc;
        exitPupil.reset(pupilAlloc.allocate(1));
        pupilAlloc.construct(exitPupil.get(), computedPupil);

        focalLengthMm = rossLens->metadata.focalLength * 10.0f;
        fieldOfViewDegrees = fovDegreesForFocalLengthMm(focalLengthMm);

        loadStatus = std::to_string(rossLens->surfaces.size()) + " surfaces"
            ", pupil bounds: " + std::to_string(exitPupil->pupilBounds.size())
            + ", focal length: " + std::to_string(focalLengthMm) + " mm"
            + (usesSensorFile ? ", sensor file" : ", manual sensor");
        LOG_INFO("RealisticCamera: " << loadStatus);
    } catch (const std::exception& e) {
        freeRossLens();
        loadStatus = e.what();
        LOG_ERROR("RealisticCamera: " << loadStatus);
    }
}
