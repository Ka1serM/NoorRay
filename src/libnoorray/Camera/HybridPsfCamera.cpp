#include "HybridPsfCamera.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "CUDA/ManagedMemory.h"
#include "CUDA/rstd/Allocator.h"
#include "Log.h"
#include "libross/foundation/gpu/types/Allocator.h"
#include "libross/foundation/physics/Wavelengths.h"
#include "libross/imaging/cameralens/lenssystemio/CameraLensSystemReader.h"
#include "libross/imaging/cameralens/raylut/FindRayThroughApertureCenterRayGenerator.h"
#include "libross/imaging/cameralens/raylut/io/read/RayLUTFileReader.h"
#include "libross/imaging/cameralens/raylut/io/write/RayLUTFileWriter.h"
#include "libross/imaging/cameralens/raytracing/exitpupil/ExitPupilCalculator.h"
#include "libross/imaging/cameralens/raytracing/findapertureray/FindRayThroughApertureCenter.h"
#include "libross/imaging/imagesensor/ImageSensorReader.h"
#include "libross/imaging/imagesensor/ImageSensorSampler.h"
#include "openlensfileio/glasscatalogs/glasscatalog/GlassCatalogLibrary.h"
#include "portable-file-dialogs.h"

namespace {
class ScopedStopwatch {
public:
    ScopedStopwatch(std::string label, std::string* status = nullptr)
        : label(std::move(label)), status(status), start(std::chrono::steady_clock::now())
    {
        if (this->status != nullptr)
            *this->status = this->label + " started";
        LOG_INFO(this->label << " started");
    }

    ~ScopedStopwatch()
    {
        const auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        LOG_INFO(label << " finished in " << elapsed << " ms");
    }

private:
    std::string label;
    std::string* status;
    std::chrono::steady_clock::time_point start;
};

std::string joinRossPsfPathsWithSemicolons(const std::vector<std::string>& paths)
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

HybridPsfCamera::HybridPsfCamera()
    : HybridPsfCamera(std::make_unique<GatherPsfSensor>())
{
}

HybridPsfCamera::HybridPsfCamera(std::unique_ptr<Sensor> ownedSensor)
    : TaggedBase(std::move(ownedSensor))
{
}

HybridPsfCamera::~HybridPsfCamera()
{
    freeRossObjects();
}

void HybridPsfCamera::freeRossObjects()
{
    if (!rayLut && !exitPupil && !rossSensor && !rossLens && !sourceRossLens)
        return;

    nr::synchronizeBeforeManagedMutation("HybridPsfCamera optics free");

    rayLut.reset();
    exitPupil.reset();
    rossSensor.reset();
    rossLens.reset();
    sourceRossLens.reset();
}

void HybridPsfCamera::updateLensSettings()
{
    opticsUpdatePending = false;
    if (!sourceRossLens || !rossSensor)
        return;

    nr::synchronizeBeforeManagedMutation("HybridPsfCamera optics update");
    try {
        ross::CameraLens updatedLens(*sourceRossLens);
        if (apertureDiameterMm > 0.0f)
            updatedLens.changeAperture_mm(apertureDiameterMm);
        updatedLens.focusLens(focusDistanceCm);

        ross::ExitPupilCalculator::CalculationSettings pupilSettings;
        ross::TaskReporter pupilReporter;
        ross::ExitPupilCalculator pupilCalculator(
            updatedLens, rossSensor->getDiagonal().centimeter(), pupilSettings, pupilReporter);
        ross::ExitPupil updatedPupil = pupilCalculator.calculate();

        const ross::Resolution resolution(
            getSensor().resolutionX(), getSensor().resolutionY());
        ross::RayLUT updatedLut(resolution, std::max(1, rayLutStepSize));
        ross::ImageSensorSampler imageSensorSampler(*rossSensor);
        ross::FindRayThroughApertureCenter findApertureRay(updatedLens);
        ross::FindRayThroughApertureCenterRayGenerator generator(
            findApertureRay, imageSensorSampler, resolution,
            std::max(1, samplesPerDimension));
        updatedLut.populate(generator, {
            ross::FraunhoferLines::C,
            ross::FraunhoferLines::d,
            ross::FraunhoferLines::F});

        nr::rstd::allocator<ross::CameraLens> lensAllocator;
        nr::rstd::unique_ptr<ross::CameraLens> newLens(lensAllocator.allocate(1));
        lensAllocator.construct(newLens.get(), updatedLens);
        nr::rstd::allocator<ross::ExitPupil> pupilAllocator;
        nr::rstd::unique_ptr<ross::ExitPupil> newPupil(pupilAllocator.allocate(1));
        pupilAllocator.construct(newPupil.get(), updatedPupil);
        nr::rstd::allocator<ross::RayLUT> lutAllocator;
        nr::rstd::unique_ptr<ross::RayLUT> newLut(lutAllocator.allocate(1));
        lutAllocator.construct(newLut.get(), std::move(updatedLut));

        rossLens = std::move(newLens);
        exitPupil = std::move(newPupil);
        rayLut = std::move(newLut);
        opticsDirty = true;
    } catch (const std::exception& error) {
        loadStatus = error.what();
        LOG_ERROR("HybridPsfCamera: " << loadStatus);
    }
}

void HybridPsfCamera::load(std::string lensPath_, std::string glassCatalogPaths_, std::string rayLutPath_)
{
    lensPath = std::move(lensPath_);
    glassCatalogPaths = std::move(glassCatalogPaths_);
    rayLutPath = std::move(rayLutPath_);
    loadLensSensorAndPsf();
}

void HybridPsfCamera::load(std::string lensPath_,
    const std::vector<std::string>& glassCatalogPaths_, std::string rayLutPath_)
{
    load(std::move(lensPath_), joinRossPsfPathsWithSemicolons(glassCatalogPaths_),
        std::move(rayLutPath_));
}

void HybridPsfCamera::setApertureDiameterMm(const float requestedApertureDiameterMm)
{
    const float clampedApertureDiameterMm = std::max(0.0f, requestedApertureDiameterMm);
    if (apertureDiameterMm == clampedApertureDiameterMm)
        return;
    nr::synchronizeBeforeManagedMutation("HybridPsfCamera aperture");
    apertureDiameterMm = clampedApertureDiameterMm;
    opticsUpdatePending = static_cast<bool>(sourceRossLens);
}

void HybridPsfCamera::setOpticalFocusDistanceCm(const float requestedFocusDistanceCm)
{
    const float minimumFocusDistanceCm = sourceRossLens
        ? std::max(0.1f, sourceRossLens->metadata.closestFocalDistance)
        : 0.1f;
    const float clampedFocusDistanceCm =
        std::max(minimumFocusDistanceCm, requestedFocusDistanceCm);
    if (focusDistanceCm == clampedFocusDistanceCm)
        return;
    nr::synchronizeBeforeManagedMutation("HybridPsfCamera focus distance");
    focusDistanceCm = clampedFocusDistanceCm;
    opticsUpdatePending = static_cast<bool>(sourceRossLens);
}

void HybridPsfCamera::prepareOptics()
{
    if (opticsUpdatePending)
        updateLensSettings();
}

HybridPsfCamera::HybridPsfCamera(const HybridPsfCamera& other)
    : TaggedBase(other)
{
    lensPath = other.lensPath;
    glassCatalogPaths = other.glassCatalogPaths;
    rayLutPath = other.rayLutPath;
    loadStatus = other.loadStatus;
    opticsDirty = other.opticsDirty;
    opticsUpdatePending = other.opticsUpdatePending;
    apertureDiameterMm = other.apertureDiameterMm;
    rayLutStepSize = other.rayLutStepSize;
    samplesPerDimension = other.samplesPerDimension;

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
    if (other.rossSensor) {
        nr::rstd::allocator<ross::ImageSensor> allocator;
        rossSensor.reset(allocator.allocate(1));
        allocator.construct(rossSensor.get(), *other.rossSensor);
    }
    if (other.rayLut) {
        nr::rstd::allocator<ross::RayLUT> allocator;
        rayLut.reset(allocator.allocate(1));
        allocator.construct(rayLut.get(), *other.rayLut);
    }
}

void HybridPsfCamera::setOpticsPaths(std::string lensPath_, std::string glassCatalogPaths_)
{
    lensPath = std::move(lensPath_);
    glassCatalogPaths = std::move(glassCatalogPaths_);
}

void HybridPsfCamera::loadLensSensorAndPsf(
    const bool buildRayLut, const bool resetLensSettings)
{
    opticsUpdatePending = false;
    Sensor& sensor = getSensor();
    freeRossObjects();
    opticsDirty = true;

    if (lensPath.empty() || sensor.getImageSensorPath().empty()) {
        loadStatus = "Lens and sensor are required";
        LOG_INFO("HybridPsfCamera: " << loadStatus);
        return;
    }

    try {
        ScopedStopwatch loadTimer("HybridPsfCamera load", &loadStatus);
        olio::GlassCatalogLibrary catalogs;
        std::string catalogList = glassCatalogPaths;
        std::ranges::replace(catalogList, ';', ',');
        if (!catalogList.empty())
            catalogs.loadCatalogsFromCommaSeperatedString(catalogList);

        ross::CameraLens loadedLens =
            ross::CameraLensSystemReader::readCameraLens(
                lensPath, catalogs, ross::ReadOptions{1.0f, false});
        nr::rstd::allocator<ross::CameraLens> lensAllocator;
        sourceRossLens.reset(lensAllocator.allocate(1));
        lensAllocator.construct(sourceRossLens.get(), loadedLens);
        if (resetLensSettings)
            focusDistanceCm = 500.0f;
        focusDistanceCm = std::max(
            focusDistanceCm, loadedLens.metadata.closestFocalDistance);
        if (resetLensSettings || apertureDiameterMm <= 0.0f) {
            apertureDiameterMm = std::max(0.0f, loadedLens.getApertureRadius() * 20.0f);
        } else {
            loadedLens.changeAperture_mm(apertureDiameterMm);
        }
        loadedLens.focusLens(focusDistanceCm);

        rossLens.reset(lensAllocator.allocate(1));
        lensAllocator.construct(rossLens.get(), loadedLens);

        const ross::ImageSensor loadedSensor =
            ross::ImageSensorReader::readFile(std::string(sensor.getImageSensorPath()));
        sensor.setDimensionsMm(
            loadedSensor.dimensions.width.millimeter(),
            loadedSensor.dimensions.height.millimeter());
        sensor.setResolution(loadedSensor.resolution.width, loadedSensor.resolution.height);
        sensor.loadImageSensorDimensions();
        const uint32_t psfBinCount = sensor.reloadPsfGrid();

        nr::rstd::allocator<ross::ImageSensor> sensorAllocator;
        rossSensor.reset(sensorAllocator.allocate(1));
        sensorAllocator.construct(rossSensor.get(), loadedSensor);

        ross::ExitPupilCalculator::CalculationSettings pupilSettings;
        ross::TaskReporter pupilReporter;
        ross::ExitPupilCalculator pupilCalculator(
            *rossLens, loadedSensor.getDiagonal().centimeter(), pupilSettings, pupilReporter);
        ross::ExitPupil computedPupil = pupilCalculator.calculate();
        nr::rstd::allocator<ross::ExitPupil> pupilAllocator;
        exitPupil.reset(pupilAllocator.allocate(1));
        pupilAllocator.construct(exitPupil.get(), computedPupil);

        const bool cacheHit = !rayLutPath.empty() && std::filesystem::exists(rayLutPath);
        if (buildRayLut) {
            ScopedStopwatch rayLutTimer(cacheHit ? "HybridPsfCamera Ray LUT read" : "HybridPsfCamera Ray LUT build",
                &loadStatus);
            const ross::Resolution resolution(sensor.resolutionX(), sensor.resolutionY());
            ross::RayLUT loadedLut;
            if (cacheHit) {
                loadedLut = ross::RayLUTFileReader().read(rayLutPath);
            } else {
                loadedLut = ross::RayLUT(resolution, std::max(1, rayLutStepSize));
                ross::ImageSensorSampler imageSensorSampler(loadedSensor);
                ross::FindRayThroughApertureCenter findApertureRay(loadedLens);
                ross::FindRayThroughApertureCenterRayGenerator generator(
                    findApertureRay, imageSensorSampler, resolution, std::max(1, samplesPerDimension));
                loadedLut.populate(generator, {ross::FraunhoferLines::C, ross::FraunhoferLines::d, ross::FraunhoferLines::F});
                if (!rayLutPath.empty())
                    ross::RayLUTFileWriter().write(rayLutPath, loadedLut);
            }

            nr::rstd::allocator<ross::RayLUT> lutAllocator;
            rayLut.reset(lutAllocator.allocate(1));
            lutAllocator.construct(rayLut.get(), std::move(loadedLut));
        }

        focalLengthMm = rossLens->metadata.focalLength * 10.0f;
        fieldOfViewDegrees = fovDegreesForFocalLengthMm(focalLengthMm);
        loadStatus = "loaded, psf bins: " + std::to_string(psfBinCount)
            + ", raylut " + (buildRayLut
                ? (cacheHit ? "cache hit" : (rayLutPath.empty() ? "built in memory" : "built and saved"))
                : "disabled, tracing directly");
        LOG_INFO("HybridPsfCamera: " << loadStatus);
    } catch (const std::exception& e) {
        freeRossObjects();
        loadStatus = e.what();
        LOG_ERROR("HybridPsfCamera: " << loadStatus);
    }
}
