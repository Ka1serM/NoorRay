#include "RossPsfCamera.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <imgui.h>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "CUDA/ManagedMemory.h"
#include "CUDA/rstd/Allocator.h"
#include "Log.h"
#include "UI/ImGuiManager.h"
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

RossPsfCamera::RossPsfCamera()
    : RossPsfCamera(std::make_unique<GatherPsfSensor>())
{
}

RossPsfCamera::RossPsfCamera(std::unique_ptr<Sensor> ownedSensor)
    : Camera(std::move(ownedSensor))
{
}

RossPsfCamera::~RossPsfCamera()
{
    freeRossObjects();
}

void RossPsfCamera::freeRossObjects()
{
    if (!rayLut && !exitPupil && !rossSensor && !rossLens && !sourceRossLens)
        return;

    nr::synchronizeBeforeManagedMutation("RossPsfCamera optics free");

    rayLut.reset();
    exitPupil.reset();
    rossSensor.reset();
    rossLens.reset();
    sourceRossLens.reset();
}

void RossPsfCamera::updateLensSettings()
{
    opticsUpdatePending = false;
    if (!sourceRossLens || !rossSensor)
        return;

    nr::synchronizeBeforeManagedMutation("RossPsfCamera optics update");
    try {
        ross::CameraLens updatedLens(*sourceRossLens);
        if (apertureDiameterMm > 0.0f)
            updatedLens.changeAperture_mm(apertureDiameterMm);
        updatedLens.focusLens(focusDistance * 100.0f);

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
        LOG_ERROR("RossPsfCamera: " << loadStatus);
    }
}

void RossPsfCamera::load(std::string lensPath_, std::string glassCatalogPaths_, std::string rayLutPath_)
{
    lensPath = std::move(lensPath_);
    glassCatalogPaths = std::move(glassCatalogPaths_);
    rayLutPath = std::move(rayLutPath_);
    loadLensSensorAndPsf();
}

void RossPsfCamera::load(std::string lensPath_,
    const std::vector<std::string>& glassCatalogPaths_, std::string rayLutPath_)
{
    load(std::move(lensPath_), joinRossPsfPathsWithSemicolons(glassCatalogPaths_),
        std::move(rayLutPath_));
}

void RossPsfCamera::setApertureDiameter(const float millimeters)
{
    const float requestedDiameter = std::max(0.0f, millimeters);
    if (apertureDiameterMm == requestedDiameter)
        return;
    apertureDiameterMm = requestedDiameter;
    opticsUpdatePending = static_cast<bool>(sourceRossLens);
}

void RossPsfCamera::setOpticalFocusDistance(const float meters)
{
    const float requestedDistance = std::max(0.001f, meters);
    if (focusDistance == requestedDistance)
        return;
    focusDistance = requestedDistance;
    opticsUpdatePending = static_cast<bool>(sourceRossLens);
}

void RossPsfCamera::prepareOptics()
{
    if (opticsUpdatePending)
        updateLensSettings();
}

RossPsfCamera::RossPsfCamera(const RossPsfCamera& other)
    : Camera(other)
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

void RossPsfCamera::setOpticsPaths(std::string lensPath_, std::string glassCatalogPaths_)
{
    lensPath = std::move(lensPath_);
    glassCatalogPaths = std::move(glassCatalogPaths_);
}

void RossPsfCamera::loadLensSensorAndPsf(
    const bool buildRayLut, const bool resetLensSettings)
{
    opticsUpdatePending = false;
    Sensor& sensor = getSensor();
    freeRossObjects();
    opticsDirty = true;

    if (lensPath.empty() || sensor.getImageSensorPath().empty()) {
        loadStatus = "Lens and sensor are required";
        LOG_INFO("RossPsfCamera: " << loadStatus);
        return;
    }

    try {
        ScopedStopwatch loadTimer("RossPsfCamera load", &loadStatus);
        olio::GlassCatalogLibrary catalogs;
        std::string catalogList = glassCatalogPaths;
        std::ranges::replace(catalogList, ';', ',');
        if (!catalogList.empty())
            catalogs.loadCatalogsFromCommaSeperatedString(catalogList);

        ross::CameraLens loadedLens =
            ross::CameraLensSystemReader::readCameraLens(lensPath, catalogs);
        nr::rstd::allocator<ross::CameraLens> lensAllocator;
        sourceRossLens.reset(lensAllocator.allocate(1));
        lensAllocator.construct(sourceRossLens.get(), loadedLens);
        if (resetLensSettings) {
            apertureDiameterMm = std::max(0.0f, loadedLens.getApertureRadius() * 20.0f);
            focusDistance = 5.0f;
        } else if (apertureDiameterMm > 0.0f) {
            loadedLens.changeAperture_mm(apertureDiameterMm);
        } else {
            apertureDiameterMm = std::max(0.0f, loadedLens.getApertureRadius() * 20.0f);
        }
        loadedLens.focusLens(focusDistance * 100.0f);

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
            ScopedStopwatch rayLutTimer(cacheHit ? "RossPsfCamera Ray LUT read" : "RossPsfCamera Ray LUT build",
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
        fieldOfView = fovForFocalLength(focalLengthMm);
        loadStatus = "loaded, psf bins: " + std::to_string(psfBinCount)
            + ", raylut " + (buildRayLut
                ? (cacheHit ? "cache hit" : (rayLutPath.empty() ? "built in memory" : "built and saved"))
                : "disabled, tracing directly");
        LOG_INFO("RossPsfCamera: " << loadStatus);
    } catch (const std::exception& e) {
        freeRossObjects();
        loadStatus = e.what();
        LOG_ERROR("RossPsfCamera: " << loadStatus);
    }
}

bool RossPsfCamera::renderUi()
{
    Sensor& sensor = getSensor();
    if (lensDialog && lensDialog->ready(0)) {
        const auto selection = lensDialog->result();
        if (!selection.empty()) {
            lensPath = selection.front();
            loadLensSensorAndPsf(true, true);
        }
        lensDialog.reset();
    }
    if (glassCatalogDialog && glassCatalogDialog->ready(0)) {
        const auto selection = glassCatalogDialog->result();
        if (!selection.empty()) {
            glassCatalogPaths = joinRossPsfPathsWithSemicolons(selection);
            loadLensSensorAndPsf();
        }
        glassCatalogDialog.reset();
    }
    if (rayLutOpenDialog && rayLutOpenDialog->ready(0)) {
        const auto selection = rayLutOpenDialog->result();
        if (!selection.empty())
            rayLutPath = selection.front();
        rayLutOpenDialog.reset();
    }
    if (rayLutSaveDialog && rayLutSaveDialog->ready(0)) {
        const auto selection = rayLutSaveDialog->result();
        if (!selection.empty())
            rayLutPath = selection;
        rayLutSaveDialog.reset();
    }

    std::array<char, 512> lensBuffer{};
    std::array<char, 1024> catalogBuffer{};
    std::array<char, 512> rayLutBuffer{};
    std::snprintf(lensBuffer.data(), lensBuffer.size(), "%s", lensPath.c_str());
    std::snprintf(catalogBuffer.data(), catalogBuffer.size(), "%s", glassCatalogPaths.c_str());
    std::snprintf(rayLutBuffer.data(), rayLutBuffer.size(), "%s", rayLutPath.c_str());

    bool changed = false;
    const float browseButtonWidth = ImGui::CalcTextSize("...").x + ImGui::GetStyle().FramePadding.x * 2.0f;

    auto pathRow = [&](const char* label, const char* id, auto& buffer, std::string& target,
                       auto openDialog) {
        ImGuiManager::tableRowLabel(label);
        ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - browseButtonWidth - ImGui::GetStyle().ItemSpacing.x);
        if (ImGui::InputText(id, buffer.data(), buffer.size())) {
            target = buffer.data();
            changed = true;
        }
        ImGui::PopItemWidth();
        ImGui::SameLine();
        if (ImGui::Button((std::string("...") + id).c_str(), ImVec2(browseButtonWidth, 0)))
            openDialog();
    };

    pathRow("Lens File", "##RossPsfLens", lensBuffer, lensPath, [&] {
        lensDialog = std::make_unique<pfd::open_file>(
            "Select Lens File", ".",
            std::vector<std::string>{"Lens Files", "*.olio *.zmx *.dat", "All Files", "*"});
    });
    pathRow("Glass Catalogs", "##RossPsfCatalogs", catalogBuffer, glassCatalogPaths, [&] {
        glassCatalogDialog = std::make_unique<pfd::open_file>(
            "Select Glass Catalogs", ".",
            std::vector<std::string>{"Glass Catalogs", "*.agf *.AGF", "All Files", "*"},
            pfd::opt::multiselect);
    });
    ImGuiManager::tableRowLabel("Ray LUT");
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x
        - ImGui::CalcTextSize("Load").x - ImGui::CalcTextSize("Save As").x
        - ImGui::GetStyle().FramePadding.x * 4.0f
        - ImGui::GetStyle().ItemSpacing.x * 2.0f);
    if (ImGui::InputText("##RossPsfRayLut", rayLutBuffer.data(), rayLutBuffer.size())) {
        rayLutPath = rayLutBuffer.data();
        changed = true;
    }
    ImGui::PopItemWidth();
    ImGui::SameLine();
    if (ImGui::Button("Load##RossPsfRayLut")) {
        rayLutOpenDialog = std::make_unique<pfd::open_file>(
            "Load Ray LUT Cache", ".",
            std::vector<std::string>{"Ray LUT", "*.raylut", "All Files", "*"});
    }
    ImGui::SameLine();
    if (ImGui::Button("Save As##RossPsfRayLut")) {
        rayLutSaveDialog = std::make_unique<pfd::save_file>(
            "Save Ray LUT Cache As", "raylut.raylut",
            std::vector<std::string>{"Ray LUT", "*.raylut", "All Files", "*"});
    }

    ImGuiManager::dragFloatRow("Aperture Diameter (mm)", apertureDiameterMm, 0.1f, 0.f, 64.f, [&](float value) {
        setApertureDiameter(value);
        changed = true;
    });
    ImGuiManager::dragFloatRow("Focus Distance", focusDistance, 0.1f, 0.001f, 10000.f, [&](float value) {
        setOpticalFocusDistance(value);
        changed = true;
    });

    ImGuiManager::tableRowLabel("Ray LUT Step");
    changed |= ImGui::InputInt("##RossPsfRayLutStep", &rayLutStepSize);
    rayLutStepSize = std::max(1, rayLutStepSize);
    ImGuiManager::tableRowLabel("Aperture Samples/Dim");
    changed |= ImGui::InputInt("##RossPsfSamplesPerDim", &samplesPerDimension);
    samplesPerDimension = std::max(1, samplesPerDimension);

    if (ImGui::Button("Reload##RossPsfCamera")) {
        loadLensSensorAndPsf(true, true);
        changed = true;
    }
    ImGui::SameLine();
    ImGui::TextUnformatted(loadStatus.c_str());

    const bool sensorChanged = sensor.renderUi();
    if (sensorChanged && !lensPath.empty())
        loadLensSensorAndPsf();
    return changed || sensorChanged;
}
