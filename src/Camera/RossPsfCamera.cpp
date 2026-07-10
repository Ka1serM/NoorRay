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

#include "CUDA/rstd/Allocator.h"
#include "Log.h"
#include "UI/ImGuiManager.h"
#include "libross/foundation/gpu/types/Allocator.h"
#include "libross/foundation/physics/Wavelengths.h"
#include "libross/imaging/cameralens/lenssystemio/CameraLensSystemReader.h"
#include "libross/imaging/cameralens/raylut/FindRayThroughApertureCenterRayGenerator.h"
#include "libross/imaging/cameralens/raylut/io/read/RayLUTFileReader.h"
#include "libross/imaging/cameralens/raylut/io/write/RayLUTFileWriter.h"
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

RossPsfCamera::~RossPsfCamera()
{
    freeRossObjects();
}

void RossPsfCamera::freeRossObjects()
{
    if (rayLut != nullptr) {
        ross::rstd::allocator<ross::RayLUT> allocator;
        allocator.destroy(rayLut);
        allocator.deallocate(rayLut, 1);
        rayLut = nullptr;
    }
    if (rossSensor != nullptr) {
        ross::rstd::allocator<ross::ImageSensor> allocator;
        allocator.destroy(rossSensor);
        allocator.deallocate(rossSensor, 1);
        rossSensor = nullptr;
    }
    if (rossLens != nullptr) {
        ross::rstd::allocator<ross::CameraLens> allocator;
        allocator.destroy(rossLens);
        allocator.deallocate(rossLens, 1);
        rossLens = nullptr;
    }
}

void RossPsfCamera::load(std::string lensPath_, std::string glassCatalogPaths_, std::string rayLutPath_)
{
    lensPath = std::move(lensPath_);
    glassCatalogPaths = std::move(glassCatalogPaths_);
    rayLutPath = std::move(rayLutPath_);
    loadLensSensorAndPsf();
}

void RossPsfCamera::setOpticsPaths(std::string lensPath_, std::string glassCatalogPaths_)
{
    lensPath = std::move(lensPath_);
    glassCatalogPaths = std::move(glassCatalogPaths_);
}

void RossPsfCamera::loadLensSensorAndPsf(const bool buildRayLut)
{
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
        if (apertureDiameterMm > 0.0f)
            loadedLens.changeAperture_mm(apertureDiameterMm);
        loadedLens.focusLens(focusDistance * 100.0f);

        ross::rstd::allocator<ross::CameraLens> lensAllocator;
        rossLens = lensAllocator.allocate(1);
        lensAllocator.construct(rossLens, loadedLens);

        const ross::ImageSensor loadedSensor =
            ross::ImageSensorReader::readFile(std::string(sensor.getImageSensorPath()));
        sensor.setDimensionsMm(
            loadedSensor.dimensions.width.millimeter(),
            loadedSensor.dimensions.height.millimeter());
        sensor.setResolution(loadedSensor.resolution.width, loadedSensor.resolution.height);
        sensor.loadImageSensorDimensions();
        uint32_t psfBinCount = 0;
        sensor.DispatchCPU([&](auto* concreteSensor) {
            using SensorType = std::remove_cvref_t<decltype(*concreteSensor)>;
            if constexpr (std::is_same_v<SensorType, ScatterPsfSensor>
                          || std::is_same_v<SensorType, GatherPsfSensor>) {
                concreteSensor->loadPsfGrid();
                psfBinCount = concreteSensor->psfBinCount();
            }
        });

        ross::rstd::allocator<ross::ImageSensor> sensorAllocator;
        rossSensor = sensorAllocator.allocate(1);
        sensorAllocator.construct(rossSensor, loadedSensor);

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

            ross::rstd::allocator<ross::RayLUT> lutAllocator;
            rayLut = lutAllocator.allocate(1);
            lutAllocator.construct(rayLut, std::move(loadedLut));
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
    if (lensDialog && lensDialog->ready(0)) {
        const auto selection = lensDialog->result();
        if (!selection.empty()) {
            lensPath = selection.front();
            loadLensSensorAndPsf();
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
    const float selectButtonWidth = ImGui::CalcTextSize("Select").x + ImGui::GetStyle().FramePadding.x * 2.0f;

    auto pathRow = [&](const char* label, const char* id, auto& buffer, std::string& target,
                       auto openDialog) {
        ImGuiManager::tableRowLabel(label);
        ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - selectButtonWidth - ImGui::GetStyle().ItemSpacing.x);
        if (ImGui::InputText(id, buffer.data(), buffer.size())) {
            target = buffer.data();
            changed = true;
        }
        ImGui::PopItemWidth();
        ImGui::SameLine();
        if (ImGui::Button((std::string("Select") + id).c_str()))
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
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - selectButtonWidth * 2.0f - ImGui::GetStyle().ItemSpacing.x * 2.0f);
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
        apertureDiameterMm = std::max(0.f, value);
        loadLensSensorAndPsf();
        changed = true;
    });
    ImGuiManager::dragFloatRow("Focus Distance", focusDistance, 0.1f, 0.001f, 10000.f, [&](float value) {
        focusDistance = std::max(0.001f, value);
        loadLensSensorAndPsf();
        changed = true;
    });

    ImGuiManager::tableRowLabel("Ray LUT Step");
    changed |= ImGui::InputInt("##RossPsfRayLutStep", &rayLutStepSize);
    rayLutStepSize = std::max(1, rayLutStepSize);
    ImGuiManager::tableRowLabel("Aperture Samples/Dim");
    changed |= ImGui::InputInt("##RossPsfSamplesPerDim", &samplesPerDimension);
    samplesPerDimension = std::max(1, samplesPerDimension);

    if (ImGui::Button("Reload##RossPsfCamera")) {
        loadLensSensorAndPsf();
        changed = true;
    }
    ImGui::SameLine();
    ImGui::TextUnformatted(loadStatus.c_str());

    const bool sensorChanged = sensor.renderUi();
    return changed || sensorChanged;
}
