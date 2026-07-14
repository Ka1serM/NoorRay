#include "RealisticCamera.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <imgui.h>
#include <string>
#include <stdexcept>
#include "CUDA/ManagedMemory.h"
#include "Raytracing/Sellmeier.h"
#include "Log.h"
#include "UI/ImGuiManager.h"
#include "libross/foundation/gpu/types/Allocator.h"
#include "libross/imaging/cameralens/lenssystemio/CameraLensSystemReader.h"
#include "libross/imaging/cameralens/raytracing/exitpupil/ExitPupilCalculator.h"
#include "openlensfileio/glasscatalogs/glasscatalog/GlassCatalogLibrary.h"

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

const char* surfaceGeometryLabel(const ross::LensSurface& surf)
{
    if (surf.isAperture())
        return "Aperture";
    switch (surf.geometry) {
        case ross::LensGeometry::SPHERICAL:   return "Spheric";
        case ross::LensGeometry::ASPHERICAL:  return "Aspheric";
        case ross::LensGeometry::CYLINDER_X:  return "Cylinder X";
        case ross::LensGeometry::CYLINDER_Y:  return "Cylinder Y";
        case ross::LensGeometry::PLANAR:      return "Planar";
    }
    return "Surface";
}
}

RealisticCamera::RealisticCamera()
    : RealisticCamera(std::make_unique<RectangularSensor>())
{
}

RealisticCamera::RealisticCamera(std::unique_ptr<Sensor> ownedSensor)
    : Camera(std::move(ownedSensor))
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
    apertureDiameterMm = clampedApertureDiameterMm;
    opticsUpdatePending = static_cast<bool>(sourceRossLens);
}

void RealisticCamera::setOpticalFocusDistanceCm(const float requestedFocusDistanceCm)
{
    const float clampedFocusDistanceCm = std::max(0.1f, requestedFocusDistanceCm);
    if (focusDistanceCm == clampedFocusDistanceCm)
        return;
    focusDistanceCm = clampedFocusDistanceCm;
    opticsUpdatePending = static_cast<bool>(sourceRossLens);
}

void RealisticCamera::prepareOptics()
{
    if (opticsUpdatePending)
        updateLensSettings();
}

RealisticCamera::RealisticCamera(const RealisticCamera& other)
    : Camera(other)
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

    if (lensPath.empty() || sensor.getImageSensorPath().empty()) {
        sensorWidthCm = 0.0f;
        sensorHeightCm = 0.0f;
        filmDiagonalCm = 0.0f;
        loadStatus = "No lens or sensor file loaded";
        LOG_INFO("RealisticCamera: no lens/sensor loaded");
        return;
    }

    try {
        olio::GlassCatalogLibrary catalogs;
        std::string catalogList = glassCatalogPaths;
        std::ranges::replace(catalogList, ';', ',');
        if (!catalogList.empty())
            catalogs.loadCatalogsFromCommaSeperatedString(catalogList);

        ross::CameraLens loaded =
            ross::CameraLensSystemReader::readCameraLens(lensPath, catalogs);
        nr::rstd::allocator<ross::CameraLens> lensAlloc;
        sourceRossLens.reset(lensAlloc.allocate(1));
        lensAlloc.construct(sourceRossLens.get(), loaded);
        if (resetLensSettings)
            focusDistanceCm = 500.0f;
        if (resetLensSettings || apertureDiameterMm <= 0.0f) {
            apertureDiameterMm = std::max(0.0f, loaded.getApertureRadius() * 20.0f);
        } else {
            loaded.changeAperture_mm(apertureDiameterMm);
        }
        loaded.focusLens(focusDistanceCm);

        rossLens.reset(lensAlloc.allocate(1));
        lensAlloc.construct(rossLens.get(), loaded);

        if (!sensor.loadImageSensorDimensions())
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
            + ", focal length: " + std::to_string(focalLengthMm) + " mm";
        LOG_INFO("RealisticCamera: " << loadStatus);
    } catch (const std::exception& e) {
        freeRossLens();
        loadStatus = e.what();
        LOG_ERROR("RealisticCamera: " << loadStatus);
    }
}

bool RealisticCamera::renderUi()
{
    Sensor& sensor = getSensor();
    if (lensDialog && lensDialog->ready(0)) {
        const auto selection = lensDialog->result();
        if (!selection.empty()) {
            lensPath = selection.front();
            loadLensAndSensor(true);
        }
        lensDialog.reset();
    }
    if (glassCatalogDialog && glassCatalogDialog->ready(0)) {
        const auto selection = glassCatalogDialog->result();
        if (!selection.empty()) {
            glassCatalogPaths = joinWithSemicolons(selection);
            loadLensAndSensor();
        }
        glassCatalogDialog.reset();
    }

    std::array<char, 512> lensBuffer{};
    std::array<char, 1024> glassCatalogBuffer{};
    std::snprintf(lensBuffer.data(), lensBuffer.size(), "%s", lensPath.c_str());
    std::snprintf(glassCatalogBuffer.data(), glassCatalogBuffer.size(), "%s", glassCatalogPaths.c_str());

    const float browseButtonWidth = ImGui::CalcTextSize("...").x + ImGui::GetStyle().FramePadding.x * 2.0f;

    ImGuiManager::tableRowLabel("Lens File");
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - browseButtonWidth - ImGui::GetStyle().ItemSpacing.x);
    if (ImGui::InputText("##RealisticLensPath", lensBuffer.data(), lensBuffer.size()))
        lensPath = lensBuffer.data();
    ImGui::PopItemWidth();
    ImGui::SameLine();
    ImGui::BeginDisabled(lensDialog != nullptr);
    if (ImGui::Button("...##RealisticLens", ImVec2(browseButtonWidth, 0))) {
        lensDialog = std::make_unique<pfd::open_file>(
            "Select Lens File", ".",
            std::vector<std::string>{"Lens Files", "*.olio *.zmx *.dat", "All Files", "*"});
    }
    ImGui::EndDisabled();

    ImGuiManager::tableRowLabel("Glass Catalogs");
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - browseButtonWidth - ImGui::GetStyle().ItemSpacing.x);
    if (ImGui::InputText("##RealisticGlassCatalogPaths", glassCatalogBuffer.data(), glassCatalogBuffer.size()))
        glassCatalogPaths = glassCatalogBuffer.data();
    ImGui::PopItemWidth();
    ImGui::SameLine();
    ImGui::BeginDisabled(glassCatalogDialog != nullptr);
    if (ImGui::Button("...##RealisticGlassCatalogs", ImVec2(browseButtonWidth, 0))) {
        glassCatalogDialog = std::make_unique<pfd::open_file>(
            "Select Glass Catalogs", ".",
            std::vector<std::string>{"Glass Catalogs", "*.agf *.AGF", "All Files", "*"},
            pfd::opt::multiselect);
    }
    ImGui::EndDisabled();

    bool changed = false;

    if (ImGui::Button("Reload##RealisticCamera")) {
        loadLensAndSensor(true);
        changed = true;
    }
    ImGui::SameLine();
    ImGui::TextUnformatted(loadStatus.c_str());

    ImGuiManager::dragFloatRow("Aperture Diameter (mm)", apertureDiameterMm, 0.1f, 0.f, 64.f, [&](float value) {
        setApertureDiameterMm(value);
        changed = true;
    });

    ImGuiManager::dragFloatRow("Focus Distance (cm)", focusDistanceCm, 10.0f, 0.1f, 1000000.f, [&](float value) {
        setOpticalFocusDistanceCm(value);
        changed = true;
    });

    const bool sensorChanged = sensor.renderUi();
    if (sensorChanged && !lensPath.empty())
        loadLensAndSensor();

    ImGuiManager::tableRowLabel("Focal Length");
    ImGui::Text("%.1f mm", focalLengthMm);

    if (rossLens && !rossLens->surfaces.empty()) {
        constexpr ImGuiTableFlags tableFlags =
            ImGuiTableFlags_Borders |
            ImGuiTableFlags_RowBg |
            ImGuiTableFlags_Resizable |
            ImGuiTableFlags_ScrollY |
            ImGuiTableFlags_SizingStretchProp;
        const float tableHeight = std::min(320.0f, 28.0f +
            static_cast<float>(rossLens->surfaces.size()) * ImGui::GetTextLineHeightWithSpacing());
        if (ImGui::BeginChild("##LensElements", ImVec2(0.0f, tableHeight), ImGuiChildFlags_Borders)) {
            if (ImGui::BeginTable("##LensTable", 7, tableFlags)) {
                ImGui::TableSetupScrollFreeze(0, 1);
                ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 32.0f);
                ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed, 84.0f);
                ImGui::TableSetupColumn("Radius mm");
                ImGui::TableSetupColumn("Thickness mm");
                ImGui::TableSetupColumn("Aperture mm");
                ImGui::TableSetupColumn("IOR");
                ImGui::TableSetupColumn("Z mm");
                ImGui::TableHeadersRow();

                for (size_t i = 0; i < rossLens->surfaces.size(); ++i) {
                    const auto& surf = rossLens->surfaces[i];
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::Text("%zu", i);
                    ImGui::TableSetColumnIndex(1);
                    ImGui::TextUnformatted(surfaceGeometryLabel(surf));
                    ImGui::TableSetColumnIndex(2);
                    ImGui::Text("%.4g", surf.curvatureRadius);
                    ImGui::TableSetColumnIndex(3);
                    ImGui::Text("%.4g", surf.thickness);
                    ImGui::TableSetColumnIndex(4);
                    ImGui::Text("%.4g", surf.apertureRadius);
                    ImGui::TableSetColumnIndex(5);
                    ImGui::Text("%.4g", surf.material.getIor(FraunhoferGreenNm));
                    ImGui::TableSetColumnIndex(6);
                    ImGui::Text("%.4g", surf.center);
                }
                ImGui::EndTable();
            }
        }
        ImGui::EndChild();
    }

    return changed || sensorChanged;
}
