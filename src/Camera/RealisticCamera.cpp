#include "RealisticCamera.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <imgui.h>
#include <string>
#include <stdexcept>
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

void RealisticCamera::load(std::string lensPath_, std::string glassCatalogPaths_)
{
    lensPath           = std::move(lensPath_);
    glassCatalogPaths  = std::move(glassCatalogPaths_);
    loadLensAndSensor();
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
    if (exitPupil != nullptr) {
        ross::rstd::allocator<ross::ExitPupil> pupilAlloc;
        pupilAlloc.destroy(exitPupil);
        pupilAlloc.deallocate(exitPupil, 1);
        exitPupil = nullptr;
    }
    if (rossLens != nullptr) {
        ross::rstd::allocator<ross::CameraLens> allocator;
        allocator.destroy(rossLens);
        allocator.deallocate(rossLens, 1);
        rossLens = nullptr;
    }
}

void RealisticCamera::loadLensAndSensor()
{
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
        if (apertureDiameterMm > 0.0f) {
            loaded.changeAperture_mm(apertureDiameterMm);
        } else {
            apertureDiameterMm = std::max(0.0f, loaded.getApertureRadius() * 20.0f);
        }
        loaded.focusLens(focusDistance * 100.0f);

        ross::rstd::allocator<ross::CameraLens> lensAlloc;
        rossLens = lensAlloc.allocate(1);
        lensAlloc.construct(rossLens, loaded);

        if (!sensor.loadImageSensorDimensions())
            throw std::runtime_error("Failed to load image sensor");

        sensorWidthCm = sensor.width() * 0.1f;
        sensorHeightCm = sensor.height() * 0.1f;
        filmDiagonalCm = std::sqrt(
            sensorWidthCm * sensorWidthCm + sensorHeightCm * sensorHeightCm);

        const float filmDiagonalCmVal = filmDiagonalCm;
        ross::ExitPupilCalculator::CalculationSettings calcSettings;
        calcSettings.samplesFilmX = 64;
        calcSettings.samplesLens = 1024;
        ross::TaskReporter taskReporter;
        ross::ExitPupilCalculator exitPupilCalc(
            *rossLens, filmDiagonalCmVal, calcSettings, taskReporter);
        ross::ExitPupil computedPupil = exitPupilCalc.calculate();

        ross::rstd::allocator<ross::ExitPupil> pupilAlloc;
        exitPupil = pupilAlloc.allocate(1);
        pupilAlloc.construct(exitPupil, computedPupil);

        effectiveFocalLengthM = rossLens->metadata.focalLength * 0.01f;
        focalLengthMm = effectiveFocalLengthM * 1000.0f;
        fieldOfView = fovForFocalLength(focalLengthMm);

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
    if (lensDialog && lensDialog->ready(0)) {
        const auto selection = lensDialog->result();
        if (!selection.empty()) {
            lensPath = selection.front();
            loadLensAndSensor();
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

    const float selectButtonWidth = ImGui::CalcTextSize("Select").x + ImGui::GetStyle().FramePadding.x * 2.0f;

    ImGuiManager::tableRowLabel("Lens File");
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - selectButtonWidth - ImGui::GetStyle().ItemSpacing.x);
    if (ImGui::InputText("##RealisticLensPath", lensBuffer.data(), lensBuffer.size()))
        lensPath = lensBuffer.data();
    ImGui::PopItemWidth();
    ImGui::SameLine();
    ImGui::BeginDisabled(lensDialog != nullptr);
    if (ImGui::Button("Select##RealisticLens", ImVec2(selectButtonWidth, 0))) {
        lensDialog = std::make_unique<pfd::open_file>(
            "Select Lens File", ".",
            std::vector<std::string>{"Lens Files", "*.olio *.zmx *.dat", "All Files", "*"});
    }
    ImGui::EndDisabled();

    ImGuiManager::tableRowLabel("Glass Catalogs");
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - selectButtonWidth - ImGui::GetStyle().ItemSpacing.x);
    if (ImGui::InputText("##RealisticGlassCatalogPaths", glassCatalogBuffer.data(), glassCatalogBuffer.size()))
        glassCatalogPaths = glassCatalogBuffer.data();
    ImGui::PopItemWidth();
    ImGui::SameLine();
    ImGui::BeginDisabled(glassCatalogDialog != nullptr);
    if (ImGui::Button("Select##RealisticGlassCatalogs", ImVec2(selectButtonWidth, 0))) {
        glassCatalogDialog = std::make_unique<pfd::open_file>(
            "Select Glass Catalogs", ".",
            std::vector<std::string>{"Glass Catalogs", "*.agf *.AGF", "All Files", "*"},
            pfd::opt::multiselect);
    }
    ImGui::EndDisabled();

    bool changed = false;

    if (ImGui::Button("Reload##RealisticCamera")) {
        loadLensAndSensor();
        changed = true;
    }
    ImGui::SameLine();
    ImGui::TextUnformatted(loadStatus.c_str());

    ImGuiManager::dragFloatRow("Aperture Diameter (mm)", apertureDiameterMm, 0.1f, 0.f, 64.f, [&](float value) {
        apertureDiameterMm = std::max(0.f, value);
        if (rossLens != nullptr && apertureDiameterMm > 0.0f)
            rossLens->changeAperture_mm(apertureDiameterMm);
        loadLensAndSensor();
        changed = true;
    });

    ImGuiManager::dragFloatRow("Focus Distance", focusDistance, 0.1f, 0.001f, 10000.f, [&](float value) {
        focusDistance = std::max(0.001f, value);
        if (rossLens != nullptr)
            rossLens->focusLens(focusDistance * 100.0f);
        loadLensAndSensor();
        changed = true;
    });

    const bool sensorChanged = sensor.renderUi();
    if (sensorChanged && rossLens != nullptr)
        loadLensAndSensor();

    ImGuiManager::tableRowLabel("Focal Length");
    ImGui::Text("%.1f mm", focalLengthMm);

    if (rossLens != nullptr && !rossLens->surfaces.empty()) {
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
