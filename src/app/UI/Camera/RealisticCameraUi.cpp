#include "Camera/RealisticCamera.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <imgui.h>
#include <string>
#include <vector>

#include "CUDA/ManagedMemory.h"
#include "Raytracing/Sellmeier.h"
#include "UI/ImGuiManager.h"

namespace
{
std::string joinWithSemicolons(const std::vector<std::string>& paths)
{
    std::string result;
    for (const std::string& path : paths) {
        if (!result.empty()) result += ';';
        result += path;
    }
    return result;
}

const char* surfaceGeometryLabel(const ross::LensSurface& surface)
{
    if (surface.isAperture()) return "Aperture";
    switch (surface.geometry) {
    case ross::LensGeometry::SPHERICAL: return "Spheric";
    case ross::LensGeometry::ASPHERICAL: return "Aspheric";
    case ross::LensGeometry::CYLINDER_X: return "Cylinder X";
    case ross::LensGeometry::CYLINDER_Y: return "Cylinder Y";
    case ross::LensGeometry::PLANAR: return "Planar";
    }
    return "Surface";
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

    ImGuiManager::dragFloatRow("Exposure", exposure, 0.01f, -100.f, 100.f, [&](float value) {
        nr::synchronizeBeforeManagedMutation("Realistic camera exposure");
        exposure = value;
        changed = true;
    });

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
