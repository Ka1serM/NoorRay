#include "Camera/HybridPsfCamera.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <imgui.h>
#include <string>
#include <vector>

#include "CUDA/ManagedMemory.h"
#include "UI/ImGuiManager.h"
#include "UI/MathInput.h"
#include "portable-file-dialogs.h"

namespace
{
std::string joinRossPsfPathsWithSemicolons(const std::vector<std::string>& paths)
{
    std::string result;
    for (const std::string& path : paths) {
        if (!result.empty()) result += ';';
        result += path;
    }
    return result;
}
}

bool HybridPsfCamera::renderUi()
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

    ImGuiManager::dragFloatRow("Exposure", exposure, 0.01f, -100.f, 100.f, [&](float value) {
        nr::synchronizeBeforeManagedMutation("Hybrid PSF camera exposure");
        exposure = value;
        changed = true;
    });

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
        setApertureDiameterMm(value);
        changed = true;
    });
    ImGuiManager::dragFloatRow("Focus Distance (cm)", focusDistanceCm, 10.0f, 0.1f, 1000000.f, [&](float value) {
        setOpticalFocusDistanceCm(value);
        changed = true;
    });

    ImGuiManager::tableRowLabel("Ray LUT Step");
    int requestedRayLutStep = rayLutStepSize;
    if (MathInput::InputInt("##RossPsfRayLutStep", &requestedRayLutStep)) {
        nr::synchronizeBeforeManagedMutation("Hybrid PSF ray LUT step");
        rayLutStepSize = std::max(1, requestedRayLutStep);
        changed = true;
    }
    ImGuiManager::tableRowLabel("Aperture Samples/Dim");
    int requestedSamplesPerDimension = samplesPerDimension;
    if (MathInput::InputInt("##RossPsfSamplesPerDim", &requestedSamplesPerDimension)) {
        nr::synchronizeBeforeManagedMutation("Hybrid PSF aperture samples");
        samplesPerDimension = std::max(1, requestedSamplesPerDimension);
        changed = true;
    }

    if (ImGui::Button("Reload##HybridPsfCamera")) {
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
