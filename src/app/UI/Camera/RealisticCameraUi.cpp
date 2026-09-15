#include "UI/Camera/CameraUi.h"

#include "Camera/RealisticCamera.h"
#include <array>
#include <cstdio>
#include <imgui.h>
#include <memory>
#include "UI/ImGuiManager.h"
#include "portable-file-dialogs.h"

namespace
{
// Pending file dialogs for the realistic camera being inspected. The owner
// keeps a late result from being applied to a camera selected afterwards.
struct PendingCameraDialog
{
    std::unique_ptr<pfd::open_file> dialog;
    const RealisticCamera* owner{};
};
PendingCameraDialog pendingLens;
PendingCameraDialog pendingGlassCatalogs;

bool ready(const PendingCameraDialog& pending, const RealisticCamera& camera)
{
    return pending.dialog && pending.owner == &camera && pending.dialog->ready(0);
}
}

bool camera_ui::render(RealisticCamera& camera) {
    bool changed = false; Sensor& sensor = camera.getSensor();
    if (ready(pendingLens, camera)) { const auto selected = pendingLens.dialog->result(); if (!selected.empty()) { camera.setOpticsPaths(selected.front(), camera.getGlassCatalogPaths()); changed |= camera.loadLensAndSensor(true); } pendingLens = {}; }
    if (ready(pendingGlassCatalogs, camera)) { const auto selected = pendingGlassCatalogs.dialog->result(); if (!selected.empty()) { std::string catalogPaths; for (const auto& path : selected) { if (!catalogPaths.empty()) catalogPaths += ';'; catalogPaths += path; } camera.setOpticsPaths(camera.getLensPath(), std::move(catalogPaths)); changed |= camera.loadLensAndSensor(); } pendingGlassCatalogs = {}; }
    std::array<char, 512> lens{}; std::array<char, 1024> catalogs{}; std::snprintf(lens.data(), lens.size(), "%s", camera.getLensPath().c_str()); std::snprintf(catalogs.data(), catalogs.size(), "%s", camera.getGlassCatalogPaths().c_str());
    ImGuiManager::tableRowLabel("Lens File"); if (ImGui::InputText("##RealisticLens", lens.data(), lens.size())) camera.setOpticsPaths(lens.data(), camera.getGlassCatalogPaths()); ImGui::SameLine(); if (ImGui::Button("Browse##Lens")) pendingLens = {std::make_unique<pfd::open_file>("Select ZMX Lens", ".", std::vector<std::string>{"Zemax lenses", "*.zmx", "All Files", "*"}), &camera};
    ImGuiManager::tableRowLabel("AGF Catalogs"); if (ImGui::InputText("##RealisticCatalogs", catalogs.data(), catalogs.size())) camera.setOpticsPaths(camera.getLensPath(), catalogs.data()); ImGui::SameLine(); if (ImGui::Button("Browse##Catalogs")) pendingGlassCatalogs = {std::make_unique<pfd::open_file>("Select AGF catalogs", ".", std::vector<std::string>{"AGF catalogs", "*.agf *.AGF", "All Files", "*"}, pfd::opt::multiselect), &camera};
    ImGuiManager::dragFloatRow("Aperture Diameter (mm)", camera.apertureDiameterMm, .1f, 0.f, 64.f, [&](float v) { camera.setApertureDiameterMm(v); changed = true; });
    ImGuiManager::dragFloatRow("Focus Distance (cm)", camera.focusDistanceCm, 10.f, .1f, 1e6f, [&](float v) { camera.setOpticalFocusDistanceCm(v); changed = true; });
    ImGuiManager::tableRowLabel(""); if (ImGui::Button("Reload##Realistic")) changed |= camera.loadLensAndSensor(true); ImGui::SameLine(); ImGui::TextUnformatted(camera.getLoadStatus().c_str());
    const bool sensorChanged = render(sensor); if (sensorChanged && !camera.getLensPath().empty()) changed |= camera.loadLensAndSensor();
    ImGuiManager::tableRowLabel("Focal Length"); ImGui::Text("%.1f mm", camera.focalLengthMm);
    ImGuiManager::tableRowLabel("Native Optics"); ImGui::Text("%u surfaces, %.2f mm exit pupil", camera.optics.surfaceCount, camera.optics.rearPupilRadius * 2.f);
    return changed || sensorChanged;
}
