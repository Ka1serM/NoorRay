#include "Rendering/Camera/RealisticCamera.h"
#include <array>
#include <cstdio>
#include <imgui.h>
#include "UI/ImGuiManager.h"
#include "portable-file-dialogs.h"
bool RealisticCamera::renderUi() {
    bool changed = false; Sensor& sensor = getSensor();
    if (lensDialog && lensDialog->ready(0)) { const auto selected = lensDialog->result(); if (!selected.empty()) { lensPath = selected.front(); changed |= loadLensAndSensor(true); } lensDialog.reset(); }
    if (glassCatalogDialog && glassCatalogDialog->ready(0)) { const auto selected = glassCatalogDialog->result(); if (!selected.empty()) { glassCatalogPaths.clear(); for (const auto& path : selected) { if (!glassCatalogPaths.empty()) glassCatalogPaths += ';'; glassCatalogPaths += path; } changed |= loadLensAndSensor(); } glassCatalogDialog.reset(); }
    std::array<char, 512> lens{}; std::array<char, 1024> catalogs{}; std::snprintf(lens.data(), lens.size(), "%s", lensPath.c_str()); std::snprintf(catalogs.data(), catalogs.size(), "%s", glassCatalogPaths.c_str());
    ImGuiManager::tableRowLabel("Lens File"); if (ImGui::InputText("##RealisticLens", lens.data(), lens.size())) lensPath = lens.data(); ImGui::SameLine(); if (ImGui::Button("Browse##Lens")) lensDialog = std::make_unique<pfd::open_file>("Select ZMX Lens", ".", std::vector<std::string>{"Zemax lenses", "*.zmx", "All Files", "*"});
    ImGuiManager::tableRowLabel("AGF Catalogs"); if (ImGui::InputText("##RealisticCatalogs", catalogs.data(), catalogs.size())) glassCatalogPaths = catalogs.data(); ImGui::SameLine(); if (ImGui::Button("Browse##Catalogs")) glassCatalogDialog = std::make_unique<pfd::open_file>("Select AGF catalogs", ".", std::vector<std::string>{"AGF catalogs", "*.agf *.AGF", "All Files", "*"}, pfd::opt::multiselect);
    ImGuiManager::dragFloatRow("Aperture Diameter (mm)", apertureDiameterMm, .1f, 0.f, 64.f, [&](float v) { setApertureDiameterMm(v); changed = true; });
    ImGuiManager::dragFloatRow("Focus Distance (cm)", focusDistanceCm, 10.f, .1f, 1e6f, [&](float v) { setOpticalFocusDistanceCm(v); changed = true; });
    ImGuiManager::tableRowLabel(""); if (ImGui::Button("Reload##Realistic")) changed |= loadLensAndSensor(true); ImGui::SameLine(); ImGui::TextUnformatted(loadStatus.c_str());
    const bool sensorChanged = sensor.renderUi(); if (sensorChanged && !lensPath.empty()) changed |= loadLensAndSensor();
    ImGuiManager::tableRowLabel("Focal Length"); ImGui::Text("%.1f mm", focalLengthMm);
    ImGuiManager::tableRowLabel("Native Optics"); ImGui::Text("%u surfaces, %.2f mm exit pupil", optics.surfaceCount, optics.rearPupilRadius * 2.f);
    return changed || sensorChanged;
}
