#include "Camera/Sensor.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <imgui.h>
#include <type_traits>
#include <vector>

#include "Camera/GatherPsfSensor.h"
#include "Camera/RectangularSensor.h"
#include "Camera/ScatterPsfSensor.h"
#include "CUDA/ManagedMemory.h"
#include "UI/ImGuiManager.h"
#include "UI/MathInput.h"
#include "portable-file-dialogs.h"

namespace
{
constexpr std::array<const char*, 3> SensorTypeNames{
    "Rectangular", "Scatter PSF", "Gather PSF"};

void synchronizeSensorMutation(const char* reason)
{
    nr::synchronizeBeforeManagedMutation(reason);
}

bool beginSensorUi(const char* label)
{
    ImGuiManager::tableRowLabel("Sensor");
    return ImGui::TreeNodeEx(label, ImGuiTreeNodeFlags_Framed);
}

bool renderSensorTypeCombo(Sensor& owner, const SensorType currentType)
{
    ImGuiManager::tableRowLabel("Type");
    int typeIndex = static_cast<int>(currentType);
    if (!ImGui::Combo("##SensorType", &typeIndex, SensorTypeNames.data(), SensorTypeNames.size()))
        return false;
    synchronizeSensorMutation("Sensor type request");
    owner.requestType(static_cast<SensorType>(typeIndex));
    return true;
}

bool renderPhysicalSensorRows(Sensor& sensor)
{
    if (sensor.imageSensorDialog && sensor.imageSensorDialog->ready(0)) {
        const auto selection = sensor.imageSensorDialog->result();
        if (!selection.empty()) {
            synchronizeSensorMutation("Sensor file selection");
            sensor.setImageSensorPath(selection.front());
            sensor.loadImageSensorDimensions();
        }
        delete sensor.imageSensorDialog;
        sensor.imageSensorDialog = nullptr;
    }

    bool changed = false;
    std::array<char, 512> path{};
    std::snprintf(path.data(), path.size(), "%s", sensor.imageSensorPath);
    const float buttonWidth = ImGui::CalcTextSize("...").x + ImGui::GetStyle().FramePadding.x * 2.f;
    ImGuiManager::tableRowLabel("Sensor File");
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - buttonWidth - ImGui::GetStyle().ItemSpacing.x);
    if (ImGui::InputText("##ImageSensorPath", path.data(), path.size())) {
        synchronizeSensorMutation("Sensor file path");
        sensor.setImageSensorPath(path.data()); changed = true;
    }
    ImGui::PopItemWidth(); ImGui::SameLine();
    ImGui::BeginDisabled(sensor.imageSensorDialog != nullptr);
    if (ImGui::Button("...##ImageSensor", ImVec2(buttonWidth, 0)))
        sensor.imageSensorDialog = new pfd::open_file("Select Sensor File", ".",
            std::vector<std::string>{"Sensor Files", "*.json", "All Files", "*"});
    ImGui::EndDisabled();

    ImGuiManager::dragFloatRow("Width (mm)", sensor.width(), 0.1f, 0.1f, 500.f, [&](float value) {
        synchronizeSensorMutation("Sensor width");
        sensor.setDimensionsMm(std::max(value, 0.1f), sensor.height()); changed = true;
    });
    ImGuiManager::dragFloatRow("Height (mm)", sensor.height(), 0.1f, 0.1f, 500.f, [&](float value) {
        synchronizeSensorMutation("Sensor height");
        sensor.setDimensionsMm(sensor.width(), std::max(value, 0.1f)); changed = true;
    });
    int resolutionX = static_cast<int>(sensor.resolutionX());
    int resolutionY = static_cast<int>(sensor.resolutionY());
    ImGuiManager::tableRowLabel("Resolution X");
    if (MathInput::InputInt("##ResolutionX", &resolutionX, 0, 0, ImGuiInputTextFlags_CharsDecimal) && resolutionX > 0) {
        synchronizeSensorMutation("Sensor horizontal resolution");
        sensor.setResolution(resolutionX, sensor.resolutionY()); changed = true;
    }
    ImGuiManager::tableRowLabel("Resolution Y");
    if (MathInput::InputInt("##ResolutionY", &resolutionY, 0, 0, ImGuiInputTextFlags_CharsDecimal) && resolutionY > 0) {
        synchronizeSensorMutation("Sensor vertical resolution");
        sensor.setResolution(sensor.resolutionX(), resolutionY); changed = true;
    }
    ImGuiManager::tableRowLabel("");
    ImGui::BeginDisabled(sensor.getImageSensorPath().empty());
    if (ImGui::Button("Reload##ImageSensor", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
        synchronizeSensorMutation("Sensor file reload");
        changed |= sensor.loadImageSensorDimensions();
    }
    ImGui::EndDisabled();
    return changed;
}

template <typename PsfSensor>
bool renderPsfGridRows(PsfSensor& sensor)
{
    if (sensor.psfGridDialog && sensor.psfGridDialog->ready(0)) {
        const auto selection = sensor.psfGridDialog->result();
        if (!selection.empty()) {
            synchronizeSensorMutation("PSF grid selection");
            sensor.psfGridPath = selection.front();
            sensor.loadPsfGrid();
        }
        sensor.psfGridDialog.reset();
    }
    bool changed = false;
    std::array<char, 512> path{};
    std::snprintf(path.data(), path.size(), "%s", sensor.psfGridPath.c_str());
    const float buttonWidth = ImGui::CalcTextSize("...").x + ImGui::GetStyle().FramePadding.x * 2.f;
    ImGuiManager::tableRowLabel("PSF Grid");
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - buttonWidth - ImGui::GetStyle().ItemSpacing.x);
    if (ImGui::InputText("##SensorPsfGrid", path.data(), path.size())) {
        synchronizeSensorMutation("PSF grid path");
        sensor.psfGridPath = path.data();
        changed = true;
    }
    ImGui::PopItemWidth(); ImGui::SameLine();
    if (ImGui::Button("...##SensorPsfGrid", ImVec2(buttonWidth, 0)))
        sensor.psfGridDialog = std::make_unique<pfd::open_file>("Select PSF Grid JSON", ".",
            std::vector<std::string>{"JSON", "*.json", "All Files", "*"});
    if (ImGui::Button("Reload PSF Grid")) {
        synchronizeSensorMutation("PSF grid reload");
        sensor.loadPsfGrid();
        changed = true;
    }
    ImGui::SameLine(); ImGui::TextUnformatted(sensor.psfLoadStatus.c_str());
    return changed;
}

template <typename Concrete>
bool renderConcreteSensor(Sensor& owner, Concrete& sensor, const char* label, const SensorType type)
{
    if (!beginSensorUi(label)) return false;
    bool changed = false;
    if (ImGui::BeginTable("SensorTable", 2, ImGuiTableFlags_SizingStretchProp)) {
        changed |= renderSensorTypeCombo(owner, type);
        changed |= renderPhysicalSensorRows(owner);
        if constexpr (!std::is_same_v<Concrete, RectangularSensor>)
            changed |= renderPsfGridRows(sensor);
        ImGui::EndTable();
    }
    ImGui::TreePop();
    return changed;
}
}

bool Sensor::renderUi()
{
    return DispatchCPU([this](auto* concrete) { return concrete->renderUi(*this); });
}

bool RectangularSensor::renderUi(Sensor& owner)
{
    return renderConcreteSensor(owner, *this, "Rectangular###SensorProperties", SensorType::Rectangular);
}

bool ScatterPsfSensor::renderUi(Sensor& owner)
{
    return renderConcreteSensor(owner, *this, "Scatter PSF###SensorProperties", SensorType::ScatterPsf);
}

bool GatherPsfSensor::renderUi(Sensor& owner)
{
    return renderConcreteSensor(owner, *this, "Gather PSF###SensorProperties", SensorType::GatherPsf);
}
