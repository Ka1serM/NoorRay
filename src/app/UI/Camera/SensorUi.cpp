#include "UI/Camera/CameraUi.h"

#include "Camera/Sensor.h"
#include <array>
#include <cstdio>
#include <imgui.h>
#include <memory>
#include "UI/ImGuiManager.h"
#include "UI/MathInput.h"
#include "portable-file-dialogs.h"

namespace
{
// The inspector shows one sensor at a time, so one pending dialog is enough.
// The owner identifies which sensor asked, so a result that arrives after the
// selection changed is not applied to a different sensor.
struct PendingSensorDialog
{
    std::unique_ptr<pfd::open_file> dialog;
    const Sensor* owner{};
};
PendingSensorDialog pendingImageSensor;
}

bool camera_ui::render(Sensor& sensor) {
    if (!ImGuiManager::accordionRow("Sensor###SensorProperties")) return false;
    bool changed = false;
    std::array<char,512> path{}; std::snprintf(path.data(),path.size(),"%s",sensor.imageSensorPath);
    ImGuiManager::tableRowLabel("Sensor File");
    if (ImGui::InputText("##ImageSensorPath",path.data(),path.size())) { sensor.setImageSensorPath(path.data()); changed=true; }
    ImGui::SameLine();
    if (ImGui::Button("Browse##ImageSensor")) pendingImageSensor = {std::make_unique<pfd::open_file>("Select Sensor File",".",std::vector<std::string>{"Sensor Files","*.json","All Files","*"}), &sensor};
    if (pendingImageSensor.dialog && pendingImageSensor.owner == &sensor && pendingImageSensor.dialog->ready(0)) { const auto result=pendingImageSensor.dialog->result(); if(!result.empty()){sensor.setImageSensorPath(result.front()); changed|=sensor.loadImageSensorDimensions();} pendingImageSensor = {}; }
    ImGuiManager::dragFloatRow("Width (mm)", sensor.widthMm, .1f, .1f, 500.f, [&](float v){sensor.setDimensionsMm(v,sensor.heightMm);changed=true;});
    ImGuiManager::dragFloatRow("Height (mm)", sensor.heightMm, .1f, .1f, 500.f, [&](float v){sensor.setDimensionsMm(sensor.widthMm,v);changed=true;});
    int x=static_cast<int>(sensor.resolutionWidth), y=static_cast<int>(sensor.resolutionHeight);
    ImGuiManager::tableRowLabel("Resolution X"); if(MathInput::InputInt("##ResolutionX",&x,0,0,ImGuiInputTextFlags_CharsDecimal)&&x>0){sensor.setResolution(x,sensor.resolutionHeight);changed=true;}
    ImGuiManager::tableRowLabel("Resolution Y"); if(MathInput::InputInt("##ResolutionY",&y,0,0,ImGuiInputTextFlags_CharsDecimal)&&y>0){sensor.setResolution(sensor.resolutionWidth,y);changed=true;}
    ImGuiManager::tableRowLabel(""); ImGui::TextUnformatted(sensor.imageSensorLoadStatus); return changed;
}
