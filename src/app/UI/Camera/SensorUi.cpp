#include "Rendering/Camera/Sensor.h"
#include <array>
#include <cstdio>
#include <imgui.h>
#include "UI/ImGuiManager.h"
#include "UI/MathInput.h"
#include "portable-file-dialogs.h"

bool Sensor::renderUi() {
    if (!ImGuiManager::accordionRow("Sensor###SensorProperties")) return false;
    bool changed = false;
    std::array<char,512> path{}; std::snprintf(path.data(),path.size(),"%s",imageSensorPath);
    ImGuiManager::tableRowLabel("Sensor File");
    if (ImGui::InputText("##ImageSensorPath",path.data(),path.size())) { setImageSensorPath(path.data()); changed=true; }
    ImGui::SameLine();
    if (ImGui::Button("Browse##ImageSensor")) imageSensorDialog = new pfd::open_file("Select Sensor File",".",{"Sensor Files","*.json","All Files","*"});
    if (imageSensorDialog && imageSensorDialog->ready(0)) { const auto result=imageSensorDialog->result(); if(!result.empty()){setImageSensorPath(result.front()); changed|=loadImageSensorDimensions();} delete imageSensorDialog; imageSensorDialog=nullptr; }
    ImGuiManager::dragFloatRow("Width (mm)", widthMm, .1f, .1f, 500.f, [&](float v){setDimensionsMm(v,heightMm);changed=true;});
    ImGuiManager::dragFloatRow("Height (mm)", heightMm, .1f, .1f, 500.f, [&](float v){setDimensionsMm(widthMm,v);changed=true;});
    int x=static_cast<int>(resolutionWidth), y=static_cast<int>(resolutionHeight);
    ImGuiManager::tableRowLabel("Resolution X"); if(MathInput::InputInt("##ResolutionX",&x,0,0,ImGuiInputTextFlags_CharsDecimal)&&x>0){setResolution(x,resolutionHeight);changed=true;}
    ImGuiManager::tableRowLabel("Resolution Y"); if(MathInput::InputInt("##ResolutionY",&y,0,0,ImGuiInputTextFlags_CharsDecimal)&&y>0){setResolution(resolutionWidth,y);changed=true;}
    ImGuiManager::tableRowLabel(""); ImGui::TextUnformatted(imageSensorLoadStatus); return changed;
}
