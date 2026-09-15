#include "UI/Camera/CameraUi.h"

#include "Camera/FisheyeCamera.h"

#include <algorithm>

#include "UI/ImGuiManager.h"

bool camera_ui::render(FisheyeCamera& camera)
{
    bool changed = renderCommon(camera);
    ImGuiManager::dragFloatRow("Aperture Diameter (mm)", camera.apertureDiameterMm, 0.1f, 0.f, 1000.f, [&](float value) {
        camera.apertureDiameterMm = std::max(0.f, value);
        changed = true;
    });
    ImGuiManager::dragFloatRow("Focus Distance (cm)", camera.focusDistanceCm, 10.0f, 0.1f, 100000.f, [&](float value) {
        camera.focusDistanceCm = std::max(0.1f, value);
        changed = true;
    });
    ImGuiManager::dragFloatRow("Bokeh Bias", camera.bokehBias, 0.01f, 0.001f, 10.f, [&](float value) {
        camera.bokehBias = std::max(0.001f, value);
        changed = true;
    });
    return render(camera.getSensor()) || changed;
}
