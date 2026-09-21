#include "UI/Camera/CameraUi.h"

#include "Camera/FisheyeCamera.h"

#include <algorithm>

#include "UI/ImGuiManager.h"

bool camera_ui::render(FisheyeCamera& camera)
{
    bool changed = renderCommon(camera);
    float aperture = camera.getApertureDiameterMm();
    ImGuiManager::dragFloatRow("Aperture Diameter (mm)", aperture, 0.1f, 0.f, 1000.f, [&](float value) {
        camera.setApertureDiameterMm(value);
        changed = true;
    });
    float focusDistance = camera.getFocusDistanceCm();
    ImGuiManager::dragFloatRow("Focus Distance (cm)", focusDistance, 10.0f, 0.1f, 100000.f, [&](float value) {
        camera.setFocusDistanceCm(value);
        changed = true;
    });
    float bokehBias = camera.getBokehBias();
    ImGuiManager::dragFloatRow("Bokeh Bias", bokehBias, 0.01f, 0.001f, 10.f, [&](float value) {
        camera.setBokehBias(value);
        changed = true;
    });
    return render(camera.getSensor()) || changed;
}
