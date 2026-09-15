#include "UI/Camera/CameraUi.h"

#include "Camera/Camera.h"

#include <algorithm>

#include "UI/ImGuiManager.h"

bool camera_ui::renderCommon(Camera& camera)
{
    bool changed = false;
    ImGuiManager::dragFloatRow("Exposure", camera.exposure, 0.01f, -100.f, 100.f, [&](float value) {
        camera.exposure = value; changed = true;
    });
    float derivedFovDegrees = camera.fovDegreesForFocalLengthMm(camera.focalLengthMm);
    ImGuiManager::dragFloatRow("Field of View (degrees)", derivedFovDegrees, 0.1f, 1.f, 179.f, [&](float value) {
        camera.setFocalLengthMm(camera.focalLengthMmForFovDegrees(value));
        changed = true;
    });
    ImGuiManager::dragFloatRow("Focal Length (mm)", camera.focalLengthMm, 0.1f, 0.001f, 500.f, [&](float value) {
        camera.setFocalLengthMm(value); changed = true;
    });
    return changed;
}
