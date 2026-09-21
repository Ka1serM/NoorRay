#include "UI/Camera/CameraUi.h"

#include "Camera/Camera.h"

#include <algorithm>

#include "UI/ImGuiManager.h"

bool camera_ui::renderCommon(Camera& camera)
{
    bool changed = false;
    float exposure = camera.getExposure();
    ImGuiManager::dragFloatRow("Exposure", exposure, 0.01f, -100.f, 100.f, [&](float value) {
        camera.setExposure(value); changed = true;
    });
    float focalLengthMm = camera.getFocalLengthMm();
    float derivedFovDegrees = camera.fovDegreesForFocalLengthMm(focalLengthMm);
    ImGuiManager::dragFloatRow("Field of View (degrees)", derivedFovDegrees, 0.1f, 1.f, 179.f, [&](float value) {
        camera.setFocalLengthMm(camera.focalLengthMmForFovDegrees(value));
        changed = true;
    });
    ImGuiManager::dragFloatRow("Focal Length (mm)", focalLengthMm, 0.1f, 0.001f, 500.f, [&](float value) {
        camera.setFocalLengthMm(value); changed = true;
    });
    return changed;
}
