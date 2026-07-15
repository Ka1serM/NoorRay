#include "Camera/Camera.h"

#include <algorithm>

#include "CUDA/ManagedMemory.h"
#include "UI/ImGuiManager.h"

bool Camera::renderUi()
{
    bool changed = false;
    ImGuiManager::dragFloatRow("Exposure", exposure, 0.01f, -100.f, 100.f, [&](float value) {
        nr::synchronizeBeforeManagedMutation("Camera exposure");
        exposure = value; changed = true;
    });
    ImGuiManager::dragFloatRow("Field of View (degrees)", fieldOfViewDegrees, 0.1f, 1.f, 179.f, [&](float value) {
        nr::synchronizeBeforeManagedMutation("Camera field of view");
        fieldOfViewDegrees = std::clamp(value, 1.f, 179.f);
        focalLengthMm = focalLengthMmForFovDegrees(fieldOfViewDegrees);
        changed = true;
    });
    ImGuiManager::dragFloatRow("Focal Length (mm)", focalLengthMm, 0.1f, 0.001f, 500.f, [&](float value) {
        setFocalLengthMm(value); changed = true;
    });
    return changed;
}
