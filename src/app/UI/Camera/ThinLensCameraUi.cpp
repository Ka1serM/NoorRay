#include "Rendering/Camera/ThinLensCamera.h"

#include <algorithm>

#include "Backend/CUDA/ManagedMemory.h"
#include "UI/ImGuiManager.h"

bool ThinLensCamera::renderUi()
{
    bool changed = Camera::renderUi();
    ImGuiManager::dragFloatRow("Aperture Diameter (mm)", apertureDiameterMm, 0.1f, 0.f, 1000.f, [&](float value) {
        nr::synchronizeBeforeManagedMutation("Thin-lens aperture");
        apertureDiameterMm = std::max(0.f, value);
        changed = true;
    });
    ImGuiManager::dragFloatRow("Focus Distance (cm)", focusDistanceCm, 10.0f, 0.1f, 100000.f, [&](float value) {
        nr::synchronizeBeforeManagedMutation("Thin-lens focus distance");
        focusDistanceCm = std::max(0.1f, value);
        changed = true;
    });
    ImGuiManager::dragFloatRow("Bokeh Bias", bokehBias, 0.01f, 0.001f, 10.f, [&](float value) {
        nr::synchronizeBeforeManagedMutation("Thin-lens bokeh bias");
        bokehBias = std::max(0.001f, value);
        changed = true;
    });
    return getSensor().renderUi() || changed;
}
