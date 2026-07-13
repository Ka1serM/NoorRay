#include "FisheyeCamera.h"

#include <algorithm>
#include <cmath>

#include "UI/ImGuiManager.h"

bool FisheyeCamera::renderUi()
{
    bool changed = Camera::renderUi();
    ImGuiManager::dragFloatRow("F-Stop", fStop, 0.1f, 0.f, 32.f, [&](float value) {
        fStop = std::max(0.f, value);
        changed = true;
    });
    ImGuiManager::dragFloatRow("Focus Distance", focusDistance, 0.1f, 0.001f, 1000.f, [&](float value) {
        focusDistance = std::max(0.001f, value);
        changed = true;
    });
    ImGuiManager::dragFloatRow("Bokeh Bias", bokehBias, 0.01f, 0.001f, 10.f, [&](float value) {
        bokehBias = std::max(0.001f, value);
        changed = true;
    });
    const bool sensorChanged = getSensor().renderUi();
    return changed || sensorChanged;
}
