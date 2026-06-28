#include "Camera/Camera.h"

#include <algorithm>
#include <cmath>

#include "Camera/CameraInstance.h"
#include "UI/ImGuiManager.h"

float Camera::focalLengthForFov(const float fovDegrees) const
{
    const float halfAngle = glm::radians(std::clamp(fovDegrees, 1.f, 179.f)) * 0.5f;
    return sensor.widthMm / (2.f * std::tan(halfAngle));
}

float Camera::fovForFocalLength(const float focalLength) const
{
    const float fov = 2.f * std::atan(sensor.widthMm / (2.f * std::max(0.001f, focalLength)))
        * (180.f / glm::pi<float>());
    return std::clamp(fov, 1.f, 179.f);
}

void Camera::setFocalLength(const float focalLength)
{
    focalLengthMm = std::max(0.001f, focalLength);
    fieldOfView = fovForFocalLength(focalLengthMm);
}

bool Camera::renderUi(CameraInstance& instance, const bool supportsDepthOfField)
{
    bool changed = false;
    ImGuiManager::dragFloatRow("Field of View", fieldOfView, 0.1f, 1.f, 179.f, [&](float value) {
        fieldOfView = std::clamp(value, 1.f, 179.f);
        focalLengthMm = focalLengthForFov(fieldOfView);
        changed = true;
    });
    ImGuiManager::dragFloatRow("Focal Length", focalLengthMm, 0.1f, 0.001f, 500.f, [&](float value) {
        setFocalLength(value);
        changed = true;
    });

    if (supportsDepthOfField) {
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
    }

    if (changed)
        instance.markDirty();
    return changed;
}
