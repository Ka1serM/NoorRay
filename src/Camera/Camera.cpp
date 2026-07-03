#include "Camera/Camera.h"

#include <algorithm>
#include <cmath>
#include <numbers>

#include "UI/ImGuiManager.h"

float Camera::focalLengthForFov(const float fovDegrees) const
{
    const float halfAngle = glm::radians(std::clamp(fovDegrees, 1.f, 179.f)) * 0.5f;
    return sensor.widthMm / (2.f * std::tan(halfAngle));
}

float Camera::fovForFocalLength(const float focalLength) const
{
    const float fov = 2.f * std::atan(sensor.widthMm / (2.f * std::max(0.001f, focalLength)))
        * (180.f / std::numbers::pi_v<float>);
    return std::clamp(fov, 1.f, 179.f);
}

void Camera::setFocalLength(const float focalLength)
{
    if (ptr()) {
        DispatchCPU([focalLength](auto* cam) {
            cam->focalLengthMm = std::max(0.001f, focalLength);
            cam->fieldOfView = cam->fovForFocalLength(cam->focalLengthMm);
        });
    } else {
        focalLengthMm = std::max(0.001f, focalLength);
        fieldOfView = fovForFocalLength(focalLengthMm);
    }
}

void Camera::setFocusDistance(const float v)
{
    const float clamped = std::max(0.001f, v);
    if (ptr())
        DispatchCPU([clamped](auto* cam) { cam->focusDistance = clamped; });
    else
        focusDistance = clamped;
}

Sensor& Camera::getSensor()
{
    if (ptr())
        return DispatchCPU([](auto* cam) -> Sensor& { return cam->sensor; });
    return sensor;
}

const Sensor& Camera::getSensor() const
{
    if (ptr())
        return DispatchCPU([](const auto* cam) -> const Sensor& { return cam->sensor; });
    return sensor;
}

bool Camera::renderUi()
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
    return changed;
}
