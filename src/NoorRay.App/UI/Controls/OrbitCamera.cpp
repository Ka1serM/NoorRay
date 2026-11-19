#include "NoorRay.App/UI/Controls/OrbitCamera.h"

#include <algorithm>
#include <cmath>

namespace NoorRay::App {

namespace {

constexpr float pi = 3.14159265f;
constexpr float pitchLimit = 0.5f * pi;
constexpr float minDistance = 1e-4f;
constexpr float snapSeconds = 0.35f;

}

void OrbitCamera::frame(const Core::Bounds& bounds, const float verticalFov) {
    if (bounds.min.x > bounds.max.x)
        return;
    target = (bounds.min + bounds.max) * 0.5f;
    const float radius = std::max(Core::length(bounds.max - bounds.min) * 0.5f, 1e-3f);
    distance = radius / std::sin(0.5f * verticalFov);
    flySpeed = radius;
    snapping = false;
}

void OrbitCamera::follow(const Core::Transform& transform) {
    const Core::float3 eye = transform.getTranslation();
    const Core::float3 offset = Core::normalize(transform.getColumn(2));
    const Core::float3 right = transform.getColumn(0);
    pitch = std::asin(std::clamp(offset.y, -1.0f, 1.0f));
    //! The right vector keeps the yaw when the camera looks straight up or down.
    yaw = std::atan2(-right.z, right.x);
    target = eye - offset * distance;
    snapping = false;
}

void OrbitCamera::orbit(const float yawRadians, const float pitchRadians) {
    yaw += yawRadians;
    pitch = std::clamp(pitch + pitchRadians, -pitchLimit, pitchLimit);
    snapping = false;
}

void OrbitCamera::look(const float yawRadians, const float pitchRadians) {
    const Core::float3 eye = getEye();
    orbit(yawRadians, pitchRadians);
    target = eye - getOffset() * distance;
}

void OrbitCamera::pan(const float right, const float up) {
    const Core::Transform transform = getTransform();
    target = target + transform.getColumn(0) * (right * distance) + transform.getColumn(1) * (up * distance);
}

void OrbitCamera::zoom(const float steps) {
    distance = std::max(distance * std::pow(1.1f, steps), minDistance);
}

void OrbitCamera::fly(const Core::float3 localDirection, const float seconds) {
    const Core::float3 moved = getTransform().transformVector(localDirection) * (flySpeed * seconds);
    target = target + moved;
}

void OrbitCamera::snap(const int axis) {
    const float yaws[] = {0.5f * pi, -0.5f * pi, yaw, yaw, 0.0f, pi};
    const float pitches[] = {0.0f, 0.0f, pitchLimit, -pitchLimit, 0.0f, 0.0f};
    if (axis < 0 || axis >= 6)
        return;
    snapFromYaw = yaw;
    snapFromPitch = pitch;
    snapToYaw = yaw + std::remainder(yaws[axis] - yaw, 2.0f * pi);
    snapToPitch = pitches[axis];
    snapStart = std::chrono::steady_clock::now();
    snapping = true;
}

bool OrbitCamera::animate() {
    if (!snapping)
        return false;
    const float t = std::min(std::chrono::duration<float>(std::chrono::steady_clock::now() - snapStart).count() / snapSeconds, 1.0f);
    const float eased = 1.0f - (1.0f - t) * (1.0f - t);
    yaw = snapFromYaw + (snapToYaw - snapFromYaw) * eased;
    pitch = snapFromPitch + (snapToPitch - snapFromPitch) * eased;
    snapping = t < 1.0f;
    return true;
}

Core::Transform OrbitCamera::getTransform() const {
    return Core::Transform::lookAt(getEye(), target, getUp());
}

Core::float3 OrbitCamera::getOffset() const {
    return {std::cos(pitch) * std::sin(yaw), std::sin(pitch), std::cos(pitch) * std::cos(yaw)};
}

//! Perpendicular to the offset at every pitch, so a view straight down stays defined.
Core::float3 OrbitCamera::getUp() const {
    return {-std::sin(pitch) * std::sin(yaw), std::cos(pitch), -std::sin(pitch) * std::cos(yaw)};
}

}
