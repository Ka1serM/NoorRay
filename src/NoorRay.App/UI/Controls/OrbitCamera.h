#pragma once

#include <chrono>

#include "NoorRay/Core/Math.h"
#include "NoorRay/Scene/Camera.h"

namespace NoorRay::App {

//! Orbit, pan, zoom and fly navigation around a target point; the result is the transform of a camera that looks down its local -Z.
class OrbitCamera {
public:
    void frame(const Core::Bounds& bounds, float verticalFov);
    void follow(const Core::Transform& transform);
    void orbit(float yawRadians, float pitchRadians);
    //! Moves the target along the view to what the camera looks at in the scene bounds, without moving the camera, so an orbit turns about a point ahead of it.
    void pivotAhead(const Core::Bounds& bounds);
    void look(float yawRadians, float pitchRadians);
    void pan(float right, float up);
    void zoom(float steps);
    void fly(Core::float3 localDirection, float seconds);
    //! Turns the view to look at the target from one axis: 0 +X, 1 -X, 2 +Y, 3 -Y, 4 +Z, 5 -Z.
    void snap(int axis);
    //! Moves a running snap on; returns whether the view changed.
    bool animate();
    bool isAnimating() const { return snapping; }

    Core::Transform getTransform() const;
    float getDistance() const { return distance; }

private:
    Core::float3 getOffset() const;
    Core::float3 getUp() const;
    Core::float3 getEye() const { return target + getOffset() * distance; }

    Core::float3 target{0.0f, 0.0f, 0.0f};
    float yaw = Scene::defaultViewYaw;
    float pitch = Scene::defaultViewPitch;
    float distance = 5.0f;
    float flySpeed = 1.0f;
    bool snapping = false;
    float snapFromYaw = 0.0f;
    float snapFromPitch = 0.0f;
    float snapToYaw = 0.0f;
    float snapToPitch = 0.0f;
    std::chrono::steady_clock::time_point snapStart;
};

}
