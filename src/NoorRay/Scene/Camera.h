#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "NoorRay/Core/Math.h"
#include "NoorRay/Scene/Lens.h"
#include "NoorRay/Scene/Sensor.h"

namespace NoorRay::Scene {

enum class CameraType : std::uint8_t {
    Perspective,
    Orthographic,
    Fisheye,
    Realistic,
    //! The panoramas of Cycles, whose parameters are in Camera::panorama.
    Equirectangular,
    FisheyeEquidistant,
    FisheyeEquisolid,
    FisheyePolynomial,
    MirrorBall,
    EquiangularCubemapFace,
    CentralCylindrical,
};

//! The default view looks at its target from the front right and above.
inline constexpr float defaultViewYaw = 0.7853982f;
inline constexpr float defaultViewPitch = 0.4636476f;

//! The camera looks down its local -Z with +Y up. An f-stop of zero is a pinhole.
struct Camera {
    std::string name;
    CameraType type = CameraType::Perspective;
    Core::Transform transform;
    Core::Transform previousTransform;
    float verticalFov = 0.6981317f;
    float orthographicHeight = 10.0f;
    Sensor sensor;
    float focusDistance = 10.0f;
    float fStop = 0.0f;
    std::uint32_t blades = 0;
    float apertureRotation = 0.0f;
    //! Cycles' anamorphic bokeh ratio.
    float apertureRatio = 1.0f;
    float near = 0.01f;
    float far = 10000.0f;
    Lens lens;
    //! Equirectangular: the longitude and latitude ranges. Fisheyes: the field of view, lens, sensor width and height, then k0 to k3 of the polynomial, whose k4 takes the place of the lens. Central cylindrical: the longitude and height ranges, then the radius.
    std::array<float, 8> panorama{};
    float pixelAspect = 1.0f;
    //! The part of the image that the film covers: left, top, right and bottom from 0 to 1.
    Core::float4 window{0.0f, 0.0f, 1.0f, 1.0f};

    bool isPanoramic() const { return type >= CameraType::Equirectangular; }
    bool isRasterizable() const { return type == CameraType::Perspective || type == CameraType::Orthographic; }
    bool hasDepthOfField() const { return type == CameraType::Realistic || (fStop > 0.0f && type != CameraType::Orthographic); }
    float getFocalLength() const { return 0.5f * sensor.height / std::tan(0.5f * verticalFov); }
    float getApertureRadius() const { return fStop > 0.0f ? 0.5f * getFocalLength() / fStop : 0.0f; }
    bool operator==(const Camera&) const = default;
};

//! Sets near and far so that the whole of the bounds lies between them from where the camera stands, with near as far out as the bounds allow.
void fitClipRange(Camera& camera, const Core::Bounds& bounds);

}
