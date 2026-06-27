#include "PerspectiveCamera.h"

#include <algorithm>
#include <cmath>

PerspectiveCamera::PerspectiveCamera(Scene& scene, const std::string& name, Transform transform, CameraSettings settings)
    : CameraBase(scene, name, transform, settings)
{
    rebuildCameraData();
}

PerspectiveCamera::PerspectiveCamera(Scene& scene, const std::string& name, Transform transform, float, float, float, float focalLength, float aperture, float focusDistance, float bokehBias)
    : CameraBase(scene, name, transform, {})
{
    settings.setFocalLength(focalLength);
    settings.fStop = aperture;
    settings.focusDistance = focusDistance;
    settings.bokehBias = bokehBias;
    rebuildCameraData();
}

PerspectiveCamera::PerspectiveCamera(const PerspectiveCamera& other) : CameraBase(other)
{
    rebuildCameraData();
}

std::unique_ptr<SceneObject> PerspectiveCamera::clone() const
{
    return std::make_unique<PerspectiveCamera>(*this);
}

void PerspectiveCamera::computeProjectionData(const vec3&, const vec3& up, const vec3& right, float aspectRatio)
{
    const float fovRad = radians(settings.fieldOfView);
    const float tanHalfFov = std::tan(fovRad * 0.5f);
    const float halfWidth = tanHalfFov;
    const float halfHeight = halfWidth / aspectRatio;
    const float focalLengthMm = CameraSettings::SensorWidthMm / (2.0f * tanHalfFov);

    cameraData.sensorScaleX = halfWidth;
    cameraData.sensorScaleY = halfHeight;
    cameraData.focalLength = focalLengthMm * 0.001f;
    cameraData.orthoHeight = 0.0f;
    cameraData.fisheyeFov = 0.0f;
    clearDepthOfField();
}
