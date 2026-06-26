#include "FisheyeCamera.h"

#include <algorithm>
#include <cmath>

FisheyeCamera::FisheyeCamera(Scene& scene, const std::string& name, Transform transform, CameraSettings settings)
    : CameraBase(scene, name, transform, settings)
{
    rebuildCameraData();
}

FisheyeCamera::FisheyeCamera(const FisheyeCamera& other) : CameraBase(other)
{
    rebuildCameraData();
}

std::unique_ptr<SceneObject> FisheyeCamera::clone() const
{
    return std::make_unique<FisheyeCamera>(*this);
}

void FisheyeCamera::computeProjectionData(const vec3&, const vec3& up, const vec3& right, float)
{
    cameraData.sensorScaleX = 0.0f;
    cameraData.sensorScaleY = 0.0f;
    cameraData.focalLength = settings.focalLengthMm * 0.001f;
    cameraData.orthoHeight = 0.0f;
    cameraData.fisheyeFov = radians(std::clamp(settings.fieldOfView, 1.0f, 179.0f));
    applyDepthOfField(settings.focalLengthMm);
}
