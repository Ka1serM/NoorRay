#include "ThinLensCamera.h"

#include <cmath>
#include "glm/gtc/constants.hpp"

ThinLensCamera::ThinLensCamera(Scene& scene, const std::string& name, Transform transform, CameraSettings settings)
    : CameraBase(scene, name, transform, settings)
{
    rebuildCameraData();
}

ThinLensCamera::ThinLensCamera(const ThinLensCamera& other) : CameraBase(other)
{
    rebuildCameraData();
}

std::unique_ptr<SceneObject> ThinLensCamera::clone() const
{
    return std::make_unique<ThinLensCamera>(*this);
}

void ThinLensCamera::computeProjectionData(const vec3&, const vec3& up, const vec3& right, float aspectRatio)
{
    const float fovRad       = glm::radians(settings.fieldOfView);
    const float tanHalfFov   = std::tan(fovRad * 0.5f);
    const float halfWidth    = tanHalfFov;
    const float halfHeight   = halfWidth / aspectRatio;
    const float focalLengthMm = CameraSettings::SensorWidthMm / (2.0f * tanHalfFov);

    cameraData.horizontal  = right * (2.0f * halfWidth);
    cameraData.vertical    = up    * (2.0f * halfHeight);
    cameraData.focalLength = focalLengthMm * 0.001f;
    cameraData.orthoHeight = 0.0f;
    cameraData.fisheyeFov  = 0.0f;
    applyDepthOfField(focalLengthMm);
}
