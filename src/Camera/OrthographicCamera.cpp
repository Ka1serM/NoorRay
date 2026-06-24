#include "OrthographicCamera.h"

#include <algorithm>
#include <cmath>

OrthographicCamera::OrthographicCamera(Scene& scene, const std::string& name, Transform transform, CameraSettings settings)
    : CameraBase(scene, name, transform, settings)
{
    rebuildCameraData();
}

OrthographicCamera::OrthographicCamera(const OrthographicCamera& other) : CameraBase(other)
{
    rebuildCameraData();
}

std::unique_ptr<SceneObject> OrthographicCamera::clone() const
{
    return std::make_unique<OrthographicCamera>(*this);
}

void OrthographicCamera::computeProjectionData(const vec3&, const vec3& up, const vec3& right, float aspectRatio)
{
    constexpr float referenceHeight = 10.0f;
    constexpr float referenceFocalLengthMm = 21.0f;
    const float orthoHeight = referenceHeight * referenceFocalLengthMm / std::max(0.001f, settings.focalLengthMm);
    const float halfHeight = orthoHeight * 0.5f;
    const float halfWidth = halfHeight * aspectRatio;

    cameraData.horizontal = right * halfWidth;
    cameraData.vertical = up * halfHeight;
    cameraData.focalLength = settings.focalLengthMm * 0.001f;
    cameraData.orthoHeight = orthoHeight;
    cameraData.fisheyeFov = 0.0f;
    clearDepthOfField();
}
