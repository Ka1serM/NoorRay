#include "Camera.h"

PerspectiveCamera::PerspectiveCamera(
    const glm::vec3& origin,
    const glm::vec3& lookAt,
    const glm::vec3& up,
    float aspect,
    float aperture,
    float focusDist,
    float focalLengthMM,
    float sensorWidthMM
) : origin(origin)
{
    float focalLengthMeters = focalLengthMM * 0.001f;
    lensRadius = (aperture > 0.0f) ? (focalLengthMeters / (2.0f * aperture * 0.1f)) : 0.0f;

    float fovRadians = 2.0f * atanf((sensorWidthMM * 0.5f) / focalLengthMM);
    float halfHeight = tanf(fovRadians * 0.5f);
    float halfWidth = aspect * halfHeight;

    viewVector   = normalize(origin - lookAt);
    rightVector  = normalize(cross(up, viewVector));
    upVector     = cross(viewVector, rightVector);

    lowerLeft = origin
              - rightVector * halfWidth * focusDist
              - upVector    * halfHeight * focusDist
              - viewVector  * focusDist;

    horizontal = rightVector * 2.0f * halfWidth * focusDist;
    vertical   = upVector    * 2.0f * halfHeight * focusDist;
}

PerspectiveCamera::PerspectiveCamera(
    const glm::vec3& origin,
    const glm::vec3& lookAt,
    const glm::vec3& up,
    float aspect,
    float fovDegrees
) : origin(origin), lensRadius(0.0f)
{
    float theta = glm::radians(fovDegrees);
    float halfHeight = tanf(theta * 0.5f);
    float halfWidth = aspect * halfHeight;
    float focusDist = 1.0f;

    viewVector   = normalize(origin - lookAt);
    rightVector  = normalize(cross(up, viewVector));
    upVector     = cross(viewVector, rightVector);

    lowerLeft = origin
              - rightVector * halfWidth * focusDist
              - upVector    * halfHeight * focusDist
              - viewVector  * focusDist;

    horizontal = rightVector * 2.0f * halfWidth * focusDist;
    vertical   = upVector    * 2.0f * halfHeight * focusDist;
}

void PerspectiveCamera::update(GLFWwindow *window, double mouseX, double mouseY)
{

}

CameraData PerspectiveCamera::getGpuCameraData() const {
    return CameraData(
        origin,
        lowerLeft,
        horizontal,
        vertical,
        rightVector,
        upVector,
        viewVector,
        lensRadius
    );
}

