#pragma once

#include <InputTracker.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "shaders/SharedStructs.h"

class PerspectiveCamera {
public:
    PerspectiveCamera(
        const glm::vec3& origin,
        const glm::vec3& lookAt,
        float aspect,
        float sensorWidth,    // mm
        float sensorHeight,   // mm
        float focalLength,    // mm
        float aperture,       // aperture diameter in mm
        float focusDistance   // meters
    );

    void update(InputTracker& inputTracker, float deltaTime);

    void setSensorSize(float width, float height);
    void setFocalLength(float focalLength);
    void setAperture(float aperture);
    void setFocusDistance(float focusDistance);

    void renderUI(bool &dirty);

    float getSensorWidth() const { return sensorWidth; }
    float getSensorHeight() const { return sensorHeight; }
    float getFocalLength() const { return cameraData.focalLength; }
    float getAperture() const { return cameraData.aperture; }
    float getFocusDistance() const { return cameraData.focusDistance; }

    const CameraData& getCameraData() const { return cameraData; }

private:
    void updateHorizontalVertical();

    glm::vec3 UP = glm::vec3(0, -1, 0);
    float aspectRatio;
    float sensorWidth;   // mm
    float sensorHeight;  // mm
    CameraData cameraData{}; // sent to GPU
};
