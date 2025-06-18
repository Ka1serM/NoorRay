#pragma once

#include <string>
#include "InputTracker.h"
#include "glm/glm.hpp"
#include "Scene/MeshInstance.h"
#include "Scene/SceneObject.h"
#include "Shaders/SharedStructs.h"

class PerspectiveCamera : public MeshInstance {

private:
    float aspectRatio;
    float sensorWidth;   // mm
    float sensorHeight;  // mm

    CameraData cameraData{};

    static constexpr glm::vec3 UP = glm::vec3(0, 1, 0);
public:
    PerspectiveCamera(Renderer& renderer, const std::string& name, Transform transform, float aspect, float sensorWidth, float sensorHeight, float focalLength, float aperture, float focusDistance);

    // Accessors for camera params
    float getFocalLength() const { return cameraData.focalLength; }
    float getAperture() const { return cameraData.aperture; }
    float getFocusDistance() const { return cameraData.focusDistance; }
    float getSensorWidth() const { return sensorWidth; }
    float getSensorHeight() const { return sensorHeight; }
    CameraData getCameraData() const { return cameraData; }

    void setFocalLength(float val);
    void setAperture(float val);
    void setFocusDistance(float val);
    void setSensorSize(float width, float height);

    void update(InputTracker& inputTracker, float deltaTime);
    void renderUi() override;

private:
    void updateHorizontalVertical();
    glm::vec3 calculateDirection() const;
    void updateCameraData();

    void setPosition(const glm::vec3& pos) override;
    void setRotation(const glm::quat& rot) override;
    void setRotationEuler(const glm::vec3& rot) override;
    void setScale(const glm::vec3& scale) override;
};
