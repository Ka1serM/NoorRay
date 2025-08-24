#pragma once

#include <string>
#include "glm/glm.hpp"
#include "Scene/MeshInstance.h"
#include "../Shaders/SharedStructs.h"

class Scene;

class PerspectiveCamera : public SceneObject {

    Scene& scene;

private:
    float aspectRatio;
    float sensorWidth;   // mm
    float sensorHeight;  // mm

    CameraData cameraData{};

    mat4 viewMatrix{};
    mat4 projectionMatrix{};

    static constexpr auto UP = vec3(0, 1, 0);
public:
    PerspectiveCamera(Scene& scene, const std::string& name, Transform transform, float aspect, float sensorWidth, float sensorHeight, float focalLength, float aperture, float focusDistance, float bokehBias);
    
    // Accessors for camera params
    float getFocalLength() const { return cameraData.focalLength; }
    float getAperture() const { return cameraData.aperture; }
    float getFocusDistance() const { return cameraData.focusDistance; }
    float getBokehBias() const { return cameraData.bokehBias; }
    float getSensorWidth() const { return sensorWidth; }
    float getSensorHeight() const { return sensorHeight; }
    CameraData getCameraData() const { return cameraData; }

    mat4 getViewMatrix() const;
    mat4 getProjectionMatrix() const;

    void setFocalLength(float val);
    void setAperture(float val);
    void setFocusDistance(float val);
    void setBokehBias(float val);
    void setSensorSize(float width, float height);

    void update();
    void renderUi() override;

    void setPosition(const vec3& pos) override;
    void setRotation(const quat& rot) override;
    void setRotationEuler(const vec3& rot) override;

private:
    void updateHorizontalVertical();
    void updateCameraData();
};
