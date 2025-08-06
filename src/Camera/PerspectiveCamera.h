#pragma once

#include <string>
#include "glm/glm.hpp"
#include "Scene/MeshInstance.h"
#include "../Shaders/PathTracing/SharedStructs.h"

class Scene;

class PerspectiveCamera : public SceneObject {

    Scene& scene;

private:
    float aspectRatio;
    float sensorWidth;   // mm
    float sensorHeight;  // mm

    CameraData cameraData{};

    mat4 viewMatrix;
    mat4 projectionMatrix;

    static constexpr vec3 UP = vec3(0, 1, 0);
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

    mat4 getViewMatrix() const { return viewMatrix; }
    mat4 getProjectionMatrix() const { return projectionMatrix; }

    void setFocalLength(float val);
    void setAperture(float val);
    void setFocusDistance(float val);
    void setBokehBias(float val);
    void setSensorSize(float width, float height);

    void update();
    void renderUi() override;

private:
    void updateHorizontalVertical();
    void updateViewMatrix();
    void updateProjectionMatrix();
    void updateCameraData();
};
