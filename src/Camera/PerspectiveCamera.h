#pragma once

#include <string>
#include "glm/glm.hpp"
#include "Scene/MeshInstance.h"
#include "../Shaders/SharedStructs.h"

class Scene;

class PerspectiveCamera : public SceneObject {

    float aspectRatio;
    float sensorWidth;   // mm
    float sensorHeight;  // mm

    CameraData cameraData{};

    mat4 viewMatrix{};
    mat4 projectionMatrix{};

    bool arcballMode = false;
    vec3 arcballPivot = vec3(0.0f);

    
    // We keep WORLD_UP for yaw calculations in the fly-camera to maintain a stable horizon,
    // but we will no longer use it to derive the camera's local orientation.
    static constexpr auto WORLD_UP = vec3(0.0f, -1.0f, 0.0f);
    static constexpr auto LOCAL_FORWARD = vec3(0.0f, 0.0f, 1.0f);
    static constexpr auto LOCAL_UP = vec3(0.0f, -1.0f, 0.0f);
    static constexpr auto LOCAL_RIGHT = vec3(1.0f, 0.0f, 0.0f);
    
public:
    
    PerspectiveCamera(Scene& scene, const std::string& name, Transform transform, float aspect, float sensorWidth, float sensorHeight, float focalLength, float aperture, float focusDistance, float bokehBias);
    PerspectiveCamera(const PerspectiveCamera& other);
    std::unique_ptr<SceneObject> clone() const override;

    // Accessors for camera params
    float getFocalLength() const { return cameraData.focalLength; }
    float getAperture() const { return cameraData.aperture; }
    float getFocusDistance() const { return cameraData.focusDistance; }
    float getBokehBias() const { return cameraData.bokehBias; }
    float getSensorWidth() const { return sensorWidth; }
    float getSensorHeight() const { return sensorHeight; }
    CameraData getCameraData() const { return cameraData; }
    bool getArcballActive() const { return arcballMode;}

    mat4 getViewMatrix() const;
    mat4 getProjectionMatrix() const;

    void setFocalLength(float val);
    void setAperture(float val);
    void setFocusDistance(float val);
    void setBokehBias(float val);
    void setSensorSize(float width, float height);
    void setArcballActive(const bool enabled) { arcballMode = enabled; }
    void setArcballPivot(const vec3& pivot) { /* arcballPivot = pivot; */ }
    vec3 getArcballPivot() const { return arcballPivot; }

    void update();
    void renderUi() override;

    void onTransformUpdated() override;
private:
    void updateHorizontalVertical();
    void updateCameraData();
};
