#pragma once

#define GLM_ENABLE_EXPERIMENTAL

#include <memory>
#include <string>
#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp"
#include "Scene/SceneObject.h"
#include "Shaders/Shared.h"

enum class CameraProjectionType : int {
    Perspective  = CAMERA_PERSPECTIVE,
    Orthographic = CAMERA_ORTHOGRAPHIC,
    Fisheye      = CAMERA_FISHEYE,
    ThinLens     = CAMERA_THINLENS,
    Realistic    = CAMERA_REALISTIC,
};

struct CameraSettings {
    static constexpr float SensorWidthMm = 32.0f;

    float fieldOfView = 90.0f;
    float focalLengthMm = 16.0f;
    float fStop = 0.0f;
    float focusDistance = 4.0f;
    float bokehBias = 1.0f;

    static float focalLengthForFov(float fovDegrees);
    static float fovForFocalLength(float focalLength);
    void setFieldOfView(float value);
    void setFocalLength(float value);
};

struct Sensor {
    float widthMm = 32.0f;
    float heightMm = 18.0f;
    uint32_t resolutionWidth = 0;   // 0 = follow raytracer viewport size
    uint32_t resolutionHeight = 0;
};

class CameraBase : public SceneObject {
public:
    static constexpr vec3 WorldUp{0.0f, 1.0f, 0.0f};
    static constexpr vec3 LocalForward{0.0f, 0.0f, -1.0f};
    static constexpr vec3 LocalUp{0.0f, 1.0f, 0.0f};
    static constexpr vec3 LocalRight{1.0f, 0.0f, 0.0f};

    CameraBase(Scene& scene, const std::string& name, Transform transform, CameraSettings settings = {});
    CameraBase(const CameraBase& other);

    std::string getType() const override { return "Camera"; }
    void renderUi() override;
    void update();
    void onTransformUpdated() override;

    virtual CameraProjectionType getProjectionType() const = 0;
    virtual const char* getProjectionName() const = 0;
    virtual std::unique_ptr<CameraBase> recreateAs(CameraProjectionType type) const;
    virtual bool getPreferredRenderSize(uint32_t& width, uint32_t& height) const;
    virtual bool supportsDOF() const { return false; }

    CameraData getCameraData() const { return cameraData; }
    mat4 getCameraToWorld() const { return cameraToWorldMatrix; }
    virtual void populateRealisticCameraSettings(RealisticCameraSettings& realisticSettings) const;
    virtual std::string getRayLutCacheFolder() const { return {}; }
    const CameraSettings& getSettings() const { return settings; }
    CameraSettings& getSettings() { return settings; }
    const Sensor& getSensor() const { return sensor; }
    Sensor& getSensor() { return sensor; }

    mat4 getViewMatrix() const;
    virtual mat4 getProjectionMatrix() const;

    void setRenderSize(uint32_t width, uint32_t height);
    bool isDataEquivalent(const CameraData& other) const;

    void setArcballPivot(const vec3& pivot);
    const vec3& getArcballPivot() const { return arcballPivot; }
    void setArcballActive(bool active) { arcballMode = active; }
    bool getArcballActive() const { return arcballMode; }

protected:
    CameraSettings settings;
    Sensor sensor;
    CameraData cameraData{};
    mat4 cameraToWorldMatrix{1.0f};
    vec3 arcballPivot{};
    bool arcballMode = false;
    uint32_t renderWidth = 1;
    uint32_t renderHeight = 1;

    void rebuildCameraData();
    void applyDepthOfField(float focalLengthMm);
    void clearDepthOfField();
    virtual void computeProjectionData(const vec3& direction, const vec3& up, const vec3& right, float aspectRatio) = 0;
};
