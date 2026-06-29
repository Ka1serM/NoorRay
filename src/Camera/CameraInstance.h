#pragma once

#define GLM_ENABLE_EXPERIMENTAL

#include <memory>
#include <string>
#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp"
#include "Camera/Camera.h"
#include "Scene/SceneObject.h"

enum class CameraProjectionType : int {
    Perspective,
    Orthographic,
    Fisheye,
    ThinLens,
    Realistic,
};

class CameraInstance : public SceneObject {
public:
    static constexpr vec3 WorldUp{0.f, 1.f, 0.f};
    static constexpr vec3 LocalForward{0.f, 0.f, -1.f};
    static constexpr vec3 LocalUp{0.f, 1.f, 0.f};
    static constexpr vec3 LocalRight{1.f, 0.f, 0.f};

    CameraInstance(Scene& scene, const std::string& name, Transform transform, Camera camera);
    CameraInstance(const CameraInstance& other);
    ~CameraInstance();

    std::string getType() const override { return "Camera"; }
    bool renderUi() override;
    void update();
    void onTransformUpdated() override;

    CameraProjectionType getProjectionType() const;
    const char* getProjectionName() const;

    Camera* getCamera() { return gpuCamera; }
    const Camera* getCamera() const { return gpuCamera; }
    mat4 getViewMatrix() const;
    virtual mat4 getProjectionMatrix() const;

    void markDirty();

    void setArcballPivot(const vec3& pivot);
    const vec3& getArcballPivot() const { return arcballPivot; }
    void setArcballActive(bool active) { arcballMode = active; }
    bool getArcballActive() const { return arcballMode; }

    void switchTo(CameraProjectionType type);
    void loadRealisticLens(const std::string& lensPath, const std::string& sensorPath,
                           const std::string& glassCatalogPaths);
    std::unique_ptr<SceneObject> clone() const override;


protected:
    void rebuildCamera();

private:
    Camera* gpuCamera{};           // Stable unified-memory handle to the active camera type
    vec3 arcballPivot{};
    bool arcballMode = false;

    void allocateCamera(CameraProjectionType type);
    void freeCamera();
};
