#pragma once

#define GLM_ENABLE_EXPERIMENTAL

#include <memory>
#include <string>
#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp"
#include "Camera/Camera.h"
#include "CUDA/rstd/UniquePtr.h"
#include "Scene/SceneObject.h"

class CameraInstance : public SceneObject {
public:
    struct InputState {
        float deltaTime = 0.f;
        bool accelerate = false;
        bool forward = false;
        bool backward = false;
        bool left = false;
        bool right = false;
        bool up = false;
        bool down = false;
    };

    static constexpr vec3 WorldUp{0.f, 1.f, 0.f};
    static constexpr vec3 LocalForward{0.f, 0.f, -1.f};
    static constexpr vec3 LocalUp{0.f, 1.f, 0.f};
    static constexpr vec3 LocalRight{1.f, 0.f, 0.f};

    explicit CameraInstance(std::unique_ptr<Camera> camera,
        const std::string& name = "Camera",
        Transform transform = {});
    CameraInstance(const CameraInstance& other);
    ~CameraInstance();

    std::string getType() const override { return "Camera"; }
    void update(float mouseDeltaX, float mouseDeltaY, const InputState& input);
    void onTransformUpdated() override;

    CameraProjectionType getProjectionType() const;
    const char* getProjectionName() const;

    Camera* getCamera() { return camera.get(); }
    const Camera* getCamera() const { return camera.get(); }
    Camera* getGpuCamera() { return camera.get(); }
    const Camera* getGpuCamera() const { return camera.get(); }
    mat4 getViewMatrix() const;
    virtual mat4 getProjectionMatrix() const;

    void markDirty();

    void setArcballPivot(const vec3& pivot);
    const vec3& getArcballPivot() const { return arcballPivot; }
    void setArcballActive(bool active) { arcballMode = active; }
    bool getArcballActive() const { return arcballMode; }

    void switchTo(CameraProjectionType type);
    std::unique_ptr<SceneObject> clone() const override;
    void accept(SceneObjectVisitor& visitor) override;

    void rebuildCamera();

private:
    nr::rstd::unique_ptr<Camera> camera;
    vec3 arcballPivot{};
    bool arcballMode = false;

    void allocateCamera(CameraProjectionType type);
    void tagCamera();
};
