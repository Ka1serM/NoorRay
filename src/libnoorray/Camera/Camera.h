#pragma once

#include <memory>

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "Camera/Sensor.h"

class PerspectiveCamera;
class ThinLensCamera;
class OrthographicCamera;
class FisheyeCamera;
class RealisticCamera;
class Camera;

enum class CameraProjectionType : int {
    Perspective,
    Orthographic,
    Fisheye,
    ThinLens,
    Realistic,
};

class Camera {
public:
    Camera() = default;
    explicit Camera(std::unique_ptr<Sensor> sensor);
    Camera(const Camera& other);
    Camera& operator=(const Camera& other);
    virtual ~Camera();
    bool renderUi();
    Camera* ptr() { return this; }
    const Camera* ptr() const { return this; }
    explicit operator bool() const { return true; }
    template <typename Concrete> bool Is() const { return dynamic_cast<const Concrete*>(this) != nullptr; }
    template <typename Concrete> Concrete* CastOrNullptr() { return dynamic_cast<Concrete*>(this); }
    template <typename Concrete> const Concrete* CastOrNullptr() const { return dynamic_cast<const Concrete*>(this); }
    template <typename F> decltype(auto) DispatchCPU(F&& f) { return f(this); }
    template <typename F> decltype(auto) DispatchCPU(F&& f) const { return f(this); }
    std::unique_ptr<Sensor> releaseSensor();
    void setSensor(std::unique_ptr<Sensor> sensor);

    glm::mat4 cameraToWorld{1.f};
    float focalLengthMm{2.892f};
    float focusDistanceCm{500.f};
    float exposure{};

private:
    std::unique_ptr<Sensor> sensor;
public:
    Sensor& getSensor() { return *sensor; }
    const Sensor& getSensor() const { return *sensor; }
    void setFocalLengthMm(float focalLengthMm);
    void setFocusDistanceCm(float focusDistanceCm);
    void setExposure(float exposure);
    float getFocusDistanceCm() const;
    float getFocalLengthMm() const;
    void setCameraToWorld(const glm::mat4& m);
    void prepareForRender();
    Camera cloneBaseState() const;
    float focalLengthMmForFovDegrees(float fovDegrees) const;
    float fovDegreesForFocalLengthMm(float focalLengthMm) const;

};

#include "Camera/PerspectiveCamera.h"
#include "Camera/ThinLensCamera.h"
#include "Camera/OrthographicCamera.h"
#include "Camera/FisheyeCamera.h"
#include "Camera/RealisticCamera.h"
