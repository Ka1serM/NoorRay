#pragma once

#include <memory>

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <noorrhi/noorrhi.hpp>

#include "Camera/Sensor.h"
#include "Shared/Camera.h"

class PerspectiveCamera;
class ThinLensCamera;
class OrthographicCamera;
class FisheyeCamera;
class RealisticCamera;
class Camera;
class CameraInstance;

enum class CameraProjectionType : int {
    Perspective,
    Orthographic,
    Fisheye,
    ThinLens,
    Realistic,
};

class Camera : public noorrhi::Shared<nr::graphics::Camera> {
    friend class CameraInstance;
    CameraInstance* owner_{};
public:
    Camera();
    explicit Camera(std::unique_ptr<Sensor> sensor);
    Camera(const Camera& other);
    Camera& operator=(const Camera& other);
    virtual ~Camera();
    template <typename Concrete> bool Is() const { return dynamic_cast<const Concrete*>(this) != nullptr; }
    template <typename Concrete> Concrete* CastOrNullptr() { return dynamic_cast<Concrete*>(this); }
    template <typename Concrete> const Concrete* CastOrNullptr() const { return dynamic_cast<const Concrete*>(this); }
    template <typename F> decltype(auto) DispatchCPU(F&& f) { return f(this); }
    template <typename F> decltype(auto) DispatchCPU(F&& f) const { return f(this); }
    std::unique_ptr<Sensor> releaseSensor();
    void setSensor(std::unique_ptr<Sensor> sensor);

protected:
    void notifyChanged();
    glm::mat4 cameraToWorld{1.f};
    float focalLengthMm{2.892f};
    float focusDistanceCm{500.f};
    float exposure{};
    std::unique_ptr<Sensor> sensor;

public:
    Sensor& getSensor() { return *sensor; }
    const Sensor& getSensor() const { return *sensor; }
    void setFocalLengthMm(float focalLengthMm);
    void setFocusDistanceCm(float focusDistanceCm);
    void setExposure(float exposure);
    const glm::mat4& getCameraToWorld() const { return cameraToWorld; }
    float getExposure() const;
    float getFocusDistanceCm() const;
    float getFocalLengthMm() const;
    void setCameraToWorld(const glm::mat4& m);
    void setProjectionType(CameraProjectionType projection);
    virtual float getApertureDiameterMm() const { return data.apertureDiameterMm; }
    virtual void setApertureDiameterMm(float value);
    virtual float getBokehBias() const { return 1.0f; }
    virtual void setBokehBias(float) {}
    // Rebuilds the shader-ready record from the authored camera state. All
    // camera and sensor setters call this automatically. Direct legacy field
    // edits require an explicit call to publish their changes.
    void updateData();
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
