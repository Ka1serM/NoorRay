#pragma once

#include "Backend/Host/Platform.h"
#include <memory>
#include <optional>

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "Rendering/Camera/Sensor.h"
#include "Rendering/Sampling/RandomSampler.h"

class PerspectiveCamera;
class ThinLensCamera;
class OrthographicCamera;
class FisheyeCamera;
class RealisticCamera;
class Camera;

struct CameraSample
{
    Ray ray{};
    float weight{1.0f};
};

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

    NR_CPU_GPU Ray transformRay(const Ray& ray) const
    {
        const glm::vec3 origin = glm::vec3(
            cameraToWorld * glm::vec4(ray.origin(), 1.0f));
        const glm::vec3 direction = glm::normalize(glm::vec3(
            cameraToWorld * glm::vec4(ray.direction(), 0.0f)));
        return Ray(origin, direction);
    }

    // A missing pinhole/fisheye sample means no sensor contribution. Optical
    // cameras use it for rays stopped by their lens model and keep alpha opaque.
    NR_CPU_GPU bool invalidRayIsOpaque() const { return false; }

    NR_CPU_GPU Sensor& getSensor() { return *sensor; }
    NR_CPU_GPU const Sensor& getSensor() const { return *sensor; }
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

#include "Rendering/Camera/PerspectiveCamera.h"
#include "Rendering/Camera/ThinLensCamera.h"
#include "Rendering/Camera/OrthographicCamera.h"
#include "Rendering/Camera/FisheyeCamera.h"
#include "Rendering/Camera/RealisticCamera.h"
