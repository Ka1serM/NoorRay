#pragma once

#include "CUDA/Annotations.h"
#include "CUDA/TaggedPointer.h"
#include "CUDA/rstd/Optional.h"
#include "CUDA/rstd/UniquePtr.h"
#include "Raytracing/Gpu/Types.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "Camera/Sensor.h"
#include "Samplers/RandomSampler.h"

class PerspectiveCamera;
class ThinLensCamera;
class OrthographicCamera;
class FisheyeCamera;
class RealisticCamera;
class HybridPsfCamera;
class Camera;

enum class CameraProjectionType : int {
    Perspective,
    Orthographic,
    Fisheye,
    ThinLens,
    Realistic,
    HybridPsf,
};

using TaggedCamera = nr::TaggedObject<Camera,
    PerspectiveCamera,
    ThinLensCamera,
    OrthographicCamera,
    FisheyeCamera,
    RealisticCamera,
    HybridPsfCamera>;

class Camera : public TaggedCamera {
public:
    using TaggedCamera::TaggedCamera;
    Camera() = default;
    explicit Camera(std::unique_ptr<Sensor> sensor);
    Camera(const Camera& other);
    Camera& operator=(const Camera& other);
    virtual ~Camera();
    std::unique_ptr<Sensor> releaseSensor();
    void setSensor(std::unique_ptr<Sensor> sensor);

    glm::mat4 cameraToWorld{1.f};
    float fieldOfViewDegrees{90.f};
    float focalLengthMm{2.892f};
    float focusDistanceCm{500.f};
    float exposure{};

private:
    nr::rstd::unique_ptr<Sensor> sensor;
    char retainedPsfGridPath[512]{};
public:

    NR_CPU_GPU Ray transformRay(const Ray& ray) const
    {
        const glm::vec3 origin = glm::vec3(
            cameraToWorld * glm::vec4(ray.origin(), 1.0f));
        const glm::vec3 direction = glm::normalize(glm::vec3(
            cameraToWorld * glm::vec4(ray.direction(), 0.0f)));
        return Ray(origin, direction, ray.minDistance(), ray.maxDistance());
    }

    // A missing pinhole/fisheye sample means no sensor contribution. Optical
    // cameras use it for rays stopped by their lens model and keep alpha opaque.
    NR_CPU_GPU bool invalidRayIsOpaque() const { return false; }

    NR_CPU_GPU Sensor& getSensor() { return *sensor; }
    NR_CPU_GPU const Sensor& getSensor() const { return *sensor; }
    void setFocalLengthMm(float focalLengthMm);
    void setFocusDistanceCm(float focusDistanceCm);
    float getFocusDistanceCm() const;
    float getFocalLengthMm() const;
    void setCameraToWorld(const glm::mat4& m);
    void prepareForRender();
    Camera cloneBaseState() const;
    bool renderUi();
    float focalLengthMmForFovDegrees(float fovDegrees) const;
    float fovDegreesForFocalLengthMm(float focalLengthMm) const;

};

#include "Camera/PerspectiveCamera.h"
#include "Camera/ThinLensCamera.h"
#include "Camera/OrthographicCamera.h"
#include "Camera/FisheyeCamera.h"
#include "Camera/RealisticCamera.h"
#include "Camera/HybridPsfCamera.h"
