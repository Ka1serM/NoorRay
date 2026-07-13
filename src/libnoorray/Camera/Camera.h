#pragma once

#include "CUDA/Annotations.h"
#include "CUDA/TaggedPointer.h"
#include "CUDA/UnifiedMemoryObject.h"
#include "CUDA/rstd/UniquePtr.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#ifndef NR_GPU_CODE
#include <memory>
#include <string>
#include <vector>
#endif

#include "Camera/Sensor.h"
#include "Samplers/RandomSampler.h"

class PerspectiveCamera;
class ThinLensCamera;
class OrthographicCamera;
class FisheyeCamera;
class RealisticCamera;
class RossPsfCamera;

enum class CameraProjectionType : int {
    Perspective,
    Orthographic,
    Fisheye,
    ThinLens,
    Realistic,
    HybridPsf,
};

using TaggedCamera = nr::TaggedPointer<
    PerspectiveCamera,
    ThinLensCamera,
    OrthographicCamera,
    FisheyeCamera,
    RealisticCamera,
    RossPsfCamera>;

class Camera : public nr::UnifiedMemoryObject, public TaggedCamera {
public:
    using TaggedCamera::TaggedCamera;
    Camera() = default;
#ifndef NR_GPU_CODE
    explicit Camera(std::unique_ptr<Sensor> sensor);
#endif
    Camera(const Camera& other);
    Camera& operator=(const Camera& other);
    virtual ~Camera();
#ifndef NR_GPU_CODE
    static std::unique_ptr<Camera> create(CameraProjectionType projectionType,
        std::unique_ptr<Sensor> sensor = {});
    std::unique_ptr<Sensor> releaseSensor();
    void setSensor(std::unique_ptr<Sensor> sensor);
#endif

    Sensor sensor;
    glm::mat4 cameraToWorld{1.f};
    float fieldOfView{90.f};
    float focalLengthMm{2.892f};
    float focusDistance{4.f};

private:
    nr::rstd::unique_ptr<Sensor> sensorOwner;
    char retainedPsfGridPath[512]{};
#ifndef NR_GPU_CODE
    void tagSensor();
#endif
public:

    NR_CPU_GPU void transformRay(glm::vec3& origin, glm::vec3& direction) const
    {
        origin = glm::vec3(cameraToWorld * glm::vec4(origin, 1.f));
        direction = glm::normalize(glm::vec3(cameraToWorld * glm::vec4(direction, 0.f)));
    }

    Sensor& getSensor();
    const Sensor& getSensor() const;
    void setFocalLength(float focalLength);
    void setFocusDistance(float v);
    float getFocusDistance() const;
    float getFocalLength() const;
    void setCameraToWorld(const glm::mat4& m);
    void prepareForRender();
    Camera cloneBaseState() const;
    bool renderUi();
    float focalLengthForFov(float fovDegrees) const;
    float fovForFocalLength(float focalLength) const;

};

#include "Camera/PerspectiveCamera.h"
#include "Camera/ThinLensCamera.h"
#include "Camera/OrthographicCamera.h"
#include "Camera/FisheyeCamera.h"
#include "Camera/RealisticCamera.h"
#include "Camera/RossPsfCamera.h"
