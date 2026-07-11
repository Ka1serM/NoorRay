#pragma once

#include "CUDA/Annotations.h"
#include "CUDA/TaggedPointer.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "Camera/Sensor.h"
#include "Samplers/RandomSampler.h"

class PerspectiveCamera;
class ThinLensCamera;
class OrthographicCamera;
class FisheyeCamera;
class RealisticCamera;
class RossPsfCamera;

using TaggedCamera = nr::TaggedPointer<
    PerspectiveCamera,
    ThinLensCamera,
    OrthographicCamera,
    FisheyeCamera,
    RealisticCamera,
    RossPsfCamera>;

class Camera : public TaggedCamera {
public:
    using TaggedCamera::TaggedCamera;

    Sensor sensor;
    glm::mat4 cameraToWorld{1.f};
    float fieldOfView{90.f};
    float focalLengthMm{2.892f};
    float focusDistance{4.f};

    NR_CPU_GPU void transformRay(glm::vec3& origin, glm::vec3& direction) const
    {
        origin = glm::vec3(cameraToWorld * glm::vec4(origin, 1.f));
        direction = glm::normalize(glm::vec3(cameraToWorld * glm::vec4(direction, 0.f)));
    }

    Sensor& getSensor();
    const Sensor& getSensor() const;
    void setFocalLength(float focalLength);
    void setFocusDistance(float v);
    float getFocalLength() const;
    void setCameraToWorld(const glm::mat4& m);
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
