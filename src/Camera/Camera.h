#pragma once

#include "GPU/Annotations.h"
#include "GPU/TaggedPointer.h"
#include "Kernels/Math.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

class CameraInstance;

struct Sensor {
    float widthMm{5.784f};
    float heightMm{3.264f};
    uint32_t resolutionWidth{1928};
    uint32_t resolutionHeight{1088};

    NR_CPU_GPU glm::uvec2 resolution() const { return {resolutionWidth, resolutionHeight}; }
    NR_CPU_GPU float aspectRatio() const { return widthMm / heightMm; }

#ifndef __CUDACC__
    void renderUi(CameraInstance& instance);
#endif
};

class PerspectiveCamera;
class ThinLensCamera;
class OrthographicCamera;
class FisheyeCamera;
class RealisticCamera;

class Camera : public nr::TaggedPointer<
    PerspectiveCamera,
    ThinLensCamera,
    OrthographicCamera,
    FisheyeCamera,
    RealisticCamera>
{
public:
    using nr::TaggedPointer<
        PerspectiveCamera, ThinLensCamera, OrthographicCamera,
        FisheyeCamera, RealisticCamera>::TaggedPointer;

    Sensor sensor;
    glm::mat4 cameraToWorld{1.f};
    float fieldOfView{90.f};
    float focalLengthMm{2.892f};
    float fStop{};
    float focusDistance{4.f};
    float bokehBias{1.f};

    NR_CPU_GPU void transformRay(glm::vec3& origin, glm::vec3& direction) const
    {
        origin = glm::vec3(cameraToWorld * glm::vec4(origin, 1.f));
        direction = normalize3(glm::vec3(cameraToWorld * glm::vec4(direction, 0.f)));
    }

#ifndef __CUDACC__
    bool renderUi(CameraInstance& instance, bool supportsDepthOfField);
    float focalLengthForFov(float fovDegrees) const;
    float fovForFocalLength(float focalLength) const;
    void setFocalLength(float focalLength);
#endif
};

#include "Camera/PerspectiveCamera.h"
#include "Camera/ThinLensCamera.h"
#include "Camera/OrthographicCamera.h"
#include "Camera/FisheyeCamera.h"
#include "Camera/RealisticCamera.h"
