#pragma once

#include <cuda/std/variant>

#include "Kernels/SceneData.h"

#include "Camera/PerspectiveCamera.h"
#include "Camera/ThinLensCamera.h"
#include "Camera/OrthographicCamera.h"
#include "Camera/FisheyeCamera.h"
#include "Camera/RealisticCamera.h"

using CameraGPU = cuda::std::variant<
    PerspectiveCamera,
    ThinLensCamera,
    OrthographicCamera,
    FisheyeCamera,
    RealisticCamera>;

struct GenerateRayVisitor {
    glm::vec3& origin;
    glm::vec3& direction;
    float nx;
    float ny;
    float aspect;
    uint32_t& rng;
    const RayLutEntry* rayLut;
    uint32_t pixelIndex;
    int rayLutEnabled;

    template<typename T>
    NR_CPU_GPU bool operator()(const T& cam) const
    {
        return cam.generateRay(origin, direction, nx, ny, aspect, rng,
            rayLut, pixelIndex, rayLutEnabled);
    }
};

NR_GPU inline CameraGPU makeCameraGPU(const CameraData& camera, const RayLutEntry* rayLut)
{
    switch (camera.cameraType) {
    case CameraPerspective:
        return PerspectiveCamera{ .sensorScaleX = camera.sensorScaleX, .sensorScaleY = camera.sensorScaleY };
    case CameraThinLens:
        return ThinLensCamera{ .sensorScaleX = camera.sensorScaleX, .sensorScaleY = camera.sensorScaleY,
            .aperture = camera.aperture, .focusDistance = camera.focusDistance };
    case CameraOrthographic:
        return OrthographicCamera{ .orthoHeight = camera.orthoHeight };
    case CameraFisheye:
        return FisheyeCamera{ .fisheyeFov = camera.fisheyeFov };
    case CameraRealistic:
        return RealisticCamera{};
    default:
        return PerspectiveCamera{ .sensorScaleX = camera.sensorScaleX, .sensorScaleY = camera.sensorScaleY };
    }
}
