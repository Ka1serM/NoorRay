#pragma once

#include "GPU/Annotations.h"

#ifndef __CUDACC__
#include "CameraBase.h"
#endif

#ifdef __CUDACC__
#include "Kernels/SceneData.h"
#endif

class PerspectiveCamera
#ifndef __CUDACC__
    final : public CameraBase
#endif
{
public:
    float sensorScaleX{};
    float sensorScaleY{};

#ifndef __CUDACC__
    PerspectiveCamera(Scene& scene, const std::string& name, Transform transform, CameraSettings settings = {});
    PerspectiveCamera(Scene& scene, const std::string& name, Transform transform, float aspect, float sensorWidth, float sensorHeight, float focalLength, float aperture, float focusDistance, float bokehBias);
    PerspectiveCamera(const PerspectiveCamera& other);

    std::unique_ptr<SceneObject> clone() const override;
    CameraProjectionType getProjectionType() const override { return CameraProjectionType::Perspective; }
    const char* getProjectionName() const override { return "Perspective"; }

protected:
    void computeProjectionData(const vec3& direction, const vec3& up, const vec3& right, float aspectRatio) override;
#endif

#ifdef __CUDACC__
    NR_GPU bool generateRay(glm::vec3& origin, glm::vec3& direction,
        float nx, float ny, float, uint32_t& rng,
        const RayLutEntry*, uint32_t, int) const
    {
        (void)rng;
        origin = glm::vec3(0.0f);
        direction = normalize3(glm::vec3(nx * sensorScaleX, ny * sensorScaleY, -1.0f));
        return true;
    }
#endif
};
