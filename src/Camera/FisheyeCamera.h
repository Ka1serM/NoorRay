#pragma once

#include "GPU/Annotations.h"

#ifndef __CUDACC__
#include "CameraBase.h"
#endif

#ifdef __CUDACC__
#include "Kernels/SceneData.h"
#endif

class FisheyeCamera
#ifndef __CUDACC__
    final : public CameraBase
#endif
{
public:
    float fisheyeFov{};

#ifndef __CUDACC__
    FisheyeCamera(Scene& scene, const std::string& name, Transform transform, CameraSettings settings = {});
    FisheyeCamera(const FisheyeCamera& other);

    std::unique_ptr<SceneObject> clone() const override;
    CameraProjectionType getProjectionType() const override { return CameraProjectionType::Fisheye; }
    const char* getProjectionName() const override { return "Fisheye"; }
    bool supportsDOF() const override { return true; }

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
        const float radius = sqrtf(nx * nx + ny * ny);
        if (radius > 1.0f)
            return false;
        const float theta = radius * fisheyeFov * 0.5f;
        const float scale = radius > 1e-6f ? sinf(theta) / radius : 0.0f;
        direction = glm::vec3(nx * scale, ny * scale, -cosf(theta));
        return true;
    }
#endif
};
