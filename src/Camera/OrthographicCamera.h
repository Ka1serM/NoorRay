#pragma once

#include "GPU/Annotations.h"

#ifndef __CUDACC__
#include "CameraBase.h"
#endif

#ifdef __CUDACC__
#include "Kernels/SceneData.h"
#endif

class OrthographicCamera
#ifndef __CUDACC__
    final : public CameraBase
#endif
{
public:
    float orthoHeight{};

#ifndef __CUDACC__
    OrthographicCamera(Scene& scene, const std::string& name, Transform transform, CameraSettings settings = {});
    OrthographicCamera(const OrthographicCamera& other);

    std::unique_ptr<SceneObject> clone() const override;
    CameraProjectionType getProjectionType() const override { return CameraProjectionType::Orthographic; }
    const char* getProjectionName() const override { return "Orthographic"; }

protected:
    void computeProjectionData(const vec3& direction, const vec3& up, const vec3& right, float aspectRatio) override;
#endif

#ifdef __CUDACC__
    NR_GPU bool generateRay(glm::vec3& origin, glm::vec3& direction,
        float nx, float ny, float aspect, uint32_t& rng,
        const RayLutEntry*, uint32_t, int) const
    {
        (void)rng;
        origin = glm::vec3(nx * orthoHeight * aspect * 0.5f, ny * orthoHeight * 0.5f, 0.0f);
        direction = glm::vec3(0.0f, 0.0f, -1.0f);
        return true;
    }
#endif
};
