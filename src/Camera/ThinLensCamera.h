#pragma once

#include "GPU/Annotations.h"

#ifndef __CUDACC__
#include "CameraBase.h"
#endif

#ifdef __CUDACC__
#include "Kernels/SceneData.h"
#include "Kernels/Samplers.h"
#include "Kernels/Math.h"
#endif

#ifndef __CUDACC__
// Perspective camera with thin-lens depth-of-field.
// Mirrors GPU ThinLensCamera — aperture, focus distance, and bokeh bias are
// active parameters here; they are ignored by PerspectiveCamera (pinhole).
#endif

class ThinLensCamera
#ifndef __CUDACC__
    final : public CameraBase
#endif
{
public:
    float sensorScaleX{};
    float sensorScaleY{};
    float aperture{};
    float focusDistance{};

#ifndef __CUDACC__
    ThinLensCamera(Scene& scene, const std::string& name, Transform transform, CameraSettings settings = {});
    ThinLensCamera(const ThinLensCamera& other);

    std::unique_ptr<SceneObject> clone() const override;
    CameraProjectionType getProjectionType() const override { return CameraProjectionType::ThinLens; }
    const char* getProjectionName() const override { return "Thin Lens"; }
    bool supportsDOF() const override { return true; }

protected:
    void computeProjectionData(const vec3& direction, const vec3& up, const vec3& right, float aspectRatio) override;
#endif

#ifdef __CUDACC__
    NR_GPU bool generateRay(glm::vec3& origin, glm::vec3& direction,
        float nx, float ny, float, uint32_t& rng,
        const RayLutEntry*, uint32_t, int) const
    {
        direction = normalize3(glm::vec3(nx * sensorScaleX, ny * sensorScaleY, -1.0f));
        if (aperture > 0.0f) {
            const float angle = 6.28318530718f * randomFloat(rng);
            const float radius = aperture * sqrtf(randomFloat(rng));
            origin = glm::vec3(cosf(angle) * radius, sinf(angle) * radius, 0.0f);
            const glm::vec3 focusPoint = multiply(direction,
                focusDistance / fmaxf(-direction.z, 1e-5f));
            direction = normalize3(subtract(focusPoint, origin));
        } else {
            origin = glm::vec3(0.0f);
        }
        return true;
    }
#endif
};
