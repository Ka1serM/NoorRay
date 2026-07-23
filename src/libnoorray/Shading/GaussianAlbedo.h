#pragma once

#include <cstddef>
#include <cstdint>

#include <glm/geometric.hpp>

#include "Raytracing/Gpu/SceneData.h"

// Gaussian colors are represented by view-dependent spherical harmonics. Keep
// their decoding shared between beauty and AOV rendering; each caller selects
// the SH order appropriate to its pass.
NR_GPU inline glm::vec3 gaussianAlbedoRgb(
    const GpuSceneData& scene,
    const Ray& ray,
    const uint32_t gaussianId,
    const SphericalHarmonicsOrder order)
{
    const __half* coefficients = scene.gaussianShCoeffs
        + static_cast<std::size_t>(gaussianId)
            * scene.gaussianShCoefficientCount
            * SphericalHarmonicsChannelCount;
    glm::vec3 rgb = glm::vec3(0.5f)
        + evaluateSphericalHarmonics(coefficients,
            order,
            glm::normalize(-ray.direction));
    rgb.x = fminf(fmaxf(rgb.x, 0.0f), 1.0f);
    rgb.y = fminf(fmaxf(rgb.y, 0.0f), 1.0f);
    rgb.z = fminf(fmaxf(rgb.z, 0.0f), 1.0f);
    return rgb;
}
