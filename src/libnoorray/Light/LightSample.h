#pragma once

#include <cmath>
#include <cstdint>

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include "CUDA/Annotations.h"
#include "Shading/Spectrum.h"
#include "Samplers/RandomSampler.h"

inline constexpr float LightPi = 3.14159265358979323846f;

NR_CPU_GPU inline float lightSelectionLuminance(const glm::vec3 color)
{
    const float luminance = glm::dot(color, glm::vec3(0.2126f, 0.7152f, 0.0722f));
    return fmaxf(luminance, 0.0f);
}

struct LightSample
{
    glm::vec3 direction{};
    float distance{};
    SampledSpectrum radiance{};
};

NR_CPU_GPU inline void makeLightBasis(
    const glm::vec3 normal, glm::vec3& tangent, glm::vec3& bitangent)
{
    const glm::vec3 helper = fabsf(normal.z) < 0.999f
        ? glm::vec3(0.0f, 0.0f, 1.0f)
        : glm::vec3(0.0f, 1.0f, 0.0f);
    tangent = glm::normalize(glm::cross(helper, normal));
    bitangent = glm::cross(normal, tangent);
}
