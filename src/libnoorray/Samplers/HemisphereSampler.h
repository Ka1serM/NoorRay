#pragma once

#include <cmath>

#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "CUDA/Annotations.h"

namespace nr::sampling
{
inline constexpr float Pi = 3.14159265358979323846f;

NR_CPU_GPU inline void buildBasis(
    const glm::vec3 normal, glm::vec3& tangent, glm::vec3& bitangent)
{
    const glm::vec3 helper = fabsf(normal.z) < 0.999f
        ? glm::vec3(0.0f, 0.0f, 1.0f)
        : glm::vec3(0.0f, 1.0f, 0.0f);
    tangent = glm::normalize(glm::cross(normal, helper));
    bitangent = glm::cross(tangent, normal);
}

NR_CPU_GPU inline glm::vec3 uniformHemisphere(const glm::vec2 sample)
{
    const float z = sample.x;
    const float radius = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    const float phi = 2.0f * Pi * sample.y;
    return {radius * cosf(phi), radius * sinf(phi), z};
}

NR_CPU_GPU inline glm::vec3 cosineHemisphere(
    const glm::vec3 normal, const glm::vec2 sample)
{
    const float radius = sqrtf(sample.x);
    const float phi = 2.0f * Pi * sample.y;
    glm::vec3 tangent{}, bitangent{};
    buildBasis(normal, tangent, bitangent);
    return glm::normalize(
        tangent * (radius * cosf(phi))
        + bitangent * (radius * sinf(phi))
        + normal * sqrtf(fmaxf(0.0f, 1.0f - sample.x)));
}
}
