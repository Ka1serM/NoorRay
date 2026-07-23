#pragma once

#include <cmath>

#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "CUDA/Annotations.h"

namespace nr::sampling
{
inline constexpr float Pi = 3.14159265358979323846f;

// Cycles' concentric square-to-disk mapping. Compared with the usual polar
// mapping it preserves the stratification of Owen-Sobol and other structured
// 2D sample sequences, which is important for both BSDF and light sampling.
NR_CPU_GPU inline glm::vec2 concentricDisk(const glm::vec2 sample)
{
    const float a = 2.0f * sample.x - 1.0f;
    const float b = 2.0f * sample.y - 1.0f;
    if (a == 0.0f && b == 0.0f)
        return glm::vec2(0.0f);

    float radius;
    float phi;
    if (a * a > b * b)
    {
        radius = a;
        phi = 0.25f * Pi * (b / a);
    }
    else
    {
        radius = b;
        phi = 0.5f * Pi - 0.25f * Pi * (a / b);
    }
    return glm::vec2(radius * cosf(phi), radius * sinf(phi));
}

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
    glm::vec2 disk = concentricDisk(sample);
    const float z = 1.0f - glm::dot(disk, disk);
    disk *= sqrtf(fmaxf(z + 1.0f, 0.0f));
    return {disk.x, disk.y, z};
}

NR_CPU_GPU inline glm::vec3 cosineHemisphere(
    const glm::vec3 normal, const glm::vec2 sample)
{
    const glm::vec2 disk = concentricDisk(sample);
    glm::vec3 tangent{}, bitangent{};
    buildBasis(normal, tangent, bitangent);
    return tangent * disk.x + bitangent * disk.y
        + normal * sqrtf(fmaxf(0.0f, 1.0f - glm::dot(disk, disk)));
}
}
