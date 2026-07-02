#pragma once

#include <cmath>

#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "CUDA/Annotations.h"
#include "Samplers/HemisphereSampler.h"
#include "Samplers/RandomSampler.h"

namespace nr::bsdf
{
inline constexpr float Pi = 3.14159265358979323846f;
inline constexpr float Epsilon = 1e-5f;

// GGX/Trowbridge-Reitz normal distribution function.
NR_CPU_GPU inline float distributionGgx(
    const glm::vec3 normal, const glm::vec3 halfVector, const float roughness)
{
    const float alpha = roughness * roughness;
    const float alpha2 = alpha * alpha;
    const float ndh = fmaxf(glm::dot(normal, halfVector), 0.0f);
    const float denominator = ndh * ndh * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / fmaxf(Pi * denominator * denominator, Epsilon);
}

// Smith masking-shadowing, single direction (G1).
NR_CPU_GPU inline float smithG1Ggx(const float nd, const float roughness)
{
    const float cosTheta = fminf(fmaxf(nd, 0.0f), 1.0f);
    if (cosTheta <= 0.0f)
        return 0.0f;
    const float alpha = roughness * roughness;
    const float alpha2 = alpha * alpha;
    const float root = sqrtf(alpha2 + (1.0f - alpha2) * cosTheta * cosTheta);
    return 2.0f * cosTheta / fmaxf(cosTheta + root, Epsilon);
}

NR_CPU_GPU inline float geometrySmith(
    const glm::vec3 normal, const glm::vec3 view, const glm::vec3 light, const float roughness)
{
    return smithG1Ggx(fmaxf(glm::dot(normal, view), 0.0f), roughness)
        * smithG1Ggx(fmaxf(glm::dot(normal, light), 0.0f), roughness);
}

// VNDF (visible normal distribution function) half-vector sampling,
// Heitz 2018, in the local frame where the shading normal is +Z.
NR_CPU_GPU inline glm::vec3 sampleGgxVndfLocal(
    const glm::vec3 view, const float roughness, const glm::vec2 sample)
{
    const float alpha = roughness * roughness;
    const glm::vec3 stretched = glm::normalize(glm::vec3(alpha * view.x, alpha * view.y, view.z));
    const float phi = 2.0f * Pi * sample.x;
    const float z = (1.0f - sample.y) * (1.0f + stretched.z) - stretched.z;
    const float sinTheta = sqrtf(fminf(fmaxf(1.0f - z * z, 0.0f), 1.0f));
    const glm::vec3 c(sinTheta * cosf(phi), sinTheta * sinf(phi), z);
    const glm::vec3 halfStretched = c + stretched;
    return glm::normalize(glm::vec3(
        alpha * halfStretched.x, alpha * halfStretched.y, halfStretched.z));
}

NR_CPU_GPU inline glm::vec3 sampleGgxHalfVector(
    const glm::vec3 view, const glm::vec3 normal, const float roughness, RandomState& rng)
{
    glm::vec3 tangent{}, bitangent{};
    nr::sampling::buildBasis(normal, tangent, bitangent);
    const glm::vec3 localView(
        glm::dot(view, tangent), glm::dot(view, bitangent), glm::dot(view, normal));
    const glm::vec3 localHalf = sampleGgxVndfLocal(
        localView, roughness, glm::vec2(randomFloat(rng), randomFloat(rng)));
    return glm::normalize(
        tangent * localHalf.x + bitangent * localHalf.y + normal * localHalf.z);
}

// VNDF sampling PDF expressed in solid angle of the reflected direction.
NR_CPU_GPU inline float pdfGgxReflection(
    const glm::vec3 view,
    const glm::vec3 normal,
    const glm::vec3 halfVector,
    const float roughness)
{
    const float ndv = fmaxf(glm::dot(normal, view), Epsilon);
    return smithG1Ggx(ndv, roughness)
        * distributionGgx(normal, halfVector, roughness) / (4.0f * ndv);
}
}
