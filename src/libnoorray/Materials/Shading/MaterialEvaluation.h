#pragma once

#include <cstdint>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

#include "Backend/CUDA/Annotations.h"

// Compact non-scattering results from material evaluation.

// Optional outputs indicated by flags.
enum class MaterialEvaluationFlags : uint32_t
{
    None = 0,
    // The program produced a shading normal (normal map or generated normal).
    // When clear, shadingNormal is undefined and the interpolated normal stands.
    HasShadingNormal = 1u << 0,
    // The program wrote a non-zero emission. Lets the integrator skip the
    // spectral upsampling of a black emitter.
    HasEmission = 1u << 1,
    // Opacity is not identically 1. Lets the integrator skip the stochastic
    // cutout test for the overwhelmingly common opaque case.
    HasCutout = 1u << 2,
};

NR_CPU_GPU inline uint32_t operator|(
    const MaterialEvaluationFlags a, const MaterialEvaluationFlags b)
{
    return static_cast<uint32_t>(a) | static_cast<uint32_t>(b);
}

// Non-scattering outputs; spectral lobes are stored in NoorRayCompositeBsdf.
struct MaterialEvaluation
{
    float opacity{1.0f};

    // View-independent closure color used by the albedo AOV. This is kept
    // alongside the non-scattering outputs because the spectral composite is
    // intentionally one-way and cannot be converted back to RGB.
    glm::vec3 albedo{0.0f};
    glm::vec3 emission{0.0f};
    float emissionStrength{0.0f};
    // Fast shadow-only estimate of the transmitted fraction. It avoids
    // constructing and preparing the full BSDF when a material is evaluated
    // only for stochastic transparent visibility.
    float transmissionEstimate{0.0f};

    glm::vec3 shadingNormal{0.0f};
    uint32_t flags{0};

    NR_CPU_GPU bool has(const MaterialEvaluationFlags flag) const
    {
        return (flags & static_cast<uint32_t>(flag)) != 0;
    }

    NR_CPU_GPU void set(const MaterialEvaluationFlags flag)
    {
        flags |= static_cast<uint32_t>(flag);
    }

};

// Geometric state supplied to material evaluation.
struct MaterialShadingContext
{
    glm::vec3 position{};
    glm::vec3 geometricNormal{};
    glm::vec3 interpolatedNormal{};
    glm::vec3 tangent{};
    glm::vec3 bitangent{};
    glm::vec2 uv{};
    glm::vec4 vertexColor{1.0f};
    glm::vec3 viewDirection{};

    glm::mat4 objectToWorld{1.0f};
    glm::mat4 worldToObject{1.0f};
    glm::mat3 normalToWorld{1.0f};

    uint32_t primitiveId{};
    float time{};
};
