#pragma once

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include "Backend/CUDA/Annotations.h"
#include "Rendering/Sampling/HemisphereSampler.h"
#include "Rendering/Sampling/RandomSampler.h"
#include "Materials/Shading/BsdfClosure.h"
#include "Materials/Shading/Spectrum.h"

// Diffuse reflection shared by MaterialX's Oren-Nayar and Burley closures.
// Burley mode adds the Disney diffuse grazing/retroreflection response used
// by MaterialX Standard Surface and OpenPBR.
namespace nr::shading::lobes
{

struct DiffuseLobe
{
    SampledSpectrum albedo{1.0f};
    float roughness{};
    bool burley{};
    // MaterialX's translucent_bsdf: pure Lambertian *transmission* (light
    // entering one side scatters back out the other, e.g. paper/leaves) --
    // physically a different lobe from diffuse reflection, not a coat of the
    // same math. Scatters into the hemisphere behind the shading normal
    // instead of in front of it; never combined with `burley` (translucent_bsdf
    // has no roughness input to drive it).
    bool translucent{};

    NR_CPU_GPU BsdfEvaluation eval(
        const glm::vec3 normal, const glm::vec3 view,
        const glm::vec3 outgoing) const
    {
        const glm::vec3 effectiveNormal = translucent ? -normal : normal;
        const float normalOutgoing = glm::dot(effectiveNormal, outgoing);
        if (normalOutgoing <= 0.0f)
            return {};
        float diffuse = 1.0f / BsdfPi;
        if (burley)
        {
            const float normalView = fmaxf(glm::dot(normal, view), 0.0f);
            const glm::vec3 halfSum = view + outgoing;
            const float halfLengthSquared = glm::dot(halfSum, halfSum);
            const float lightHalf = halfLengthSquared > BsdfEpsilon
                ? fmaxf(glm::dot(outgoing,
                    halfSum * (1.0f / sqrtf(halfLengthSquared))), 0.0f)
                : 0.0f;
            const float fd90 =
                0.5f + 2.0f * roughness * lightHalf * lightHalf;
            const float lightGrazing = pow5(1.0f - normalOutgoing);
            const float viewGrazing = pow5(1.0f - normalView);
            diffuse *= (1.0f + (fd90 - 1.0f) * lightGrazing)
                * (1.0f + (fd90 - 1.0f) * viewGrazing);
        }
        BsdfEvaluation result;
        result.value = albedo * diffuse;
        result.pdf = normalOutgoing / BsdfPi;
        return result;
    }

    template <typename Rng>
    NR_CPU_GPU BsdfSample sample(
        const glm::vec3 normal, const glm::vec3 view, Rng& rng) const
    {
        BsdfSample result;
        result.event = translucent ? BsdfEvent::Transmission : BsdfEvent::Diffuse;
        const glm::vec3 sampleNormal = translucent ? -normal : normal;
        result.direction = nr::sampling::cosineHemisphere(
            sampleNormal, glm::vec2(randomFloat(rng), randomFloat(rng)));
        const BsdfEvaluation evaluation = eval(normal, view, result.direction);
        if (evaluation.pdf <= 0.0f)
            return {};
        result.weight = evaluation.value
            * (fabsf(glm::dot(normal, result.direction)) / evaluation.pdf);
        result.pdf = evaluation.pdf;
        return result;
    }

    NR_CPU_GPU SampledSpectrum albedoEstimate(
        const glm::vec3 normal, const glm::vec3 view) const
    {
        if (burley && !translucent)
        {
            // Cosine-integrate the Burley retro-reflection factor over the
            // incident hemisphere.  This is the closed-form polynomial of
            // the Disney diffuse furnace integral (x = 1 - N.V), avoiding a
            // per-hit 2D quadrature while retaining the grazing correction
            // needed by the composite energy guard.
            const float normalView = fmaxf(glm::dot(normal, view), 0.0f);
            const float x = 1.0f - fminf(normalView, 1.0f);
            const float x2 = x * x;
            const float x3 = x2 * x;
            const float x4 = x3 * x;
            const float x5 = x4 * x;
            const float x6 = x5 * x;
            const float a0 = 0.976190476f - 0.488095238f * x5;
            const float a1 = 0.059523810f - 0.011904762f * x
                + 1.607142857f * x5 - 0.654760863f * x6;
            const float a2 = -0.000001844f + 0.000212882f * x
                - 0.003366655f * x2 + 0.020045492f * x3
                - 0.057259231f * x4 + 0.159789532f * x5
                - 0.049970620f * x6;
            const float roughnessSquared = roughness * roughness;
            const float burleyAlbedo = fmaxf(
                a0 + roughness * a1 + roughnessSquared * a2, 0.0f);
            return albedo * burleyAlbedo;
        }
        return albedo;
    }

private:
    NR_CPU_GPU static float pow5(const float value)
    {
        const float squared = value * value;
        return squared * squared * value;
    }
};

}
