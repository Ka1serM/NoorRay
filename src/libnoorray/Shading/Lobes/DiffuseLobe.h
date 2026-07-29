#pragma once

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include "CUDA/Annotations.h"
#include "Samplers/HemisphereSampler.h"
#include "Samplers/RandomSampler.h"
#include "Shading/BsdfClosure.h"
#include "Shading/Spectrum.h"

// Diffuse reflection shared by MaterialX's Oren-Nayar and Burley closures.
// Burley mode adds the Disney diffuse grazing/retroreflection response used
// by Standard Surface and Blender's Principled BSDF.
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

    NR_CPU_GPU BsdfSample sample(
        const glm::vec3 normal, const glm::vec3 view, RandomState& rng) const
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
        const glm::vec3 /*normal*/, const glm::vec3 /*view*/) const
    {
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
